"""Drive one run against whatever provider `RunPlan.model` names, via litellm, and
normalize it onto run events.

A UI written against one agent should work against any other, so every step becomes one
of a small set of event shapes:

    {"type": "init",     "model": "anthropic/claude-sonnet-5"}
    {"type": "text",     "text": "..."}
    {"type": "tool_use", "name": "read_file", "summary": "skills/write-hook/SKILL.md"}
    {"type": "result",   "usage": {"input_tokens": 1200, "output_tokens": 340}}

`RunPlan.model` is a litellm "provider/model" string (e.g. "anthropic/claude-sonnet-5",
"openai/gpt-4o", "gemini/gemini-flash-latest"): the prefix picks both the client and the
API key env var litellm reads, so a run works with whatever provider the caller already
has a key for.

Two request shapes come out of one `RunPlan`, because plain chat completion does not
mix well with schema-constrained output on every provider once tools are in play:

- No tools: a single call, optionally constrained to `response_schema`. Used by agents
  that only write text.
- Tools set: an exploration call, driving the model's own tool-call loop by hand (litellm
  does not auto-execute Python callables the way some native SDKs do), then, if a schema
  was also given, a second tool-free call that asks for the structured result. Used by
  agents that need to look at files before saying anything structured about them.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable
from typing import Any

import litellm

from .registry import Run
from .spec import AgentSpec, RunOutcome, RunPlan

log = logging.getLogger(__name__)

_NUM_RETRIES = 3


def _check_key_present(model: str) -> None:
    check = litellm.validate_environment(model)
    if not check["keys_in_environment"]:
        raise RuntimeError(
            f"no API key found for {model!r}. Set one of: {', '.join(check['missing_keys'])}"
        )


_ARG_LINE = re.compile(r"^\s*(\w+)\s*:\s*(.+)$")

_TYPE_MAP: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _arg_descriptions(fn: Callable) -> dict[str, str]:
    """Pull `name: description` lines out of a Google-style docstring `Args:` block,
    folding wrapped continuation lines into the same description."""
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()
    out: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for line in lines:
        if line.strip() == "Args:":
            in_args = True
            continue
        if in_args:
            if not line.strip() or not line.startswith((" ", "\t")):
                break
            if m := _ARG_LINE.match(line):
                current = m.group(1)
                out[current] = m.group(2).strip()
            elif current is not None:
                out[current] += " " + line.strip()
    return out


def _tool_schema(fn: Callable) -> dict[str, Any]:
    """Turn a plain Python function (type hints + Google-style docstring) into an
    OpenAI-style tool schema, since that's what litellm/most providers expect. The
    functions themselves (see e.g. codebase-cartographer's tools.py) are ordinary
    functions written to be read by a human, not tool declarations."""
    sig = inspect.signature(fn)
    descriptions = _arg_descriptions(fn)
    doc = inspect.getdoc(fn) or ""
    summary = doc.splitlines()[0] if doc else fn.__name__

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        properties[name] = {
            "type": _TYPE_MAP.get(param.annotation, "string"),
            "description": descriptions.get(name, ""),
        }
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": summary,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _summarize_call(name: str, args: dict) -> str:
    for key in ("path", "pattern", "query"):
        if isinstance(args.get(key), str):
            value = args[key]
            return f"{value[:150]}{'...' if len(value) > 150 else ''}"
    return name


def _usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": usage.prompt_tokens or 0,
        "output_tokens": usage.completion_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }


def _add_usage(total: dict[str, int] | None, response: Any) -> dict[str, int] | None:
    more = _usage(response)
    if more is None:
        return total
    if total is None:
        return more
    return {k: total[k] + more[k] for k in total}


async def _run_tool_loop(
    model: str, messages: list[dict[str, Any]], tools: list[Callable], max_calls: int, run: Run,
) -> tuple[str, dict[str, int] | None]:
    """Drive the model's own tool-call loop by hand, emitting tool_use/text events as it
    goes. Returns the final assistant text and accumulated token usage."""
    by_name = {fn.__name__: fn for fn in tools}
    schemas = [_tool_schema(fn) for fn in tools]
    usage: dict[str, int] | None = None
    calls_made = 0
    final_text = ""

    while True:
        response = await litellm.acompletion(
            model=model, messages=messages, tools=schemas, num_retries=_NUM_RETRIES,
        )
        usage = _add_usage(usage, response)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if message.content:
            final_text = message.content
            run.emit({"type": "text", "text": message.content})

        if not tool_calls or calls_made >= max_calls:
            return final_text, usage

        messages.append(message.model_dump())
        for call in tool_calls:
            calls_made += 1
            args = json.loads(call.function.arguments or "{}")
            fn = by_name.get(call.function.name)
            result = fn(**args) if fn is not None else f"error: no such tool: {call.function.name}"
            run.emit({
                "type": "tool_use",
                "name": call.function.name,
                "summary": _summarize_call(call.function.name, args),
            })
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": json.dumps(result),
            })
            if calls_made >= max_calls:
                break


async def execute(spec: AgentSpec, plan: RunPlan, run: Run) -> None:
    final_text = ""
    structured: dict | None = None
    usage: dict[str, int] | None = None

    try:
        # A missing key raises before anything else. It has to happen inside this try:
        # this coroutine runs as a bare asyncio.Task, and an exception raised outside
        # every except clause here would propagate to the task and vanish silently
        # (nothing awaits the task except a done-callback that only discards it), which
        # leaves the run stuck reporting "running" forever instead of "error".
        _check_key_present(plan.model)
        run.emit({"type": "init", "model": plan.model})
        messages = [
            {"role": "system", "content": plan.system_instruction},
            {"role": "user", "content": plan.prompt},
        ]

        if plan.tools:
            final_text, usage = await _run_tool_loop(
                plan.model, messages, plan.tools, plan.max_tool_calls, run,
            )

            if plan.response_schema is not None:
                summary = await litellm.acompletion(
                    model=plan.model,
                    messages=[
                        {"role": "system", "content": plan.system_instruction},
                        {
                            "role": "user",
                            "content": (
                                f"{plan.prompt}\n\n"
                                f"Here is what you found and did:\n\n{final_text}\n\n"
                                "Return the structured result now."
                            ),
                        },
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "result", "schema": plan.response_schema},
                    },
                    num_retries=_NUM_RETRIES,
                )
                structured = json.loads(summary.choices[0].message.content)
                final_text = summary.choices[0].message.content or final_text
                usage = _add_usage(usage, summary)
        else:
            kwargs: dict[str, Any] = {}
            if plan.response_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "schema": plan.response_schema},
                }
            response = await litellm.acompletion(
                model=plan.model, messages=messages, num_retries=_NUM_RETRIES, **kwargs,
            )
            final_text = response.choices[0].message.content or ""
            usage = _usage(response)
            structured = json.loads(final_text) if plan.response_schema is not None else None
            run.emit({"type": "text", "text": final_text})

        run.usage = usage
        run.emit({"type": "result", "usage": usage})

        # collect() runs here, before cleanup below, because some specs (the
        # cartographer) read their result off disk out of the same checkout that
        # cleanup is about to delete — collect first, delete second, not the other
        # way around.
        outcome = RunOutcome(
            plan=plan, structured_output=structured, final_text=final_text,
            events=run.events, usage=usage,
        )
        try:
            result = spec.collect(outcome)
        except Exception as exc:  # noqa: BLE001
            log.exception("collect failed for run %s", run.id)
            run.finish("error", error=f"could not assemble result: {type(exc).__name__}: {exc}")
            return
        run.finish("done", result=result)
    except Exception as exc:  # noqa: BLE001 - surface any failure as run state
        log.exception("run %s failed", run.id)
        run.finish("error", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Always runs, including on the two `return`s above and on cancellation
        # (asyncio.CancelledError is a BaseException, so `except Exception` above does
        # not catch it — it propagates out of this function through this `finally`,
        # and the DELETE handler stamps run.status == "cancelled" afterward).
        if plan.cleanup is not None:
            try:
                plan.cleanup()
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                log.exception("cleanup failed for run %s", run.id)
