"""Drive one run against the Gemini API and normalize it onto run events.

A UI written against one agent should work against any other, so every step becomes one
of a small set of event shapes:

    {"type": "init",     "model": "gemini-flash-latest"}
    {"type": "text",     "text": "..."}
    {"type": "tool_use", "name": "read_file", "summary": "skills/write-hook/SKILL.md"}
    {"type": "result",   "usage": {"input_tokens": 1200, "output_tokens": 340}}

Two request shapes come out of one `RunPlan`, because the API does not accept `tools` and
`response_schema` in the same call:

- No tools: a single schema-constrained call. Used by agents that only write text.
- Tools set: an exploration call with automatic function calling (plain text output),
  then, if a schema was also given, a second call with no tools that asks for the
  structured result. Used by agents that need to look at files before saying anything
  structured about them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from google import genai
from google.genai import errors, types

from .registry import Run
from .spec import AgentSpec, RunOutcome, RunPlan

log = logging.getLogger(__name__)

_API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
_MAX_429_RETRIES = 3
_DEFAULT_RETRY_DELAY = 20.0


def _api_key() -> str:
    for var in _API_KEY_VARS:
        if key := os.environ.get(var):
            return key
    raise RuntimeError(
        f"no Gemini API key found. Set one of {_API_KEY_VARS} — get one free at "
        "https://aistudio.google.com/apikey"
    )


def _summarize_call(name: str, args: dict) -> str:
    for key in ("path", "pattern", "query"):
        if isinstance(args.get(key), str):
            value = args[key]
            return f"{value[:150]}{'...' if len(value) > 150 else ''}"
    return name


def _usage(response: types.GenerateContentResponse) -> dict[str, int] | None:
    meta = response.usage_metadata
    if meta is None:
        return None
    return {
        "input_tokens": meta.prompt_token_count or 0,
        "output_tokens": meta.candidates_token_count or 0,
        "total_tokens": meta.total_token_count or 0,
    }


def _emit_function_history(run: Run, response: types.GenerateContentResponse) -> None:
    """Turn the automatic-function-calling transcript into tool_use / text events."""
    for turn in response.automatic_function_calling_history or []:
        for part in turn.parts or []:
            if part.function_call is not None:
                run.emit({
                    "type": "tool_use",
                    "name": part.function_call.name,
                    "summary": _summarize_call(part.function_call.name, dict(part.function_call.args or {})),
                })
            elif part.text:
                run.emit({"type": "text", "text": part.text})


def _retry_delay_seconds(exc: errors.ClientError) -> float:
    """The server's own suggested wait, from the 429 response's RetryInfo detail.

    Confirmed shape against two live 429s during development:
    {"error": {..., "details": [..., {"@type": "...RetryInfo", "retryDelay": "31s"}]}}
    """
    try:
        for detail in exc.response_json["error"]["details"]:
            if detail.get("@type", "").endswith("RetryInfo"):
                raw = detail.get("retryDelay", "")
                if raw.endswith("s"):
                    return float(raw[:-1])
    except (KeyError, TypeError, ValueError):
        pass
    return _DEFAULT_RETRY_DELAY


async def _generate(client: genai.Client, run: Run, **kwargs) -> types.GenerateContentResponse:
    """generate_content with retry on 429.

    Automatic function calling fires its internal tool-calling round trips back-to-back
    with no pacing of its own, so a free-tier per-minute quota is routinely exhausted
    partway through a single call on anything but a trivial exploration — this is not a
    rare edge case for this design, it is the steady state. A 429 here restarts the whole
    call (there is no way to resume mid-loop), so retrying is a real cost, but the
    alternative is failing every run that happens to need more than a handful of tool
    calls.
    """
    for attempt in range(_MAX_429_RETRIES + 1):
        try:
            return await client.aio.models.generate_content(**kwargs)
        except errors.ClientError as exc:
            if exc.code != 429 or attempt == _MAX_429_RETRIES:
                raise
            delay = _retry_delay_seconds(exc)
            run.emit({
                "type": "text",
                "text": f"Rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{_MAX_429_RETRIES})...",
            })
            await asyncio.sleep(delay + 1)  # +1s buffer past the server's own estimate
    raise AssertionError("unreachable")  # the loop above always returns or raises


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
        client = genai.Client(api_key=_api_key())
        run.emit({"type": "init", "model": plan.model})
        if plan.tools:
            explore = await _generate(
                client, run,
                model=plan.model,
                contents=plan.prompt,
                config=types.GenerateContentConfig(
                    system_instruction=plan.system_instruction,
                    tools=plan.tools,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        maximum_remote_calls=plan.max_tool_calls
                    ),
                ),
            )
            _emit_function_history(run, explore)
            final_text = explore.text or ""
            usage = _usage(explore)
            if final_text.strip():
                run.emit({"type": "text", "text": final_text})

            if plan.response_schema is not None:
                summary = await _generate(
                    client, run,
                    model=plan.model,
                    contents=(
                        f"{plan.prompt}\n\n"
                        f"Here is what you found and did:\n\n{final_text}\n\n"
                        "Return the structured result now."
                    ),
                    config=types.GenerateContentConfig(
                        system_instruction=plan.system_instruction,
                        response_mime_type="application/json",
                        response_json_schema=plan.response_schema,
                    ),
                )
                structured = json.loads(summary.text)
                final_text = summary.text or final_text
                if summary.usage_metadata is not None and usage is not None:
                    usage["output_tokens"] += summary.usage_metadata.candidates_token_count or 0
                    usage["input_tokens"] += summary.usage_metadata.prompt_token_count or 0
                    usage["total_tokens"] += summary.usage_metadata.total_token_count or 0
        else:
            config = types.GenerateContentConfig(system_instruction=plan.system_instruction)
            if plan.response_schema is not None:
                config.response_mime_type = "application/json"
                config.response_json_schema = plan.response_schema
            response = await _generate(
                client, run, model=plan.model, contents=plan.prompt, config=config
            )
            final_text = response.text or ""
            usage = _usage(response)
            structured = json.loads(final_text) if plan.response_schema is not None else None
            run.emit({"type": "text", "text": final_text})

        run.usage = usage
        run.emit({"type": "result", "usage": usage})
    except Exception as exc:  # noqa: BLE001 - surface any failure as run state
        log.exception("run %s failed", run.id)
        run.finish("error", error=f"{type(exc).__name__}: {exc}")
        return
    finally:
        if plan.cleanup is not None:
            try:
                plan.cleanup()
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                log.exception("cleanup failed for run %s", run.id)

    # A cancellation is asyncio.CancelledError, a BaseException the `except Exception`
    # above does not catch, so it propagates out of this function through `finally`
    # (which still runs cleanup) and this line is never reached. The DELETE handler
    # stamps run.status == "cancelled" afterward, once this coroutine has exited.
    outcome = RunOutcome(
        plan=plan, structured_output=structured, final_text=final_text,
        events=run.events, usage=usage,
    )
    try:
        run.finish("done", result=spec.collect(outcome))
    except Exception as exc:  # noqa: BLE001
        log.exception("collect failed for run %s", run.id)
        run.finish("error", error=f"could not assemble result: {type(exc).__name__}: {exc}")
