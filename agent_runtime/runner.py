"""Bridge the Claude Agent SDK message stream onto normalized run events.

A UI written against one agent should work against any other, so every SDK message
becomes one of a small set of event shapes:

    {"type": "init",     "skills": [...]}
    {"type": "text",     "text": "..."}
    {"type": "tool_use", "name": "Skill", "summary": "content-skills:write-hook"}
    {"type": "result",   "cost_usd": 0.42, "num_turns": 18}
"""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)

from .registry import Run
from .spec import AgentSpec, RunOutcome, RunPlan

log = logging.getLogger(__name__)

# Inputs worth showing in a progress line, most specific first.
_SUMMARY_KEYS = ("skill", "command", "file_path", "pattern", "path", "description")


def _summarize(name: str, tool_input: dict[str, Any]) -> str:
    for key in _SUMMARY_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value if len(value) <= 200 else value[:197] + "..."
    return name


async def execute(spec: AgentSpec, plan: RunPlan, run: Run) -> None:
    """Drive one agent run to completion, updating `run` as it goes."""
    structured: dict[str, Any] | None = None
    final_text = ""
    cost: float | None = None
    turns: int | None = None
    try:
        async with ClaudeSDKClient(options=plan.options) as client:
            run._cancel = client
            await client.query(plan.prompt)
            async for message in client.receive_response():
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    run.emit({"type": "init", "skills": message.data.get("skills", [])})
                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            run.emit({"type": "text", "text": block.text})
                        elif isinstance(block, ToolUseBlock):
                            run.emit({
                                "type": "tool_use",
                                "name": block.name,
                                "summary": _summarize(block.name, block.input or {}),
                            })
                elif isinstance(message, ResultMessage):
                    structured = message.structured_output
                    final_text = message.result or ""
                    cost = message.total_cost_usd
                    turns = message.num_turns
                    run.cost_usd = cost
                    run.emit({"type": "result", "cost_usd": cost, "num_turns": turns})
                    if message.is_error:
                        run.finish("error", error=final_text or "agent reported an error")
                        return
    except Exception as exc:  # noqa: BLE001 - surface any failure as run state
        log.exception("run %s failed", run.id)
        run.finish("error", error=f"{type(exc).__name__}: {exc}")
        return
    finally:
        run._cancel = None
        if plan.cleanup is not None:
            try:
                plan.cleanup()
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                log.exception("cleanup failed for run %s", run.id)

    if run.status == "cancelled":
        return

    outcome = RunOutcome(
        plan=plan,
        structured_output=structured,
        final_text=final_text,
        events=run.events,
        cost_usd=cost,
        num_turns=turns,
    )
    try:
        run.finish("done", result=spec.collect(outcome))
    except Exception as exc:  # noqa: BLE001
        log.exception("collect failed for run %s", run.id)
        run.finish("error", error=f"could not assemble result: {type(exc).__name__}: {exc}")
