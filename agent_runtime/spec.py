"""The contract an agent service implements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions
from pydantic import BaseModel


@dataclass
class RunPlan:
    """Everything one run needs: what to ask, how to configure the agent, how to clean up."""

    prompt: str
    options: ClaudeAgentOptions
    cleanup: Callable[[], None] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    """Scratch space for the spec. `collect` reads it, the runtime never touches it."""


@dataclass
class RunOutcome:
    """What the agent loop produced, handed to `collect`."""

    plan: RunPlan
    structured_output: dict[str, Any] | None
    final_text: str
    events: list[dict[str, Any]]
    cost_usd: float | None
    num_turns: int | None


@dataclass
class AgentSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    build: Callable[[Any], RunPlan]
    collect: Callable[[RunOutcome], dict[str, Any]]
    extra_routes: Callable[[Any], None] | None = None
    """Optional hook to mount agent-specific routes. Receives the FastAPI app."""


def plugin_skills(plugin_root: str) -> list[str]:
    """Every skill in a plugin directory, namespaced the way the SDK exposes them.

    Scoping `skills` to just this list keeps the bundled Claude Code skills out of the
    agent's context, so a run only sees the pipeline it is supposed to drive.
    """
    from pathlib import Path

    root = Path(plugin_root)
    name = root.name
    manifest = root / ".claude-plugin" / "plugin.json"
    if manifest.is_file():
        import json

        name = json.loads(manifest.read_text()).get("name", name)
    return sorted(
        f"{name}:{d.name}" for d in (root / "skills").iterdir()
        if (d / "SKILL.md").is_file()
    )
