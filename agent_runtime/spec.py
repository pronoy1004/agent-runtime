"""The contract an agent service implements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

DEFAULT_MODEL = "gemini-flash-latest"
"""A rolling alias rather than a pinned version: Google retires dated models out from
under existing keys (confirmed live — gemini-2.5-flash 404s as "no longer available to
new users" while still appearing in models.list()), and the alias tracks whatever
replaces it."""


@dataclass
class RunPlan:
    """Everything one run needs.

    Two shapes fall out of this:

    - No `tools`: one `generate_content` call, optionally schema-constrained. This is
      the content-producer shape — it only ever writes text.
    - `tools` set: an exploration call with automatic function calling and no schema
      (Gemini does not accept `tools` and `response_schema` together), followed by a
      second, tool-free call that asks for the structured result, if `response_schema`
      is also set. This is the cartographer shape — it needs to read files before it
      can say anything structured about them.
    """

    prompt: str
    system_instruction: str
    tools: list[Callable[..., Any]] | None = None
    max_tool_calls: int = 40
    response_schema: dict[str, Any] | None = None
    model: str = DEFAULT_MODEL
    cleanup: Callable[[], None] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    """Scratch space for the spec. `collect` reads it, the runtime never touches it."""


@dataclass
class RunOutcome:
    """What the run produced, handed to `collect`."""

    plan: RunPlan
    structured_output: dict[str, Any] | None
    final_text: str
    events: list[dict[str, Any]]
    usage: dict[str, int] | None


@dataclass
class AgentSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    build: Callable[[Any], RunPlan]
    collect: Callable[[RunOutcome], dict[str, Any]]
    extra_routes: Callable[[Any], None] | None = None
    """Optional hook to mount agent-specific routes. Receives the FastAPI app."""
