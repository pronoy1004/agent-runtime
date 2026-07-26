"""In-memory run store with event replay.

SSE has no replay of its own, and these runs last minutes, so a client that connects
late or reconnects after a drop needs the events it missed. Every event is buffered on
the run and handed out from a caller-held cursor.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["running", "done", "error", "cancelled"]

TERMINAL: frozenset[str] = frozenset({"done", "error", "cancelled"})


@dataclass
class Run:
    id: str
    status: Status = "running"
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    cost_usd: float | None = None
    artifact: bytes | None = None
    _tick: asyncio.Event = field(default_factory=asyncio.Event)
    _cancel: Any = None
    """Set to the live ClaudeSDKClient so DELETE can interrupt it."""

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self._tick.set()

    def finish(self, status: Status, *, result: dict | None = None, error: str | None = None) -> None:
        self.status = status
        self.result = result
        self.error = error
        self._tick.set()

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Buffered events from the beginning, then live ones until the run ends."""
        cursor = 0
        while True:
            while cursor < len(self.events):
                yield self.events[cursor]
                cursor += 1
            if self.status in TERMINAL:
                # Drain anything appended between the last check and the status flip.
                while cursor < len(self.events):
                    yield self.events[cursor]
                    cursor += 1
                return
            self._tick.clear()
            await self._tick.wait()

    def public(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "cost_usd": self.cost_usd,
            "created_at": self.created_at,
        }


class Registry:
    """Runs live in process memory and are evicted by age.

    One process, no shared store. That is deliberate: a run is bound to a live
    ClaudeSDKClient in this process, so it could not be served from another replica
    anyway. Run one instance per agent, or add sticky routing before scaling out.
    """

    def __init__(self, ttl_seconds: float = 3600, max_runs: int = 200) -> None:
        self._runs: dict[str, Run] = {}
        self._ttl = ttl_seconds
        self._max = max_runs

    def create(self) -> Run:
        self._evict()
        run = Run(id=uuid.uuid4().hex)
        self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def _evict(self) -> None:
        now = time.time()
        stale = [
            rid for rid, r in self._runs.items()
            if r.status in TERMINAL and now - r.created_at > self._ttl
        ]
        for rid in stale:
            del self._runs[rid]
        if len(self._runs) > self._max:
            done = sorted(
                (r for r in self._runs.values() if r.status in TERMINAL),
                key=lambda r: r.created_at,
            )
            for r in done[: len(self._runs) - self._max]:
                self._runs.pop(r.id, None)
