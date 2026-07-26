#!/usr/bin/env python3
"""Self-check for the pieces with real logic in them. Run: python3 test_runtime.py

Covers event replay (the reason the registry buffers at all) and plugin skill naming.
Everything else in the package is wiring that fails loudly on import.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from agent_runtime.registry import Registry
from agent_runtime.spec import plugin_skills


async def test_replay_for_a_late_subscriber() -> None:
    run = Registry().create()
    run.emit({"type": "text", "text": "one"})
    run.emit({"type": "text", "text": "two"})
    run.finish("done", result={"ok": True})

    seen = [e async for e in run.stream()]
    assert [e["text"] for e in seen] == ["one", "two"], seen


async def test_live_events_then_terminate() -> None:
    run = Registry().create()
    seen = []

    async def consume() -> None:
        async for event in run.stream():
            seen.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    run.emit({"type": "text", "text": "a"})
    await asyncio.sleep(0)
    run.emit({"type": "text", "text": "b"})
    run.finish("done", result={})
    await asyncio.wait_for(task, timeout=2)
    assert [e["text"] for e in seen] == ["a", "b"], seen


async def test_event_emitted_with_the_finish_is_not_dropped() -> None:
    """The race the drain loop in Run.stream exists for."""
    run = Registry().create()
    seen = []

    async def consume() -> None:
        async for event in run.stream():
            seen.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    run.emit({"type": "result", "cost_usd": 1.0})
    run.finish("done", result={})
    await asyncio.wait_for(task, timeout=2)
    assert len(seen) == 1, seen


def test_plugin_skills_are_namespaced_by_manifest_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "some-checkout-dir"
        (root / ".claude-plugin").mkdir(parents=True)
        (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "my-plugin"}))
        for skill in ("write-hook", "write-script"):
            (root / "skills" / skill).mkdir(parents=True)
            (root / "skills" / skill / "SKILL.md").write_text("---\nname: x\n---\n")
        (root / "skills" / "not-a-skill").mkdir()  # no SKILL.md, must be ignored

        got = plugin_skills(str(root))
        assert got == ["my-plugin:write-hook", "my-plugin:write-script"], got


def test_plugin_skills_falls_back_to_directory_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "bare-plugin"
        (root / "skills" / "greet").mkdir(parents=True)
        (root / "skills" / "greet" / "SKILL.md").write_text("---\nname: greet\n---\n")
        assert plugin_skills(str(root)) == ["bare-plugin:greet"]


async def main() -> None:
    await test_replay_for_a_late_subscriber()
    await test_live_events_then_terminate()
    await test_event_emitted_with_the_finish_is_not_dropped()
    test_plugin_skills_are_namespaced_by_manifest_name()
    test_plugin_skills_falls_back_to_directory_name()
    print("all checks passed")


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
