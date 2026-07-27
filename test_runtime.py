#!/usr/bin/env python3
"""Self-check for the pieces with real logic in them. Run: python3 test_runtime.py

Covers event replay (the reason the registry buffers at all) and skill inlining.
Everything else in the package is wiring that fails loudly on import, or needs a live
Gemini API key to exercise (see runner.py — not covered here).
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from agent_runtime.registry import Registry
from agent_runtime.skills import load_skill, load_skills


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
    run.emit({"type": "result", "usage": {"input_tokens": 1}})
    run.finish("done", result={})
    await asyncio.wait_for(task, timeout=2)
    assert len(seen) == 1, seen


def test_load_skill_strips_frontmatter_and_appends_references() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        skill = Path(tmp) / "write-hook"
        (skill / "references").mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: write-hook\n---\nBody text.\n")
        (skill / "references" / "formulas.md").write_text("Formula content.\n")

        got = load_skill(skill)
        assert "name: write-hook" not in got, got  # frontmatter gone
        assert "Body text." in got
        assert "Formula content." in got
        assert "Reference: formulas.md" in got


def test_load_skills_concatenates_in_order_with_headings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, body in (("a", "First skill."), ("b", "Second skill.")):
            d = root / name
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n{body}\n")

        got = load_skills(root / "a", root / "b")
        assert got.index("First skill.") < got.index("Second skill."), got
        assert "# Skill: a" in got and "# Skill: b" in got


async def main() -> None:
    await test_replay_for_a_late_subscriber()
    await test_live_events_then_terminate()
    await test_event_emitted_with_the_finish_is_not_dropped()
    test_load_skill_strips_frontmatter_and_appends_references()
    test_load_skills_concatenates_in_order_with_headings()
    print("all checks passed")


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
