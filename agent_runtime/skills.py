"""Inline a skill's instructions instead of loading it as a plugin.

Gemini has no Claude Code-style skill/plugin mechanism: there is nothing that discovers a
SKILL.md at startup and hands it to the model on demand. So a skill's instructions are read
here and folded straight into the system prompt. The skill files themselves are untouched;
only how they reach the model changes.
"""

from __future__ import annotations

import re
from pathlib import Path

_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)


def load_skill(skill_dir: str | Path) -> str:
    """One skill's SKILL.md (frontmatter stripped) plus every file under references/."""
    skill_dir = Path(skill_dir)
    parts = [_FRONTMATTER.sub("", (skill_dir / "SKILL.md").read_text(encoding="utf-8"), count=1).strip()]
    refs = skill_dir / "references"
    if refs.is_dir():
        for f in sorted(refs.glob("*.md")):
            parts.append(f"### Reference: {f.name}\n\n{f.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(parts)


def load_skills(*skill_dirs: str | Path) -> str:
    """Several skills concatenated, each under its own heading, in the order given."""
    return "\n\n---\n\n".join(
        f"# Skill: {Path(d).name}\n\n{load_skill(d)}" for d in skill_dirs
    )
