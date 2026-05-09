#!/usr/bin/env python3
"""A tiny static benchmark for AgentSkill quality.

The benchmark scores SKILL.md files against practical criteria that matter for
agent systems: trigger clarity, executable instructions, safety boundaries,
progressive disclosure, and failure recovery.

It is deliberately dependency-free so the examples can live in a GitHub Pages
repository and be reused in blog posts.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    weight: int
    patterns: tuple[str, ...]
    description: str


@dataclass
class CheckResult:
    id: str
    title: str
    weight: int
    passed: bool
    evidence: str


@dataclass
class SkillScore:
    skill: str
    score: int
    max_score: int
    grade: str
    results: List[CheckResult]


CHECKS: tuple[Check, ...] = (
    Check(
        id="metadata",
        title="Machine-readable identity",
        weight=10,
        patterns=(r"^---[\s\S]+?^---", r"\bname\s*:", r"\bdescription\s*:"),
        description="Frontmatter exposes name and description so the agent can discover the skill.",
    ),
    Check(
        id="triggering",
        title="Clear activation boundary",
        weight=15,
        patterns=(r"when to use|use (this )?skill when|✅", r"when not to use|don't use|do not use|❌"),
        description="The skill says when it applies and when it should stay silent.",
    ),
    Check(
        id="progressive_disclosure",
        title="Progressive disclosure",
        weight=10,
        patterns=(r"quick start|common commands|workflow|steps", r"example|examples"),
        description="The top-level file gives a short path first, then deeper operational details.",
    ),
    Check(
        id="executable_protocol",
        title="Executable protocol",
        weight=20,
        patterns=(r"\x60\x60\x60(?:bash|sh|python|json|text)?", r"\b(run|call|execute|command|cli|api|tool)\b"),
        description="Instructions are concrete enough for an agent to execute, not just conceptual prose.",
    ),
    Check(
        id="inputs_outputs",
        title="Inputs and outputs are specified",
        weight=10,
        patterns=(r"\binput|arguments?|parameters?\b", r"\boutput|returns?|result\b"),
        description="The skill describes what information it needs and what it produces.",
    ),
    Check(
        id="safety",
        title="Safety and permission boundaries",
        weight=15,
        patterns=(r"ask|confirm|approval|permission", r"secret|token|password|credential|private|destructive|delete"),
        description="The skill tells the agent when to pause, especially around secrets, privacy, and destructive work.",
    ),
    Check(
        id="failure_recovery",
        title="Failure recovery",
        weight=10,
        patterns=(r"fallback|retry|recover|if .*fails|error|timeout",),
        description="The skill includes recovery behavior for failed tools, stale state, or partial results.",
    ),
    Check(
        id="rate_limits",
        title="External-system discipline",
        weight=5,
        patterns=(r"rate limit|429|retry-after|throttle|batch|serialize",),
        description="The skill avoids tight loops and respects external APIs.",
    ),
    Check(
        id="verification",
        title="Verification gate",
        weight=5,
        patterns=(r"verify|test|check|validate|dry[- ]run",),
        description="The skill asks the agent to verify before claiming success.",
    ),
)


def normalise(text: str) -> str:
    return text.lower()


def first_evidence(text: str, patterns: Iterable[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            return text[line_start:line_end].strip()[:160]
    return ""


def passes_all(text: str, patterns: Iterable[str]) -> bool:
    return all(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)


def grade(percent: float) -> str:
    if percent >= 0.90:
        return "A"
    if percent >= 0.75:
        return "B"
    if percent >= 0.60:
        return "C"
    if percent >= 0.40:
        return "D"
    return "F"


def score_skill(path: Path) -> SkillScore:
    text = path.read_text(encoding="utf-8")
    results: List[CheckResult] = []
    total = 0
    max_score = sum(check.weight for check in CHECKS)

    for check in CHECKS:
        passed = passes_all(text, check.patterns)
        if passed:
            total += check.weight
        results.append(
            CheckResult(
                id=check.id,
                title=check.title,
                weight=check.weight,
                passed=passed,
                evidence=first_evidence(text, check.patterns),
            )
        )

    return SkillScore(
        skill=str(path),
        score=total,
        max_score=max_score,
        grade=grade(total / max_score),
        results=results,
    )


def render_markdown(scores: list[SkillScore]) -> str:
    lines = ["| Skill | Score | Grade |", "|---|---:|:---:|"]
    for score in scores:
        path = Path(score.skill)
        label = f"{path.parent.name}/{path.name}"
        lines.append(f"| `{label}` | {score.score}/{score.max_score} | {score.grade} |")
    lines.append("")
    for score in scores:
        lines.append(f"### `{score.skill}`")
        lines.append("")
        for result in score.results:
            marker = "✅" if result.passed else "❌"
            evidence = f" — {result.evidence}" if result.passed and result.evidence else ""
            lines.append(f"- {marker} **{result.title}** ({result.weight} pts){evidence}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score AgentSkill SKILL.md files.")
    parser.add_argument("skills", nargs="+", type=Path, help="Path(s) to SKILL.md files")
    parser.add_argument("--markdown", action="store_true", help="Render a blog-friendly markdown table")
    args = parser.parse_args()

    scores = [score_skill(path) for path in args.skills]
    if args.markdown:
        print(render_markdown(scores))
    else:
        print(json.dumps([asdict(score) for score in scores], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
