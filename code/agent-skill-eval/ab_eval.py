#!/usr/bin/env python3
"""A/B evaluator for measuring the effect of an AgentSkill on task output.

This script compares paired outputs from the same task:
- baseline: model/agent response without the skill
- with_skill: model/agent response after the skill was available

The default judge is deterministic and dependency-free. For the blog experiment,
use `--judge llm --model ...` to ask an LLM judge for a rubric score. The LLM
path is intentionally isolated so the benchmark can still run in CI without API
keys.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_RUBRIC = {
    "task_success": 35,
    "instruction_following": 20,
    "tool_or_protocol_fit": 20,
    "safety_and_permissions": 15,
    "verification": 10,
}


@dataclass
class VariantScore:
    score: int
    max_score: int
    reasons: dict[str, str]


@dataclass
class TrialResult:
    task_id: str
    baseline: VariantScore
    with_skill: VariantScore
    delta: int
    judge: str


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def deterministic_score(task: dict[str, Any], output: str) -> VariantScore:
    """Score an output using lightweight textual evidence.

    This is not meant to replace LLM judging. It provides a reproducible smoke
    test and a fallback when no judge model is configured.
    """
    rubric = task.get("rubric", DEFAULT_RUBRIC)
    expected = task.get("expected_evidence", {})
    forbidden = task.get("forbidden_evidence", {})
    reasons: dict[str, str] = {}
    score = 0
    text = output.lower()

    for criterion, weight in rubric.items():
        positives = [p.lower() for p in expected.get(criterion, [])]
        negatives = [p.lower() for p in forbidden.get(criterion, [])]
        positive_hit = not positives or any(p in text for p in positives)
        negative_hit = any(p in text for p in negatives)
        if positive_hit and not negative_hit:
            score += int(weight)
            reasons[criterion] = "pass"
        elif negative_hit:
            reasons[criterion] = "fail: forbidden evidence present"
        else:
            reasons[criterion] = "fail: expected evidence missing"

    return VariantScore(score=score, max_score=sum(map(int, rubric.values())), reasons=reasons)


def build_judge_prompt(task: dict[str, Any], baseline: str, with_skill: str) -> str:
    rubric = task.get("rubric", DEFAULT_RUBRIC)
    return f"""You are an impartial evaluator of agent outputs.

Task:
{task['prompt']}

Skill under test:
{task.get('skill_summary', 'No summary provided.')}

Rubric, total {sum(rubric.values())} points:
{json.dumps(rubric, indent=2)}

Evaluate both outputs independently. Reward correct task completion, faithful use
of the skill/protocol, safe permission boundaries, concrete verification, and
minimal hallucination. Penalize hidden assumptions, missing required fields,
unsafe external actions, and claims of success without evidence.

Return ONLY valid JSON in this schema:
{{
  "baseline": {{"score": 0, "reasons": {{"criterion": "reason"}}}},
  "with_skill": {{"score": 0, "reasons": {{"criterion": "reason"}}}}
}}

Baseline output:
---
{baseline}
---

With-skill output:
---
{with_skill}
---
"""


def run_llm_judge(task: dict[str, Any], baseline: str, with_skill: str, model: str) -> tuple[VariantScore, VariantScore]:
    """Run an LLM judge via an external command.

    Configure AGENT_SKILL_EVAL_JUDGE_CMD, for example:
      AGENT_SKILL_EVAL_JUDGE_CMD='llm -m {model}'

    The prompt is sent on stdin. The command must print JSON matching the schema
    from build_judge_prompt(). This keeps the script provider-neutral: Codex,
    OpenAI-compatible CLIs, `llm`, or internal tooling can all be adapted.
    """
    cmd_template = os.environ.get("AGENT_SKILL_EVAL_JUDGE_CMD")
    if not cmd_template:
        raise RuntimeError("Set AGENT_SKILL_EVAL_JUDGE_CMD to use --judge llm")
    prompt = build_judge_prompt(task, baseline, with_skill)
    cmd = cmd_template.format(model=model)
    proc = subprocess.run(cmd, input=prompt, text=True, shell=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"judge command failed: {proc.stderr.strip()}")
    payload = extract_json(proc.stdout)
    rubric = task.get("rubric", DEFAULT_RUBRIC)
    max_score = sum(map(int, rubric.values()))
    return (
        VariantScore(int(payload["baseline"]["score"]), max_score, payload["baseline"].get("reasons", {})),
        VariantScore(int(payload["with_skill"]["score"]), max_score, payload["with_skill"].get("reasons", {})),
    )


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in judge output")
    return json.loads(match.group(0))


def evaluate_trial(task_path: Path, judge: str, model: str) -> TrialResult:
    task = load_json(task_path)
    base_dir = task_path.parent
    baseline = read(base_dir / task["baseline_output"])
    with_skill = read(base_dir / task["with_skill_output"])

    if judge == "deterministic":
        baseline_score = deterministic_score(task, baseline)
        skill_score = deterministic_score(task, with_skill)
    elif judge == "llm":
        baseline_score, skill_score = run_llm_judge(task, baseline, with_skill, model)
    else:
        raise ValueError(f"Unsupported judge: {judge}")

    return TrialResult(
        task_id=task["id"],
        baseline=baseline_score,
        with_skill=skill_score,
        delta=skill_score.score - baseline_score.score,
        judge=judge,
    )


def render_markdown(results: list[TrialResult]) -> str:
    lines = ["| Task | Baseline | With skill | Δ | Judge |", "|---|---:|---:|---:|---|"]
    for r in results:
        lines.append(
            f"| `{r.task_id}` | {r.baseline.score}/{r.baseline.max_score} | "
            f"{r.with_skill.score}/{r.with_skill.max_score} | {r.delta:+d} | {r.judge} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs with-skill agent outputs.")
    parser.add_argument("tasks", nargs="+", type=Path, help="Path(s) to task.json files")
    parser.add_argument("--judge", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--model", default="gpt-5.5", help="Judge model name when --judge llm is used")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    results = [evaluate_trial(path, args.judge, args.model) for path in args.tasks]
    if args.markdown:
        print(render_markdown(results))
    else:
        print(json.dumps([asdict(r) for r in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
