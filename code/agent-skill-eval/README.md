# Agent Skill Evaluation Benchmark

This is a tiny benchmark for evaluating `SKILL.md` files used by agent systems.
It has two layers:

1. a dependency-free static score for the skill document itself;
2. an A/B task eval that compares agent output without the skill vs. with the skill, optionally judged by an LLM.

It is intentionally small enough to paste into a blog post and strict enough to catch common skill-writing failures.

## What it measures

The benchmark checks whether a skill gives an agent the operational contract it needs:

- machine-readable identity
- clear activation and non-activation boundaries
- progressive disclosure
- executable commands or protocols
- explicit inputs and outputs
- safety / permission boundaries
- failure recovery
- external-system discipline
- verification before success claims

This is not a semantic model benchmark. It is a static benchmark for the *artifact* that guides the agent.
That makes it cheap, deterministic, and useful as a first gate before running expensive end-to-end agent evals.

## Run

```bash
python3 code/agent-skill-eval/benchmark.py \
  code/agent-skill-eval/fixtures/good_skill/SKILL.md \
  code/agent-skill-eval/fixtures/problematic_skill/SKILL.md \
  --markdown
```

Expected shape:

```text
| Skill | Score | Grade |
|---|---:|:---:|
| `good_skill/SKILL.md` | 100/100 | A |
| `problematic_skill/SKILL.md` | 0/100 | F |
```

## Use against a real skill

```bash
python3 code/agent-skill-eval/benchmark.py /path/to/skill/SKILL.md --markdown
```

## A/B task eval

The second layer measures whether the skill changes the output of an agent/coding model. Each trial has:

- the task prompt;
- a baseline output produced without the skill;
- an output produced with the skill available;
- a rubric for scoring the difference.

Run the bundled examples:

```bash
python3 code/agent-skill-eval/ab_eval.py \
  code/agent-skill-eval/agent_trials/receipt_task.json \
  code/agent-skill-eval/agent_trials/code_task.json \
  --markdown
```

Expected shape:

```text
| Task | Baseline | With skill | Δ | Judge |
|---|---:|---:|---:|---|
| `receipt_ingest` | 35/100 | 100/100 | +65 | deterministic |
| `codex_refactor` | 35/100 | 100/100 | +65 | deterministic |
```

For the blog experiment, replace the deterministic smoke judge with an LLM-as-judge:

```bash
export AGENT_SKILL_EVAL_JUDGE_CMD='llm -m {model}'
python3 code/agent-skill-eval/ab_eval.py \
  code/agent-skill-eval/agent_trials/receipt_task.json \
  --judge llm \
  --model gpt-5.5 \
  --markdown
```

The judge command is provider-neutral: any CLI that accepts the prompt on stdin and returns JSON can be used, including a Codex/OpenAI-compatible wrapper.

## Why static first?

Agent skills are compact operational documents. A surprising amount of agent reliability comes from whether the skill
has sharp edges: when to trigger, when to refuse, what to run, what to verify, and how to recover. Static checks cannot
prove the skill works, but they catch missing structure before a model ever sees the task.

The A/B layer is where the actual blog claim should come from: same task, same model family, skill absent/present, then score the delta with a rubric and an LLM judge.
