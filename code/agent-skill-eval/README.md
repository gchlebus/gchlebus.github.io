# Agent Skill Evaluation Benchmark

This is a tiny, dependency-free benchmark for evaluating `SKILL.md` files used by agent systems.
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

## Why static first?

Agent skills are compact operational documents. A surprising amount of agent reliability comes from whether the skill
has sharp edges: when to trigger, when to refuse, what to run, what to verify, and how to recover. Static checks cannot
prove the skill works, but they catch missing structure before a model ever sees the task.
