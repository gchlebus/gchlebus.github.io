---
layout: post
title: "A tiny benchmark for evaluating agent skills"
excerpt: "Before testing an agent end to end, test the operational document that tells it what to do."
comments: true
---

Agent skills look deceptively simple: a short `SKILL.md`, maybe a helper script, maybe some examples.
But for an agent loop, that file is not documentation. It is part of the runtime contract.

If the skill is vague, the agent gets vague. If the skill does not say when to stop, the agent will improvise.
If it lacks recovery instructions, every transient tool failure becomes a new adventure. Fun for a demo, terrible for a system.

So the benchmark has two layers:

1. a static evaluator for the skill file itself;
2. an A/B task eval that measures how much the skill changes the output of an agent or coding model.

The static layer is boring on purpose. The A/B layer is the interesting part.

## Layer 1: the skill document score

The static evaluator scores nine things:

1. machine-readable identity
2. clear activation boundary
3. progressive disclosure
4. executable protocol
5. specified inputs and outputs
6. safety and permission boundaries
7. failure recovery
8. external-system discipline
9. verification gate

Run it like this:

```bash
python3 code/agent-skill-eval/benchmark.py \
  code/agent-skill-eval/fixtures/good_skill/SKILL.md \
  code/agent-skill-eval/fixtures/problematic_skill/SKILL.md \
  --markdown
```

Example output:

```markdown
| Skill | Score | Grade |
|---|---:|:---:|
| `good_skill/SKILL.md` | 100/100 | A |
| `problematic_skill/SKILL.md` | 0/100 | F |
```

This does not prove the skill works. It proves the skill has the minimum structure an agent needs before we spend money on model calls.

## Layer 2: does the skill improve the agent?

The more useful question is not "is this skill well written?" It is:

> Given the same task and roughly the same model, does the agent do better when the skill is available?

That is the A/B eval.

Each trial contains:

- the task prompt;
- a baseline output produced without the skill;
- an output produced with the skill available;
- a rubric;
- a judge.

For local smoke tests, the judge is deterministic. For the blog experiment, the judge should be an LLM-as-judge.

```bash
python3 code/agent-skill-eval/ab_eval.py \
  code/agent-skill-eval/agent_trials/receipt_task.json \
  code/agent-skill-eval/agent_trials/code_task.json \
  --markdown
```

Example output:

```markdown
| Task | Baseline | With skill | Δ | Judge |
|---|---:|---:|---:|---|
| `receipt_ingest` | 35/100 | 100/100 | +65 | deterministic |
| `codex_refactor` | 35/100 | 100/100 | +65 | deterministic |
```

The bundled examples are intentionally tiny:

- **receipt_ingest**: user sends a receipt and says "Do wydatków";
- **codex_refactor**: a coding agent is asked to make a small CLI change and report completion.

The second one is the Codex-style case: does a skill change the behavior of a coding agent from "done, trust me" to "implemented, tested, here is the command and result"?

## LLM-as-judge

The evaluator is provider-neutral. It expects a command that receives the judge prompt on stdin and returns JSON.

```bash
export AGENT_SKILL_EVAL_JUDGE_CMD='llm -m {model}'
python3 code/agent-skill-eval/ab_eval.py \
  code/agent-skill-eval/agent_trials/receipt_task.json \
  --judge llm \
  --model gpt-5.5 \
  --markdown
```

The judge prompt asks for independent scores for baseline and with-skill outputs. The rubric rewards:

- task success;
- instruction following;
- fit to the required tool/protocol;
- safety and permission boundaries;
- verification evidence.

That makes the eval useful for a blog post: we can show not just that a skill file looks better, but that the skill produces a measurable delta in task outcomes.

## Why this matters

Most agent benchmarks focus on the model. That makes sense, but it misses a practical layer: the artifacts we hand to the model.

A skill is good when it compresses operational judgment into a form the model can reliably follow:

- when to use it;
- when not to use it;
- what information is required;
- what exact commands or APIs to call;
- when to ask for permission;
- how to recover from partial failure;
- how to verify completion.

The static benchmark catches missing pieces fast. The A/B benchmark measures whether those pieces actually change behavior.

The code lives in [`code/agent-skill-eval/`](../code/agent-skill-eval/).
