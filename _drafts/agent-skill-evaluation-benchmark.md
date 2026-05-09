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

So I started with a deliberately boring benchmark: a static evaluator for `SKILL.md` files.

## What the benchmark checks

The first version scores nine things:

1. machine-readable identity
2. clear activation boundary
3. progressive disclosure
4. executable protocol
5. specified inputs and outputs
6. safety and permission boundaries
7. failure recovery
8. external-system discipline
9. verification gate

This is not meant to replace end-to-end evals. It is the cheap gate before them.

## Run it

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

## Why this matters

Most agent benchmarks focus on the model. That makes sense, but it misses a practical layer: the artifacts we hand to the model.

A skill is good when it compresses operational judgment into a form the model can reliably follow:

- when to use it
- when not to use it
- what information is required
- what exact commands or APIs to call
- when to ask for permission
- how to recover from partial failure
- how to verify completion

The static benchmark catches missing pieces fast. Then the interesting work begins: task-level evals where an agent receives realistic user prompts and has to decide whether to invoke the skill, follow it, recover from errors, and produce the right final state.

The code lives in [`code/agent-skill-eval/`](../code/agent-skill-eval/).
