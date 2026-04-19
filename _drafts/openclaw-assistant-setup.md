---
layout: post
title: "The Best Personal Assistant Setup for Under $100/mo"
description: "A look at the most cost-effective AI models for running a personal assistant like OpenClaw, based on Artificial Analysis leaderboards and API pricing."
date: 2026-04-19
categories: [ai, engineering]
tags: [openclaw, agents, llm]
---

Running a personal AI assistant like OpenClaw all day requires a balance of intelligence and cost. If you rely on it for coding, research, email triage, and constant background tasks, API costs can add up quickly. 

Based on the latest data from the [Artificial Analysis Leaderboard](https://artificialanalysis.ai/leaderboards/models) and current API pricing (as of April 2026), here are my top 3 recommended setups to keep your monthly bill comfortably under $100 without sacrificing capability.

### 1. The Sweet Spot (Value + Intelligence): Gemini 2.0 Flash / Gemini 2.5 Flash
For the vast majority of daily assistant tasks, you want a fast, cheap model with a massive context window. Google's Flash line remains the undisputed king of value.

*   **Intelligence:** Very high. Gemini models consistently rank near the top of the leaderboards, with the newer 2.5 and 3.x generation pushing even further.
*   **Cost:** Gemini 2.0 Flash is practically free at **$0.10 / 1M input and $0.40 / 1M output tokens**. Even the newer Gemini 2.5 Flash is incredibly cheap at **$0.15 / 1M input and $0.60 / 1M output**.
*   **Why it works:** OpenClaw relies heavily on context (reading memory files, workspace scanning). The 1M+ token context window means you can dump massive amounts of data into the prompt without breaking the bank. You could process 100 million tokens a month and still spend less than $50.

### 2. The Heavy Lifter (Complex Coding & Reasoning): Claude 3.7 Sonnet / o3-mini
When you need your assistant to write complex code, refactor a project, or perform deep reasoning, you need to step up from the "Flash" tier.

*   **Intelligence:** Top tier. Claude 3.7 Sonnet and OpenAI's o3-mini are some of the best reasoning and coding models available right now.
*   **Cost:** 
    *   **Claude 3.7 Sonnet:** $3.00 / 1M input, $15.00 / 1M output.
    *   **o3-mini:** ~$1.10 / 1M input, $4.40 / 1M output.
*   **Why it works:** While significantly more expensive than Flash, using these strategically keeps costs down. You can configure OpenClaw to use Gemini Flash as the default background engine, and explicitly request Claude 3.7 Sonnet or o3-mini (via `/reasoning` or model overrides) only when you need deep technical work done. 

### 3. The Open Source Disruptor: DeepSeek V3 / R1
If you want frontier-level performance at a fraction of the cost of Western models, DeepSeek is impossible to ignore. It has completely disrupted the API pricing market in 2025/2026.

*   **Intelligence:** Extremely high, often matching or beating GPT-4 class models on coding and reasoning benchmarks.
*   **Cost:** 
    *   **DeepSeek V3:** ~$0.27 / 1M input, $1.10 / 1M output.
    *   **DeepSeek R1:** ~$0.55 / 1M input, $2.19 / 1M output.
*   **Why it works:** DeepSeek V3 is roughly 5-15x cheaper than Claude Sonnet or GPT-4o, while offering comparable intelligence. If you are comfortable routing your data through a Chinese provider (or a third-party host serving the OSS weights), this is the most cost-effective way to get frontier-level intelligence for your assistant.

### The $100/mo Strategy
To stay under $100/mo, don't use a single expensive model for everything. Use a **hybrid approach**:
1.  **Default Engine:** Set `Gemini 2.5 Flash` or `DeepSeek V3` as your default for constant heartbeat polling, memory retrieval, and simple text tasks. This will handle 80% of your requests for pennies.
2.  **Coding / Deep Work:** Map an override for `Claude 3.7 Sonnet` or `o3-mini` for complex coding tasks or when you need guaranteed top-tier reasoning. 

By offloading the high-volume, low-complexity work to a cheap, large-context model, you save your budget for the tasks that actually require expensive compute.
