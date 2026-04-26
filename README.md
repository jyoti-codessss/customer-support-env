# Building CustomerSupportEnv: A 3-Layer Multi-Agent System with GRPO Training

**OpenEnv Hackathon Submission — Jyoti Yadav**

---

*"It started with a frustrating customer support call."*

I was on hold for 45 minutes. The agent forgot everything I said in the previous call. I had to repeat my entire problem from scratch. And at the end — wrong refund amount.

That moment made me think — **can AI actually do this better?**

Not just answer one question. But actually *remember* me. *Follow* company rules. *Escalate* when needed. And *improve* when it makes mistakes.

That's why I built **CustomerSupportEnv.**

---

## The Real Challenge Nobody Talks About

Everyone builds chatbots that answer one question perfectly. But real customer support is messy:

- The customer is angry from a previous bad experience
- The agent must verify identity *before* issuing a refund
- Sometimes the problem needs an engineer — not just an apology
- A VIP customer needs different treatment than a new user

Single-turn Q&A can't handle this. You need memory, planning, and judgment.

**So I built a 3-layer system that thinks like a team — not just one bot.**

---

## The 5 Tasks — From Simple to Brutal

| Task | Difficulty | Max Turns | Challenge |
|------|-----------|-----------|-----------|
| billing_dispute_easy | Easy | 6 | Correct refund + plan confirmation |
| technical_outage_medium | Medium | 8 | Diagnostics + escalation to Engineering |
| fraud_complaint_hard | Hard | 10 | Identity verify → refund → Fraud & Security |
| subscription_cancellation_hard | Hard | 12 | Retention negotiation with 2-year customer |
| vip_account_recovery_expert | Expert | 15 | Multi-method verify → audit → VIP escalation |

---

## Layer 1 — The Frontline Agent

*"Think of this as the junior support rep."*

It reads the customer's message, checks their history, and decides what to do. Every response comes with a confidence score:

| Confidence | What happens |
|-----------|-------------|
| >= 0.6 | Accepted directly |
| 0.4 - 0.6 | Supervisor reviews |
| < 0.4 | Orchestrator takes over |

---

## Layer 2 — The Supervisor

*"Every company has that one experienced manager who catches mistakes."*

Before any response goes out — the Supervisor checks it:

- Did the agent try to issue a refund without verifying identity? **Blocked.**
- Did the agent close the ticket without escalating a technical issue? **Rejected.**
- No empathy in the first response? **Flagged for correction.**

This is not just prompting. This is real policy enforcement.

---

## Layer 3 — The Orchestrator

*"This is the part I'm most proud of."*

When the agent fails — the Orchestrator doesn't just retry. It asks: *"What exactly went wrong?"* Then it fixes that specific thing and tries again.

It's like a coach giving targeted feedback — not just saying "try harder."

For the fraud task, the correct sequence is:

```
Turn 1: RESPOND — verify identity (DOB + last 4 of card)
Turn 2: RESPOND — confirm verified, lock/freeze account
Turn 3: REFUND — issue full $448 refund
Turn 4: ESCALATE — hand off to Fraud & Security team
```

Skip step 1 and jump to refund? Supervisor blocks it immediately.

---

## The Memory System

*"This changed everything."*

When Sarah calls back after a frustrating experience last week — the agent already knows her history, her sentiment, her previous issues. It greets her by name. It escalates faster. It shows extra empathy.

Without memory — every call starts cold.
With memory — it feels human.

---

## GRPO Training

I fine-tuned **Qwen2.5-1.5B-Instruct** using GRPO (Group Relative Policy Optimization) with LoRA directly on CustomerSupportEnv.

**How it works:**
1. Generate 4 different responses for the same customer prompt
2. Run each through the environment to get real rewards
3. Figure out which responses scored above average
4. Update the model to make high-reward responses more likely

**Reward signal:**

| Event | Reward |
|-------|--------|
| Helpful response | +0.05 |
| Empathy detected | +0.03 |
| Correct escalation | +0.20 |
| Correct refund | +0.20 |
| Refund without verification | -0.15 |
| Correct closure | +0.25 |
| Premature close | -0.20 |

---

## The Results

Honestly? I didn't expect perfect scores.

But the system scored **1.000 on all 5 tasks** — from a simple billing dispute to a VIP account security breach.

| Task | Score | Turns Used |
|------|-------|-----------|
| billing_dispute_easy | 1.000 | 3 |
| technical_outage_medium | 1.000 | 4 |
| fraud_complaint_hard | 1.000 | 4 |
| subscription_cancellation_hard | 1.000 | 4 |
| vip_account_recovery_expert | 1.000 | 4 |
| **Average** | **1.000** | -- |

---

## What I Learned

The biggest lesson? **Supervision beats prompting.**

You can write the perfect prompt — but a rule-based supervisor will always catch what the LLM misses. Especially safety-critical things like "never refund without verifying identity."

1. **Typed action spaces matter** — structured JSON is far more reliable than free-form text
2. **Memory changes everything** — returning customers need personalized handling
3. **Turn hints are powerful** — telling the LLM what action is expected dramatically improves coherence
4. **GRPO on small models works** — even Qwen2.5-1.5B learns when the reward signal is well-designed

---

## Demo Video

Watch the 2-minute walkthrough:
👉 [Watch Demo](YOUR_LOOM_LINK_HERE)

---

*"If you want to try it yourself — the live demo is one click away."*

👉 [Try it live](https://huggingface.co/spaces/Jyoti-6/customer-support-env)

---

## Links

- HuggingFace Space: https://huggingface.co/spaces/Jyoti-6/customer-support-env
- GitHub: https://github.com/jyoti-codessss/customer-support-env
- Trained Model: https://huggingface.co/Jyoti-6/customer-support-grpo-qwen
- Training Notebook: https://colab.research.google.com/drive/1Jt0O9v-0FVoRAiGAvIokV5Qgy3u9RESD

---

*Built for the OpenEnv Hackathon. All feedback welcome!*