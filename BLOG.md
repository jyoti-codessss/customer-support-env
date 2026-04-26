# Building CustomerSupportEnv: A 3-Layer Multi-Agent System with GRPO Training

**OpenEnv Hackathon Submission — Abhaykumar jha /Jyoti Yadav**

---

## The Problem

Customer support is one of the hardest real-world tasks for AI agents. It requires:

- **Multi-turn reasoning** across 6–15 conversation turns
- **Long-horizon planning** — knowing what to do 3 steps ahead
- **Policy compliance** — never issue a refund without verifying identity
- **World modeling** — remembering frustrated customers from previous sessions
- **Self-improvement** — learning from mistakes in real-time

Most LLM demos handle single-turn Q&A. CustomerSupportEnv goes much deeper.

---

## What I Built

**CustomerSupportEnv** is an OpenEnv-compliant simulation environment where a 3-layer AI agent system resolves customer complaints through typed actions — responding, escalating, issuing refunds, and closing tickets.

### 5 Tasks — Easy to Expert

| Task | Difficulty | Max Turns | Challenge |
|------|-----------|-----------|-----------|
| `billing_dispute_easy` | Easy | 6 | Correct refund + plan confirmation |
| `technical_outage_medium` | Medium | 8 | Diagnostics + escalation to Engineering |
| `fraud_complaint_hard` | Hard | 10 | Identity verify → refund → Fraud & Security |
| `subscription_cancellation_hard` | Hard | 12 | Retention negotiation with 2-year customer |
| `vip_account_recovery_expert` | Expert | 15 | Multi-method verify → audit → VIP escalation |

### Final Score: **1.000 / 1.000** on all 5 tasks ✅

---

## The Architecture

```
┌──────────────────────────────────────────┐
│           CUSTOMER INPUT                 │
│  5 task types across 3 difficulty levels │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│         WORLD MODEL (Memory)             │
│  Customer history • Sentiment tracking   │
│  Repeat issue detection • VIP status     │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│    LAYER 1 — CUSTOMER AGENT              │
│  LLM-powered response generation         │
│  meta-llama/Llama-3.1-8B-Instruct        │
│  Confidence scoring (0.0 → 1.0)          │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│    LAYER 2 — SUPERVISOR AGENT            │
│  Policy compliance checks                │
│  Fraud safety (identity before refund)   │
│  Empathy verification                    │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│    LAYER 3 — ORCHESTRATOR AGENT          │
│  Self-improvement loop                   │
│  Failure analysis → targeted fixes       │
│  Up to 3 retries per task                │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│         GRPO REWARD SIGNAL               │
│  JSON format • Empathy • Action • Result │
└──────────────────────────────────────────┘
```

---

## System 1: Hierarchical Multi-Agent Architecture

Three agents collaborate on every interaction:

**Layer 1 — CustomerAgent**
The frontline LLM. Receives customer message + memory context + turn hint, generates a JSON action with confidence score and reasoning.

```json
{
  "action_type": "refund",
  "response_text": "I sincerely apologize for the incorrect charge...",
  "refund_amount": 20.00,
  "confidence": 0.92,
  "reasoning": "Customer on Basic plan charged Pro rate, $20 difference"
}
```

**Layer 2 — SupervisorAgent**
Rule-based policy enforcer. Checks every response before execution:
- Is identity verified before a fraud refund? If not — **BLOCK**
- Is the agent closing without escalating a technical issue? **REJECT**
- Is there empathy in the first response? **FLAG** for correction

**Layer 3 — OrchestratorAgent**
Strategic self-improvement. After a failed attempt, it:
1. Analyzes exactly which grading criteria failed
2. Generates targeted improvement instructions
3. Injects them into the next attempt's system prompt
4. Retries up to 3 times, tracking score progression

---

## System 2: Confidence-Based Routing

| Confidence | Routing |
|-----------|---------|
| `≥ 0.6` | Accept directly |
| `0.4 – 0.6` | SupervisorAgent reviews |
| `< 0.4` | OrchestratorAgent takes over |

Critical actions (refunds, escalations, closures) are **always** reviewed regardless of confidence.

---

## System 3: World Modeling (Conversation Memory)

Before every interaction, the agent loads the customer's history:

```
CUSTOMER MEMORY — Sarah Mitchell (ACC-88421):
  Total contacts: 3
  Avg sentiment: frustrated
  [2d ago] billing: refund $20.00 → resolved (score: 0.92)
  [5d ago] technical: escalated to Engineering (score: 0.88)
  → RETURNING CUSTOMER — use personalized greeting
  → REPEAT ISSUE — consider faster escalation
```

Memory uses a **sliding window of 10 interactions** per customer, stored as JSON in `/tmp` for HuggingFace Spaces compatibility.

---

## System 4: GRPO Training

Fine-tuned **Qwen2.5-1.5B-Instruct** using GRPO + LoRA on CustomerSupportEnv.

**Training Setup:**
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Method: GRPO + LoRA (r=16, alpha=32)
- Epochs: 20
- Group size: 4 rollouts per prompt
- Tasks: All 5
- Hardware: T4 GPU (Google Colab)

**How GRPO works:**
1. Generate 4 different responses for the same customer prompt
2. Run each through CustomerSupportEnv to get rewards
3. Compute advantages (which responses scored above average)
4. Update model to make high-reward responses more likely

**Trained model:** [Jyoti-6/customer-support-grpo-qwen](https://huggingface.co/Jyoti-6/customer-support-grpo-qwen)

---

## System 5: Long-Horizon Planning

The fraud task requires this exact sequence:

```
Turn 1: RESPOND  — verify identity (DOB + last 4 of card)
Turn 2: RESPOND  — confirm verified, lock/freeze account
Turn 3: REFUND   — issue full $448 refund
Turn 4: ESCALATE — hand off to Fraud & Security team
```

Skipping identity verification triggers a SupervisorAgent block — mirroring real-world compliance.

---

## Reward Function

| Event | Reward |
|-------|--------|
| Helpful, substantive response | +0.05 |
| Empathy keywords detected | +0.03 |
| Correct escalation | +0.20 |
| Escalation when not needed | -0.10 |
| Correct refund amount | +0.20 |
| Refund without identity verification | -0.15 |
| Correct episode closure | +0.25 |
| Premature close | -0.20 |
| Repeated/loop response | -0.08 |
| Exceeding max turns | -0.10 |

---

## Results

| Task | Score | Turns Used |
|------|-------|-----------|
| billing_dispute_easy | **1.000** | 3 |
| technical_outage_medium | **1.000** | 4 |
| fraud_complaint_hard | **1.000** | 4 |
| subscription_cancellation_hard | **1.000** | 4 |
| vip_account_recovery_expert | **1.000** | 4 |
| **Average** | **1.000** | — |

---

## What I Learned

1. **Typed action spaces matter.** Structured JSON output dramatically improves reliability.
2. **Supervision beats prompting alone.** SupervisorAgent catches violations well-prompted LLMs miss.
3. **Memory changes everything.** Without it, every interaction starts cold.
4. **Turn hints are powerful.** Telling the LLM what action is expected dramatically improves coherence.
5. **GRPO on small models works.** Even Qwen2.5-1.5B learns with a well-designed reward signal.

---

## Links

- 🤗 **Live Demo:** https://huggingface.co/spaces/Jyoti-6/customer-support-env
- 💻 **GitHub:** https://github.com/jyoti-codessss/customer-support-env
- 🤖 **Trained Model:** https://huggingface.co/Jyoti-6/customer-support-grpo-qwen
- 📓 **Training Notebook:** https://colab.research.google.com/drive/1Jt0O9v-0FVoRAiGAvIokV5Qgy3u9RESD

---

## 📹 Demo Video

👉 [Watch 2-min Demo on Loom](https://loom.com/share/YOUR_LINK)

> Complete walkthrough of the 3-layer agent system in action.

*Built for the OpenEnv Hackathon. All feedback welcome!*