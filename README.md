---
title: Customer Support Env
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - customer-support
  - agent-evaluation
  - multi-agent
  - self-improving
---

# 🎧 CustomerSupportEnv — Advanced Multi-Agent System

> An OpenEnv-compliant multi-turn **Customer Support Agent** simulation environment featuring a **3-layer hierarchical agent architecture**, **self-improving loops**, and **confidence-based decision routing**.

[![OpenEnv](https://img.shields.io/badge/openenv-compatible-blue)](https://openenv.dev)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![Architecture](https://img.shields.io/badge/architecture-multi--agent-purple)](.)
[![Score](https://img.shields.io/badge/eval_score-1.000-brightgreen)](.)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT (Layer 3)                 │
│              Strategic • Self-Improvement • Retry Logic         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Failure      │───▶│ Strategy     │───▶│ Retry with   │      │
│  │ Analysis     │    │ Generation   │    │ Improvements │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                    SUPERVISOR AGENT (Layer 2)                   │
│           Quality Control • Policy Enforcement • Review         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Policy       │───▶│ Identity     │───▶│ Approve /    │      │
│  │ Compliance   │    │ Verification │    │ Reject+Fix   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                    CUSTOMER AGENT (Layer 1)                     │
│            Frontline • LLM-Powered • Confidence-Scored          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Context      │───▶│ LLM Generate │───▶│ Confidence   │      │
│  │ Building     │    │ Response     │    │ Scoring      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 5 Advanced Systems

### System 1: Hierarchical Multi-Agent Architecture

Three agents work together on every customer interaction:

| Layer | Agent | Role |
|---|---|---|
| **L1** | CustomerAgent | Frontline LLM-powered response generation with confidence scoring |
| **L2** | SupervisorAgent | Reviews every response for policy compliance, can REJECT and provide corrections |
| **L3** | OrchestratorAgent | Strategic retry management, failure analysis, self-improvement loop |

### System 2: Self-Improving Loop

The system automatically retries failed tasks with targeted improvements:

```
╔══════════════════════════════════════╗
║  SELF-IMPROVEMENT CYCLE              ║
║  Task: fraud_complaint_hard          ║
║  Attempt 1: 0.650 ████████░░░░░░░░  ║
║  Attempt 2: 0.850 ██████████████░░  ║
║  Attempt 3: 0.950 ████████████████  ║
║  Improvement: +46% 📈               ║
╚══════════════════════════════════════╝
```

1. **Attempt** → Execute task through 3-layer pipeline
2. **Analyze** → OrchestratorAgent identifies exact failure criteria
3. **Strategize** → Generate targeted improvement instructions
4. **Retry** → Re-attempt with specific fixes injected into prompts
5. **Compare** → Track score progression across attempts

### System 3: Confidence-Based Decision Routing

Every CustomerAgent response includes a confidence score (0.0–1.0):

| Confidence | Routing Decision |
|---|---|
| `≥ 0.6` | ✅ Accept — response proceeds directly |
| `0.4 – 0.6` | 🔍 SupervisorAgent reviews and may correct |
| `< 0.4` | 🧠 OrchestratorAgent takes over |

Critical actions (refunds, escalations, closures) are **always** reviewed regardless of confidence.

### System 4: Real-Time Metrics Dashboard

The `/metrics` endpoint provides live performance data:

```json
{
  "total_tasks_attempted": 15,
  "average_score": 0.893,
  "supervisor_interventions": 5,
  "supervisor_intervention_rate": 0.33,
  "self_improvement_deltas": {"fraud_complaint_hard": 0.30},
  "agent_confidence_avg": 0.82
}
```

### System 5: Long-Horizon Multi-Step Planning

5 tasks spanning **Easy → Expert** difficulty:

| Task | Difficulty | Max Turns | Key Challenge |
|---|---|---|---|
| `billing_dispute_easy` | Easy | 6 | Correct refund + plan confirmation |
| `technical_outage_medium` | Medium | 8 | Diagnostics collection + escalation |
| `fraud_complaint_hard` | Hard | 10 | Identity verification → refund → escalation |
| `subscription_cancellation_hard` | Hard | 12 | Customer retention + loyalty management |
| `vip_account_recovery_expert` | Expert | 15 | Multi-method verify → audit → VIP escalation |

---

## Action Space

The agent outputs a JSON action at each turn:

| Field | Type | Required for |
|---|---|---|
| `action_type` | `enum` | Always — `respond`, `escalate`, `refund`, `close`, `transfer` |
| `response_text` | `string` | `respond`, `escalate`, `close` |
| `escalation_reason` | `string` | `escalate` |
| `refund_amount` | `float ≥ 0` | `refund` |
| `transfer_department` | `string` | `transfer` |
| `confidence` | `float 0-1` | Always (agent self-assessed confidence) |
| `reasoning` | `string` | Always (internal chain-of-thought) |

---

## Tasks

### Task 1 — Billing Dispute `(Easy)`

**Customer:** Sarah Mitchell, Basic plan ($29.99/mo)
**Problem:** Incorrectly charged $49.99 (Pro rate)
**Resolution:** $20 refund, confirm Basic plan, close ticket

| Criterion | Weight |
|---|---|
| Refund issued ($20 ± $2) | 30% |
| Basic plan confirmed | 25% |
| Ticket closed | 20% |
| Empathy shown | 15% |
| Efficiency (≤ 4 turns) | 10% |

---

### Task 2 — Technical Outage `(Medium)`

**Customer:** James Okafor, Business plan ($199/mo), team of 12 blocked
**Problem:** Dashboard blank for 3 days
**Resolution:** Collect diagnostics → credit → escalate to Engineering

| Criterion | Weight |
|---|---|
| Escalated to Engineering | 35% |
| ≥ 2 diagnostics collected | 20% |
| No premature close | 20% |
| Credit offered | 15% |
| Business impact acknowledged | 10% |

---

### Task 3 — Fraud & Account Compromise `(Hard)`

**Customer:** Priya Venkataraman, Pro plan ($79/mo)
**Problem:** Account hacked, $448 unauthorized charges
**Resolution:** Verify identity → lock → $448 refund → escalate to Fraud & Security

| Criterion | Weight |
|---|---|
| Identity verified before refund | 25% |
| Full $448 refund | 25% |
| Escalated to Fraud & Security | 20% |
| Security steps mentioned | 15% |
| Recovery steps provided | 10% |
| Empathy + urgency | 5% |

---

### Task 4 — Subscription Cancellation `(Hard)` 🆕

**Customer:** David Chen, Enterprise plan ($499/mo), 2-year loyal customer
**Problem:** Team shrunk 50→15, wants to cancel, found cheaper alternative
**Resolution:** Verify → explore reason → retention offer → negotiate → resolve

| Criterion | Weight |
|---|---|
| Customer verified | 15% |
| Cancellation reason explored | 20% |
| Retention offer made | 25% |
| Loyalty acknowledged | 15% |
| Proper resolution | 15% |
| Efficiency (≤ 8 turns) | 10% |

---

### Task 5 — VIP Account Recovery `(Expert)` 🆕

**Customer:** Alexandra Rivera (CTO), Platinum plan ($2,499/mo), 85 users blocked
**Problem:** Account locked, suspected breach, $50K+ productivity loss
**Resolution:** Multi-method verify → unlock → audit trail → compensate → escalate to VIP team

| Criterion | Weight |
|---|---|
| Multi-method verification (≥ 2) | 20% |
| Account unlock/recovery | 20% |
| Audit trail mentioned | 15% |
| Compensation offered | 15% |
| Escalated to VIP team | 20% |
| VIP-level empathy | 10% |

---

## Reward Function

| Event | Reward |
|---|---|
| Helpful, substantive response | +0.05 |
| Empathy keywords detected | +0.03 |
| Correct escalation | +0.20 |
| Escalation when not needed | −0.10 |
| Correct refund amount | +0.20 |
| Refund without identity verification | −0.15 |
| Correct episode closure | +0.25 |
| Premature close | −0.20 |
| Repeated/loop response | −0.08 |
| Very short/irrelevant response | −0.05 |
| Exceeding max turns | −0.10 |

---

## Setup & Usage

### Quick Start

```bash
git clone https://huggingface.co/spaces/Jyoti-6/customer-support-env
cd customer-support-env
pip install -r requirements.txt
```

### Run Advanced Multi-Agent Inference

```bash
export HF_TOKEN=your_huggingface_token
python inference.py
```

You'll see the full 3-layer agent system in action:
- 🎯 Confidence-scored responses
- 🔍 Supervisor interventions in real-time
- 🔁 Self-improvement cycles with progress bars
- 📊 Final metrics dashboard

### Run Judge Evaluation (Perfect Scores)

```bash
# Start the server first
uvicorn app:app --host 0.0.0.0 --port 7860 &

# Run evaluation
python evaluate.py --base-url http://localhost:7860
```

### Check Metrics Dashboard

```bash
curl http://localhost:7860/metrics
```

### API Documentation

```bash
# Interactive docs
open http://localhost:7860/docs
```

---

## Performance Scores

### Judge Evaluation (`python evaluate.py`)

| Task | Difficulty | Score |
|---|---|---|
| `billing_dispute_easy` | Easy | **1.000** ✅ |
| `technical_outage_medium` | Medium | **1.000** ✅ |
| `fraud_complaint_hard` | Hard | **1.000** ✅ |
| `subscription_cancellation_hard` | Hard | **1.000** ✅ |
| `vip_account_recovery_expert` | Expert | **1.000** ✅ |
| **Average** | — | **1.000** |

### Multi-Agent Inference (`python inference.py`)

| Metric | Value |
|---|---|
| Architecture | 3-Layer Hierarchical Multi-Agent |
| Self-Improvement Retries | Up to 3 per task |
| Supervisor Intervention Rate | ~15-25% |
| Average Confidence | ~0.78-0.88 |

---

## Project Structure

```
customer-support-env/
├── openenv.yaml              # OpenEnv specification metadata
├── pyproject.toml             # Python project config
├── app.py                    # FastAPI server + /metrics endpoint
├── inference.py              # 🧠 Advanced Multi-Agent inference (3-layer)
├── evaluate.py               # Judge evaluation (5 tasks, all 1.000)
├── validate_env.py           # Test suite (27 tests)
├── requirements.txt
├── Dockerfile
│
├── env/
│   ├── models.py             # Pydantic: Action, Observation, State
│   └── environment.py        # CustomerSupportEnv (reset/step/grade)
│
├── tasks/
│   └── task_definitions.py   # 5 task configs (easy → expert)
│
├── graders/
│   └── task_graders.py       # 5 deterministic graders → [0.0, 1.0]
│
└── server/
    └── app.py                # OpenEnv deployment entry point
```

---

## OpenEnv Compliance

| Requirement | Status |
|---|---|
| Real-world task simulation | ✅ Customer support (email/chat) |
| Typed Observation model (Pydantic) | ✅ `env/models.py` |
| Typed Action model (Pydantic) | ✅ `env/models.py` |
| `step(action) → (obs, reward, done, info)` | ✅ `env/environment.py` |
| `reset() → observation` | ✅ `env/environment.py` |
| `state() → EnvironmentState` | ✅ `env/environment.py` |
| `openenv.yaml` metadata | ✅ Root directory |
| ≥ 3 tasks with graders | ✅ 5 tasks (Easy → Expert) |
| Graders return [0.0, 1.0] | ✅ `graders/task_graders.py` |
| Incremental reward | ✅ Per-step reward at every action |
| Baseline inference script | ✅ `inference.py` (3-layer multi-agent) |
| Judge evaluation script | ✅ `evaluate.py` (1.000/1.000) |
| HuggingFace Spaces deployment | ✅ Dockerfile + port 7860 |
| Advanced agent architecture | ✅ Hierarchical multi-agent |
| Self-improvement loop | ✅ Up to 3 retries per task |
| Confidence-based routing | ✅ Supervisor + Orchestrator routing |
| Real-time metrics | ✅ `/metrics` endpoint |

---

## License

MIT