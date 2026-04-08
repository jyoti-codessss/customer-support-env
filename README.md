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
---

# 🎧 CustomerSupportEnv

> An OpenEnv-compliant multi-turn **Customer Support Agent** simulation environment for evaluating AI agents on real-world support scenarios.

[![OpenEnv](https://img.shields.io/badge/openenv-compatible-blue)](https://openenv.dev)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://python.org)

---

## Overview & Motivation

Customer support is one of the highest-value, highest-volume real-world tasks that AI agents are being deployed for today. Yet most agent benchmarks test reasoning in isolation — not the **multi-turn, multi-action decision-making** that real support work demands.

**CustomerSupportEnv** places an AI agent in the seat of a live support representative. The agent must:

- Read incoming complaints and conversation history
- Choose from a typed action space (respond, escalate, refund, close, transfer)
- Manage multi-turn dialogue across 6–10 turns per episode
- Navigate escalation decisions, identity verification, and refund eligibility
- Earn incremental rewards for correct actions — and penalties for mistakes

Three tasks span **easy → medium → hard** and test fundamentally different capabilities: billing resolution, technical triage, and fraud investigation.

---

## Action Space

The agent outputs a JSON action at each turn. All fields are optional except `action_type`.

| Field | Type | Required for |
|---|---|---|
| `action_type` | `enum` | Always — one of `respond`, `escalate`, `refund`, `close`, `transfer` |
| `response_text` | `string` | `respond`, `escalate`, `close` |
| `escalation_reason` | `string` | `escalate` |
| `refund_amount` | `float ≥ 0` | `refund` |
| `transfer_department` | `string` | `transfer` |

**Example action:**
```json
{
  "action_type": "refund",
  "refund_amount": 20.00,
  "response_text": "I've issued a $20 refund. Your Basic plan remains active."
}
```

---

## Observation Space

At each step the agent receives:

| Field | Type | Description |
|---|---|---|
| `ticket_id` | `string` | Unique ticket identifier |
| `user_message` | `string` | Latest customer message |
| `conversation_history` | `list[Turn]` | Full prior turns (role + content + turn number) |
| `ticket_status` | `enum` | `open`, `pending`, `escalated`, `resolved`, `closed` |
| `turn_number` | `int` | Current turn index |
| `metadata` | `dict` | Customer name, account ID, plan, issue category, prior contacts |

> **Note:** Internal grader-only fields (expected_resolution, max_refund, etc.) are held in `EnvironmentState` and are **not** exposed in the observation.

---

## Tasks

### Task 1 — Billing Dispute `(Easy)`

**Ticket:** `TKT-001`  
**Customer:** Sarah Mitchell, Basic plan ($29.99/mo)  
**Problem:** Incorrectly charged $49.99 (Pro rate) with no upgrade requested.  
**Correct resolution:** Issue a $20 refund, confirm Basic plan, close ticket.  
**Max turns:** 6  

**Scoring rubric:**

| Criterion | Weight |
|---|---|
| Refund issued in correct amount ($20 ± $2) | 30% |
| Basic plan confirmed in conversation | 25% |
| Ticket properly closed/resolved | 20% |
| Empathetic acknowledgment | 15% |
| Resolved within 4 turns (efficiency) | 10% |

**Expected baseline score:** 0.70–0.90 (strong models), 0.30–0.55 (weak models)

---

### Task 2 — Technical Outage `(Medium)`

**Ticket:** `TKT-002`  
**Customer:** James Okafor, Business plan ($199/mo), team of 12 blocked  
**Problem:** Dashboard blank for 3 days. Multi-browser, cache-cleared. Prior contacts: 2.  
**Correct resolution:** Collect diagnostics (console errors, URL, date), escalate to Engineering, offer partial credit.  
**Max turns:** 8  

**Scoring rubric:**

| Criterion | Weight |
|---|---|
| Escalated to Engineering team | 35% |
| Collected ≥ 2 diagnostic data points before escalating | 20% |
| Did NOT close ticket without escalating | 20% |
| Offered partial credit/compensation | 15% |
| Acknowledged business impact | 10% |

**Expected baseline score:** 0.45–0.70 (strong models), 0.10–0.35 (weak models)

---

### Task 3 — Fraud & Account Compromise `(Hard)`

**Ticket:** `TKT-003`  
**Customer:** Priya Venkataraman, Pro plan ($79/mo)  
**Problem:** Account compromised, email changed, $448 in unauthorized charges ($149 + $89 + $210).  
**Correct resolution:** Verify identity FIRST, lock account, issue full $448 refund, escalate to Fraud & Security, provide recovery steps.  
**Max turns:** 10  

**Scoring rubric:**

| Criterion | Weight |
|---|---|
| Identity verified **before** any refund | 25% |
| Full $448 refund initiated (±$10) | 25% |
| Escalated to Fraud & Security team | 20% |
| Account lock/security steps mentioned | 15% |
| Recovery steps provided | 10% |
| Empathy + urgency matched | 5% |

**Expected baseline score:** 0.30–0.60 (strong models), 0.05–0.25 (weak models)

---

## Reward Function

Rewards are **incremental** — issued at every step, not just at episode completion.

| Event | Reward |
|---|---|
| Helpful, substantive response | +0.05 |
| Empathy keywords detected | +0.03 |
| Correct escalation (right department) | +0.20 |
| Escalation when not needed | −0.10 |
| Correct refund amount | +0.20 |
| Refund without identity verification (fraud) | −0.15 |
| Correct episode closure | +0.25 |
| Premature close (unresolved) | −0.20 |
| Repeated/loop response | −0.08 |
| Very short/irrelevant response | −0.05 |
| Exceeding max turns | −0.10 |

All per-step rewards are clipped to **[−1.0, 1.0]**.  
Final episode **grade scores** are in **[0.0, 1.0]** and computed independently by task-specific graders.

---

## Setup & Usage

### Local Development

```bash
git clone https://github.com/your-org/customer-support-env
cd customer-support-env
pip install -r requirements.txt
```

**Run the validation suite:**
```bash
python validate_env.py
```

**Run baseline inference:**
```bash
export HF_TOKEN=your_huggingface_token
python inference.py
# Run a specific task:
python inference.py --task billing_dispute_easy
# Use a different model:
python inference.py --model meta-llama/Llama-3.1-70B-Instruct
```

**Start the API server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
# API docs at: http://localhost:7860/docs
```

---

### Docker

```bash
# Build
docker build -t customer-support-env .

# Run
docker run -p 7860:7860 customer-support-env

# Run inference inside container
docker run -e HF_TOKEN=your_token customer-support-env python inference.py
```

---

### API Usage

**Reset (start a new episode):**
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "billing_dispute_easy"}'
```

**Step (take an action):**
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "action": {
      "action_type": "respond",
      "response_text": "I apologize for this billing error. Let me issue a refund right away."
    }
  }'
```

**Get current state:**
```bash
curl http://localhost:7860/state/your-session-id
```

---

### Python API

```python
from env.environment import CustomerSupportEnv
from env.models import Action, ActionType

env = CustomerSupportEnv("billing_dispute_easy")
obs = env.reset()

while True:
    # Your agent logic here
    action = Action(
        action_type=ActionType.RESPOND,
        response_text="I'm sorry for the overcharge. I'll fix this right away."
    )
    obs, reward, done, info = env.step(action)
    print(f"Turn {obs.turn_number} | Reward: {reward:+.3f} | Done: {done}")
    if done:
        break

score, breakdown = env.grade()
print(f"Final score: {score:.3f}")
print(breakdown)
```

---

## Project Structure

```
customer-support-env/
├── openenv.yaml              # OpenEnv specification metadata
├── app.py                    # FastAPI server (HuggingFace Spaces entrypoint)
├── inference.py              # Baseline inference script (HF_TOKEN)
├── validate_env.py           # Standalone test suite (no extra deps)
├── requirements.txt
├── Dockerfile
│
├── env/
│   ├── __init__.py
│   ├── models.py             # Pydantic: Action, Observation, State, Reward
│   └── environment.py        # CustomerSupportEnv (reset/step/state/grade)
│
├── tasks/
│   ├── __init__.py
│   └── task_definitions.py   # 3 task configs (easy/medium/hard)
│
├── graders/
│   ├── __init__.py
│   └── task_graders.py       # Deterministic graders → score [0.0, 1.0]
│
└── tests/
    └── test_environment.py   # pytest test suite
```

---

## Baseline Performance Scores

Evaluated using `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Inference API:

| Task | Difficulty | Baseline Score | Notes |
|---|---|---|---|
| `billing_dispute_easy` | Easy | **0.72** | Refunds correctly but misses empathy |
| `technical_outage_medium` | Medium | **0.48** | Escalates but skips diagnostics |
| `fraud_complaint_hard` | Hard | **0.31** | Often refunds before verifying identity |
| **Average** | — | **0.50** | — |

*Scores from `baseline_results.json` after `python inference.py`.*

---

## HuggingFace Spaces Deployment

This environment is tagged with `openenv` and deployable directly as a HuggingFace Space.

**`README.md` Space header (add to top of README for HF):**
```yaml
---
title: CustomerSupportEnv
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - customer-support
  - agent-evaluation
---
```

The FastAPI server starts on port 7860 (HuggingFace default) via the `CMD` in `Dockerfile`.  
API documentation is auto-generated at `/docs`.

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
| ≥ 3 tasks with agent graders | ✅ Easy / Medium / Hard |
| Graders return score in [0.0, 1.0] | ✅ `graders/task_graders.py` |
| Incremental reward (not just terminal) | ✅ Per-step reward at every action |
| Penalties for bad behavior | ✅ Loops, premature close, unverified refunds |
| Baseline inference script | ✅ `inference.py` (HF_TOKEN) |
| HuggingFace Spaces deployment | ✅ Dockerfile + port 7860 |
| `openenv validate` compatible | ✅ |

---

## License

MIT