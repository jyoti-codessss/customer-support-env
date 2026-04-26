---
title: Customer Support Env
emoji: "\U0001F3A7"
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
  - conversation-memory
---

# CustomerSupportEnv -- Advanced Multi-Agent System

> An OpenEnv-compliant multi-turn **Customer Support Agent** simulation environment featuring a **3-layer hierarchical agent architecture**, **conversation memory**, **self-improving loops**, and **confidence-based decision routing**.

[![OpenEnv](https://img.shields.io/badge/openenv-compatible-blue)](https://openenv.dev)
[![HuggingFace Spaces](https://img.shields.io/badge/HF-Spaces-yellow)](https://huggingface.co/spaces/Jyoti-6/customer-support-env)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![Architecture](https://img.shields.io/badge/architecture-multi--agent-purple)](.)
[![Score](https://img.shields.io/badge/eval_score-1.000-brightgreen)](.)
[![Colab](https://img.shields.io/badge/Google-Colab-orange)](.)

---

## Architecture Overview

```
                    ORCHESTRATOR AGENT (Layer 3)
              Strategic - Self-Improvement - Retry Logic

   Failure Analysis --> Strategy Generation --> Retry with Improvements
-----------------------------------------------------------------
                    SUPERVISOR AGENT (Layer 2)
           Quality Control - Policy Enforcement - Review

   Policy Compliance --> Identity Verification --> Approve / Reject+Fix
-----------------------------------------------------------------
                 CONVERSATION MEMORY (World Model)
           Customer History - Sentiment Tracking - Repeat Detection

   Load Profile --> Inject Context --> Personalize Response
-----------------------------------------------------------------
                    CUSTOMER AGENT (Layer 1)
            Frontline - LLM-Powered - Confidence-Scored

   Context Building --> LLM Generate Response --> Confidence Scoring
```

---

## 6 Advanced Systems

### System 1: Hierarchical Multi-Agent Architecture

Three agents work together on every customer interaction:

| Layer | Agent | Role |
|---|---|---|
| **L1** | CustomerAgent | Frontline LLM-powered response generation with confidence scoring |
| **L2** | SupervisorAgent | Reviews responses for policy compliance, can REJECT and provide corrections |
| **L3** | OrchestratorAgent | Strategic retry management, failure analysis, self-improvement loop |

### System 2: Conversation Memory (World Model)

Persistent customer memory across interactions (`env/memory.py`):

- **Per-customer profiles** stored in JSON (sliding window: max 10 interactions)
- **Sentiment tracking**: frustrated / neutral / satisfied
- **Repeat issue detection**: flags returning customers with same problem
- **Agent context injection**: memory context added to LLM prompts for personalization
- **Proactive empathy**: previously frustrated customers get extra care

```
CUSTOMER MEMORY -- Sarah Johnson (ACC-88421):
  Total contacts: 3
  Avg sentiment: satisfied
  [2d ago] billing: refund $20.00 -> resolved (sentiment: satisfied)
  [5d ago] technical: escalated to Engineering -> escalated (sentiment: frustrated)
  RETURNING CUSTOMER -- personalize greeting
```

Memory API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/memory/stats` | Aggregate memory statistics |
| `GET` | `/memory/{account_id}` | Customer profile and context |
| `DELETE` | `/memory/{account_id}` | Delete customer memory |

### System 3: Self-Improving Loop

The system automatically retries failed tasks with targeted improvements:

```
[SELF-IMPROVEMENT CYCLE] Task: fraud_complaint_hard
  Attempt 1: 0.650 ##########......
  Attempt 2: 0.850 ##############..
  Attempt 3: 0.950 ################
  Improvement: +46%
```

1. **Attempt** -> Execute task through 3-layer pipeline
2. **Analyze** -> OrchestratorAgent identifies exact failure criteria
3. **Strategize** -> Generate targeted improvement instructions
4. **Retry** -> Re-attempt with specific fixes injected into prompts
5. **Compare** -> Track score progression across attempts

### System 4: Confidence-Based Decision Routing

Every CustomerAgent response includes a confidence score (0.0-1.0):

| Confidence | Routing Decision |
|---|---|
| `>= 0.6` | Accept -- response proceeds directly |
| `0.4 - 0.6` | SupervisorAgent reviews and may correct |
| `< 0.4` | OrchestratorAgent takes over |

Critical actions (refunds, escalations, closures) are **always** reviewed regardless of confidence.

### System 5: Interactive Gradio Demo

Live interactive demo at `/demo` with real LLM inference:

- **Task selector**: Choose from 5 difficulty levels
- **Real-time agent execution**: Watch the LLM make decisions step-by-step
- **Memory visualization**: See world model state for each customer
- **Score breakdown**: Detailed reward analysis per criterion
- **Cumulative reward chart**: Visual performance tracking
- **Architecture tab**: Full system diagram
- **API Reference tab**: All endpoints documented

### System 6: Long-Horizon Multi-Step Planning

5 tasks spanning **Easy -> Expert** difficulty:

| Task | Difficulty | Max Turns | Key Challenge |
|---|---|---|---|
| `billing_dispute_easy` | Easy | 6 | Correct refund + plan confirmation |
| `technical_outage_medium` | Medium | 8 | Diagnostics collection + escalation |
| `fraud_complaint_hard` | Hard | 10 | Identity verification -> refund -> escalation |
| `subscription_cancellation_hard` | Hard | 12 | Customer retention + loyalty management |
| `vip_account_recovery_expert` | Expert | 15 | Multi-method verify -> audit -> VIP escalation |

---

## Action Space

The agent outputs a JSON action at each turn:

| Field | Type | Required for |
|---|---|---|
| `action_type` | `enum` | Always -- `respond`, `escalate`, `refund`, `close`, `transfer` |
| `response_text` | `string` | `respond`, `escalate`, `close` |
| `escalation_reason` | `string` | `escalate` |
| `refund_amount` | `float >= 0` | `refund` |
| `transfer_department` | `string` | `transfer` |
| `confidence` | `float 0-1` | Always (agent self-assessed confidence) |
| `reasoning` | `string` | Always (internal chain-of-thought) |

---

## Tasks

### Task 1 -- Billing Dispute (Easy)

**Customer:** Sarah Mitchell, Basic plan ($29.99/mo)
**Problem:** Incorrectly charged $49.99 (Pro rate)
**Resolution:** $20 refund, confirm Basic plan, close ticket

| Criterion | Weight |
|---|---|
| Refund issued ($20 +/- $2) | 30% |
| Basic plan confirmed | 25% |
| Ticket closed | 20% |
| Empathy shown | 15% |
| Efficiency (<= 4 turns) | 10% |

### Task 2 -- Technical Outage (Medium)

**Customer:** James Okafor, Business plan ($199/mo), team of 12 blocked
**Problem:** Dashboard blank for 3 days
**Resolution:** Collect diagnostics -> credit -> escalate to Engineering

| Criterion | Weight |
|---|---|
| Escalated to Engineering | 35% |
| >= 2 diagnostics collected | 20% |
| No premature close | 20% |
| Credit offered | 15% |
| Business impact acknowledged | 10% |

### Task 3 -- Fraud & Account Compromise (Hard)

**Customer:** Priya Venkataraman, Pro plan ($79/mo)
**Problem:** Account hacked, $448 unauthorized charges
**Resolution:** Verify identity -> lock -> $448 refund -> escalate to Fraud & Security

| Criterion | Weight |
|---|---|
| Identity verified before refund | 25% |
| Full $448 refund | 25% |
| Escalated to Fraud & Security | 20% |
| Security steps mentioned | 15% |
| Recovery steps provided | 10% |
| Empathy + urgency | 5% |

### Task 4 -- Subscription Cancellation (Hard)

**Customer:** David Chen, Enterprise plan ($499/mo), 2-year loyal customer
**Problem:** Team shrunk 50->15, wants to cancel, found cheaper alternative
**Resolution:** Verify -> explore reason -> retention offer -> negotiate -> resolve

| Criterion | Weight |
|---|---|
| Customer verified | 15% |
| Cancellation reason explored | 20% |
| Retention offer made | 25% |
| Loyalty acknowledged | 15% |
| Proper resolution | 15% |
| Efficiency (<= 8 turns) | 10% |

### Task 5 -- VIP Account Recovery (Expert)

**Customer:** Alexandra Rivera (CTO), Platinum plan ($2,499/mo), 85 users blocked
**Problem:** Account locked, suspected breach, $50K+ productivity loss
**Resolution:** Multi-method verify -> unlock -> audit trail -> compensate -> escalate to VIP team

| Criterion | Weight |
|---|---|
| Multi-method verification (>= 2) | 20% |
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
| Escalation when not needed | -0.10 |
| Correct refund amount | +0.20 |
| Refund without identity verification | -0.15 |
| Correct episode closure | +0.25 |
| Premature close | -0.20 |
| Repeated/loop response | -0.08 |
| Very short/irrelevant response | -0.05 |
| Exceeding max turns | -0.10 |

---

## Setup & Usage

### Quick Start

```bash
git clone https://github.com/jyoti-codessss/customer-support-env.git
cd customer-support-env
pip install -r requirements.txt
touch env/__init__.py tasks/__init__.py graders/__init__.py
```

### Run Multi-Agent Inference

```bash
export HF_TOKEN=your_huggingface_token
python inference.py
```

You'll see the full 3-layer agent system in action:
- Confidence-scored responses with routing
- Supervisor interventions in real-time
- Self-improvement cycles with progress tracking
- Memory stats after completion

### Run Single Task

```bash
python inference.py --task billing_dispute_easy
```

### Launch Gradio Demo

```bash
python demo.py
# Opens at http://localhost:7860 with real LLM inference
```

### Launch FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
# API at http://localhost:7860
# Demo at http://localhost:7860/demo
# Docs at http://localhost:7860/docs
```

### Run in Google Colab

Open `CustomerSupportEnv_Colab.ipynb` in Google Colab:
1. Clone repo and install dependencies
2. Set your HF_TOKEN
3. Test environment (no LLM needed)
4. Launch Gradio with public share link
5. Run full LLM inference

### Check Memory System

```bash
# Memory stats
curl http://localhost:7860/memory/stats

# Customer profile
curl http://localhost:7860/memory/ACC-88421

# Delete memory
curl -X DELETE http://localhost:7860/memory/ACC-88421
```

### API Documentation

```bash
# Interactive Swagger docs
open http://localhost:7860/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Landing page with links |
| `POST` | `/reset` | Start new episode: `{"task_id": "billing_dispute_easy"}` |
| `POST` | `/step` | Take action: `{"session_id": "...", "action_type": "respond", "content": "..."}` |
| `POST` | `/grade` | Get final score: `{"session_id": "..."}` |
| `GET` | `/metrics` | Real-time performance dashboard |
| `GET` | `/memory/stats` | Memory system statistics |
| `GET` | `/memory/{account_id}` | Customer memory profile |
| `DELETE` | `/memory/{account_id}` | Delete customer memory |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger API documentation |
| `GET` | `/demo` | Interactive Gradio demo |

---

## Performance Scores

### Evaluation Results

| Task | Difficulty | Score |
|---|---|---|
| `billing_dispute_easy` | Easy | **1.000** |
| `technical_outage_medium` | Medium | **1.000** |
| `fraud_complaint_hard` | Hard | **1.000** |
| `subscription_cancellation_hard` | Hard | **1.000** |
| `vip_account_recovery_expert` | Expert | **1.000** |
| **Average** | -- | **1.000** |

### Multi-Agent Metrics

| Metric | Value |
|---|---|
| Architecture | 3-Layer Hierarchical Multi-Agent |
| Self-Improvement Retries | Up to 3 per task |
| Supervisor Intervention Rate | ~15-25% |
| Average Confidence | ~0.78-0.88 |
| Memory Slots per Customer | 10 (sliding window) |
| Sentiment Tracking | frustrated / neutral / satisfied |

---

## Project Structure

```
customer-support-env/
|-- README.md
|-- openenv.yaml                    # OpenEnv specification metadata
|-- pyproject.toml                  # Python project config
|-- requirements.txt                # Dependencies
|-- Dockerfile                      # HuggingFace Spaces deployment
|-- CustomerSupportEnv_Colab.ipynb  # Google Colab notebook
|
|-- app.py                          # FastAPI server + memory endpoints + Gradio mount
|-- demo.py                         # Gradio interactive UI (real LLM inference)
|-- inference.py                    # 3-layer multi-agent inference system
|-- evaluate.py                     # Judge evaluation (5 tasks, all 1.000)
|-- validate_env.py                 # Test suite
|
|-- env/
|   |-- __init__.py
|   |-- models.py                   # Pydantic: Action, Observation, State
|   |-- environment.py              # CustomerSupportEnv (reset/step/grade + memory)
|   |-- memory.py                   # ConversationMemory (JSON persistence)
|
|-- tasks/
|   |-- __init__.py
|   |-- task_definitions.py         # 5 task configs (Easy -> Expert)
|
|-- graders/
|   |-- __init__.py
|   |-- task_graders.py             # 5 deterministic graders -> [0.0, 1.0]
```

---

## OpenEnv Compliance

| Requirement | Status |
|---|---|
| Real-world task simulation | Customer support (email/chat) |
| Typed Observation model (Pydantic) | `env/models.py` |
| Typed Action model (Pydantic) | `env/models.py` |
| `step(action) -> (obs, reward, done, info)` | `env/environment.py` |
| `reset() -> observation` | `env/environment.py` |
| `state() -> EnvironmentState` | `env/environment.py` |
| `openenv.yaml` metadata | Root directory |
| >= 3 tasks with graders | 5 tasks (Easy -> Expert) |
| Graders return [0.0, 1.0] | `graders/task_graders.py` |
| Incremental reward | Per-step reward at every action |
| Baseline inference script | `inference.py` (3-layer multi-agent) |
| Judge evaluation script | `evaluate.py` (1.000/1.000) |
| HuggingFace Spaces deployment | Dockerfile + port 7860 |
| Advanced agent architecture | Hierarchical multi-agent |
| Self-improvement loop | Up to 3 retries per task |
| Confidence-based routing | Supervisor + Orchestrator routing |
| Conversation memory | `env/memory.py` (JSON persistence) |
| Interactive demo | Gradio UI at `/demo` |
| Google Colab support | `CustomerSupportEnv_Colab.ipynb` |

---

## Trained Model (GRPO Fine-tuned)

- **Model:** [Jyoti-6/customer-support-grpo-qwen](https://huggingface.co/Jyoti-6/customer-support-grpo-qwen)
- **Base:** Qwen/Qwen2.5-1.5B-Instruct
- **Method:** GRPO with LoRA (r=16)
- **Epochs:** 10
- **Environment:** CustomerSupportEnv


## Links

- **GitHub:** https://github.com/jyoti-codessss/customer-support-env
- **HuggingFace Space:** https://huggingface.co/spaces/Jyoti-6/customer-support-env
- **Live Demo:** https://jyoti-6-customer-support-env.hf.space/demo
- **Trained Model:** https://huggingface.co/Jyoti-6/customer-support-grpo-qwen
- **Training Notebook (Colab):** https://colab.research.google.com/drive/1Jt0O9v-0FVoRAiGAvIokV5Qgy3u9RESD?usp=sharing
---



## License

MIT