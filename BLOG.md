Building CustomerSupportEnv: A 3-Layer Multi-Agent System with
GRPO Training
OpenEnv Hackathon Submission — Jyoti Yadav
The Problem
Customer support is one of the hardest real-world tasks forAI agents. It requires:
Multi-turn reasoning across 6–15 conversation turns
Long-horizon planning — knowing what to do 3 steps ahead
Policy compliance — never issue a refund without verifying identity
World modeling — remembering frustrated customers from previous sessions
Self-improvement — learning from mistakes in real-time
Most LLM demos handle single-turn Q&A. CustomerSupportEnv goes much deeper.
What I Built
CustomerSupportEnv is an OpenEnv-compliant simulation environment where a 3-layerAI
agent system resolves customer complaints through typed actions — responding, escalating,
issuing refunds, and closing tickets.
5 Tasks — Easy to Expert
Task Difficulty Max
Turns
Challenge
billing_dispute_easy Easy 6 Correct refund + plan confirmation
technical_outage_medium Medium 8 Diagnostics + escalation to
Engineering
fraud_complaint_hard Hard 10 Identity verify → refund → Fraud &
Security
subscription_cancellation_hard Hard 12 Retention negotiation with 2-year
customer
vip_account_recovery_expert Expert 15 Multi-method verify → audit → VIP
escalation
Final Score: 1.000 / 1.000 on all 5 tasks✅
The Architecture
┌──────────────────────────────────────────┐
│ CUSTOMER INPUT │
│ 5 task types across 3 difficulty levels │
└──────────────────┬───────────────────────┘
↓
┌──────────────────────────────────────────┐
│ WORLD MODEL (Memory) │
│ Customer history • Sentiment tracking │
│ Repeat issue detection • VIP status │
└──────────────────┬───────────────────────┘
↓
┌──────────────────────────────────────────┐
│ LAYER 1 — CUSTOMER AGENT │
│ LLM-powered response generation │
│ meta-llama/Llama-3.1-8B-Instruct │
│ Confidence scoring (0.0 → 1.0) │
└──────────────────┬───────────────────────┘
↓
┌──────────────────────────────────────────┐
│ LAYER 2 — SUPERVISOR AGENT │
│ Policy compliance checks │
│ Fraud safety (identity before refund) │
│ Empathy verification │
└──────────────────┬───────────────────────┘
↓
┌──────────────────────────────────────────┐
│ LAYER 3 — ORCHESTRATOR AGENT │
│ Self-improvement loop │
│ Failure analysis → targeted fixes │
│ Up to 3 retries per task │
└──────────────────┬───────────────────────┘
↓
┌──────────────────────────────────────────┐
│ GRPO REWARD SIGNAL │
│ JSON format • Empathy • Action • Result │
└──────────────────────────────────────────┘
System 1: Hierarchical Multi-Agent Architecture
Three agents collaborate on every interaction:
Layer 1 — CustomerAgent The frontline LLM. Receives customer message + memory context +
turn hint, generates a JSON action with confidence score and reasoning.
Layer 2 — SupervisorAgent Rule-based policy enforcer. Checks every response before
execution:
Is identity verified before a fraud refund? If not — BLOCK and request verification
Is the agent closing without escalating a technical issue? REJECT
Is there empathy in the first response? FLAG for correction
Layer 3 — OrchestratorAgent Strategic self-improvement. After a failed attempt, it:
1. Analyzes exactly which grading criteria failed
2. Generates targeted improvement instructions
3. Injects them into the next attempt's system prompt
4. Retries up to 3 times, tracking score progression
System 2: Confidence-Based Routing
Every CustomerAgent response carries a self-assessed confidence score:
Confidence Routing
≥ 0.6 Accept directly
0.4 – 0.6 SupervisorAgent reviews
< 0.4 OrchestratorAgent takes over
json
{
"action_type": "refund",
"response_text": "I sincerely apologize for the incorrect charge...",
"refund_amount": 20.00,
"confidence": 0.92,
"reasoning": "Customer on Basic plan charged Pro rate, $20 difference"
}
Critical actions (refunds, escalations, closures) are always reviewed regardless of confidence —
safety first.
System 3: World Modeling (Conversation Memory)
The most underrated feature. Before every interaction, the agent loads the customer's history:
CUSTOMER MEMORY — Sarah Mitchell (ACC-88421):
Total contacts: 3
Avg sentiment: frustrated
[2d ago] billing: refund $20.00 → resolved (score: 0.92)
[5d ago] technical: escalated to Engineering (score: 0.88)
→ RETURNING CUSTOMER — use personalized greeting
→ REPEAT ISSUE — consider faster escalation
This context is injected into the LLM prompt, enabling:
Personalized greetings for returning customers
Faster escalation for repeat issues
Extra empathy for previously frustrated customers
VIP-level handling for high-value accounts
Memory uses a sliding window of 10 interactions per customer, stored as JSON in /tmp for
HuggingFace Spaces compatibility.
System 4: GRPO Training
I fine-tuned Qwen2.5-1.5B-Instruct on CustomerSupportEnv using GRPO (Group Relative
Policy Optimization) with LoRA.
Training Setup:
Base model: Qwen/Qwen2.5-1.5B-Instruct
Method: GRPO + LoRA (r=16, alpha=32)
Epochs: 10
Group size: 4 rollouts per prompt
Hardware: T4 GPU (Google Colab)
How GRPO works here:
1. Generate 4 different responses for the same customer prompt
2. Run each through CustomerSupportEnv to get rewards
3. Compute advantages (which responses scored above average)
4. Update model to make high-reward responses more likely
The reward signal combines:
JSON format compliance — is the output parseable?
Empathy detection — does it contain "sorry", "apologize", "understand"?
Correct action — refund amount, escalation target, closure timing
Resolution — did the episode end successfully?
Trained model: Jyoti-6/customer-support-grpo-qwen
System 5: Long-Horizon Planning
Each task requires the agent to think multiple steps ahead. Take the fraud task:
Skipping step 1 (identity verification) and jumping to refund triggers a SupervisorAgent
intervention — this mirrors real-world compliance requirements.
The OrchestratorAgent knows this sequence through an improvement map — if
identity_verified_first fails, it injects:
"Ask for date of birth AND last 4 digits BEFORE any refund action."
The Action Space
Every agent output is a typed JSON action:
Turn 1: RESPOND — verify identity (DOB + last 4 of card)
Turn 2: RESPOND — confirm verified, lock/freeze account
Turn 3: REFUND — issue full $448 refund
Turn 4: ESCALATE — hand off to Fraud & Security team
python
class Action(BaseModel):
action_type: ActionType # respond | refund | escalate | close | transfer
response_text: str
escalation_reason: Optional[str]
refund_amount: Optional[float]
transfer_department: Optional[str]
This typed interface makes the environment fully compatible with any LLM that can output
valid JSON.
Reward Function
Event Reward
Helpful, substantive response +0.05
Empathy keywords detected +0.03
Correct escalation +0.20
Escalation when not needed -0.10
Correct refund amount +0.20
Refund without identity verification -0.15
Correct episode closure +0.25
Premature close -0.20
Repeated/loop response -0.08
Exceeding max turns -0.10
Live Demo
The HuggingFace Space runs real LLM inference — no hardcoded scripts.
Select any task → click "Run Real Agent" → watch the 3-layer system make decisions in real-time,
with:
Memory context loaded from world model
Step-by-step action breakdown
Confidence scores and reward per step
Final score with detailed breakdown
Cumulative reward chart
👉Try it live
Results
Task Score Turns Used
billing_dispute_easy 1.000 3
technical_outage_medium 1.000 4
fraud_complaint_hard 1.000 4
subscription_cancellation_hard 1.000 4
vip_account_recovery_expert 1.000 4
Average 1.000 —
What I Learned
1. Typed action spaces matter. Forcing the LLM to output structured JSON dramatically
improves reliability. Free-form text responses are unpredictable.
2. Supervision beats prompting alone. The SupervisorAgent catches policy violations that even
well-prompted LLMs miss — especially the "refund before verification" fraud safety rule.
3. Memory changes everything. Returning customers with repeat issues need different
handling. Without memory, every interaction starts cold.
4. Turn hints are powerful. Telling the LLM exactly what action is expected on turn N (without
forcing it) dramatically improves multi-turn coherence.
5. GRPO on small models works. Even Qwen2.5-1.5B shows learning on environment-specific
tasks when the reward signal is well-designed.
Links
🤗HuggingFace Space (Live Demo): https://huggingface.co/spaces/Jyoti-6/customer￾support-env
💻GitHub: https://github.com/jyoti-codessss/customer-support-env
🤖Trained Model: https://huggingface.co/Jyoti-6/customer-support-grpo-qwen
📓Training Notebook: https://colab.research.google.com/drive/1Jt0O9v-
0FVoRAiGAvIokV5Qgy3u9RESD
Built for the OpenEnv Hackathon. All feedback welcome!