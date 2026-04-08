"""
inference.py — Baseline inference script for CustomerSupportEnv.

Evaluates a model across all three tasks using the OpenAI API client
(compatible with HuggingFace Inference Endpoints via HF_TOKEN).

Usage:
    HF_TOKEN=your_token python inference.py
    HF_TOKEN=your_token python inference.py --task billing_dispute_easy
    HF_TOKEN=your_token python inference.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import os
import sys
import json
import argparse
from typing import Optional, List

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))

from env.environment import CustomerSupportEnv
from env.models import Action, ActionType
from tasks.task_definitions import TASKS

# ── Config ─────────────────────────────────────────────────────────────────
# Defaults set for API_BASE_URL and MODEL_NAME (not HF_TOKEN)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")  # No default — must be set by user

BENCHMARK = "customer-support-env"

SYSTEM_PROMPT = """You are a professional customer support agent. Your goal is to resolve customer issues efficiently and empathetically.

At each turn, you must respond with a JSON object containing your action. Valid action types:
- respond: Send a message to the customer
- escalate: Escalate to a specialist team
- refund: Issue a refund
- close: Close the ticket as resolved
- transfer: Transfer to another department

Response format (ALWAYS valid JSON, no other text):
{
  "action_type": "respond" | "escalate" | "refund" | "close" | "transfer",
  "response_text": "Your message to the customer (for respond/escalate/close)",
  "escalation_reason": "Reason for escalation (for escalate)",
  "refund_amount": 0.00,
  "transfer_department": "Department name (for transfer)"
}

Guidelines:
- Always verify customer identity before processing refunds on fraud cases
- Gather diagnostic information before escalating technical issues
- Show empathy and acknowledge the customer's frustration
- Be concise but thorough
- Escalate when you cannot resolve the issue directly"""


# ── Logging helpers (exact hackathon format) ────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Prompt builder ──────────────────────────────────────────────────────────
def build_user_prompt(obs, task_context: str) -> str:
    history_text = ""
    for turn in obs.conversation_history:
        prefix = "Customer" if turn.role == "user" else "Agent"
        history_text += f"{prefix}: {turn.content}\n\n"

    return f"""TASK CONTEXT: {task_context}

CUSTOMER INFO:
- Name: {obs.metadata.get('customer_name', 'Unknown')}
- Account: {obs.metadata.get('account_id', 'N/A')}
- Plan: {obs.metadata.get('plan', 'N/A')}
- Issue Category: {obs.metadata.get('issue_category', 'N/A')}
- Prior Contacts: {obs.metadata.get('prior_contacts', 0)}

CONVERSATION HISTORY:
{history_text}

CURRENT STATUS: {obs.ticket_status}
TURN: {obs.turn_number}

Respond with your next action as a JSON object."""


# ── Action parser ───────────────────────────────────────────────────────────
def parse_action(response_text: str) -> Action:
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        data = json.loads(text)
        return Action(**data)
    except Exception:
        return Action(
            action_type=ActionType.RESPOND,
            response_text=response_text[:500] if response_text else "I'm looking into this for you."
        )


# ── Task runner ─────────────────────────────────────────────────────────────
def run_task(client: OpenAI, model: str, task_id: str) -> dict:
    env = CustomerSupportEnv(task_id)
    task_def = TASKS[task_id]
    task_context = task_def["system_context"]

    obs = env.reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    rewards: List[float] = []
    done = False
    step = 0

    # [START] log
    log_start(task=task_id, env=BENCHMARK, model=model)

    while not done:
        step += 1
        user_prompt = build_user_prompt(obs, task_context)
        messages.append({"role": "user", "content": user_prompt})

        error = None
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            raw_response = completion.choices[0].message.content
        except Exception as e:
            error = str(e)
            raw_response = '{"action_type": "respond", "response_text": "I apologize for the delay. Let me look into this for you right away."}'

        messages.append({"role": "assistant", "content": raw_response})

        action = parse_action(raw_response)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)

        # [STEP] log
        log_step(step=step, action=action.action_type, reward=reward, done=done, error=error)

    # Grade
    final_score, breakdown = env.grade()
    success = final_score >= 0.5

    # [END] log
    log_end(success=success, steps=step, score=final_score, rewards=rewards)

    return {
        "task_id": task_id,
        "difficulty": task_def["difficulty"],
        "final_score": final_score,
        "cumulative_reward": round(sum(rewards), 4),
        "turns_used": step,
        "max_turns": task_def["max_turns"],
        "breakdown": breakdown,
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CustomerSupportEnv Inference Script")
    parser.add_argument("--task", default=None, help="Specific task to run (default: all)")
    parser.add_argument("--model", default=MODEL_NAME, help="Model ID")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Usage: HF_TOKEN=your_token python inference.py")
        sys.exit(1)

    # OpenAI client configured via API_BASE_URL and HF_TOKEN
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks_to_run = [args.task] if args.task else list(TASKS.keys())
    results = []

    for task_id in tasks_to_run:
        result = run_task(client, args.model, task_id)
        results.append(result)

    # Summary
    avg_score = sum(r["final_score"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    for r in results:
        bar = "█" * int(r["final_score"] * 20)
        print(f"  {r['task_id']:<30} {r['final_score']:.3f}  {bar}")
    print(f"  {'AVERAGE':<30} {avg_score:.3f}")

    with open("baseline_results.json", "w") as f:
        json.dump({"model": args.model, "results": results, "average_score": round(avg_score, 4)}, f, indent=2)
    print(f"\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()