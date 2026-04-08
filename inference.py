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
from typing import Optional

from openai import OpenAI

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from env.environment import CustomerSupportEnv
from env.models import Action, ActionType
from tasks.task_definitions import TASKS

# ── Config ────────────────────────────────────────────────────────────────────

HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

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


def build_user_prompt(obs, task_context: str) -> str:
    """Build the prompt for the model from current observation."""
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


def parse_action(response_text: str) -> Optional[Action]:
    """Parse model response into an Action object."""
    try:
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        data = json.loads(text)
        return Action(**data)
    except Exception as e:
        print(f"  [Parse error] {e} | Raw: {response_text[:100]}")
        # Fallback: respond with the raw text
        return Action(
            action_type=ActionType.RESPOND,
            response_text=response_text[:500] if response_text else "I'm looking into this for you."
        )


def run_task(client: OpenAI, model: str, task_id: str, verbose: bool = True) -> dict:
    """Run a single task and return results."""
    env = CustomerSupportEnv(task_id)
    task_def = TASKS[task_id]
    task_context = task_def["system_context"]

    obs = env.reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    total_reward = 0.0
    done = False
    turn = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK: {task_id} | Difficulty: {task_def['difficulty'].upper()}")
        print(f"{'='*60}")
        print(f"Customer: {obs.user_message}\n")

    while not done:
        turn += 1
        user_prompt = build_user_prompt(obs, task_context)
        messages.append({"role": "user", "content": user_prompt})

        # Call the model
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            raw_response = completion.choices[0].message.content
        except Exception as e:
            print(f"  [API Error turn {turn}]: {e}")
            raw_response = '{"action_type": "respond", "response_text": "I apologize for the delay. Let me look into this for you right away."}'

        messages.append({"role": "assistant", "content": raw_response})

        # Parse and execute action
        action = parse_action(raw_response)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if verbose:
            print(f"--- Turn {turn} | Action: {action.action_type} | Reward: {reward:+.3f} ---")
            if action.response_text:
                print(f"Agent: {action.response_text[:200]}")
            if action.refund_amount:
                print(f"  💰 Refund: ${action.refund_amount:.2f}")
            if action.escalation_reason:
                print(f"  📤 Escalation: {action.escalation_reason[:100]}")
            if obs.user_message and not done:
                print(f"Customer: {obs.user_message}")
            print()

    # Grade the episode
    final_score, breakdown = env.grade()

    if verbose:
        print(f"\n📊 RESULTS — {task_id}")
        print(f"   Task Score:       {final_score:.3f} / 1.000")
        print(f"   Cumulative Reward:{total_reward:+.3f}")
        print(f"   Turns Used:       {turn} / {task_def['max_turns']}")
        print(f"   Ticket Status:    {env.state().ticket_status}")
        print(f"   Scoring Breakdown:")
        for k, v in breakdown.items():
            if k != "final_score":
                print(f"     {k}: {v}")

    return {
        "task_id": task_id,
        "difficulty": task_def["difficulty"],
        "final_score": final_score,
        "cumulative_reward": round(total_reward, 4),
        "turns_used": turn,
        "max_turns": task_def["max_turns"],
        "breakdown": breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="CustomerSupportEnv Inference Script")
    parser.add_argument("--task", default=None, help="Specific task to run (default: all)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    # Auth
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Usage: HF_TOKEN=your_token python inference.py")
        sys.exit(1)

    client = OpenAI(
        base_url=HF_BASE_URL,
        api_key=hf_token,
    )

    tasks_to_run = [args.task] if args.task else list(TASKS.keys())
    results = []

    print(f"\n🤖 CustomerSupportEnv Baseline Evaluation")
    print(f"   Model: {args.model}")
    print(f"   Tasks: {tasks_to_run}")

    for task_id in tasks_to_run:
        result = run_task(client, args.model, task_id, verbose=not args.quiet)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("📈 BASELINE SUMMARY")
    print(f"{'='*60}")
    avg_score = sum(r["final_score"] for r in results) / len(results)
    for r in results:
        bar = "█" * int(r["final_score"] * 20)
        print(f"  {r['task_id']:<30} {r['final_score']:.3f}  {bar}")
    print(f"  {'AVERAGE':<30} {avg_score:.3f}")

    # Save results
    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "results": results,
            "average_score": round(avg_score, 4),
        }, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return avg_score


if __name__ == "__main__":
    main()