"""
evaluate.py — Complete automated evaluation script for judges.

Tests all 3 tasks end-to-end via the REST API.
No manual session_id needed — everything is automated.

Usage:
    # Local test:
    python evaluate.py --base-url http://localhost:7860

    # HuggingFace Space test:
    python evaluate.py --base-url https://jyoti-6-customer-support-env.hf.space
"""

import requests
import argparse
import json
import sys

# ── Color output ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

# ── Pre-defined optimal actions for each task ─────────────────
# These simulate a perfect agent so judges can see max scores

TASK_SCRIPTS = {
    "billing_dispute_easy": [
        {
            "action_type": "respond",
            "response_text": "I'm so sorry for this billing error Sarah. I completely understand your frustration. I can see you are on the Basic plan at $29.99 and were incorrectly charged $49.99. I will fix this right away."
        },
        {
            "action_type": "refund",
            "response_text": "I have issued a $20.00 refund for the overcharge. Your Basic plan at $29.99 per month remains active and unchanged.",
            "refund_amount": 20.00
        },
        {
            "action_type": "close",
            "response_text": "Your billing issue is now fully resolved. The $20 refund will appear in 3-5 business days. Your Basic plan is confirmed. I apologize again for the inconvenience!"
        }
    ],

    "technical_outage_medium": [
        {
            "action_type": "respond",
            "response_text": "I understand how frustrating this must be for your entire team. I apologize for the impact on your business. To help our Engineering team fix this, could you please check the browser console for any error messages? Also please confirm the exact URL that is broken."
        },
        {
            "action_type": "respond",
            "response_text": "Thank you for the console error details. I can see this is a known issue affecting Business accounts created before 2024. Our Engineering team needs to fix this on the server side. I will also apply a service credit to your account for the 3 days of downtime."
        },
        {
            "action_type": "refund",
            "response_text": "I have applied a $20 service credit to your account for the downtime.",
            "refund_amount": 20.00
        },
        {
            "action_type": "escalate",
            "response_text": "I am escalating this to our Engineering team right now as highest priority. They will contact you within 2 hours.",
            "escalation_reason": "Engineering team escalation - Business account dashboard 403 error affecting 12 users for 3 days. Console shows 403 forbidden. Account created March 2023. URL: app.example.com/dashboard"
        }
    ],

    "fraud_complaint_hard": [
        {
            "action_type": "respond",
            "response_text": "I am so sorry this happened to you. This is our highest priority. Before I can take any action on your account, I need to verify your identity to protect you. Could you please confirm your date of birth and the last 4 digits of your card on file?"
        },
        {
            "action_type": "respond",
            "response_text": "Thank you for verifying your identity. I have confirmed your account. I am immediately locking and suspending this account to prevent any further unauthorized access or charges. Your account is now secured."
        },
        {
            "action_type": "refund",
            "response_text": "I have initiated a full refund of $448.00 for all three unauthorized charges ($149 + $89 + $210). This will appear in 3-5 business days.",
            "refund_amount": 448.00
        },
        {
            "action_type": "escalate",
            "response_text": "I am escalating this to our Fraud and Security team. They will investigate the breach and send you a secure link to recover your account with a new email and password. You will receive an email at your original address within 1 hour with next steps to regain access.",
            "escalation_reason": "Fraud & Security team - Full account compromise. Unauthorized email change and $448 in fraudulent charges. Identity verified. Account locked. Full refund issued. Customer needs account recovery assistance."
        }
    ]
}


def run_task_evaluation(base_url: str, task_id: str) -> dict:
    """Run a complete task evaluation and return results."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TASK: {task_id}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    # ── Step 1: Reset ────────────────────────────────────────
    try:
        reset_resp = requests.post(
            f"{base_url}/reset",
            json={"task_id": task_id},
            timeout=30
        )
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
    except Exception as e:
        print(f"{RED}❌ /reset failed: {e}{RESET}")
        return {"task_id": task_id, "score": 0.0, "error": str(e)}

    session_id = reset_data["session_id"]
    obs = reset_data["observation"]
    print(f"{GREEN}✅ Session started: {session_id[:16]}...{RESET}")
    print(f"   Customer: {obs['user_message'][:80]}...")

    # ── Step 2: Execute action script ────────────────────────
    actions = TASK_SCRIPTS[task_id]
    total_reward = 0.0
    final_score = 0.0
    breakdown = {}
    done = False

    for i, action_data in enumerate(actions):
        if done:
            break

        try:
            step_resp = requests.post(
                f"{base_url}/step",
                json={
                    "session_id": session_id,
                    "action": action_data
                },
                timeout=30
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()

            reward = step_data["reward"]
            done = step_data["done"]
            total_reward += reward

            action_type = action_data["action_type"]
            reward_color = GREEN if reward >= 0 else RED
            print(f"   Turn {i+1} | {action_type:<10} | Reward: {reward_color}{reward:+.3f}{RESET}")

            # If done, get final score
            if done and "final_score" in step_data:
                final_score = step_data["final_score"]
                breakdown = step_data.get("grade_breakdown", {})

        except Exception as e:
            print(f"{RED}   ❌ Step {i+1} failed: {e}{RESET}")

    # ── Step 3: Grade if not done yet ───────────────────────
    if not done:
        try:
            grade_resp = requests.post(
                f"{base_url}/grade/{session_id}",
                timeout=30
            )
            if grade_resp.status_code == 200:
                grade_data = grade_resp.json()
                final_score = grade_data["score"]
                breakdown = grade_data["breakdown"]
        except:
            pass

    # ── Print results ────────────────────────────────────────
    score_color = GREEN if final_score >= 0.7 else YELLOW if final_score >= 0.4 else RED
    print(f"\n   {score_color}Score: {final_score:.3f} / 1.000{RESET}")
    print(f"   Cumulative Reward: {total_reward:+.3f}")
    if breakdown:
        print(f"   Breakdown:")
        for k, v in breakdown.items():
            if k != "final_score":
                icon = "✅" if v is True or (isinstance(v, (int, float)) and v > 0) else "❌"
                print(f"     {icon} {k}: {v}")

    return {
        "task_id": task_id,
        "score": final_score,
        "cumulative_reward": round(total_reward, 4),
        "breakdown": breakdown
    }


def main():
    parser = argparse.ArgumentParser(description="CustomerSupportEnv Judge Evaluation Script")
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="Base URL of the environment API"
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Run specific task only (default: all)"
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    print(f"\n{BLUE}🎧 CustomerSupportEnv — Judge Evaluation{RESET}")
    print(f"   API: {base_url}")

    # ── Health check ─────────────────────────────────────────
    try:
        health = requests.get(f"{base_url}/health", timeout=10)
        if health.status_code == 200:
            print(f"   {GREEN}✅ API is healthy{RESET}")
        else:
            print(f"   {RED}❌ API health check failed{RESET}")
            sys.exit(1)
    except Exception as e:
        print(f"   {RED}❌ Cannot connect to API: {e}{RESET}")
        print(f"   Make sure the server is running at {base_url}")
        sys.exit(1)

    # ── Run tasks ─────────────────────────────────────────────
    tasks = [args.task] if args.task else list(TASK_SCRIPTS.keys())
    results = []

    for task_id in tasks:
        result = run_task_evaluation(base_url, task_id)
        results.append(result)

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}📊 FINAL RESULTS{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    avg = sum(r["score"] for r in results) / len(results)
    for r in results:
        bar = "█" * int(r["score"] * 20)
        color = GREEN if r["score"] >= 0.7 else YELLOW if r["score"] >= 0.4 else RED
        print(f"  {r['task_id']:<35} {color}{r['score']:.3f}{RESET}  {bar}")

    avg_color = GREEN if avg >= 0.7 else YELLOW if avg >= 0.4 else RED
    print(f"  {'AVERAGE':<35} {avg_color}{avg:.3f}{RESET}")

    # Save results
    output = {
        "api_url": base_url,
        "results": results,
        "average_score": round(avg, 4)
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to evaluation_results.json")
    print(f"\n{GREEN}🎉 Evaluation complete!{RESET}\n")

    return avg


if __name__ == "__main__":
    main()