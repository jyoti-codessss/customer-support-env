"""
evaluate.py — Complete automated evaluation script for judges.

Tests all 5 tasks end-to-end via the REST API.
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
MAGENTA = "\033[95m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
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
    ],

    "subscription_cancellation_hard": [
        {
            "action_type": "respond",
            "response_text": "I'm sorry to hear you're considering cancelling, David. As a valued customer of over 2 years, your loyalty means a lot to us. I can confirm your account ACC-12750 on the Enterprise plan. Before I process anything, could you help me understand what's prompting this decision? I want to make sure we explore all options for you."
        },
        {
            "action_type": "respond",
            "response_text": "I completely understand. With your team size going from 50 to 15, the Enterprise plan at $499/month doesn't fit anymore. That makes total sense. I'd like to offer you a few options: We can downgrade you to our Business plan at $199/month which would save you $300/month, or I can offer you a special loyalty discount of 40% off your current Enterprise plan for the next 6 months, bringing it to $299/month. Would either of those work for you?"
        },
        {
            "action_type": "respond",
            "response_text": "I appreciate your dedication to finding the right solution. Let me also offer 3 months completely free on the Business plan at $199/month as a thank-you for your 2 years of loyalty. This way you can try the adjusted plan with zero risk. If it still doesn't meet your needs after 3 months, we'll process the cancellation immediately with no questions asked."
        },
        {
            "action_type": "close",
            "response_text": "I've processed your plan change. Your account has been updated to reflect the new arrangement. Thank you for giving us a chance to continue serving your team, David. I'll schedule a follow-up check-in for 30 days from now to make sure everything is working well. If you need anything before then, don't hesitate to reach out. We truly appreciate your loyalty."
        }
    ],

    "vip_account_recovery_expert": [
        {
            "action_type": "respond",
            "response_text": "I deeply apologize for this critical situation, Ms. Rivera. As a Platinum customer, this is being treated as our highest priority — absolutely top priority. I understand your team of 85 people is blocked and the business impact is significant. For security purposes, I need to verify your identity through multiple methods before I can take action on this account. Could you please confirm: (1) the phone number on file for this account, and (2) the last 4 digits of the corporate card used for billing?"
        },
        {
            "action_type": "respond",
            "response_text": "Thank you for confirming your identity, Ms. Rivera. I've verified you are the authorized account holder. I am immediately initiating the account unlock and recovery process. I'm also flagging this for an urgent security audit — our team will review the full audit trail of all recent changes to your account, including the unauthorized email change. I'm working on getting your team interim access right away while we restore full access."
        },
        {
            "action_type": "refund",
            "response_text": "Given the severity of this situation and the $50,000+ in productivity losses your team has experienced, I am applying an immediate service credit of $2,499.00 — a full month's compensation — to your Platinum account. This is the least we can do for the downtime you've suffered.",
            "refund_amount": 2499.00
        },
        {
            "action_type": "escalate",
            "response_text": "I am escalating this immediately to our VIP Support and Priority Engineering team. They will: (1) complete the full account recovery within the hour, (2) conduct a forensic security review of the breach, (3) restore your original email and credentials, and (4) provide a detailed incident report. You will receive a direct call from our senior VIP team lead within 30 minutes. Again, I'm deeply sorry for this experience — this is completely unacceptable for a valued Platinum customer.",
            "escalation_reason": "VIP Support & Priority Engineering - Platinum account ACC-VIP-0042 compromised. CTO Alexandra Rivera, NexaFlow Technologies. 85 users locked out, $50K+ productivity loss. Recovery email changed without authorization — suspected breach. Identity verified via phone + card. Account unlock initiated. $2,499 service credit applied. Urgent: full forensic audit trail review needed. Restore access within 1 hour."
        }
    ],
}


def run_task_evaluation(base_url: str, task_id: str) -> dict:
    """Run a complete task evaluation and return results."""
    diff_colors = {
        "easy": GREEN, "medium": YELLOW, "hard": RED, "expert": MAGENTA
    }

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

    print(f"\n{CYAN}{BOLD}🎧 CustomerSupportEnv — Advanced Judge Evaluation{RESET}")
    print(f"   API: {base_url}")
    print(f"   Tasks: {len(TASK_SCRIPTS)} (Easy → Expert)")

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
    print(f"\n{CYAN}{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}📊 FINAL RESULTS{RESET}")
    print(f"{CYAN}{BOLD}{'='*60}{RESET}")

    avg = sum(r["score"] for r in results) / len(results)
    for r in results:
        bar = "█" * int(r["score"] * 20)
        color = GREEN if r["score"] >= 0.7 else YELLOW if r["score"] >= 0.4 else RED
        print(f"  {r['task_id']:<40} {color}{r['score']:.3f}{RESET}  {bar}")

    avg_color = GREEN if avg >= 0.7 else YELLOW if avg >= 0.4 else RED
    print(f"  {'AVERAGE':<40} {avg_color}{avg:.3f}{RESET}")

    # Check original 3 tasks specifically
    original_tasks = ["billing_dispute_easy", "technical_outage_medium", "fraud_complaint_hard"]
    original_results = [r for r in results if r["task_id"] in original_tasks]
    if original_results:
        orig_avg = sum(r["score"] for r in original_results) / len(original_results)
        print(f"\n  {CYAN}Original 3 Tasks Average: {GREEN if orig_avg >= 0.9 else YELLOW}{orig_avg:.3f}{RESET}")

    # Save results
    output = {
        "api_url": base_url,
        "results": results,
        "average_score": round(avg, 4),
        "total_tasks": len(results),
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to evaluation_results.json")
    print(f"\n{GREEN}{BOLD}🎉 Evaluation complete!{RESET}\n")

    return avg


if __name__ == "__main__":
    main()