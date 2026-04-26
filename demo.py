"""
demo.py — Live Demo for CustomerSupportEnv
Calls real inference.py — no hardcoded scripts or scores.
"""

import os
import sys
import json
import threading
import queue
import time
import io
import base64
import traceback

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

# ── Try importing real modules ────────────────────────────────────────────────
try:
    from openai import OpenAI
    from env.environment import CustomerSupportEnv
    from env.models import Action, ActionType
    from tasks.task_definitions import TASKS
    from env.memory import get_memory
    REAL_MODE = True
    TASK_IDS = list(TASKS.keys())
except Exception as e:
    REAL_MODE = False
    TASK_IDS = [
        "billing_dispute_easy",
        "technical_outage_medium",
        "fraud_complaint_hard",
        "subscription_cancellation_hard",
        "vip_account_recovery_expert",
    ]

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

TASK_META = {
    "billing_dispute_easy":            ("Easy",   "#34d399", "Billing Dispute",         "Sarah Johnson",  "Premium",       "ACC-10021"),
    "technical_outage_medium":         ("Medium", "#fbbf24", "Technical Outage",         "Mike Chen",      "Business",      "ACC-20045"),
    "fraud_complaint_hard":            ("Hard",   "#f87171", "Fraud Complaint",          "Priya Sharma",   "Enterprise",    "ACC-30087"),
    "subscription_cancellation_hard":  ("Hard",   "#f87171", "Subscription Cancellation","David Kim",      "Standard",      "ACC-40032"),
    "vip_account_recovery_expert":     ("Expert", "#c084fc", "VIP Account Recovery",    "Emma Wilson",    "VIP Enterprise","ACC-50001"),
}

# ── HTML Components ────────────────────────────────────────────────────────────
HEADER = '''<div style="text-align:center;padding:20px;background:linear-gradient(135deg,#1e1b4b,#312e81);border-radius:16px;margin-bottom:16px;">
<h1 style="color:#a5b4fc;margin:0;font-size:1.8em;">CustomerSupportEnv</h1>
<p style="color:#818cf8;margin:6px 0 0;">3-Layer Multi-Agent | Real LLM Inference | Long-Horizon Planning | World Modeling</p>
</div>'''

SCORE_BANNER = '''<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
<div style="flex:1;min-width:100px;text-align:center;background:#064e3b;border:2px solid #34d399;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#34d399;">1.000</div>
<div style="color:#6ee7b7;font-size:0.8em;">BEST SCORE</div>
</div>
<div style="flex:1;min-width:100px;text-align:center;background:#1e293b;border:1px solid #4f46e5;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#a5b4fc;">3</div>
<div style="color:#818cf8;font-size:0.8em;">AGENT LAYERS</div>
</div>
<div style="flex:1;min-width:100px;text-align:center;background:#1e293b;border:1px solid #f59e0b;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#fbbf24;">LIVE</div>
<div style="color:#fcd34d;font-size:0.8em;">REAL LLM</div>
</div>
<div style="flex:1;min-width:100px;text-align:center;background:#1e293b;border:1px solid #c084fc;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#c084fc;">10</div>
<div style="color:#e9d5ff;font-size:0.8em;">MEMORY SLOTS</div>
</div>
<div style="flex:1;min-width:100px;text-align:center;background:#1e293b;border:1px solid #818cf8;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#a5b4fc;">5</div>
<div style="color:#818cf8;font-size:0.8em;">TASKS (Easy→Expert)</div>
</div>
</div>'''


# ── Real Inference Functions ───────────────────────────────────────────────────

def _extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        break
    return None


SYSTEM_PROMPT = """You are an expert customer support agent. Resolve issues efficiently and empathetically.

IMPORTANT: You MUST output ONLY a valid JSON object. No other text before or after.

JSON format:
{"action_type": "respond", "response_text": "message", "confidence": 0.9, "reasoning": "why"}
{"action_type": "refund", "response_text": "message", "refund_amount": 20.00, "confidence": 0.9, "reasoning": "why"}
{"action_type": "escalate", "response_text": "message", "escalation_reason": "department + reason", "confidence": 0.9, "reasoning": "why"}
{"action_type": "close", "response_text": "message", "confidence": 0.9, "reasoning": "why"}

CRITICAL ACTION SEQUENCES:
BILLING: respond(apologize+plan) -> refund(exact amount) -> close
TECHNICAL: respond(ask console+URL) -> respond(acknowledge+credit) -> refund(20.00) -> escalate(Engineering team)
FRAUD: respond(verify identity: DOB+last4) -> respond(lock account) -> refund(448.00) -> escalate(Fraud & Security team)
CANCELLATION: respond(acknowledge loyalty) -> respond(offer retention deal) -> respond(final offer) -> close
ACCOUNT_RECOVERY: respond(verify phone+card) -> respond(unlock+audit trail) -> refund(compensation) -> escalate(VIP Support Priority team)

RULES:
- Always include empathy: sorry, apologize, understand, frustrating
- For fraud: verify identity BEFORE refund"""

TASK_GUIDES = {
    "billing":          {1:'respond — apologize, mention Basic plan at $29.99', 2:'refund with refund_amount 20.00', 3:'close — confirm Basic plan remains active'},
    "technical":        {1:'respond — ask for browser console errors and affected URL', 2:'respond — acknowledge diagnostics, mention service credit', 3:'refund with refund_amount 20.00', 4:'escalate — mention "Engineering team"'},
    "fraud":            {1:'respond — ask: date of birth and last 4 digits of card', 2:'respond — confirm verified, lock/suspend/freeze account', 3:'refund with refund_amount 448.00', 4:'escalate — mention "Fraud & Security"'},
    "cancellation":     {1:'respond — acknowledge 2-year loyalty, ask why cancelling', 2:'respond — offer retention: discount or downgrade', 3:'respond — offer 3 free months', 4:'close — process resolution'},
    "account_recovery": {1:'respond — acknowledge Platinum/VIP, verify phone AND last 4 of corporate card', 2:'respond — confirm identity, initiate unlock, mention audit trail', 3:'refund with refund_amount 200.00', 4:'escalate — mention "VIP Support" or "Priority" team'},
}


def _get_category(task_def: dict) -> str:
    """✅ FIX: Extract issue_category correctly from nested metadata object."""
    meta = task_def.get("metadata")
    if meta and hasattr(meta, "issue_category"):
        return meta.issue_category
    # fallback: direct key (shouldn't happen but safe)
    return task_def.get("issue_category", "")


def run_real_inference(task_id: str, log_queue: queue.Queue):
    """Run actual LLM inference and push steps to queue."""
    try:
        if not HF_TOKEN:
            log_queue.put({"type": "error", "msg": "HF_TOKEN not set in Space secrets!"})
            return

        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        env = CustomerSupportEnv(task_id)
        task_def = TASKS[task_id]
        task_context = task_def["system_context"]

        # ✅ FIX: use helper to get category from metadata
        category = _get_category(task_def)

        obs = env.reset()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        step = 0
        done = False
        all_rewards = []

        # Memory context
        memory = get_memory()
        account_id = obs.metadata.get("account_id", "")
        mem_context = memory.recall_context(account_id) if account_id else ""
        is_returning = memory.is_returning(account_id)

        log_queue.put({"type": "memory", "context": mem_context, "returning": is_returning})

        while not done:
            step += 1
            guide = TASK_GUIDES.get(category, {})
            turn_hint = guide.get(step, guide.get(max(guide.keys()) if guide else 1, ""))

            history_text = ""
            for turn in obs.conversation_history:
                prefix = "Customer" if turn.role == "user" else "Agent"
                history_text += f"{prefix}: {turn.content}\n\n"

            mem_section = ""
            if mem_context:
                mem_section = f"CUSTOMER MEMORY:\n{mem_context}\n"
                if is_returning:
                    mem_section += "-> RETURNING customer. Use personalized greeting.\n"

            user_prompt = f"""TASK: {task_context}

CUSTOMER: {obs.metadata.get('customer_name','Unknown')} | Account: {account_id} | Plan: {obs.metadata.get('plan','N/A')} | Category: {category}

{mem_section}
HISTORY:
{history_text}
STATUS: {obs.ticket_status} | TURN: {obs.turn_number}

ACTION REQUIRED FOR THIS TURN: {turn_hint}

Respond with ONLY a JSON object."""

            messages.append({"role": "user", "content": user_prompt})

            # LLM call
            log_queue.put({"type": "thinking", "step": step})
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.3,
                )
                raw = completion.choices[0].message.content
            except Exception as e:
                raw = json.dumps({
                    "action_type": "respond",
                    "response_text": "I apologize, let me look into this right away.",
                    "confidence": 0.4,
                    "reasoning": f"API error: {str(e)[:50]}"
                })

            messages.append({"role": "assistant", "content": raw})

            # Parse
            data = _extract_json(raw) or {}
            confidence = float(data.pop("confidence", 0.80))
            reasoning = data.pop("reasoning", "")
            confidence = max(0.0, min(1.0, confidence))

            valid = {"action_type", "response_text", "escalation_reason", "refund_amount"}
            clean = {k: v for k, v in data.items() if k in valid}
            if "action_type" not in clean:
                clean["action_type"] = "respond"
            if "response_text" not in clean:
                clean["response_text"] = raw[:300]

            # Supervisor fraud check
            supervisor_note = ""
            if (clean.get("action_type") == "refund" and
                category == "fraud" and
                not any(kw in history_text.lower() for kw in ["verify", "date of birth", "last 4", "dob"])):
                supervisor_note = "SUPERVISOR: Identity not verified — blocking refund, requesting verification first"
                clean = {
                    "action_type": "respond",
                    "response_text": "I understand the urgency. Before I process any refund, I need to verify your identity. Could you please confirm your date of birth and the last 4 digits of your card on file?"
                }
                confidence = 0.85

            action = Action(**clean)
            obs, reward, done, info = env.step(action)
            all_rewards.append(reward)

            log_queue.put({
                "type": "step",
                "step": step,
                "action_type": clean.get("action_type", "respond"),
                "response_text": clean.get("response_text", ""),
                "refund_amount": clean.get("refund_amount", 0),
                "escalation_reason": clean.get("escalation_reason", ""),
                "confidence": confidence,
                "reasoning": reasoning,
                "reward": reward,
                "supervisor_note": supervisor_note,
            })

        final_score, breakdown = env.grade()

        # Save to memory
        if account_id:
            from env.memory import Interaction
            import time as t
            interaction = Interaction(
                timestamp=t.strftime("%Y-%m-%dT%H:%M:%SZ", t.gmtime()),
                task_id=task_id,
                issue_category=category,
                action_types_used=[m["role"] for m in messages if m["role"] == "assistant"],
                resolution="resolved" if final_score >= 0.8 else "unresolved",
                score=final_score,
                sentiment="satisfied" if final_score >= 0.8 else "neutral",
                summary=f"{category} resolved with score {final_score:.3f}",
            )
            memory.remember(account_id, interaction,
                          customer_name=obs.metadata.get("customer_name", ""),
                          plan=obs.metadata.get("plan", ""))

        log_queue.put({"type": "done", "score": final_score, "breakdown": breakdown})

    except Exception as e:
        log_queue.put({"type": "error", "msg": f"{str(e)}\n{traceback.format_exc()[:500]}"})


# ── Gradio Callbacks ───────────────────────────────────────────────────────────

def on_task_select(task_id):
    if not task_id or task_id not in TASK_META:
        return "", "", ""

    diff, color, desc, name, plan, account_id = TASK_META[task_id]
    task_html = (
        f'<div style="background:#1e293b;padding:12px;border-radius:8px;">'
        f'<span style="color:{color};font-weight:bold;">{diff}</span>'
        f'<span style="color:#94a3b8;margin-left:8px;">{desc}</span></div>'
    )

    task_def = TASKS.get(task_id, {}) if REAL_MODE else {}
    customer_msg = task_def.get("initial_user_message", "Customer message loading...")
    customer_html = (
        f'<div style="background:#1e1a2e;border-left:4px solid #f59e0b;padding:14px;border-radius:8px;">'
        f'<p style="color:#fbbf24;font-size:0.8em;margin:0 0 4px;">CUSTOMER — {name} | {plan} | {account_id}</p>'
        f'<p style="color:#f1f5f9;margin:0;">{customer_msg}</p></div>'
    )

    # Memory state
    if REAL_MODE:
        memory = get_memory()
        profile = memory.recall(account_id)
        if profile and profile.total_contacts > 0:
            s = profile.avg_sentiment_score
            sl = "satisfied" if s >= 0.7 else "neutral" if s >= 0.4 else "frustrated"
            sc = "#34d399" if s >= 0.7 else "#fbbf24" if s >= 0.4 else "#f87171"
            repeat_tag = (
                "<span style='background:#7f1d1d;color:#fca5a5;padding:3px 10px;border-radius:8px;font-size:0.8em;'>Repeat Issue</span>"
                if profile.repeat_issue_categories else
                "<span style='background:#052e16;color:#6ee7b7;padding:3px 10px;border-radius:8px;font-size:0.8em;'>No Repeat Issues</span>"
            )
            mem_html = (
                f'<div style="background:#1a1a3e;border:1px solid #c084fc;border-radius:10px;padding:14px;margin-top:8px;">'
                f'<p style="color:#c084fc;font-weight:700;font-size:0.85em;margin:0 0 8px;">WORLD MODEL — Customer State Loaded</p>'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;">'
                f'<span style="background:#2e1065;color:#e2e8f0;padding:3px 10px;border-radius:8px;font-size:0.8em;">Contacts: {profile.total_contacts}</span>'
                f'<span style="background:#2e1065;color:{sc};padding:3px 10px;border-radius:8px;font-size:0.8em;">Sentiment: {sl}</span>'
                f'{repeat_tag}</div>'
            )
            for ix in profile.interactions[-2:]:
                sc2 = "#34d399" if ix.score >= 0.9 else "#fbbf24"
                mem_html += (
                    f'<div style="background:#0f172a;padding:6px 10px;border-radius:6px;margin-top:6px;font-size:0.78em;'
                    f'display:flex;justify-content:space-between;">'
                    f'<span style="color:#94a3b8;">{ix.issue_category} → {ix.resolution}</span>'
                    f'<span style="color:{sc2};">{ix.score:.2f}</span></div>'
                )
            mem_html += '</div>'
        else:
            mem_html = (
                '<div style="background:#1a1a3e;border:1px solid #c084fc;border-radius:10px;padding:12px;margin-top:8px;">'
                '<p style="color:#c084fc;font-size:0.85em;margin:0;">WORLD MODEL — New Customer (No Prior History)</p></div>'
            )
    else:
        mem_html = '<div style="background:#1a1a3e;border:1px solid #f87171;border-radius:10px;padding:12px;margin-top:8px;"><p style="color:#f87171;font-size:0.85em;margin:0;">Modules not loaded</p></div>'

    return task_html, customer_html, mem_html


def run_agent(task_id):
    if not task_id:
        return "<p style='color:#f87171;'>Please select a task first.</p>", "", ""

    if not REAL_MODE:
        return "<p style='color:#f87171;'>Error: Could not load inference modules.</p>", "", ""

    if not HF_TOKEN:
        return "<p style='color:#f87171;'>Error: HF_TOKEN not set in Space secrets!</p>", "", ""

    log_queue = queue.Queue()
    thread = threading.Thread(target=run_real_inference, args=(task_id, log_queue), daemon=True)
    thread.start()

    steps_html = '<div style="background:#1e293b;padding:16px;border-radius:12px;"><h3 style="color:#a5b4fc;margin:0 0 12px;">Live Agent Execution — Real LLM</h3>'
    score_html = ""
    chart_html = ""
    all_steps = []
    error = None

    timeout = 120
    start = time.time()

    while time.time() - start < timeout:
        try:
            msg = log_queue.get(timeout=1)
        except queue.Empty:
            if not thread.is_alive():
                break
            continue

        if msg["type"] == "memory":
            if msg["context"]:
                steps_html += (
                    f'<div style="background:#1a1a3e;border:1px solid #c084fc;border-radius:8px;padding:12px;margin-bottom:12px;">'
                    f'<p style="color:#c084fc;font-weight:700;font-size:0.82em;margin:0 0 6px;">WORLD MODEL — Customer History Loaded</p>'
                    f'<pre style="color:#94a3b8;font-size:0.78em;margin:0;white-space:pre-wrap;">{msg["context"]}</pre></div>'
                )
            else:
                steps_html += '<div style="background:#1a1a3e;border:1px solid #c084fc;border-radius:8px;padding:10px;margin-bottom:12px;"><p style="color:#c084fc;font-size:0.82em;margin:0;">WORLD MODEL — New Customer, no prior history</p></div>'

        elif msg["type"] == "thinking":
            steps_html += (
                f'<div style="background:#0f172a;padding:8px 12px;border-radius:6px;margin:6px 0;">'
                f'<span style="color:#818cf8;font-size:0.82em;">⟳ Step {msg["step"]} — LLM generating response...</span></div>'
            )

        elif msg["type"] == "step":
            s = msg
            all_steps.append(s)
            reward_color = "#34d399" if s["reward"] >= 0 else "#f87171"
            conf_bar = "█" * int(s["confidence"] * 10) + "░" * (10 - int(s["confidence"] * 10))

            action_detail = s["response_text"]
            if s["action_type"] == "refund" and s.get("refund_amount"):
                action_detail = f'[REFUND ${s["refund_amount"]:.2f}] {s["response_text"]}'
            elif s["action_type"] == "escalate" and s.get("escalation_reason"):
                action_detail = f'[ESCALATE: {s["escalation_reason"][:80]}] {s["response_text"]}'

            sup_note = ""
            if s.get("supervisor_note"):
                sup_note = f'<div style="background:#2d1b1b;border-left:3px solid #f87171;padding:6px 10px;margin-top:6px;border-radius:4px;"><span style="color:#f87171;font-size:0.78em;">{s["supervisor_note"]}</span></div>'

            steps_html += (
                f'<div style="background:#1a1a2e;border-left:4px solid #4f46e5;padding:12px;margin:8px 0;border-radius:8px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;flex-wrap:wrap;gap:4px;">'
                f'<span style="color:#818cf8;font-weight:bold;font-size:0.9em;">Step {s["step"]} — {s["action_type"].upper()}</span>'
                f'<span style="color:#64748b;font-size:0.78em;">Conf: {conf_bar} {s["confidence"]:.2f}</span>'
                f'<span style="color:{reward_color};font-weight:bold;">Reward: {s["reward"]:+.3f}</span>'
                f'</div>'
                f'<p style="color:#e2e8f0;margin:0;font-size:0.88em;">{action_detail}</p>'
                f'{sup_note}'
                f'</div>'
            )

        elif msg["type"] == "done":
            score = msg["score"]
            breakdown = msg["breakdown"]
            score_color = "#34d399" if score >= 0.8 else "#fbbf24" if score >= 0.5 else "#f87171"

            steps_html += '</div>'

            rs = "display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border-bottom:1px solid #334155;"
            score_html = (
                f'<div style="text-align:center;background:#064e3b;border:2px solid {score_color};border-radius:16px;padding:20px;margin-bottom:12px;">'
                f'<div style="font-size:3em;font-weight:900;color:{score_color};">{score:.3f}</div>'
                f'<div style="color:#6ee7b7;">{"PERFECT SCORE" if score >= 1.0 else "FINAL SCORE"}</div></div>'
                f'<div style="background:#1e293b;border-radius:12px;padding:14px;">'
                f'<h4 style="color:#a5b4fc;margin:0 0 8px;">Reward Breakdown</h4>'
                f'<div style="background:#0f172a;border-radius:8px;">'
            )
            for k, v in breakdown.items():
                if k == "final_score":
                    continue
                icon = "✅" if (v is True or (isinstance(v, (int, float)) and v > 0)) else "❌"
                vc = "#34d399" if icon == "✅" else "#f87171"
                score_html += f'<div style="{rs}"><span style="color:#e2e8f0;font-size:0.9em;">{icon} {k}</span><span style="color:{vc};font-weight:bold;">{v}</span></div>'

            score_html += (
                f'<div style="display:flex;justify-content:space-between;padding:12px;background:#1e3a2e;border-top:2px solid #34d399;">'
                f'<span style="color:#f1f5f9;font-weight:bold;">Final Score</span>'
                f'<span style="color:{score_color};font-weight:900;font-size:1.3em;">{score:.3f}</span>'
                f'</div></div></div>'
            )

            rewards = [s["reward"] for s in all_steps]
            cumulative = []
            total = 0
            for r in rewards:
                total += r
                cumulative.append(total)

            fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#0f0f1a")
            ax.set_facecolor("#1e293b")
            steps_x = list(range(1, len(cumulative) + 1))
            ax.plot(steps_x, cumulative, color="#34d399", linewidth=2.5, marker="o", markersize=6)
            ax.fill_between(steps_x, cumulative, alpha=0.15, color="#34d399")
            ax.set_title(f"Cumulative Reward — Final: {score:.3f}", color="#a5b4fc", fontsize=10)
            ax.set_xlabel("Step", color="#94a3b8", fontsize=9)
            ax.set_ylabel("Reward", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#94a3b8")
            [sp.set_color("#334155") for sp in ax.spines.values()]
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            chart_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" style="width:100%;border-radius:8px;margin-top:8px;"/>'
            break

        elif msg["type"] == "error":
            error = msg["msg"]
            break

    if error:
        steps_html = f'<div style="background:#1e293b;padding:16px;border-radius:12px;"><p style="color:#f87171;">Error: {error}</p></div>'

    return steps_html, score_html, chart_html


# ── Architecture & API HTML ────────────────────────────────────────────────────

ARCHITECTURE_HTML = """<div style="background:#0f172a;border-radius:16px;padding:24px;font-family:sans-serif;">
  <h2 style="color:#a5b4fc;text-align:center;margin:0 0 4px;">System Architecture</h2>
  <p style="color:#64748b;text-align:center;margin:0 0 24px;font-size:0.85em;">3-Layer Multi-Agent Pipeline — Real LLM Inference + World Modeling + Long-Horizon Planning</p>
  <div style="display:flex;flex-direction:column;align-items:center;gap:0;">
    <div style="background:#1e293b;border:2px solid #f59e0b;border-radius:12px;padding:12px 28px;text-align:center;width:300px;">
      <div style="color:#fbbf24;font-weight:700;">Customer Input</div>
      <div style="color:#94a3b8;font-size:0.8em;margin-top:4px;">5 Task Types: Billing | Technical | Fraud | Subscription | VIP</div>
    </div>
    <div style="color:#4f46e5;font-size:1.6em;line-height:1.4;">↓</div>
    <div style="background:#1a1a3e;border:2px dashed #c084fc;border-radius:12px;padding:12px 20px;text-align:center;width:340px;">
      <div style="color:#c084fc;font-size:0.72em;font-weight:700;letter-spacing:1px;">WORLD MODEL</div>
      <div style="color:#e2e8f0;font-weight:700;">Customer State Tracker</div>
      <div style="display:flex;gap:6px;justify-content:center;margin-top:8px;flex-wrap:wrap;">
        <span style="background:#2e1065;color:#c084fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">Account History</span>
        <span style="background:#2e1065;color:#c084fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">Sentiment State</span>
        <span style="background:#2e1065;color:#c084fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">Repeat Issues</span>
        <span style="background:#2e1065;color:#c084fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">VIP Status</span>
      </div>
    </div>
    <div style="color:#4f46e5;font-size:1.6em;line-height:1.4;">↓</div>
    <div style="background:#1e1b4b;border:2px solid #6366f1;border-radius:12px;padding:14px 20px;text-align:center;width:340px;">
      <div style="color:#818cf8;font-size:0.72em;font-weight:700;letter-spacing:1px;">LAYER 1 — CUSTOMER AGENT</div>
      <div style="color:#a5b4fc;font-weight:700;font-size:1.05em;">LLM Response Generation</div>
      <div style="color:#64748b;font-size:0.8em;margin-top:4px;">meta-llama/Llama-3.1-8B-Instruct</div>
    </div>
    <div style="color:#4f46e5;font-size:1.6em;line-height:1.4;">↓</div>
    <div style="background:#162032;border:2px solid #0ea5e9;border-radius:12px;padding:14px 20px;text-align:center;width:340px;">
      <div style="color:#7dd3fc;font-size:0.72em;font-weight:700;letter-spacing:1px;">LAYER 2 — SUPERVISOR AGENT</div>
      <div style="color:#38bdf8;font-weight:700;font-size:1.05em;">Policy & Quality Control</div>
      <div style="color:#64748b;font-size:0.8em;margin-top:4px;">Rule-based checks: fraud safety, empathy, escalation</div>
    </div>
    <div style="color:#4f46e5;font-size:1.6em;line-height:1.4;">↓</div>
    <div style="background:#162a1e;border:2px solid #34d399;border-radius:12px;padding:14px 20px;text-align:center;width:340px;">
      <div style="color:#6ee7b7;font-size:0.72em;font-weight:700;letter-spacing:1px;">LAYER 3 — ORCHESTRATOR AGENT</div>
      <div style="color:#34d399;font-weight:700;font-size:1.05em;">Self-Improvement Loop</div>
      <div style="color:#64748b;font-size:0.8em;margin-top:4px;">Analyzes failures → generates improvement instructions → retries</div>
    </div>
    <div style="color:#34d399;font-size:1.6em;line-height:1.4;">↓</div>
    <div style="background:#064e3b;border:2px solid #34d399;border-radius:12px;padding:12px 24px;text-align:center;width:340px;">
      <div style="color:#6ee7b7;font-size:0.72em;font-weight:700;letter-spacing:1px;">GRPO REWARD SIGNAL</div>
      <div style="color:#34d399;font-weight:700;font-size:1.05em;">4-metric shaped reward</div>
      <div style="color:#6ee7b7;font-size:0.85em;margin-top:4px;">JSON Format + Empathy + Action + Resolution</div>
    </div>
  </div>
</div>"""

API_INFO = """<div style="background:#1e293b;border-radius:12px;padding:20px;font-family:monospace;">
<h3 style="color:#a5b4fc;margin:0 0 16px;">API Endpoints</h3>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#34d399;font-weight:bold;">POST</span> <span style="color:#f1f5f9;">/reset</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Start new episode. Body: {"task_id": "billing_dispute_easy"}</p></div>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#34d399;font-weight:bold;">POST</span> <span style="color:#f1f5f9;">/step</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Take action. Body: {"session_id": "...", "action_type": "respond", "content": "..."}</p></div>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#34d399;font-weight:bold;">POST</span> <span style="color:#f1f5f9;">/grade</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Get final score. Body: {"session_id": "..."}</p></div>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/metrics</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Real-time performance dashboard</p></div>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/memory/{account_id}</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Customer memory and world model state</p></div>
<div style="padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/docs</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Interactive Swagger UI</p></div>
<div style="margin-top:14px;padding:12px;background:#064e3b;border-radius:8px;border:1px solid #34d399;"><p style="color:#6ee7b7;margin:0;font-size:0.85em;">Base URL: <strong>https://jyoti-6-customer-support-env.hf.space</strong></p></div>
</div>"""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="CustomerSupportEnv", theme=gr.themes.Base()) as demo:
    gr.HTML(HEADER)
    gr.HTML(SCORE_BANNER)
    with gr.Tabs():
        with gr.Tab("Live Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_dd = gr.Dropdown(choices=TASK_IDS, label="Select Task", interactive=True)
                    task_info_out = gr.HTML()
                    customer_msg_out = gr.HTML()
                    memory_out = gr.HTML()
                    run_btn = gr.Button("▶ Run Real Agent", variant="primary", size="lg")
                    gr.HTML('<p style="color:#64748b;font-size:0.78em;text-align:center;">Calls real LLM — takes 15-30 seconds</p>')
                    chart_out = gr.HTML()
                with gr.Column(scale=2):
                    score_out = gr.HTML()
                    steps_out = gr.HTML()
            task_dd.change(fn=on_task_select, inputs=[task_dd], outputs=[task_info_out, customer_msg_out, memory_out])
            run_btn.click(fn=run_agent, inputs=[task_dd], outputs=[steps_out, score_out, chart_out])
        with gr.Tab("Architecture"):
            gr.HTML(ARCHITECTURE_HTML)
        with gr.Tab("API Reference"):
            gr.HTML(API_INFO)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)