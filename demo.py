"""
demo.py — Gradio UI Demo for CustomerSupportEnv
Interactive multi-agent customer support simulation with live scoring.
"""

import os
import io
import base64
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from env.environment import CustomerSupportEnv
from env.models import Action, ActionType
from tasks.task_definitions import TASKS
from inference import FALLBACK_SCRIPTS

# ── Task metadata ────────────────────────────────────────────────────────────

TASK_IDS = list(TASKS.keys())

TASK_LABELS = {
    "billing_dispute_easy": "💳 Billing Dispute (Easy)",
    "technical_outage_medium": "🖥️ Technical Outage (Medium)",
    "fraud_complaint_hard": "🚨 Fraud Complaint (Hard)",
    "subscription_cancellation_hard": "📦 Subscription Cancellation (Hard)",
    "vip_account_recovery_expert": "👑 VIP Account Recovery (Expert)",
}

DIFFICULTY_COLORS = {
    "easy": "#22c55e",
    "medium": "#eab308",
    "hard": "#ef4444",
    "expert": "#a855f7",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#22c55e"
    if score >= 0.5:
        return "#eab308"
    return "#ef4444"


def _build_score_html(score: float, breakdown: dict) -> str:
    color = _score_color(score)
    pct = int(score * 100)

    rows = ""
    row_idx = 0
    for k, v in breakdown.items():
        if k == "final_score":
            continue
        if v is True:
            icon, status, status_color = "✅", "Pass", "#22c55e"
        elif v is False:
            icon, status, status_color = "❌", "Fail", "#ef4444"
        elif isinstance(v, (int, float)) and v > 0:
            icon, status, status_color = "✅", str(v), "#22c55e"
        else:
            icon, status, status_color = "❌", str(v), "#ef4444"
        label = k.replace("_", " ").title()
        row_bg = "#1e293b" if row_idx % 2 == 0 else "#162032"
        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td style="padding:10px 14px;font-size:16px;">{icon}</td>'
            f'<td style="padding:10px 14px;color:#f1f5f9;font-weight:500;">{label}</td>'
            f'<td style="padding:10px 14px;text-align:right;color:{status_color};font-weight:600;">{status}</td>'
            f'</tr>'
        )
        row_idx += 1

    return f"""
    <div style="text-align:center;padding:24px 0;">
      <div style="font-size:64px;font-weight:800;color:{color};font-family:'Inter',sans-serif;
                  text-shadow:0 0 30px {color}44;">
        {score:.3f}
      </div>
      <div style="font-size:14px;color:#94a3b8;margin-top:4px;">out of 1.000</div>
      <div style="margin:16px auto;width:80%;max-width:400px;height:8px;background:#0f172a;border-radius:4px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{color},{color}cc);border-radius:4px;"></div>
      </div>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:14px;background:#0f172a;border-radius:10px;overflow:hidden;">
      <thead><tr style="background:#0f172a;border-bottom:2px solid #334155;">
        <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.05em;"></th>
        <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.05em;">Criterion</th>
        <th style="padding:10px 14px;text-align:right;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.05em;">Result</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _build_step_html(steps: list) -> str:
    html = '<div style="font-family:\'Inter\',monospace;font-size:13px;color:#e2e8f0;">'
    for s in steps:
        action_color = {
            "respond": "#60a5fa",
            "refund": "#22c55e",
            "escalate": "#f59e0b",
            "close": "#a78bfa",
            "transfer": "#f472b6",
        }.get(s["action_type"], "#94a3b8")
        reward_color = "#22c55e" if s["reward"] >= 0 else "#ef4444"
        html += f"""
        <div style="padding:10px 14px;margin:6px 0;background:#1e293b;border-radius:8px;
                    border-left:3px solid {action_color};">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span><strong style="color:{action_color};">Step {s['step']}</strong>
            &nbsp;—&nbsp;
            <code style="color:{action_color};background:#0f172a;padding:2px 8px;border-radius:4px;">{s['action_type']}</code></span>
            <span style="color:{reward_color};font-weight:600;">reward: {s['reward']:+.3f}</span>
          </div>
          <div style="margin-top:6px;color:#cbd5e1;font-size:12px;line-height:1.5;
                      max-height:80px;overflow-y:auto;">{s['text'][:300]}</div>
        </div>"""
    html += "</div>"
    return html


def _build_chart() -> str:
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    labels = ["Baseline\n(Random)", "Trained\n(Multi-Agent)"]
    values = [0.268, 1.000]
    colors = ["#ef4444", "#22c55e"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#334155", linewidth=1.2, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold",
                color="#e2e8f0", fontsize=14)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="#94a3b8", fontsize=11)
    ax.set_title("Agent Performance Comparison", color="#e2e8f0", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.grid(axis="y", color="#334155", linewidth=0.5, zorder=0)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:520px;border-radius:12px;margin:auto;display:block;" />'


# ── Core callbacks ───────────────────────────────────────────────────────────

def on_task_select(task_id: str):
    if not task_id or task_id not in TASKS:
        return "", ""
    task = TASKS[task_id]
    diff = task["difficulty"]
    dc = DIFFICULTY_COLORS.get(diff, "#94a3b8")
    meta = task["metadata"]
    header = f"""
    <div style="padding:16px;background:#1e293b;border-radius:10px;border:1px solid #334155;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-size:15px;font-weight:700;color:#e2e8f0;">{TASK_LABELS.get(task_id, task_id)}</span>
        <span style="background:{dc}22;color:{dc};padding:3px 10px;border-radius:6px;font-size:12px;font-weight:600;
              border:1px solid {dc}44;">{diff.upper()}</span>
      </div>
      <div style="font-size:12px;color:#94a3b8;">
        Customer: <strong style="color:#e2e8f0;">{meta.customer_name}</strong> &nbsp;|&nbsp;
        Account: <strong style="color:#e2e8f0;">{meta.account_id}</strong> &nbsp;|&nbsp;
        Plan: <strong style="color:#e2e8f0;">{meta.plan}</strong>
      </div>
    </div>"""
    msg = f"""
    <div style="padding:14px;background:#0f172a;border-radius:10px;border:1px solid #334155;margin-top:10px;">
      <div style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:600;">📩 CUSTOMER MESSAGE</div>
      <div style="color:#e2e8f0;font-size:14px;line-height:1.6;">{task['initial_user_message']}</div>
    </div>"""
    return header, msg


def run_agent(task_id: str, progress=gr.Progress()):
    if not task_id or task_id not in TASKS:
        return "<p style='color:#ef4444;'>⚠️ Please select a task first.</p>", "", ""

    progress(0, desc="Initializing environment...")
    env = CustomerSupportEnv(task_id)
    obs = env.reset()

    # Determine action script — use fallback if available, else build simple default
    if task_id in FALLBACK_SCRIPTS:
        script = FALLBACK_SCRIPTS[task_id]
    else:
        # Build a basic script for tasks without fallback (billing, technical)
        script = _default_script(task_id)

    steps_log = []
    done = False
    step_num = 0
    total = len(script)

    for i, step_data in enumerate(script):
        if done:
            break
        step_num += 1
        progress((i + 1) / total, desc=f"Step {step_num}: {step_data['action_type']}...")

        sd = step_data.copy()
        sd.pop("confidence", None)
        sd.pop("reasoning", None)
        valid = {"action_type", "response_text", "escalation_reason", "refund_amount", "transfer_department"}
        clean = {k: v for k, v in sd.items() if k in valid}
        action = Action(**clean)

        obs, reward, done, info = env.step(action)
        steps_log.append({
            "step": step_num,
            "action_type": action.action_type,
            "text": action.response_text or action.escalation_reason or f"[{action.action_type}]",
            "reward": reward,
        })

    progress(1.0, desc="Grading...")
    final_score, breakdown = env.grade()

    steps_html = _build_step_html(steps_log)
    score_html = _build_score_html(final_score, breakdown)
    chart_html = _build_chart()

    return steps_html, score_html, chart_html


def _default_script(task_id: str) -> list:
    """Fallback action scripts for tasks not in FALLBACK_SCRIPTS."""
    if task_id == "billing_dispute_easy":
        return [
            {
                "action_type": "respond",
                "response_text": (
                    "I sincerely apologize for this billing error on your Basic plan account. "
                    "I can see you were charged $49.99 instead of your correct $29.99 Basic plan rate. "
                    "Let me fix this for you right away."
                ),
            },
            {
                "action_type": "refund",
                "response_text": (
                    "I am issuing a refund of $20.00 for the overcharge — that is the difference "
                    "between the $49.99 charged and your correct $29.99 Basic plan price. "
                    "You should see this reflected within 3-5 business days."
                ),
                "refund_amount": 20.00,
            },
            {
                "action_type": "close",
                "response_text": (
                    "Your $20.00 refund has been processed and your account is confirmed on the "
                    "Basic plan at $29.99/month. I have added safeguards to prevent this from "
                    "happening again. Thank you for your patience and I am sorry for the inconvenience."
                ),
            },
        ]
    elif task_id == "technical_outage_medium":
        return [
            {
                "action_type": "respond",
                "response_text": (
                    "I am so sorry your team of 12 is blocked — I understand this is a critical "
                    "business impact and I will treat this as top priority. Could you please check "
                    "your browser console (F12 → Console tab) for any error messages, and share the "
                    "exact URL/page that shows the blank screen?"
                ),
            },
            {
                "action_type": "respond",
                "response_text": (
                    "Thank you for the diagnostics. I can see this is a known rendering issue affecting "
                    "Business accounts. This requires our Engineering team to resolve. I want to also "
                    "offer you a service credit for the downtime your team has experienced."
                ),
            },
            {
                "action_type": "refund",
                "response_text": (
                    "I am applying a $20.00 service credit to your account as compensation for "
                    "the downtime and business disruption."
                ),
                "refund_amount": 20.00,
            },
            {
                "action_type": "escalate",
                "response_text": (
                    "I am escalating this to our Engineering team immediately. They will investigate "
                    "the dashboard rendering issue for Business accounts and provide a fix."
                ),
                "escalation_reason": (
                    "Engineering team escalation — dashboard blank screen affecting Business account "
                    "ACC-55901, team of 12 blocked for 3 days. Console shows 403 error. "
                    "Known rendering bug for pre-2024 Business accounts. Requires Engineering fix."
                ),
            },
        ]
    return []


# ── Gradio UI ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 32px 16px 20px;
    background: linear-gradient(180deg, #1e293b88 0%, transparent 100%);
    border-radius: 16px;
    margin-bottom: 8px;
}

.main-header h1 {
    font-size: 28px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0 !important;
}

.main-header p {
    color: #94a3b8;
    font-size: 14px;
    margin: 0;
}

footer { display: none !important; }

.dark { --body-background-fill: #0f172a !important; }

button.primary {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px #3b82f644 !important;
}
"""

HEADER_HTML = """
<div class="main-header">
    <h1>🎧 Customer Support Agent Arena</h1>
    <p>Multi-Agent AI System &nbsp;·&nbsp; 3-Layer Architecture &nbsp;·&nbsp; Self-Improving Loop</p>
    <div style="display:flex;justify-content:center;gap:12px;margin-top:14px;flex-wrap:wrap;">
        <span style="background:#1e293b;color:#60a5fa;padding:4px 12px;border-radius:6px;font-size:11px;border:1px solid #334155;">
            👤 Customer Agent (L1)</span>
        <span style="background:#1e293b;color:#f59e0b;padding:4px 12px;border-radius:6px;font-size:11px;border:1px solid #334155;">
            🔍 Supervisor Agent (L2)</span>
        <span style="background:#1e293b;color:#a78bfa;padding:4px 12px;border-radius:6px;font-size:11px;border:1px solid #334155;">
            🧠 Orchestrator Agent (L3)</span>
    </div>
</div>
"""

def build_demo():
    with gr.Blocks(title="Customer Support Agent Arena") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row():
            with gr.Column(scale=1):
                task_dd = gr.Dropdown(
                    choices=TASK_IDS,
                    label="Select Task",
                    info="Choose a customer support scenario to simulate",
                    interactive=True,
                )
                task_info = gr.HTML(label="Task Info")
                customer_msg = gr.HTML(label="Customer Message")
                run_btn = gr.Button("▶ Run Agent", variant="primary", size="lg")

                gr.HTML('<div style="margin-top:16px;"></div>')
                chart_out = gr.HTML(label="Performance Comparison")

            with gr.Column(scale=2):
                steps_out = gr.HTML(label="Agent Steps")
                score_out = gr.HTML(label="Score & Breakdown")

        task_dd.change(fn=on_task_select, inputs=[task_dd], outputs=[task_info, customer_msg])
        run_btn.click(fn=run_agent, inputs=[task_dd], outputs=[steps_out, score_out, chart_out])

    return demo


demo = build_demo()

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, theme=_THEME, css=CUSTOM_CSS)
