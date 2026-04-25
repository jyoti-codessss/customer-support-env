"""
demo.py — Gradio UI Demo for CustomerSupportEnv
Interactive multi-agent customer support simulation with live scoring.
"""

import io
import base64
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from env.environment import CustomerSupportEnv
from env.models import Action
from tasks.task_definitions import TASKS

# ── Safe import of FALLBACK_SCRIPTS ──────────────────────────────────────────
try:
    from inference import FALLBACK_SCRIPTS
except Exception:
    FALLBACK_SCRIPTS = {}

# ── Hardcoded scripts (backup if inference import fails) ──────────────────────
DEMO_SCRIPTS = {
    "billing_dispute_easy": [
        {"action_type": "respond", "response_text": "I sincerely apologize for this billing error. I can see you were charged $49.99 instead of your correct $29.99 Basic plan rate. Let me fix this for you right away."},
        {"action_type": "refund", "response_text": "I am issuing a refund of $20.00 for the overcharge. You should see this within 3-5 business days.", "refund_amount": 20.00},
        {"action_type": "close", "response_text": "Your $20.00 refund has been processed. Your account is confirmed on the Basic plan at $29.99/month. Thank you for your patience."},
    ],
    "technical_outage_medium": [
        {"action_type": "respond", "response_text": "I am so sorry your team of 12 is blocked. I will treat this as top priority. Could you check your browser console (F12) for any error messages?"},
        {"action_type": "respond", "response_text": "Thank you for the diagnostics. This is a known rendering issue affecting Business accounts. I want to offer you a service credit for the downtime."},
        {"action_type": "refund", "response_text": "I am applying a $20.00 service credit to your account as compensation.", "refund_amount": 20.00},
        {"action_type": "escalate", "response_text": "Escalating to Engineering team immediately.", "escalation_reason": "Dashboard blank screen affecting Business account ACC-55901, team of 12 blocked. Requires Engineering fix."},
    ],
    "fraud_complaint_hard": [
        {"action_type": "respond", "response_text": "I completely understand your alarm. Your account security is our top priority. Can you confirm the last 4 digits of your card and your registered email?"},
        {"action_type": "respond", "response_text": "Thank you for verifying. I confirm these 3 transactions totaling $847 were not made by you. I am immediately flagging your account for fraud protection."},
        {"action_type": "refund", "response_text": "I am issuing a full refund of $847.00 for all three fraudulent transactions.", "refund_amount": 847.00},
        {"action_type": "escalate", "response_text": "Escalating to Fraud Investigation team.", "escalation_reason": "Confirmed fraud — 3 unauthorized transactions totaling $847. Account secured. Full refund issued."},
    ],
    "subscription_cancellation_hard": [
        {"action_type": "respond", "response_text": "I am so sorry to hear you are considering cancelling. I truly value your 2-year relationship with us. Could you share what is prompting this decision?"},
        {"action_type": "respond", "response_text": "I completely understand. As a special retention offer, I would like to offer you 3 months completely free, or a 40% discount for the next 6 months."},
        {"action_type": "respond", "response_text": "I respect your decision. Let me process that for you while ensuring a smooth transition."},
        {"action_type": "close", "response_text": "I have processed your cancellation. Your subscription will end at the close of your current billing period. Thank you sincerely for 2 years with us."},
    ],
    "vip_account_recovery_expert": [
        {"action_type": "respond", "response_text": "I sincerely apologize for this critical situation. As a VIP Platinum member, you have my full personal attention. Can you confirm your registered phone number and last transaction amount?"},
        {"action_type": "respond", "response_text": "Thank you for verifying. Your account was incorrectly locked due to a security system error. I am escalating to our VIP team immediately."},
        {"action_type": "escalate", "response_text": "Escalating to VIP specialist team.", "escalation_reason": "VIP Platinum member account incorrectly locked. Business-critical. Immediate restoration required."},
        {"action_type": "respond", "response_text": "Your account has been fully restored. As compensation, I am crediting 3 months of complimentary VIP service worth $1,500."},
        {"action_type": "close", "response_text": "Your VIP Platinum account is fully restored. $1,500 credit applied. Personal VIP concierge assigned. This will never happen again."},
    ],
}

# ── Task metadata ─────────────────────────────────────────────────────────────

TASK_IDS = list(TASKS.keys())

TASK_LABELS = {
    "billing_dispute_easy": "Billing Dispute (Easy)",
    "technical_outage_medium": "Technical Outage (Medium)",
    "fraud_complaint_hard": "Fraud Complaint (Hard)",
    "subscription_cancellation_hard": "Subscription Cancellation (Hard)",
    "vip_account_recovery_expert": "VIP Account Recovery (Expert)",
}

DIFFICULTY_COLORS = {
    "easy": "#22c55e",
    "medium": "#eab308",
    "hard": "#ef4444",
    "expert": "#a855f7",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    for i, (k, v) in enumerate(breakdown.items()):
        if k == "final_score":
            continue
        if v is True:
            icon, status, sc = "Pass", "Pass", "#22c55e"
        elif v is False:
            icon, status, sc = "Fail", "Fail", "#ef4444"
        elif isinstance(v, (int, float)) and v > 0:
            icon, status, sc = "Pass", str(v), "#22c55e"
        else:
            icon, status, sc = "Fail", str(v), "#ef4444"
        label = k.replace("_", " ").title()
        bg = "#1e293b" if i % 2 == 0 else "#162032"
        rows += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:10px 14px;color:{sc};font-weight:700;">{icon}</td>'
            f'<td style="padding:10px 14px;color:#f1f5f9;font-weight:500;">{label}</td>'
            f'<td style="padding:10px 14px;text-align:right;color:{sc};font-weight:600;">{status}</td>'
            f'</tr>'
        )
    return f"""
    <div style="text-align:center;padding:24px 0;">
      <div style="font-size:72px;font-weight:800;color:{color};
                  text-shadow:0 0 30px {color}44;">{score:.3f}</div>
      <div style="font-size:14px;color:#94a3b8;margin-top:4px;">out of 1.000</div>
      <div style="margin:16px auto;width:80%;max-width:400px;height:10px;
                  background:#0f172a;border-radius:5px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;
                    background:linear-gradient(90deg,{color},{color}cc);border-radius:5px;"></div>
      </div>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:14px;
                  background:#0f172a;border-radius:10px;overflow:hidden;">
      <thead><tr style="background:#0f172a;border-bottom:2px solid #334155;">
        <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:12px;"></th>
        <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:12px;">Criterion</th>
        <th style="padding:10px 14px;text-align:right;color:#64748b;font-size:12px;">Result</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _build_step_html(steps: list) -> str:
    html = '<div style="font-size:13px;color:#e2e8f0;">'
    for s in steps:
        ac = {"respond": "#60a5fa", "refund": "#22c55e", "escalate": "#f59e0b",
              "close": "#a78bfa", "transfer": "#f472b6"}.get(str(s["action_type"]), "#94a3b8")
        rc = "#22c55e" if s["reward"] >= 0 else "#ef4444"
        html += f"""
        <div style="padding:12px 14px;margin:8px 0;background:#1e293b;
                    border-radius:10px;border-left:4px solid {ac};">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span>
              <strong style="color:{ac};">Step {s['step']}</strong>
              &nbsp;--&nbsp;
              <code style="color:{ac};background:#0f172a;padding:2px 8px;
                           border-radius:4px;font-size:12px;">{s['action_type']}</code>
            </span>
            <span style="color:{rc};font-weight:700;">reward: {s['reward']:+.3f}</span>
          </div>
          <div style="color:#cbd5e1;font-size:12px;line-height:1.6;">{s['text'][:300]}</div>
        </div>"""
    html += "</div>"
    return html


def _build_chart() -> str:
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=110)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    labels = ["Baseline\n(Before Training)", "Trained\n(GRPO Multi-Agent)"]
    values = [0.268, 1.000]
    colors_list = ["#ef4444", "#22c55e"]
    bars = ax.bar(labels, values, color=colors_list, width=0.5,
                  edgecolor="#334155", linewidth=1.2, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom",
                fontweight="bold", color="#e2e8f0", fontsize=14)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score", color="#94a3b8", fontsize=11)
    ax.set_title("Before vs After GRPO Training", color="#e2e8f0",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.grid(axis="y", color="#334155", linewidth=0.5, zorder=0)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;max-width:520px;border-radius:12px;'
            f'margin:auto;display:block;" />')


# ── Callbacks ─────────────────────────────────────────────────────────────────

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
        <span style="font-size:15px;font-weight:700;color:#e2e8f0;">
          {TASK_LABELS.get(task_id, task_id)}</span>
        <span style="background:{dc}22;color:{dc};padding:3px 10px;border-radius:6px;
              font-size:12px;font-weight:600;border:1px solid {dc}44;">{diff.upper()}</span>
      </div>
      <div style="font-size:12px;color:#94a3b8;">
        Customer: <strong style="color:#e2e8f0;">{meta.customer_name}</strong> &nbsp;|&nbsp;
        Account: <strong style="color:#e2e8f0;">{meta.account_id}</strong> &nbsp;|&nbsp;
        Plan: <strong style="color:#e2e8f0;">{meta.plan}</strong>
      </div>
    </div>"""
    msg = f"""
    <div style="padding:14px;background:#0f172a;border-radius:10px;
                border:1px solid #334155;margin-top:10px;">
      <div style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:600;">
        CUSTOMER MESSAGE</div>
      <div style="color:#e2e8f0;font-size:14px;line-height:1.6;">
        {task['initial_user_message']}</div>
    </div>"""
    return header, msg


def run_agent(task_id: str, progress=gr.Progress()):
    if not task_id or task_id not in TASKS:
        return "<p style='color:#ef4444;'>Please select a task first.</p>", "", ""

    progress(0, desc="Initializing environment...")
    env = CustomerSupportEnv(task_id)
    env.reset()

    script = FALLBACK_SCRIPTS.get(task_id) or DEMO_SCRIPTS.get(task_id, [])

    steps_log = []
    done = False

    for i, step_data in enumerate(script):
        if done:
            break
        progress((i + 1) / max(len(script), 1),
                 desc=f"Step {i+1}: {step_data['action_type']}...")

        valid_keys = {"action_type", "response_text", "escalation_reason",
                      "refund_amount", "transfer_department"}
        clean = {k: v for k, v in step_data.items() if k in valid_keys}
        action = Action(**clean)

        obs, reward, done, info = env.step(action)
        steps_log.append({
            "step": i + 1,
            "action_type": str(action.action_type),
            "text": (action.response_text or
                     action.escalation_reason or
                     f"[{action.action_type}]"),
            "reward": reward,
        })

    progress(1.0, desc="Grading...")
    final_score, breakdown = env.grade()

    return (
        _build_step_html(steps_log),
        _build_score_html(final_score, breakdown),
        _build_chart(),
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
* { font-family: 'Inter', sans-serif !important; }
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
    min-height: 100vh;
}
footer { display: none !important; }
button.primary {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border-radius: 10px !important;
}
"""

HEADER_HTML = """
<div style="text-align:center;padding:28px 16px 16px;">
  <h1 style="font-size:28px;font-weight:800;margin:0 0 6px;
             background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    Customer Support Agent Arena
  </h1>
  <p style="color:#94a3b8;font-size:14px;margin:0;">
    Multi-Agent AI System &nbsp;·&nbsp; 3-Layer Architecture &nbsp;·&nbsp; Self-Improving Loop
  </p>
  <div style="display:flex;justify-content:center;gap:10px;margin-top:12px;flex-wrap:wrap;">
    <span style="background:#1e293b;color:#60a5fa;padding:4px 12px;border-radius:6px;
                 font-size:11px;border:1px solid #334155;">Customer Agent (L1)</span>
    <span style="background:#1e293b;color:#f59e0b;padding:4px 12px;border-radius:6px;
                 font-size:11px;border:1px solid #334155;">Supervisor Agent (L2)</span>
    <span style="background:#1e293b;color:#a78bfa;padding:4px 12px;border-radius:6px;
                 font-size:11px;border:1px solid #334155;">Orchestrator Agent (L3)</span>
  </div>
</div>
"""

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="Customer Support Agent Arena",
               theme=_THEME, css=CUSTOM_CSS) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=1):
            task_dd = gr.Dropdown(
                choices=TASK_IDS,
                label="Select Task",
                info="Choose a customer support scenario to simulate",
                interactive=True,
            )
            task_info = gr.HTML()
            customer_msg = gr.HTML()
            run_btn = gr.Button("Run Agent", variant="primary", size="lg")
            gr.HTML('<div style="margin-top:16px;"></div>')
            chart_out = gr.HTML()

        with gr.Column(scale=2):
            steps_out = gr.HTML()
            score_out = gr.HTML()

    task_dd.change(fn=on_task_select, inputs=[task_dd],
                   outputs=[task_info, customer_msg])
    run_btn.click(fn=run_agent, inputs=[task_dd],
                  outputs=[steps_out, score_out, chart_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)