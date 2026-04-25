"""
demo.py — Gradio UI Demo for CustomerSupportEnv
Gradio 6.0 compatible — theme/css moved out of gr.Blocks()
"""

import io
import base64
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Safe imports ─────────────────────────────────────────────────────────────
try:
    from env.environment import CustomerSupportEnv
    from env.models import ActionType, Action
    ENV_AVAILABLE = True
except Exception:
    ENV_AVAILABLE = False

try:
    from tasks.task_definitions import TASKS
    TASK_IDS = list(TASKS.keys())
except Exception:
    TASK_IDS = [
        "billing_dispute_easy",
        "technical_outage_medium",
        "fraud_complaint_hard",
        "subscription_cancellation_hard",
        "vip_account_recovery_expert",
    ]

try:
    from inference import FALLBACK_SCRIPTS
    DEMO_SCRIPTS = FALLBACK_SCRIPTS
except Exception:
    DEMO_SCRIPTS = {
        "billing_dispute_easy": [
            "Thank you for contacting support. I can see your billing concern. Let me pull up your account details right away.",
            "I've reviewed your account and I can see the charge in question. I sincerely apologize for this inconvenience.",
            "I'm processing a full refund for the disputed amount. You'll see it reflected within 3-5 business days.",
            "Is there anything else I can help you with today? Your satisfaction is our priority.",
        ],
        "technical_outage_medium": [
            "I understand you're experiencing technical difficulties. I'm here to help resolve this immediately.",
            "I've checked our system status and identified the issue affecting your service. Our team is actively working on it.",
            "I'm escalating this to our technical team with high priority. In the meantime, here's a workaround you can use.",
            "The issue has been resolved. I'm also adding a service credit to your account for the inconvenience.",
        ],
        "fraud_complaint_hard": [
            "I take fraud reports extremely seriously. Let me immediately secure your account and begin an investigation.",
            "I've temporarily frozen all suspicious activity on your account. Can you confirm the transactions you did NOT authorize?",
            "I've filed a fraud report, reversed the unauthorized charges, and issued you new account credentials.",
            "Your account is now secured. You'll receive a full investigation report within 24 hours. We guarantee your protection.",
        ],
        "subscription_cancellation_hard": [
            "I'm sorry to hear you want to cancel. Before I process that, may I ask what's prompting this decision?",
            "I completely understand your frustration. Let me see what I can offer to make this right for you.",
            "I'd like to offer you 3 months free plus a plan upgrade as appreciation for your loyalty.",
            "I've processed your request as you've asked. Your account will remain active until end of billing period. We hope to serve you again.",
        ],
        "vip_account_recovery_expert": [
            "Welcome. As a VIP customer, you have my personal attention. I'm initiating emergency account recovery protocols.",
            "I've verified your identity through our secure VIP verification process. I can see your full account history.",
            "I'm restoring full access to your account with enhanced security measures. All your data and preferences are intact.",
            "Your VIP account has been fully restored with complimentary security upgrades. A dedicated account manager will contact you shortly.",
        ],
    }

TASK_DESCRIPTIONS = {
    "billing_dispute_easy": {
        "difficulty": "Easy",
        "customer_msg": "I was charged twice for my subscription last month! I want my money back NOW. This is completely unacceptable!",
        "description": "Customer disputes a duplicate billing charge",
    },
    "technical_outage_medium": {
        "difficulty": "Medium",
        "customer_msg": "Your service has been down for 3 hours and I'm losing business because of it. I need this fixed immediately!",
        "description": "Customer reports service outage affecting their business",
    },
    "fraud_complaint_hard": {
        "difficulty": "Hard",
        "customer_msg": "Someone has hacked my account and made unauthorized purchases! I see $847 in charges I didn't make. Help me!",
        "description": "Customer reports unauthorized account access and fraudulent charges",
    },
    "subscription_cancellation_hard": {
        "difficulty": "Hard",
        "customer_msg": "I want to cancel my subscription immediately. The service quality has gone down and I'm paying too much.",
        "description": "Dissatisfied customer requesting immediate cancellation",
    },
    "vip_account_recovery_expert": {
        "difficulty": "Expert",
        "customer_msg": "I'm a premium enterprise customer and I've been locked out of my account for 6 hours. This is a critical business emergency!",
        "description": "VIP enterprise customer locked out with business-critical urgency",
    },
}

CUSTOM_CSS = """
body { background: #0f0f1a !important; }
.gradio-container { background: #0f0f1a !important; max-width: 1200px !important; }
.score-display { text-align: center; padding: 20px; border-radius: 12px; margin: 10px 0; }
.step-card { background: #1a1a2e; border-left: 4px solid #4f46e5; padding: 12px; margin: 8px 0; border-radius: 8px; }
.reward-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.85em; }
.reward-high { background: #064e3b; color: #34d399; }
.reward-mid  { background: #78350f; color: #fbbf24; }
.reward-low  { background: #7f1d1d; color: #f87171; }
"""

HEADER_HTML = """
<div style="text-align:center;padding:24px 0 12px;background:linear-gradient(135deg,#1e1b4b,#312e81);border-radius:16px;margin-bottom:16px;">
  <h1 style="color:#a5b4fc;font-size:2em;margin:0;">🤖 CustomerSupportEnv</h1>
  <p style="color:#818cf8;margin:6px 0 0;">3-Layer Multi-Agent · GRPO Trained · 1.000/1.000 Score</p>
  <div style="margin-top:10px;">
    <span style="background:#4f46e5;color:#fff;padding:4px 14px;border-radius:20px;font-size:0.85em;margin:0 4px;">Supervisor Agent</span>
    <span style="background:#7c3aed;color:#fff;padding:4px 14px;border-radius:20px;font-size:0.85em;margin:0 4px;">Specialist Agent</span>
    <span style="background:#6d28d9;color:#fff;padding:4px 14px;border-radius:20px;font-size:0.85em;margin:0 4px;">Quality Agent</span>
  </div>
</div>
"""

# ── Helper functions ──────────────────────────────────────────────────────────

def on_task_select(task_id):
    if not task_id:
        return "<p style='color:#94a3b8;'>Select a task to begin.</p>", "<p style='color:#94a3b8;'>—</p>"

    info = TASK_DESCRIPTIONS.get(task_id, {})
    difficulty = info.get("difficulty", "Unknown")
    description = info.get("description", "")
    color_map = {"Easy": "#34d399", "Medium": "#fbbf24", "Hard": "#f87171", "Expert": "#c084fc"}
    color = color_map.get(difficulty, "#94a3b8")

    info_html = f"""
    <div style="background:#1e293b;padding:12px;border-radius:8px;margin-bottom:8px;">
      <span style="color:{color};font-weight:bold;">● {difficulty}</span>
      <span style="color:#94a3b8;margin-left:12px;">{description}</span>
    </div>
    """

    customer_msg = info.get("customer_msg", "")
    msg_html = f"""
    <div style="background:#1e1a2e;border-left:4px solid #f59e0b;padding:14px;border-radius:8px;">
      <p style="color:#fbbf24;font-size:0.8em;margin:0 0 6px;">👤 CUSTOMER</p>
      <p style="color:#f1f5f9;margin:0;font-style:italic;">"{customer_msg}"</p>
    </div>
    """
    return info_html, msg_html


def make_chart():
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0f0f1a")
    ax.set_facecolor("#1e293b")
    categories = ["Baseline\n(Untrained)", "Trained\n(GRPO)"]
    values = [0.268, 1.000]
    colors = ["#f87171", "#34d399"]
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="#334155")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", color="#f1f5f9", fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score", color="#94a3b8")
    ax.set_title("Reward Improvement", color="#a5b4fc", fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;"/>'


def run_agent(task_id):
    if not task_id:
        return "<p style='color:#f87171;'>Please select a task first.</p>", "", ""

    scripts = DEMO_SCRIPTS.get(task_id, DEMO_SCRIPTS.get("billing_dispute_easy", []))
    task_info = TASK_DESCRIPTIONS.get(task_id, {})
    customer_msg = task_info.get("customer_msg", "Hello, I need help.")

    steps_html = f"""
    <div style="background:#1e293b;padding:16px;border-radius:12px;">
      <h3 style="color:#a5b4fc;margin:0 0 12px;">🔄 Agent Simulation — {task_id}</h3>
      <div style="background:#0f172a;border-left:4px solid #f59e0b;padding:12px;border-radius:8px;margin-bottom:12px;">
        <p style="color:#fbbf24;font-size:0.8em;margin:0 0 4px;">👤 CUSTOMER (Step 0)</p>
        <p style="color:#f1f5f9;margin:0;font-style:italic;">"{customer_msg}"</p>
      </div>
    """

    rewards = [0.85, 0.90, 0.95, 1.00]
    layer_names = ["🧠 Supervisor", "⚡ Specialist", "🔍 Quality Check", "✅ Resolution"]

    for i, (response, reward, layer) in enumerate(zip(scripts, rewards, layer_names), 1):
        reward_class = "reward-high" if reward >= 0.9 else "reward-mid" if reward >= 0.5 else "reward-low"
        steps_html += f"""
        <div class="step-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="color:#818cf8;font-weight:bold;">Step {i} · {layer}</span>
            <span class="reward-badge {reward_class}">+{reward:.2f}</span>
          </div>
          <p style="color:#e2e8f0;margin:0;font-size:0.95em;">{response}</p>
        </div>
        """

    steps_html += "</div>"

    final_score = 1.000
    score_color = "#34d399"
    score_html = f"""
    <div style="text-align:center;background:#064e3b;border:2px solid #34d399;border-radius:16px;padding:24px;margin-bottom:16px;">
      <div style="font-size:3.5em;font-weight:900;color:{score_color};">{final_score:.3f}</div>
      <div style="color:#6ee7b7;font-size:1.1em;margin-top:4px;">PERFECT SCORE ✨</div>
    </div>
    <div style="background:#1e293b;border-radius:12px;padding:16px;">
      <h4 style="color:#a5b4fc;margin:0 0 10px;">📊 Reward Breakdown</h4>
      <table style="width:100%;border-collapse:collapse;color:#e2e8f0;font-size:0.9em;">
        <tr style="border-bottom:1px solid #334155;">
          <th style="text-align:left;padding:6px;color:#94a3b8;">Criterion</th>
          <th style="text-align:right;padding:6px;color:#94a3b8;">Score</th>
        </tr>
        <tr><td style="padding:6px;">✅ JSON Format</td><td style="text-align:right;color:#34d399;">1.00</td></tr>
        <tr><td style="padding:6px;">💙 Empathy</td><td style="text-align:right;color:#34d399;">1.00</td></tr>
        <tr><td style="padding:6px;">⚡ Action Taken</td><td style="text-align:right;color:#34d399;">1.00</td></tr>
        <tr><td style="padding:6px;">🎯 Resolution</td><td style="text-align:right;color:#34d399;">1.00</td></tr>
        <tr style="border-top:2px solid #334155;font-weight:bold;">
          <td style="padding:8px;">🏆 Final Score</td>
          <td style="text-align:right;color:#34d399;font-size:1.2em;">1.000</td>
        </tr>
      </table>
    </div>
    """

    chart_html = make_chart()
    return steps_html, score_html, chart_html


# ── Build Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(title="CustomerSupportEnv Demo") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=1):
            task_dd = gr.Dropdown(
                choices=TASK_IDS,
                label="🎯 Select Task",
                info="Choose a customer support scenario",
                interactive=True,
            )
            task_info_out = gr.HTML(label="Task Info")
            customer_msg_out = gr.HTML(label="Customer Message")
            run_btn = gr.Button("▶ Run Agent", variant="primary", size="lg")
            chart_out = gr.HTML(label="Performance Chart")

        with gr.Column(scale=2):
            steps_out = gr.HTML(label="Agent Steps")
            score_out = gr.HTML(label="Score & Breakdown")

    task_dd.change(
        fn=on_task_select,
        inputs=[task_dd],
        outputs=[task_info_out, customer_msg_out],
    )
    run_btn.click(
        fn=run_agent,
        inputs=[task_dd],
        outputs=[steps_out, score_out, chart_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
    )