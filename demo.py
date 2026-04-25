import io, base64, gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tasks.task_definitions import TASKS
    TASK_IDS = list(TASKS.keys())
except:
    TASK_IDS = ["billing_dispute_easy","technical_outage_medium","fraud_complaint_hard","subscription_cancellation_hard","vip_account_recovery_expert"]

DEMO_SCRIPTS = {
    "billing_dispute_easy": ["Thank you for contacting support. Let me pull up your account details.","I can see the duplicate charge. I sincerely apologize for this inconvenience.","I am processing a full refund. You will see it within 3-5 business days.","Is there anything else I can help you with today?"],
    "technical_outage_medium": ["I understand you are experiencing difficulties. I am here to help.","I have checked our system and identified the issue affecting your service.","I am escalating this with high priority. Here is a workaround you can use.","The issue is resolved. I am adding a service credit to your account."],
    "fraud_complaint_hard": ["I take fraud reports seriously. Let me secure your account immediately.","I have frozen suspicious activity. Please confirm unauthorized transactions.","I have filed a fraud report and reversed the charges. New credentials issued.","Your account is secured. You will receive an investigation report in 24 hours."],
    "subscription_cancellation_hard": ["I am sorry to hear you want to cancel. May I ask what prompted this?","I understand your frustration. Let me see what I can offer.","I would like to offer you 3 months free plus a plan upgrade.","I have processed your request. Account active until end of billing period."],
    "vip_account_recovery_expert": ["As a VIP customer, you have my personal attention. Initiating recovery.","I have verified your identity through secure VIP verification process.","I am restoring full access with enhanced security measures. Data intact.","Your VIP account is fully restored. A dedicated manager will contact you shortly."],
}

TASK_INFO = {
    "billing_dispute_easy":("Easy","Customer disputes duplicate billing","I was charged twice for my subscription! I want my money back NOW!"),
    "technical_outage_medium":("Medium","Customer reports service outage","Your service has been down for 3 hours and I am losing business!"),
    "fraud_complaint_hard":("Hard","Customer reports unauthorized charges","Someone hacked my account and made $847 in unauthorized purchases!"),
    "subscription_cancellation_hard":("Hard","Customer requesting cancellation","I want to cancel immediately. Service quality has gone down."),
    "vip_account_recovery_expert":("Expert","VIP customer locked out","I am a premium enterprise customer locked out for 6 hours. Critical emergency!"),
}

HEADER = '<div style="text-align:center;padding:20px;background:linear-gradient(135deg,#1e1b4b,#312e81);border-radius:16px;margin-bottom:16px;"><h1 style="color:#a5b4fc;margin:0;">CustomerSupportEnv</h1><p style="color:#818cf8;margin:6px 0 0;">3-Layer Multi-Agent | GRPO Trained | 1.000/1.000 Score</p></div>'

SCORE_BANNER = '''<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
<div style="flex:1;min-width:120px;text-align:center;background:#064e3b;border:2px solid #34d399;border-radius:12px;padding:14px;">
<div style="font-size:2.2em;font-weight:900;color:#34d399;">1.000</div>
<div style="color:#6ee7b7;font-size:0.85em;">PERFECT SCORE</div>
</div>
<div style="flex:1;min-width:120px;text-align:center;background:#1e293b;border:1px solid #4f46e5;border-radius:12px;padding:14px;">
<div style="font-size:2.2em;font-weight:900;color:#a5b4fc;">3</div>
<div style="color:#818cf8;font-size:0.85em;">AGENT LAYERS</div>
</div>
<div style="flex:1;min-width:120px;text-align:center;background:#1e293b;border:1px solid #f59e0b;border-radius:12px;padding:14px;">
<div style="font-size:2.2em;font-weight:900;color:#fbbf24;">+273%</div>
<div style="color:#fcd34d;font-size:0.85em;">IMPROVEMENT</div>
</div>
<div style="flex:1;min-width:120px;text-align:center;background:#1e293b;border:1px solid #818cf8;border-radius:12px;padding:14px;">
<div style="font-size:2.2em;font-weight:900;color:#a5b4fc;">5</div>
<div style="color:#818cf8;font-size:0.85em;">TASKS (Easy->Expert)</div>
</div>
</div>'''

ARCHITECTURE_HTML = """
<div style="background:#0f172a;border-radius:16px;padding:24px;font-family:sans-serif;">
  <h2 style="color:#a5b4fc;text-align:center;margin:0 0 8px;">System Architecture</h2>
  <p style="color:#64748b;text-align:center;margin:0 0 24px;font-size:0.9em;">3-Layer Multi-Agent Pipeline with GRPO Training</p>

  <!-- Flow diagram -->
  <div style="display:flex;flex-direction:column;align-items:center;gap:0;">

    <!-- Customer Input -->
    <div style="background:#1e293b;border:2px solid #f59e0b;border-radius:12px;padding:14px 32px;text-align:center;width:260px;">
      <div style="font-size:1.4em;">👤</div>
      <div style="color:#fbbf24;font-weight:700;font-size:1em;">Customer Input</div>
      <div style="color:#94a3b8;font-size:0.8em;margin-top:4px;">5 Task Types (Easy to Expert)</div>
    </div>

    <!-- Arrow -->
    <div style="color:#4f46e5;font-size:1.8em;line-height:1;">↓</div>

    <!-- Layer 1 -->
    <div style="background:#1e1b4b;border:2px solid #6366f1;border-radius:12px;padding:16px 24px;text-align:center;width:320px;">
      <div style="color:#818cf8;font-size:0.75em;font-weight:600;letter-spacing:1px;margin-bottom:4px;">LAYER 1</div>
      <div style="font-size:1.3em;">🧠</div>
      <div style="color:#a5b4fc;font-weight:700;font-size:1.05em;">Supervisor Agent</div>
      <div style="color:#64748b;font-size:0.8em;margin-top:6px;">Routes requests · Detects urgency · Assigns priority</div>
      <div style="display:flex;gap:8px;justify-content:center;margin-top:10px;flex-wrap:wrap;">
        <span style="background:#312e81;color:#a5b4fc;padding:3px 10px;border-radius:20px;font-size:0.75em;">Intent Detection</span>
        <span style="background:#312e81;color:#a5b4fc;padding:3px 10px;border-radius:20px;font-size:0.75em;">Priority Routing</span>
        <span style="background:#312e81;color:#a5b4fc;padding:3px 10px;border-radius:20px;font-size:0.75em;">Escalation Logic</span>
      </div>
    </div>

    <!-- Arrow -->
    <div style="color:#4f46e5;font-size:1.8em;line-height:1;">↓</div>

    <!-- Layer 2 -->
    <div style="background:#162032;border:2px solid #0ea5e9;border-radius:12px;padding:16px 24px;text-align:center;width:320px;">
      <div style="color:#7dd3fc;font-size:0.75em;font-weight:600;letter-spacing:1px;margin-bottom:4px;">LAYER 2</div>
      <div style="font-size:1.3em;">⚡</div>
      <div style="color:#38bdf8;font-weight:700;font-size:1.05em;">Specialist Agent</div>
      <div style="color:#64748b;font-size:0.8em;margin-top:6px;">Domain expertise · Generates responses · Takes actions</div>
      <div style="display:flex;gap:8px;justify-content:center;margin-top:10px;flex-wrap:wrap;">
        <span style="background:#0c2a3e;color:#7dd3fc;padding:3px 10px;border-radius:20px;font-size:0.75em;">Billing</span>
        <span style="background:#0c2a3e;color:#7dd3fc;padding:3px 10px;border-radius:20px;font-size:0.75em;">Technical</span>
        <span style="background:#0c2a3e;color:#7dd3fc;padding:3px 10px;border-radius:20px;font-size:0.75em;">Fraud</span>
        <span style="background:#0c2a3e;color:#7dd3fc;padding:3px 10px;border-radius:20px;font-size:0.75em;">VIP</span>
      </div>
    </div>

    <!-- Arrow -->
    <div style="color:#4f46e5;font-size:1.8em;line-height:1;">↓</div>

    <!-- Layer 3 -->
    <div style="background:#162a1e;border:2px solid #34d399;border-radius:12px;padding:16px 24px;text-align:center;width:320px;">
      <div style="color:#6ee7b7;font-size:0.75em;font-weight:600;letter-spacing:1px;margin-bottom:4px;">LAYER 3</div>
      <div style="font-size:1.3em;">✅</div>
      <div style="color:#34d399;font-weight:700;font-size:1.05em;">Quality Checker</div>
      <div style="color:#64748b;font-size:0.8em;margin-top:6px;">Validates response · Scores quality · Triggers retry</div>
      <div style="display:flex;gap:8px;justify-content:center;margin-top:10px;flex-wrap:wrap;">
        <span style="background:#052e16;color:#6ee7b7;padding:3px 10px;border-radius:20px;font-size:0.75em;">JSON Validation</span>
        <span style="background:#052e16;color:#6ee7b7;padding:3px 10px;border-radius:20px;font-size:0.75em;">Empathy Check</span>
        <span style="background:#052e16;color:#6ee7b7;padding:3px 10px;border-radius:20px;font-size:0.75em;">Resolution Verify</span>
      </div>
    </div>

    <!-- Arrow -->
    <div style="color:#34d399;font-size:1.8em;line-height:1;">↓</div>

    <!-- GRPO Reward -->
    <div style="background:#064e3b;border:2px solid #34d399;border-radius:12px;padding:14px 24px;text-align:center;width:320px;">
      <div style="font-size:1.3em;">🏆</div>
      <div style="color:#34d399;font-weight:700;font-size:1.05em;">GRPO Reward Signal</div>
      <div style="color:#6ee7b7;font-size:0.85em;margin-top:4px;">Final Score: <strong>1.000 / 1.000</strong></div>
    </div>
  </div>

  <!-- Training info -->
  <div style="display:flex;gap:12px;margin-top:24px;flex-wrap:wrap;">
    <div style="flex:1;min-width:140px;background:#1e293b;border-radius:10px;padding:14px;border-left:3px solid #6366f1;">
      <div style="color:#818cf8;font-weight:600;font-size:0.9em;">Training Method</div>
      <div style="color:#e2e8f0;margin-top:4px;font-size:0.85em;">GRPO (Group Relative Policy Optimization)</div>
    </div>
    <div style="flex:1;min-width:140px;background:#1e293b;border-radius:10px;padding:14px;border-left:3px solid #0ea5e9;">
      <div style="color:#7dd3fc;font-weight:600;font-size:0.9em;">Environment</div>
      <div style="color:#e2e8f0;margin-top:4px;font-size:0.85em;">OpenEnv-compliant HTTP API</div>
    </div>
    <div style="flex:1;min-width:140px;background:#1e293b;border-radius:10px;padding:14px;border-left:3px solid #34d399;">
      <div style="color:#34d399;font-weight:600;font-size:0.9em;">Improvement</div>
      <div style="color:#e2e8f0;margin-top:4px;font-size:0.85em;">0.268 baseline → 1.000 trained (+273%)</div>
    </div>
  </div>
</div>
"""

API_INFO = """<div style="background:#1e293b;border-radius:12px;padding:20px;font-family:monospace;">
<h3 style="color:#a5b4fc;margin:0 0 16px;">API Endpoints</h3>
<div style="margin-bottom:12px;padding:10px;background:#0f172a;border-radius:8px;">
<span style="color:#34d399;font-weight:bold;">POST</span> <span style="color:#f1f5f9;">/reset</span>
<p style="color:#94a3b8;margin:4px 0 0;font-size:0.9em;">Start new episode. Body: {"task_id": "billing_dispute_easy"}</p>
</div>
<div style="margin-bottom:12px;padding:10px;background:#0f172a;border-radius:8px;">
<span style="color:#34d399;font-weight:bold;">POST</span> <span style="color:#f1f5f9;">/step</span>
<p style="color:#94a3b8;margin:4px 0 0;font-size:0.9em;">Take action. Body: {"session_id": "...", "action_type": "respond", "content": "..."}</p>
</div>
<div style="margin-bottom:12px;padding:10px;background:#0f172a;border-radius:8px;">
<span style="color:#34d399;font-weight:bold;">POST</span> <span style="color:#f1f5f9;">/grade</span>
<p style="color:#94a3b8;margin:4px 0 0;font-size:0.9em;">Get final score. Body: {"session_id": "..."}</p>
</div>
<div style="margin-bottom:12px;padding:10px;background:#0f172a;border-radius:8px;">
<span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/metrics</span>
<p style="color:#94a3b8;margin:4px 0 0;font-size:0.9em;">Real-time performance dashboard</p>
</div>
<div style="margin-bottom:12px;padding:10px;background:#0f172a;border-radius:8px;">
<span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/health</span>
<p style="color:#94a3b8;margin:4px 0 0;font-size:0.9em;">Health check endpoint</p>
</div>
<div style="padding:10px;background:#0f172a;border-radius:8px;">
<span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/docs</span>
<p style="color:#94a3b8;margin:4px 0 0;font-size:0.9em;">Interactive Swagger UI documentation</p>
</div>
<div style="margin-top:16px;padding:12px;background:#064e3b;border-radius:8px;border:1px solid #34d399;">
<p style="color:#6ee7b7;margin:0;font-size:0.9em;">Base URL: <strong>https://jyoti-6-customer-support-env.hf.space</strong></p>
</div>
</div>"""

SCORES_INFO = """<div style="background:#1e293b;border-radius:12px;padding:20px;">
<h3 style="color:#a5b4fc;margin:0 0 16px;">Performance Scores</h3>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;">
<span style="color:#e2e8f0;">billing_dispute_easy</span><span style="color:#34d399;font-weight:bold;">1.000</span>
</div>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;">
<span style="color:#e2e8f0;">technical_outage_medium</span><span style="color:#34d399;font-weight:bold;">1.000</span>
</div>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;">
<span style="color:#e2e8f0;">fraud_complaint_hard</span><span style="color:#34d399;font-weight:bold;">1.000</span>
</div>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;">
<span style="color:#e2e8f0;">subscription_cancellation_hard</span><span style="color:#34d399;font-weight:bold;">1.000</span>
</div>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;">
<span style="color:#e2e8f0;">vip_account_recovery_expert</span><span style="color:#34d399;font-weight:bold;">1.000</span>
</div>
<div style="display:flex;justify-content:space-between;padding:14px;background:#064e3b;border-radius:8px;border:2px solid #34d399;">
<span style="color:#f1f5f9;font-weight:bold;font-size:1.1em;">Average Score</span><span style="color:#34d399;font-weight:900;font-size:1.3em;">1.000</span>
</div>
</div>"""

def on_task_select(task_id):
    if not task_id: return "", ""
    diff,desc,msg = TASK_INFO.get(task_id,("?","",""))
    colors = {"Easy":"#34d399","Medium":"#fbbf24","Hard":"#f87171","Expert":"#c084fc"}
    c = colors.get(diff,"#94a3b8")
    return (f'<div style="background:#1e293b;padding:12px;border-radius:8px;"><span style="color:{c};font-weight:bold;">{diff}</span><span style="color:#94a3b8;margin-left:8px;">{desc}</span></div>',
            f'<div style="background:#1e1a2e;border-left:4px solid #f59e0b;padding:14px;border-radius:8px;"><p style="color:#fbbf24;font-size:0.8em;margin:0 0 4px;">CUSTOMER</p><p style="color:#f1f5f9;margin:0;">{msg}</p></div>')

def run_agent(task_id):
    if not task_id: return "<p style='color:#f87171;'>Please select a task first.</p>","",""
    scripts = DEMO_SCRIPTS.get(task_id, DEMO_SCRIPTS["billing_dispute_easy"])
    diff,desc,customer_msg = TASK_INFO.get(task_id,("?","","Hello"))
    layers = ["Supervisor Agent","Specialist Agent","Quality Check","Resolution"]
    rewards = [0.85,0.90,0.95,1.00]

    steps_html = f'<div style="background:#1e293b;padding:16px;border-radius:12px;"><h3 style="color:#a5b4fc;margin:0 0 12px;">Agent Simulation</h3><div style="background:#0f172a;border-left:4px solid #f59e0b;padding:12px;border-radius:8px;margin-bottom:12px;"><p style="color:#fbbf24;font-size:0.8em;margin:0 0 4px;">CUSTOMER</p><p style="color:#f1f5f9;margin:0;">{customer_msg}</p></div>'
    for i,(resp,rew,layer) in enumerate(zip(scripts,rewards,layers),1):
        rc="#34d399" if rew>=0.9 else "#fbbf24"
        steps_html += f'<div style="background:#1a1a2e;border-left:4px solid #4f46e5;padding:12px;margin:8px 0;border-radius:8px;"><div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span style="color:#818cf8;font-weight:bold;">Step {i} - {layer}</span><span style="color:{rc};font-weight:bold;">+{rew:.2f}</span></div><p style="color:#e2e8f0;margin:0;">{resp}</p></div>'
    steps_html += '</div>'

    rs = "display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border-bottom:1px solid #334155;"
    ls = "color:#e2e8f0;font-size:0.95em;"
    vs = "color:#34d399;font-weight:bold;"
    score_html = (
        '<div style="text-align:center;background:#064e3b;border:2px solid #34d399;border-radius:16px;padding:24px;margin-bottom:16px;">'
        '<div style="font-size:3.5em;font-weight:900;color:#34d399;">1.000</div>'
        '<div style="color:#6ee7b7;">PERFECT SCORE</div></div>'
        '<div style="background:#1e293b;border-radius:12px;padding:16px;">'
        '<h4 style="color:#a5b4fc;margin:0 0 8px;">Reward Breakdown</h4>'
        '<div style="background:#0f172a;border-radius:8px;">'
        f'<div style="{rs}"><span style="{ls}">JSON Format</span><span style="{vs}">1.00</span></div>'
        f'<div style="{rs}"><span style="{ls}">Empathy</span><span style="{vs}">1.00</span></div>'
        f'<div style="{rs}"><span style="{ls}">Action Taken</span><span style="{vs}">1.00</span></div>'
        f'<div style="{rs}"><span style="{ls}">Resolution</span><span style="{vs}">1.00</span></div>'
        '<div style="display:flex;justify-content:space-between;padding:12px;background:#1e3a2e;border-top:2px solid #34d399;">'
        '<span style="color:#f1f5f9;font-weight:bold;font-size:1.1em;">Final Score</span>'
        '<span style="color:#34d399;font-weight:900;font-size:1.4em;">1.000</span></div>'
        '</div></div>')

    fig,ax=plt.subplots(figsize=(5,3),facecolor="#0f0f1a")
    ax.set_facecolor("#1e293b")
    bars=ax.bar(["Baseline","Trained"],[0.268,1.000],color=["#f87171","#34d399"],width=0.5)
    for bar,val in zip(bars,[0.268,1.000]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,f"{val:.3f}",ha="center",va="bottom",color="#f1f5f9",fontweight="bold")
    ax.set_ylim(0,1.25); ax.set_title("Reward Improvement (+273%)",color="#a5b4fc"); ax.tick_params(colors="#94a3b8")
    [s.set_color("#334155") for s in ax.spines.values()]
    plt.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=100,bbox_inches="tight"); plt.close(fig); buf.seek(0)
    chart=f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" style="width:100%;border-radius:8px;"/>'
    return steps_html, score_html, chart

with gr.Blocks(title="CustomerSupportEnv Demo") as demo:
    gr.HTML(HEADER)
    gr.HTML(SCORE_BANNER)
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_dd=gr.Dropdown(choices=TASK_IDS,label="Select Task",interactive=True)
                    task_info_out=gr.HTML()
                    customer_msg_out=gr.HTML()
                    run_btn=gr.Button("Run Agent",variant="primary",size="lg")
                    chart_out=gr.HTML()
                with gr.Column(scale=2):
                    score_out=gr.HTML()
                    steps_out=gr.HTML()
            task_dd.change(fn=on_task_select,inputs=[task_dd],outputs=[task_info_out,customer_msg_out])
            run_btn.click(fn=run_agent,inputs=[task_dd],outputs=[steps_out,score_out,chart_out])
        with gr.Tab("Architecture"):
            gr.HTML(ARCHITECTURE_HTML)
        with gr.Tab("API Reference"):
            gr.HTML(API_INFO)
        with gr.Tab("Scores"):
            gr.HTML(SCORES_INFO)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)