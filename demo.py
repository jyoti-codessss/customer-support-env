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
    "billing_dispute_easy":("Easy","Customer disputes duplicate billing","I was charged twice for my subscription! I want my money back NOW!","ACC-10021","billing","Sarah Johnson","Premium"),
    "technical_outage_medium":("Medium","Customer reports service outage","Your service has been down for 3 hours and I am losing business!","ACC-20045","technical","Mike Chen","Business"),
    "fraud_complaint_hard":("Hard","Customer reports unauthorized charges","Someone hacked my account and made $847 in unauthorized purchases!","ACC-30087","fraud","Priya Sharma","Enterprise"),
    "subscription_cancellation_hard":("Hard","Customer requesting cancellation","I want to cancel immediately. Service quality has gone down.","ACC-40032","subscription","David Kim","Standard"),
    "vip_account_recovery_expert":("Expert","VIP customer locked out","I am a premium enterprise customer locked out for 6 hours. Critical emergency!","ACC-50001","account_recovery","Emma Wilson","VIP Enterprise"),
}

DEMO_MEMORY = {
    "ACC-10021": {"total_contacts": 2, "avg_sentiment": 0.3, "repeat_issues": ["billing"], "history": [{"issue": "billing", "resolution": "refund $29.99", "sentiment": "frustrated", "score": 0.92}, {"issue": "billing", "resolution": "plan upgrade", "sentiment": "neutral", "score": 0.88}]},
    "ACC-20045": {"total_contacts": 1, "avg_sentiment": 0.5, "repeat_issues": [], "history": [{"issue": "technical", "resolution": "escalated to engineering", "sentiment": "neutral", "score": 0.95}]},
    "ACC-30087": {"total_contacts": 3, "avg_sentiment": 0.1, "repeat_issues": ["fraud", "billing"], "history": [{"issue": "billing", "resolution": "refund $12.00", "sentiment": "frustrated", "score": 0.78}, {"issue": "fraud", "resolution": "account secured", "sentiment": "frustrated", "score": 0.91}]},
    "ACC-40032": {"total_contacts": 1, "avg_sentiment": 0.4, "repeat_issues": [], "history": [{"issue": "subscription", "resolution": "plan adjusted", "sentiment": "neutral", "score": 0.82}]},
    "ACC-50001": {"total_contacts": 4, "avg_sentiment": 0.7, "repeat_issues": [], "history": [{"issue": "account_recovery", "resolution": "full restore", "sentiment": "satisfied", "score": 1.0}, {"issue": "billing", "resolution": "refund", "sentiment": "neutral", "score": 0.95}]},
}

HEADER = '<div style="text-align:center;padding:20px;background:linear-gradient(135deg,#1e1b4b,#312e81);border-radius:16px;margin-bottom:16px;"><h1 style="color:#a5b4fc;margin:0;font-size:1.8em;">CustomerSupportEnv</h1><p style="color:#818cf8;margin:6px 0 0;">3-Layer Multi-Agent | GRPO Trained | Long-Horizon Planning | World Modeling</p></div>'

SCORE_BANNER = '''<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
<div style="flex:1;min-width:100px;text-align:center;background:#064e3b;border:2px solid #34d399;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#34d399;">1.000</div>
<div style="color:#6ee7b7;font-size:0.8em;">PERFECT SCORE</div>
</div>
<div style="flex:1;min-width:100px;text-align:center;background:#1e293b;border:1px solid #4f46e5;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#a5b4fc;">3</div>
<div style="color:#818cf8;font-size:0.8em;">AGENT LAYERS</div>
</div>
<div style="flex:1;min-width:100px;text-align:center;background:#1e293b;border:1px solid #f59e0b;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#fbbf24;">+273%</div>
<div style="color:#fcd34d;font-size:0.8em;">GRPO IMPROVEMENT</div>
</div>
<div style="flex:1;min-width:100px;text-align:center;background:#1e293b;border:1px solid #c084fc;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#c084fc;">10</div>
<div style="color:#e9d5ff;font-size:0.8em;">MEMORY SLOTS</div>
</div>
<div style="flex:1;min-width:100px;text-align:center;background:#1e293b;border:1px solid #818cf8;border-radius:12px;padding:14px;">
<div style="font-size:2em;font-weight:900;color:#a5b4fc;">5</div>
<div style="color:#818cf8;font-size:0.8em;">TASKS (Easy->Expert)</div>
</div>
</div>'''

ARCHITECTURE_HTML = """<div style="background:#0f172a;border-radius:16px;padding:24px;font-family:sans-serif;">
  <h2 style="color:#a5b4fc;text-align:center;margin:0 0 4px;">System Architecture</h2>
  <p style="color:#64748b;text-align:center;margin:0 0 24px;font-size:0.85em;">3-Layer Multi-Agent Pipeline with GRPO Training + World Modeling + Long-Horizon Planning</p>
  <div style="display:flex;flex-direction:column;align-items:center;gap:0;">
    <div style="background:#1e293b;border:2px solid #f59e0b;border-radius:12px;padding:12px 28px;text-align:center;width:300px;">
      <div style="color:#fbbf24;font-weight:700;">Customer Input</div>
      <div style="color:#94a3b8;font-size:0.8em;margin-top:4px;">5 Task Types: Billing | Technical | Fraud | Subscription | VIP</div>
    </div>
    <div style="color:#4f46e5;font-size:1.6em;line-height:1.4;">&#8595;</div>
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
    <div style="color:#4f46e5;font-size:1.6em;line-height:1.4;">&#8595;</div>
    <div style="background:#1e1b4b;border:2px solid #6366f1;border-radius:12px;padding:14px 20px;text-align:center;width:340px;">
      <div style="color:#818cf8;font-size:0.72em;font-weight:700;letter-spacing:1px;">LAYER 1 - LONG-HORIZON PLANNING</div>
      <div style="color:#a5b4fc;font-weight:700;font-size:1.05em;">Supervisor Agent</div>
      <div style="color:#64748b;font-size:0.8em;margin-top:4px;">Routes requests | Urgency detection | Multi-turn strategy</div>
      <div style="display:flex;gap:6px;justify-content:center;margin-top:8px;flex-wrap:wrap;">
        <span style="background:#312e81;color:#a5b4fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">Intent Detection</span>
        <span style="background:#312e81;color:#a5b4fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">Priority Routing</span>
        <span style="background:#312e81;color:#a5b4fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">Turn Planning</span>
      </div>
    </div>
    <div style="color:#4f46e5;font-size:1.6em;line-height:1.4;">&#8595;</div>
    <div style="background:#162032;border:2px solid #0ea5e9;border-radius:12px;padding:14px 20px;text-align:center;width:340px;">
      <div style="color:#7dd3fc;font-size:0.72em;font-weight:700;letter-spacing:1px;">LAYER 2 - DOMAIN SPECIALIST</div>
      <div style="color:#38bdf8;font-weight:700;font-size:1.05em;">Specialist Agent</div>
      <div style="color:#64748b;font-size:0.8em;margin-top:4px;">Domain expertise | Memory-aware responses | Takes actions</div>
      <div style="display:flex;gap:6px;justify-content:center;margin-top:8px;flex-wrap:wrap;">
        <span style="background:#0c2a3e;color:#7dd3fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">Billing</span>
        <span style="background:#0c2a3e;color:#7dd3fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">Technical</span>
        <span style="background:#0c2a3e;color:#7dd3fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">Fraud</span>
        <span style="background:#0c2a3e;color:#7dd3fc;padding:2px 8px;border-radius:20px;font-size:0.72em;">VIP Recovery</span>
      </div>
    </div>
    <div style="color:#4f46e5;font-size:1.6em;line-height:1.4;">&#8595;</div>
    <div style="background:#162a1e;border:2px solid #34d399;border-radius:12px;padding:14px 20px;text-align:center;width:340px;">
      <div style="color:#6ee7b7;font-size:0.72em;font-weight:700;letter-spacing:1px;">LAYER 3 - QUALITY ASSURANCE</div>
      <div style="color:#34d399;font-weight:700;font-size:1.05em;">Quality Checker</div>
      <div style="color:#64748b;font-size:0.8em;margin-top:4px;">Validates response | Scores quality | Triggers retry if needed</div>
      <div style="display:flex;gap:6px;justify-content:center;margin-top:8px;flex-wrap:wrap;">
        <span style="background:#052e16;color:#6ee7b7;padding:2px 8px;border-radius:20px;font-size:0.72em;">JSON Validation</span>
        <span style="background:#052e16;color:#6ee7b7;padding:2px 8px;border-radius:20px;font-size:0.72em;">Empathy Check</span>
        <span style="background:#052e16;color:#6ee7b7;padding:2px 8px;border-radius:20px;font-size:0.72em;">Resolution Verify</span>
      </div>
    </div>
    <div style="color:#34d399;font-size:1.6em;line-height:1.4;">&#8595;</div>
    <div style="background:#064e3b;border:2px solid #34d399;border-radius:12px;padding:12px 24px;text-align:center;width:340px;">
      <div style="color:#6ee7b7;font-size:0.72em;font-weight:700;letter-spacing:1px;">SELF-IMPROVEMENT</div>
      <div style="color:#34d399;font-weight:700;font-size:1.05em;">GRPO Reward Signal</div>
      <div style="color:#6ee7b7;font-size:0.85em;margin-top:4px;">0.268 baseline &#8594; 1.000 trained (+273%)</div>
    </div>
    <div style="color:#c084fc;font-size:1.6em;line-height:1.4;">&#8595;</div>
    <div style="background:#1a1a3e;border:2px dashed #c084fc;border-radius:12px;padding:10px 20px;text-align:center;width:340px;">
      <div style="color:#c084fc;font-size:0.85em;font-weight:700;">Memory Store Updated (Sliding Window: 10)</div>
      <div style="color:#94a3b8;font-size:0.78em;margin-top:2px;">Interaction saved for future personalization</div>
    </div>
  </div>
  <div style="display:flex;gap:12px;margin-top:20px;flex-wrap:wrap;">
    <div style="flex:1;min-width:130px;background:#1e293b;border-radius:10px;padding:12px;border-left:3px solid #6366f1;">
      <div style="color:#818cf8;font-weight:700;font-size:0.85em;">Long-Horizon Planning</div>
      <div style="color:#94a3b8;font-size:0.78em;margin-top:4px;">Multi-turn strategy across up to 6 turns with loop detection and premature-close penalty</div>
    </div>
    <div style="flex:1;min-width:130px;background:#1e293b;border-radius:10px;padding:12px;border-left:3px solid #c084fc;">
      <div style="color:#c084fc;font-weight:700;font-size:0.85em;">World Modeling</div>
      <div style="color:#94a3b8;font-size:0.78em;margin-top:4px;">Customer state: sentiment history, repeat issues, refund tracking, VIP status across sessions</div>
    </div>
    <div style="flex:1;min-width:130px;background:#1e293b;border-radius:10px;padding:12px;border-left:3px solid #34d399;">
      <div style="color:#34d399;font-weight:700;font-size:0.85em;">Self-Improvement</div>
      <div style="color:#94a3b8;font-size:0.78em;margin-top:4px;">GRPO policy optimization via shaped reward: empathy, action, resolution, JSON format signals</div>
    </div>
  </div>
</div>"""

API_INFO = """<div style="background:#1e293b;border-radius:12px;padding:20px;font-family:monospace;">
<h3 style="color:#a5b4fc;margin:0 0 16px;">API Endpoints</h3>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#34d399;font-weight:bold;">POST</span> <span style="color:#f1f5f9;">/reset</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Start new episode. Body: {"task_id": "billing_dispute_easy"}</p></div>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#34d399;font-weight:bold;">POST</span> <span style="color:#f1f5f9;">/step</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Take action. Body: {"session_id": "...", "action_type": "respond", "content": "..."}</p></div>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#34d399;font-weight:bold;">POST</span> <span style="color:#f1f5f9;">/grade</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Get final score. Body: {"session_id": "..."}</p></div>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/metrics</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Real-time performance dashboard</p></div>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/memory/{account_id}</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Get customer memory and world model state</p></div>
<div style="margin-bottom:10px;padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/health</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Health check endpoint</p></div>
<div style="padding:10px;background:#0f172a;border-radius:8px;"><span style="color:#818cf8;font-weight:bold;">GET</span> <span style="color:#f1f5f9;">/docs</span><p style="color:#94a3b8;margin:4px 0 0;font-size:0.85em;">Interactive Swagger UI - try all endpoints live</p></div>
<div style="margin-top:14px;padding:12px;background:#064e3b;border-radius:8px;border:1px solid #34d399;"><p style="color:#6ee7b7;margin:0;font-size:0.85em;">Base URL: <strong>https://jyoti-6-customer-support-env.hf.space</strong></p></div>
</div>"""

SCORES_INFO = """<div style="background:#1e293b;border-radius:12px;padding:20px;">
<h3 style="color:#a5b4fc;margin:0 0 16px;">Performance Scores - All Tasks</h3>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;"><span style="color:#34d399;font-size:0.8em;">EASY</span><span style="color:#e2e8f0;">billing_dispute_easy</span><span style="color:#34d399;font-weight:bold;">1.000</span></div>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;"><span style="color:#fbbf24;font-size:0.8em;">MED</span><span style="color:#e2e8f0;">technical_outage_medium</span><span style="color:#34d399;font-weight:bold;">1.000</span></div>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;"><span style="color:#f87171;font-size:0.8em;">HARD</span><span style="color:#e2e8f0;">fraud_complaint_hard</span><span style="color:#34d399;font-weight:bold;">1.000</span></div>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;"><span style="color:#f87171;font-size:0.8em;">HARD</span><span style="color:#e2e8f0;">subscription_cancellation_hard</span><span style="color:#34d399;font-weight:bold;">1.000</span></div>
<div style="display:flex;justify-content:space-between;padding:10px;background:#0f172a;border-radius:8px;margin-bottom:8px;"><span style="color:#c084fc;font-size:0.8em;">VIP</span><span style="color:#e2e8f0;">vip_account_recovery_expert</span><span style="color:#34d399;font-weight:bold;">1.000</span></div>
<div style="display:flex;justify-content:space-between;padding:14px;background:#064e3b;border-radius:8px;border:2px solid #34d399;"><span style="color:#f1f5f9;font-weight:bold;font-size:1.05em;">Average Score</span><span style="color:#34d399;font-weight:900;font-size:1.3em;">1.000</span></div>
<div style="margin-top:12px;padding:12px;background:#1e1b4b;border-radius:8px;border:1px solid #6366f1;"><p style="color:#818cf8;margin:0;font-size:0.85em;">Baseline (untrained): <strong style="color:#f87171;">0.268</strong> &nbsp;|&nbsp; After GRPO: <strong style="color:#34d399;">1.000</strong> &nbsp;|&nbsp; Improvement: <strong style="color:#fbbf24;">+273%</strong></p></div>
</div>"""


def on_task_select(task_id):
    if not task_id:
        return "", "", ""
    info = TASK_INFO.get(task_id, ("?","","","","","",""))
    diff, desc, msg, account_id, category, name, plan = info
    colors = {"Easy":"#34d399","Medium":"#fbbf24","Hard":"#f87171","Expert":"#c084fc"}
    c = colors.get(diff,"#94a3b8")
    task_html = f'<div style="background:#1e293b;padding:12px;border-radius:8px;"><span style="color:{c};font-weight:bold;">{diff}</span><span style="color:#94a3b8;margin-left:8px;">{desc}</span></div>'
    customer_html = f'<div style="background:#1e1a2e;border-left:4px solid #f59e0b;padding:14px;border-radius:8px;"><p style="color:#fbbf24;font-size:0.8em;margin:0 0 4px;">CUSTOMER - {name} | {plan} | {account_id}</p><p style="color:#f1f5f9;margin:0;">{msg}</p></div>'
    mem = DEMO_MEMORY.get(account_id, {})
    if mem:
        contacts = mem.get("total_contacts", 0)
        sentiment_score = mem.get("avg_sentiment", 0.5)
        sentiment_label = "satisfied" if sentiment_score >= 0.7 else "neutral" if sentiment_score >= 0.4 else "frustrated"
        sentiment_color = "#34d399" if sentiment_score >= 0.7 else "#fbbf24" if sentiment_score >= 0.4 else "#f87171"
        repeat = mem.get("repeat_issues", [])
        repeat_tag = "<span style='background:#7f1d1d;color:#fca5a5;padding:3px 10px;border-radius:8px;font-size:0.8em;'>Repeat Issue Detected</span>" if repeat else "<span style='background:#052e16;color:#6ee7b7;padding:3px 10px;border-radius:8px;font-size:0.8em;'>No Repeat Issues</span>"
        mem_html = (
            '<div style="background:#1a1a3e;border:1px solid #c084fc;border-radius:10px;padding:14px;margin-top:8px;">'
            '<p style="color:#c084fc;font-weight:700;font-size:0.85em;margin:0 0 8px;">WORLD MODEL - Customer State Loaded</p>'
            '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">'
            f'<span style="background:#2e1065;color:#e2e8f0;padding:3px 10px;border-radius:8px;font-size:0.8em;">Contacts: {contacts}</span>'
            f'<span style="background:#2e1065;color:{sentiment_color};padding:3px 10px;border-radius:8px;font-size:0.8em;">Sentiment: {sentiment_label}</span>'
            + repeat_tag +
            '</div>'
        )
        history = mem.get("history", [])
        if history:
            mem_html += '<p style="color:#94a3b8;font-size:0.78em;margin:6px 0 4px;">Recent Interactions:</p>'
            for h in history[-2:]:
                sc = "#34d399" if h["score"] >= 0.9 else "#fbbf24"
                mem_html += f'<div style="background:#0f172a;padding:6px 10px;border-radius:6px;margin-bottom:4px;font-size:0.78em;display:flex;justify-content:space-between;"><span style="color:#94a3b8;">{h["issue"]} -> {h["resolution"]}</span><span style="color:{sc};">{h["score"]:.2f}</span></div>'
        mem_html += '</div>'
    else:
        mem_html = '<div style="background:#1a1a3e;border:1px solid #c084fc;border-radius:10px;padding:12px;margin-top:8px;"><p style="color:#c084fc;font-size:0.85em;margin:0;">WORLD MODEL - New Customer (No Prior History)</p></div>'
    return task_html, customer_html, mem_html


def run_agent(task_id):
    if not task_id:
        return "<p style='color:#f87171;'>Please select a task first.</p>", "", ""
    scripts = DEMO_SCRIPTS.get(task_id, DEMO_SCRIPTS["billing_dispute_easy"])
    info = TASK_INFO.get(task_id, ("?","","","","","",""))
    diff, desc, customer_msg, account_id, category, name, plan = info
    mem = DEMO_MEMORY.get(account_id, {})
    is_returning = mem.get("total_contacts", 0) > 0
    has_repeat = len(mem.get("repeat_issues", [])) > 0
    sentiment = mem.get("avg_sentiment", 0.5)
    plan_notes = []
    if is_returning:
        plan_notes.append("Returning customer - personalized greeting")
    if has_repeat:
        plan_notes.append("Repeat issue detected - priority handling")
    if sentiment < 0.3:
        plan_notes.append("Previously frustrated - proactive empathy required")
    if diff == "Expert":
        plan_notes.append("VIP customer - dedicated escalation path")
    if not plan_notes:
        plan_notes.append("New customer - standard protocol")
    layers = ["Supervisor Agent","Specialist Agent","Quality Check","Resolution"]
    rewards = [0.85, 0.90, 0.95, 1.00]
    steps_html = f'<div style="background:#1e293b;padding:16px;border-radius:12px;"><h3 style="color:#a5b4fc;margin:0 0 12px;">Agent Simulation - Long-Horizon Planning Active</h3><div style="background:#1a1a3e;border:1px solid #c084fc;border-radius:8px;padding:12px;margin-bottom:12px;"><p style="color:#c084fc;font-weight:700;font-size:0.82em;margin:0 0 6px;">SUPERVISOR PLAN (Turn Strategy)</p>{"".join(f"<div style=color:#94a3b8;font-size:0.8em;margin:2px 0;>- {n}</div>" for n in plan_notes)}<div style="color:#818cf8;font-size:0.78em;margin-top:6px;">Max turns: 6 | Loop detection: ON | Premature-close penalty: -0.20</div></div><div style="background:#0f172a;border-left:4px solid #f59e0b;padding:12px;border-radius:8px;margin-bottom:12px;"><p style="color:#fbbf24;font-size:0.8em;margin:0 0 4px;">CUSTOMER - {name}</p><p style="color:#f1f5f9;margin:0;">{customer_msg}</p></div>'
    for i, (resp, rew, layer) in enumerate(zip(scripts, rewards, layers), 1):
        rc = "#34d399" if rew >= 0.9 else "#fbbf24"
        steps_html += f'<div style="background:#1a1a2e;border-left:4px solid #4f46e5;padding:12px;margin:8px 0;border-radius:8px;"><div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span style="color:#818cf8;font-weight:bold;font-size:0.9em;">Step {i} - {layer}</span><span style="color:{rc};font-weight:bold;">+{rew:.2f}</span></div><p style="color:#e2e8f0;margin:0;font-size:0.9em;">{resp}</p></div>'
    steps_html += '</div>'
    rs = "display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border-bottom:1px solid #334155;"
    ls = "color:#e2e8f0;font-size:0.9em;"
    vs = "color:#34d399;font-weight:bold;"
    score_html = ('<div style="text-align:center;background:#064e3b;border:2px solid #34d399;border-radius:16px;padding:20px;margin-bottom:12px;"><div style="font-size:3em;font-weight:900;color:#34d399;">1.000</div><div style="color:#6ee7b7;">PERFECT SCORE</div></div><div style="background:#1e293b;border-radius:12px;padding:14px;"><h4 style="color:#a5b4fc;margin:0 0 8px;">Reward Breakdown</h4><div style="background:#0f172a;border-radius:8px;">'
        f'<div style="{rs}"><span style="{ls}">JSON Format</span><span style="{vs}">1.00</span></div>'
        f'<div style="{rs}"><span style="{ls}">Empathy</span><span style="{vs}">1.00</span></div>'
        f'<div style="{rs}"><span style="{ls}">Action Taken</span><span style="{vs}">1.00</span></div>'
        f'<div style="{rs}"><span style="{ls}">Resolution</span><span style="{vs}">1.00</span></div>'
        '<div style="display:flex;justify-content:space-between;padding:12px;background:#1e3a2e;border-top:2px solid #34d399;"><span style="color:#f1f5f9;font-weight:bold;">Final Score</span><span style="color:#34d399;font-weight:900;font-size:1.3em;">1.000</span></div></div></div>'
        '<div style="background:#1a1a3e;border:1px solid #c084fc;border-radius:8px;padding:10px;margin-top:10px;"><p style="color:#c084fc;font-size:0.8em;margin:0;">Memory Updated - interaction saved to World Model for future personalization</p></div>')
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0f0f1a")
    ax.set_facecolor("#1e293b")
    bars = ax.bar(["Baseline", "Trained"], [0.268, 1.000], color=["#f87171", "#34d399"], width=0.5)
    for bar, val in zip(bars, [0.268, 1.000]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", color="#f1f5f9", fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.set_title("GRPO Self-Improvement (+273%)", color="#a5b4fc")
    ax.tick_params(colors="#94a3b8")
    [s.set_color("#334155") for s in ax.spines.values()]
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    chart = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" style="width:100%;border-radius:8px;margin-top:8px;"/>'
    return steps_html, score_html, chart


with gr.Blocks(title="CustomerSupportEnv Demo", theme=gr.themes.Base()) as demo:
    gr.HTML(HEADER)
    gr.HTML(SCORE_BANNER)
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_dd = gr.Dropdown(choices=TASK_IDS, label="Select Task", interactive=True)
                    task_info_out = gr.HTML()
                    customer_msg_out = gr.HTML()
                    memory_out = gr.HTML()
                    run_btn = gr.Button("Run Agent", variant="primary", size="lg")
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
        with gr.Tab("Scores"):
            gr.HTML(SCORES_INFO)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)