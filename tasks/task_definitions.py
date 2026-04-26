"""
Task definitions for the Customer Support Agent environment.
"""
from env.models import TicketMetadata

TASKS = {
    "billing_dispute_easy": {
        "ticket_id": "TKT-001",
        "difficulty": "easy",
        "initial_user_message": "Hi, I was charged \.99 on my account this month but I'm on the \.99 Basic plan. I never upgraded. Can you fix this please?",
        "metadata": TicketMetadata(customer_name="Sarah Mitchell", account_id="ACC-88421", plan="Basic (\.99/mo)", prior_contacts=0, issue_category="billing", expected_resolution="refund", refund_eligible=True, max_refund=20.00, requires_escalation=False),
        "max_turns": 6,
        "system_context": "Customer was accidentally charged the Pro plan rate. The correct resolution is to issue a \.00 refund and confirm the account remains on the Basic plan.",
    },
    "technical_outage_medium": {
        "ticket_id": "TKT-002",
        "difficulty": "medium",
        "initial_user_message": "This is the third day our dashboard isn't loading. It just shows a blank screen after login. Our whole team of 12 is blocked. We've tried Chrome, Firefox, cleared cache — nothing works.",
        "metadata": TicketMetadata(customer_name="James Okafor", account_id="ACC-55901", plan="Business (\/mo)", prior_contacts=2, issue_category="technical", expected_resolution="escalate", refund_eligible=True, max_refund=20.00, requires_escalation=True, escalation_department="Engineering"),
        "max_turns": 8,
        "system_context": "Known rendering bug affecting Business accounts. Agent must gather diagnostics then escalate to Engineering. A partial credit is appropriate.",
    },
    "fraud_complaint_hard": {
        "ticket_id": "TKT-003",
        "difficulty": "hard",
        "initial_user_message": "Someone has hacked my account! I see three charges I did NOT make: \, \, and \ — all from last night. My email was also changed without my permission.",
        "metadata": TicketMetadata(customer_name="Priya Venkataraman", account_id="ACC-33107", plan="Pro (\/mo)", prior_contacts=0, issue_category="fraud", expected_resolution="escalate+refund", refund_eligible=True, max_refund=448.00, requires_escalation=True, escalation_department="Fraud & Security"),
        "max_turns": 10,
        "system_context": "Confirmed account compromise. Agent must verify identity, lock account, initiate full \ refund, and escalate to Fraud & Security team.",
    },
    "subscription_cancellation_hard": {
        "ticket_id": "TKT-004",
        "difficulty": "hard",
        "initial_user_message": "I've been a customer for over 2 years now and I want to cancel my subscription. The Enterprise plan at \/month is just too expensive. Our team has shrunk from 50 to 15 people.",
        "metadata": TicketMetadata(customer_name="David Chen", account_id="ACC-12750", plan="Enterprise (\/mo)", prior_contacts=1, issue_category="cancellation", expected_resolution="retain_or_cancel", refund_eligible=True, max_refund=499.00, requires_escalation=False),
        "max_turns": 12,
        "system_context": "Long-term Enterprise customer wants to cancel due to cost. Agent must acknowledge loyalty, offer retention deal, and negotiate based on their concerns.",
    },
    "vip_account_recovery_expert": {
        "ticket_id": "TKT-005",
        "difficulty": "expert",
        "initial_user_message": "This is extremely urgent. I'm the CTO of NexaFlow Technologies and our company account has been completely locked out since yesterday morning. We're on the Platinum plan at \,499/month.",
        "metadata": TicketMetadata(customer_name="Alexandra Rivera", account_id="ACC-VIP-0042", plan="Platinum (\,499/mo)", prior_contacts=0, issue_category="account_recovery", expected_resolution="escalate+refund", refund_eligible=True, max_refund=2499.00, requires_escalation=True, escalation_department="VIP Support"),
        "max_turns": 15,
        "system_context": "VIP Platinum customer — CTO-level executive. Account locked, suspected breach. Agent must verify identity through 2 methods, unlock account, offer compensation, and escalate to VIP Support.",
    },
}
