"""
Task definitions for the Customer Support Agent environment.
Each task has an initial user message, metadata, and max turns.

Tasks:
  1. billing_dispute_easy        — Simple billing overcharge
  2. technical_outage_medium     — Technical outage with multi-symptom triage
  3. fraud_complaint_hard        — Fraud + account compromise
  4. subscription_cancellation_hard — Long-term customer retention challenge
  5. vip_account_recovery_expert — VIP multi-step account recovery
"""

from env.models import TicketMetadata

TASKS = {
    # ── EASY: Simple billing overcharge ─────────────────────────────────────
    "billing_dispute_easy": {
        "ticket_id": "TKT-001",
        "difficulty": "easy",
        "initial_user_message": (
            "Hi, I was charged $49.99 on my account this month but I'm on the "
            "$29.99 Basic plan. I never upgraded. Can you fix this please?"
        ),
        "metadata": TicketMetadata(
            customer_name="Sarah Mitchell",
            account_id="ACC-88421",
            plan="Basic ($29.99/mo)",
            prior_contacts=0,
            issue_category="billing",
            expected_resolution="refund",
            refund_eligible=True,
            max_refund=20.00,
            requires_escalation=False,
        ),
        "max_turns": 6,
        "system_context": (
            "Customer was accidentally charged the Pro plan rate. "
            "The correct resolution is to issue a $20.00 refund (the difference) "
            "and confirm the account remains on the Basic plan. "
            "No escalation is needed."
        ),
    },

    # ── MEDIUM: Technical outage with multi-symptom triage ───────────────────
    "technical_outage_medium": {
        "ticket_id": "TKT-002",
        "difficulty": "medium",
        "initial_user_message": (
            "This is the third day our dashboard isn't loading. "
            "It just shows a blank screen after login. Our whole team of 12 is blocked. "
            "We've tried Chrome, Firefox, cleared cache — nothing works. "
            "This is costing us real money."
        ),
        "metadata": TicketMetadata(
            customer_name="James Okafor",
            account_id="ACC-55901",
            plan="Business ($199/mo)",
            prior_contacts=2,
            issue_category="technical",
            expected_resolution="escalate",
            refund_eligible=True,
            max_refund=20.00,   # partial credit for downtime
            requires_escalation=True,
            escalation_department="Engineering",
        ),
        "max_turns": 8,
        "system_context": (
            "This is a known rendering bug affecting Business accounts created before 2024. "
            "Tier-1 support cannot fix it. Agent must gather: browser console errors, "
            "account creation date, and affected URL — then escalate to Engineering. "
            "A partial credit is appropriate. Closing without escalation = incorrect resolution."
        ),
    },

    # ── HARD: Fraud + account compromise ────────────────────────────────────
    "fraud_complaint_hard": {
        "ticket_id": "TKT-003",
        "difficulty": "hard",
        "initial_user_message": (
            "Someone has hacked my account! I see three charges I did NOT make: "
            "$149, $89, and $210 — all from last night. "
            "My email was also changed without my permission. "
            "I can't log in anymore. I need this fixed NOW. "
            "These charges total $448 and I want them all reversed."
        ),
        "metadata": TicketMetadata(
            customer_name="Priya Venkataraman",
            account_id="ACC-33107",
            plan="Pro ($79/mo)",
            prior_contacts=0,
            issue_category="fraud",
            expected_resolution="escalate+refund",
            refund_eligible=True,
            max_refund=448.00,
            requires_escalation=True,
            escalation_department="Fraud & Security",
        ),
        "max_turns": 10,
        "system_context": (
            "Confirmed account compromise. Agent must: (1) verify identity through "
            "alternative means (phone/DOB/last 4 of card), (2) lock the account, "
            "(3) initiate full $448 refund, (4) escalate to Fraud & Security team, "
            "(5) provide next steps to regain account access. "
            "Skipping identity verification before refund = severe error. "
            "Not escalating = incomplete resolution."
        ),
    },

    # ── HARD: Subscription cancellation with retention ──────────────────────
    "subscription_cancellation_hard": {
        "ticket_id": "TKT-004",
        "difficulty": "hard",
        "initial_user_message": (
            "I've been a customer for over 2 years now and I want to cancel my "
            "subscription. The Enterprise plan at $499/month is just too expensive "
            "for what we're getting. Our team has shrunk from 50 to 15 people and "
            "we don't use half the features anymore. I've already found a cheaper "
            "alternative. Please process my cancellation effective immediately."
        ),
        "metadata": TicketMetadata(
            customer_name="David Chen",
            account_id="ACC-12750",
            plan="Enterprise ($499/mo)",
            prior_contacts=1,
            issue_category="cancellation",
            expected_resolution="retain_or_cancel",
            refund_eligible=True,
            max_refund=499.00,
            requires_escalation=False,
        ),
        "max_turns": 12,
        "system_context": (
            "Long-term Enterprise customer wants to cancel due to cost vs. team shrinkage. "
            "Agent must: (1) verify customer identity, (2) ask for detailed cancellation reason, "
            "(3) acknowledge 2-year loyalty, (4) offer a retention deal (discounted plan, "
            "downgrade to Business at $199/mo, or 3 months free), (5) negotiate based on "
            "their specific concerns (team size, unused features), (6) if customer insists: "
            "process cancellation gracefully with confirmation and follow-up schedule. "
            "Extra credit for successful retention. Show empathy for their loyalty."
        ),
    },

    # ── EXPERT: VIP account recovery ────────────────────────────────────────
    "vip_account_recovery_expert": {
        "ticket_id": "TKT-005",
        "difficulty": "expert",
        "initial_user_message": (
            "This is extremely urgent. I'm the CTO of NexaFlow Technologies and our "
            "company account has been completely locked out since yesterday morning. "
            "We're on the Platinum plan at $2,499/month and our entire engineering "
            "team of 85 people cannot access any of our dashboards, API keys, or "
            "deployment pipelines. We've lost over $50,000 in productivity already. "
            "I've tried resetting the password but the recovery email was changed "
            "to an address I don't recognize. I suspect a security breach. "
            "Our account ID is ACC-VIP-0042 and I need this resolved within the hour."
        ),
        "metadata": TicketMetadata(
            customer_name="Alexandra Rivera",
            account_id="ACC-VIP-0042",
            plan="Platinum ($2,499/mo)",
            prior_contacts=0,
            issue_category="account_recovery",
            expected_resolution="escalate+refund",
            refund_eligible=True,
            max_refund=2499.00,
            requires_escalation=True,
            escalation_department="VIP Support",
        ),
        "max_turns": 15,
        "system_context": (
            "VIP Platinum customer — CTO-level executive. Account locked, suspected breach. "
            "$2,499/mo subscription, 85 users blocked, $50K+ productivity loss claimed. "
            "Agent must: (1) verify identity through at least 2 methods (company domain email, "
            "phone on file, last 4 of corporate card, date of account creation), "
            "(2) acknowledge VIP status and business impact urgency, "
            "(3) initiate account unlock/recovery process, "
            "(4) mention audit trail review of recent account changes, "
            "(5) offer service credit or compensation for downtime ($500-$2499), "
            "(6) escalate to VIP Support / Priority Engineering team, "
            "(7) provide immediate interim access workaround if possible. "
            "This requires the highest level of urgency, professionalism, and thoroughness. "
            "Failure to verify identity = security risk. Failure to escalate = VIP churn risk."
        ),
    },
}
