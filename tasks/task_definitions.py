"""
Task definitions for the Customer Support Agent environment.
Each task has an initial user message, metadata, and max turns.
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
}
