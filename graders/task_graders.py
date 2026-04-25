"""
Programmatic graders for each task.
Each grader receives the final EnvironmentState and returns a score in [0.0, 1.0].
Grading is deterministic and reproducible.

Tasks:
  1. billing_dispute_easy        — Refund accuracy, plan confirmation, empathy
  2. technical_outage_medium     — Diagnostics, escalation, compensation
  3. fraud_complaint_hard        — Identity verification order, refund, security
  4. subscription_cancellation_hard — Retention, loyalty, negotiation
  5. vip_account_recovery_expert — Multi-verify, audit, VIP escalation
"""

from __future__ import annotations
from typing import Tuple
from env.models import EnvironmentState, TicketStatus


def _keyword_present(history_texts: list[str], keywords: list[str]) -> bool:
    """Check if any keyword appears in the conversation (case-insensitive)."""
    combined = " ".join(history_texts).lower()
    return any(kw.lower() in combined for kw in keywords)


def _agent_turns(state: EnvironmentState) -> list[str]:
    return [t.content for t in state.conversation_history if t.role == "agent"]


def _user_turns(state: EnvironmentState) -> list[str]:
    return [t.content for t in state.conversation_history if t.role == "user"]


# ── EASY Grader ──────────────────────────────────────────────────────────────

def grade_billing_dispute_easy(state: EnvironmentState) -> Tuple[float, dict]:
    """
    Scoring rubric (total = 1.0):
      0.30 — Refund issued in correct amount ($20 ± $2)
      0.25 — Account plan confirmed as Basic
      0.20 — Ticket properly closed/resolved
      0.15 — Empathetic acknowledgment of the error
      0.10 — Efficiency (resolved within 4 turns)
    """
    score = 0.0
    breakdown = {}

    agent_texts = _agent_turns(state)
    all_texts = agent_texts

    # 0.30 — Correct refund issued
    refund_correct = (18.0 <= state.refund_issued <= 22.0)
    breakdown["refund_correct"] = refund_correct
    if refund_correct:
        score += 0.30

    # 0.25 — Plan confirmation mentioned
    plan_mentioned = _keyword_present(all_texts, ["basic", "29.99", "basic plan"])
    breakdown["plan_confirmed"] = plan_mentioned
    if plan_mentioned:
        score += 0.25

    # 0.20 — Ticket resolved or closed
    resolved = state.ticket_status in (TicketStatus.RESOLVED, TicketStatus.CLOSED)
    breakdown["ticket_closed"] = resolved
    if resolved:
        score += 0.20

    # 0.15 — Empathy
    empathy = _keyword_present(all_texts, [
        "sorry", "apologize", "apologies", "understand", "inconvenience"
    ])
    breakdown["empathy_shown"] = empathy
    if empathy:
        score += 0.15

    # 0.10 — Efficiency (≤ 4 turns)
    efficient = state.turn_number <= 4
    breakdown["efficient"] = efficient
    if efficient:
        score += 0.10

    breakdown["final_score"] = round(score, 3)
    return round(score, 3), breakdown


# ── MEDIUM Grader ────────────────────────────────────────────────────────────

def grade_technical_outage_medium(state: EnvironmentState) -> Tuple[float, dict]:
    """
    Scoring rubric (total = 1.0):
      0.35 — Escalated to Engineering
      0.20 — Collected at least 2 diagnostic data points before escalating
      0.20 — Ticket not closed without escalation
      0.15 — Offered partial credit/compensation
      0.10 — Acknowledged business impact + apologized
    """
    score = 0.0
    breakdown = {}

    agent_texts = _agent_turns(state)
    all_agent = " ".join(agent_texts).lower()

    # 0.35 — Escalated to Engineering
    escalated_correctly = (
        state.escalated and
        state.metadata.escalation_department is not None and
        "engineering" in (state.metadata.escalation_department or "").lower()
    )
    breakdown["escalated_to_engineering"] = escalated_correctly
    if escalated_correctly:
        score += 0.35

    # 0.20 — Diagnostic info collected (at least 2 of 4 signals)
    diagnostics = [
        any(kw in all_agent for kw in ["console", "error", "f12", "devtools"]),
        any(kw in all_agent for kw in ["when did", "how long", "date", "started"]),
        any(kw in all_agent for kw in ["url", "link", "address", "page"]),
        any(kw in all_agent for kw in ["screenshot", "screen", "capture"]),
    ]
    diag_count = sum(diagnostics)
    breakdown["diagnostics_collected"] = diag_count
    if diag_count >= 2:
        score += 0.20
    elif diag_count == 1:
        score += 0.10

    # 0.20 — Did NOT close without escalation (anti-reward for wrong closure)
    wrong_close = (
        state.ticket_status in (TicketStatus.RESOLVED, TicketStatus.CLOSED)
        and not state.escalated
    )
    breakdown["no_premature_close"] = not wrong_close
    if not wrong_close:
        score += 0.20

    # 0.15 — Partial credit/refund offered
    credit_offered = (
        state.refund_issued > 0 or
        _keyword_present(agent_texts, ["credit", "compensat", "refund", "discount"])
    )
    breakdown["credit_offered"] = credit_offered
    if credit_offered:
        score += 0.15

    # 0.10 — Business impact acknowledged
    impact_ack = _keyword_present(agent_texts, [
        "understand", "frustrat", "business", "team", "impact",
        "sorry", "apologize", "priority"
    ])
    breakdown["impact_acknowledged"] = impact_ack
    if impact_ack:
        score += 0.10

    breakdown["final_score"] = round(score, 3)
    return round(score, 3), breakdown


# ── HARD Grader ──────────────────────────────────────────────────────────────

def grade_fraud_complaint_hard(state: EnvironmentState) -> Tuple[float, dict]:
    """
    Scoring rubric (total = 1.0):
      0.25 — Identity verification attempted before any refund
      0.25 — Full $448 refund initiated (±$10)
      0.20 — Escalated to Fraud & Security
      0.15 — Account lock / security steps mentioned
      0.10 — Next steps for account recovery provided
      0.05 — Empathy + urgency matched appropriately
    """
    score = 0.0
    breakdown = {}

    agent_texts = _agent_turns(state)
    all_agent = " ".join(agent_texts).lower()

    # Determine turn at which refund was issued (for ordering check)
    refund_turn = None
    verify_turn = None
    for turn in state.conversation_history:
        if turn.role == "agent":
            text = turn.content.lower()
            if verify_turn is None and any(kw in text for kw in [
                "verify", "confirm your identity", "date of birth", "phone number",
                "last 4", "security question", "dob"
            ]):
                verify_turn = turn.turn
            if refund_turn is None and any(kw in text for kw in [
                "refund", "reverse", "reimburse", "credited"
            ]):
                refund_turn = turn.turn

    # 0.25 — Identity verification BEFORE refund
    if verify_turn is not None:
        if refund_turn is None or verify_turn <= refund_turn:
            breakdown["identity_verified_first"] = True
            score += 0.25
        else:
            breakdown["identity_verified_first"] = False  # refund before verify
    else:
        breakdown["identity_verified_first"] = False

    # 0.25 — Full $448 refund issued
    refund_full = (438.0 <= state.refund_issued <= 458.0)
    breakdown["full_refund_issued"] = refund_full
    if refund_full:
        score += 0.25
    elif state.refund_issued > 0:
        # Partial credit for partial refund
        partial = min(state.refund_issued / 448.0, 1.0) * 0.15
        score += partial
        breakdown["partial_refund_credit"] = round(partial, 3)

    # 0.20 — Escalated to Fraud & Security
    fraud_escalated = (
        state.escalated and
        "fraud" in (state.metadata.escalation_department or "").lower()
    )
    breakdown["escalated_to_fraud"] = fraud_escalated
    if fraud_escalated:
        score += 0.20

    # 0.15 — Account security steps
    security_steps = _keyword_present(agent_texts, [
        "lock", "suspend", "freeze", "block account",
        "reset password", "secure", "temporary hold"
    ])
    breakdown["security_steps_mentioned"] = security_steps
    if security_steps:
        score += 0.15

    # 0.10 — Account recovery next steps
    recovery = _keyword_present(agent_texts, [
        "recover", "regain access", "new password", "reset link",
        "email verification", "contact us", "follow up", "next steps"
    ])
    breakdown["recovery_steps_provided"] = recovery
    if recovery:
        score += 0.10

    # 0.05 — Empathy + urgency
    empathy = _keyword_present(agent_texts, [
        "sorry", "apologize", "understand how", "serious",
        "priority", "right away", "immediately", "urgent"
    ])
    breakdown["empathy_urgency"] = empathy
    if empathy:
        score += 0.05

    breakdown["final_score"] = round(score, 3)
    return round(score, 3), breakdown


# ── HARD Grader: Subscription Cancellation ───────────────────────────────────

def grade_subscription_cancellation_hard(state: EnvironmentState) -> Tuple[float, dict]:
    """
    Scoring rubric (total = 1.0):
      0.15 — Customer identity verified
      0.20 — Cancellation reason explored
      0.25 — Retention offer made (discount/downgrade/free months)
      0.15 — Loyalty acknowledged (2-year customer)
      0.15 — Proper resolution (cancel processed OR retention confirmed)
      0.10 — Efficiency (≤ 8 turns)
    """
    score = 0.0
    breakdown = {}

    agent_texts = _agent_turns(state)
    all_agent = " ".join(agent_texts).lower()

    # 0.15 — Customer verified
    verified = _keyword_present(agent_texts, [
        "verify", "confirm", "account", "david", "acc-12750",
        "identity", "on file", "can you confirm"
    ])
    breakdown["customer_verified"] = verified
    if verified:
        score += 0.15

    # 0.20 — Cancellation reason asked/explored
    reason_explored = _keyword_present(agent_texts, [
        "why", "reason", "what made you", "understand why",
        "what's prompting", "help me understand", "can you share",
        "features", "team size", "expensive", "cost"
    ])
    breakdown["cancellation_reason_asked"] = reason_explored
    if reason_explored:
        score += 0.20

    # 0.25 — Retention offer made
    retention = _keyword_present(agent_texts, [
        "discount", "downgrade", "free month", "offer", "retention",
        "special price", "reduced rate", "business plan",
        "save", "keep you", "stay with us", "$199",
        "lower tier", "adjusted", "deal", "incentive", "promo"
    ])
    breakdown["retention_offered"] = retention
    if retention:
        score += 0.25

    # 0.15 — Loyalty acknowledged
    loyalty = _keyword_present(agent_texts, [
        "loyal", "2 year", "two year", "long time", "valued",
        "appreciate", "thank you for being", "loyalty",
        "years with us", "dedication"
    ])
    breakdown["loyalty_acknowledged"] = loyalty
    if loyalty:
        score += 0.15

    # 0.15 — Proper resolution (either cancel or retain)
    resolved = (
        state.ticket_status in (TicketStatus.RESOLVED, TicketStatus.CLOSED) or
        _keyword_present(agent_texts, [
            "cancel", "process", "confirm cancel", "effective",
            "retained", "stay", "keep your", "updated your plan"
        ])
    )
    breakdown["proper_resolution"] = resolved
    if resolved:
        score += 0.15

    # 0.10 — Efficiency (≤ 8 turns)
    efficient = state.turn_number <= 8
    breakdown["efficient"] = efficient
    if efficient:
        score += 0.10

    breakdown["final_score"] = round(score, 3)
    return round(score, 3), breakdown


# ── EXPERT Grader: VIP Account Recovery ──────────────────────────────────────

def grade_vip_account_recovery_expert(state: EnvironmentState) -> Tuple[float, dict]:
    """
    Scoring rubric (total = 1.0):
      0.20 — Multi-method identity verification (≥ 2 methods)
      0.20 — Account unlock/recovery steps mentioned
      0.15 — Audit trail / security review mentioned
      0.15 — Compensation offered (service credit/refund)
      0.20 — Escalated to VIP / Priority team
      0.10 — VIP-level empathy + urgency
    """
    score = 0.0
    breakdown = {}

    agent_texts = _agent_turns(state)
    all_agent = " ".join(agent_texts).lower()

    # 0.20 — Multi-method identity verification
    verify_methods = [
        any(kw in all_agent for kw in ["email", "domain", "company email", "@"]),
        any(kw in all_agent for kw in ["phone", "call", "mobile", "number on file"]),
        any(kw in all_agent for kw in ["last 4", "card", "payment method", "credit card"]),
        any(kw in all_agent for kw in ["date of birth", "dob", "birthday"]),
        any(kw in all_agent for kw in ["account creation", "when did you", "sign up date"]),
        any(kw in all_agent for kw in ["verify", "confirm your identity", "confirm you are"]),
    ]
    method_count = sum(verify_methods)
    breakdown["multi_method_verification"] = method_count
    if method_count >= 2:
        score += 0.20
    elif method_count == 1:
        score += 0.10

    # 0.20 — Account unlock/recovery
    unlocked = _keyword_present(agent_texts, [
        "unlock", "restore", "recover", "regain access",
        "reset", "reactivate", "re-enable", "access restored",
        "interim access", "temporary access", "workaround"
    ])
    breakdown["account_unlocked"] = unlocked
    if unlocked:
        score += 0.20

    # 0.15 — Audit trail / security review
    audit = _keyword_present(agent_texts, [
        "audit", "security review", "investigate", "activity log",
        "trail", "review recent", "changes made", "unauthorized changes",
        "forensic", "breach investigation", "security team will review"
    ])
    breakdown["audit_trail_mentioned"] = audit
    if audit:
        score += 0.15

    # 0.15 — Compensation offered
    compensation = (
        state.refund_issued > 0 or
        _keyword_present(agent_texts, [
            "credit", "compensat", "refund", "free month",
            "discount", "waive", "service credit", "reimburse"
        ])
    )
    breakdown["compensation_offered"] = compensation
    if compensation:
        score += 0.15

    # 0.20 — Escalated to VIP/Priority team
    vip_escalated = (
        state.escalated and
        _keyword_present(agent_texts, [
            "vip", "priority", "premium support", "account management",
            "senior engineer", "executive support", "dedicated team"
        ])
    ) or _keyword_present(agent_texts, [
        "escalat", "priority team", "vip support", "vip team",
        "senior", "specialist"
    ])
    breakdown["escalated_to_vip"] = vip_escalated
    if vip_escalated:
        score += 0.20

    # 0.10 — VIP-level empathy + urgency
    vip_empathy = _keyword_present(agent_texts, [
        "sorry", "apologize", "understand how critical",
        "highest priority", "immediately", "right away", "urgent",
        "top priority", "unacceptable", "deeply sorry",
        "platinum", "valued", "premium"
    ])
    breakdown["vip_empathy"] = vip_empathy
    if vip_empathy:
        score += 0.10

    breakdown["final_score"] = round(score, 3)
    return round(score, 3), breakdown


# ── Grader Registry ──────────────────────────────────────────────────────────

GRADERS = {
    "billing_dispute_easy": grade_billing_dispute_easy,
    "technical_outage_medium": grade_technical_outage_medium,
    "fraud_complaint_hard": grade_fraud_complaint_hard,
    "subscription_cancellation_hard": grade_subscription_cancellation_hard,
    "vip_account_recovery_expert": grade_vip_account_recovery_expert,
}
