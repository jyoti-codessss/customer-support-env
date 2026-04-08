"""
Programmatic graders for each task.
Each grader receives the final EnvironmentState and returns a score in [0.0, 1.0].
Grading is deterministic and reproducible.
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


# ── Grader Registry ──────────────────────────────────────────────────────────

GRADERS = {
    "billing_dispute_easy": grade_billing_dispute_easy,
    "technical_outage_medium": grade_technical_outage_medium,
    "fraud_complaint_hard": grade_fraud_complaint_hard,
}
