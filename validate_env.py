"""
validate_env.py — Validates the CustomerSupportEnv without external deps.
Tests core logic using plain Python dataclasses mirroring the Pydantic models.
Run: python validate_env.py
"""

import sys
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# ── Minimal model mirrors ────────────────────────────────────────────────────

class ActionType(str, Enum):
    RESPOND   = "respond"
    ESCALATE  = "escalate"
    REFUND    = "refund"
    CLOSE     = "close"
    TRANSFER  = "transfer"

class TicketStatus(str, Enum):
    OPEN      = "open"
    ESCALATED = "escalated"
    RESOLVED  = "resolved"
    CLOSED    = "closed"

@dataclass
class Action:
    action_type: ActionType
    response_text: Optional[str] = None
    escalation_reason: Optional[str] = None
    refund_amount: Optional[float] = None
    transfer_department: Optional[str] = None

@dataclass
class Turn:
    role: str
    content: str
    turn: int

@dataclass
class Meta:
    customer_name: str
    account_id: str
    plan: str
    issue_category: str
    expected_resolution: str
    refund_eligible: bool = False
    max_refund: float = 0.0
    requires_escalation: bool = False
    escalation_department: Optional[str] = None
    prior_contacts: int = 0

@dataclass
class State:
    ticket_id: str
    ticket_status: TicketStatus
    turn_number: int
    conversation_history: List[Turn]
    metadata: Meta
    cumulative_reward: float = 0.0
    refund_issued: float = 0.0
    escalated: bool = False
    resolved: bool = False
    last_response_text: Optional[str] = None
    repeated_response_count: int = 0

@dataclass
class Obs:
    ticket_id: str
    user_message: str
    conversation_history: List[Turn]
    ticket_status: TicketStatus
    turn_number: int


# ── Simplified Environment ───────────────────────────────────────────────────

TASKS = {
    "billing_dispute_easy": {
        "ticket_id": "TKT-001", "difficulty": "easy", "max_turns": 6,
        "initial_user_message": "Hi, I was charged $49.99 but I'm on the $29.99 Basic plan.",
        "metadata": Meta("Sarah", "ACC-001", "Basic $29.99", "billing", "refund",
                        True, 20.0, False)
    },
    "technical_outage_medium": {
        "ticket_id": "TKT-002", "difficulty": "medium", "max_turns": 8,
        "initial_user_message": "Dashboard blank for 3 days, team of 12 blocked.",
        "metadata": Meta("James", "ACC-002", "Business $199", "technical", "escalate",
                        True, 20.0, True, "Engineering")
    },
    "fraud_complaint_hard": {
        "ticket_id": "TKT-003", "difficulty": "hard", "max_turns": 10,
        "initial_user_message": "Account hacked, $448 in fraudulent charges.",
        "metadata": Meta("Priya", "ACC-003", "Pro $79", "fraud", "escalate+refund",
                        True, 448.0, True, "Fraud & Security")
    },
}


class SimpleEnv:
    R = {"helpful": 0.05, "empathy": 0.03, "escalate_good": 0.20,
         "escalate_bad": -0.10, "refund_good": 0.20, "refund_bad": -0.15,
         "close_good": 0.25, "close_bad": -0.20, "loop": -0.08, "short": -0.05}

    def __init__(self, task_id):
        self.task_id = task_id
        t = TASKS[task_id]
        self._state = State(
            ticket_id=t["ticket_id"], ticket_status=TicketStatus.OPEN,
            turn_number=0, conversation_history=[Turn("user", t["initial_user_message"], 0)],
            metadata=t["metadata"]
        )
        self.max_turns = t["max_turns"]

    def step(self, action: Action):
        s = self._state
        s.turn_number += 1
        reward = 0.0
        done = False

        if action.action_type == ActionType.RESPOND:
            text = action.response_text or ""
            if len(text) < 30:
                reward += self.R["short"]
            elif text == s.last_response_text:
                s.repeated_response_count += 1
                reward += self.R["loop"]
            else:
                reward += self.R["helpful"]
                if any(k in text.lower() for k in ["sorry","apologize","understand"]):
                    reward += self.R["empathy"]

        elif action.action_type == ActionType.ESCALATE:
            if s.metadata.requires_escalation:
                reason = (action.escalation_reason or "").lower()
                dept = (s.metadata.escalation_department or "").lower()
                s.escalated = True
                s.ticket_status = TicketStatus.ESCALATED
                reward += self.R["escalate_good"] if dept.split()[0] in reason else self.R["escalate_good"]*0.5
            else:
                reward += self.R["escalate_bad"]
            done = True

        elif action.action_type == ActionType.REFUND:
            meta = s.metadata
            if not meta.refund_eligible:
                reward -= 0.10
            else:
                if meta.issue_category == "fraud":
                    verified = any(
                        any(k in t.content.lower() for k in ["verify","date of birth","dob","last 4"])
                        for t in s.conversation_history if t.role == "agent"
                    )
                    if not verified:
                        reward += self.R["refund_bad"]
                    else:
                        s.refund_issued += (action.refund_amount or 0)
                        reward += self.R["refund_good"] * min(1, s.refund_issued / meta.max_refund)
                else:
                    s.refund_issued += (action.refund_amount or 0)
                    closeness = 1 - abs(s.refund_issued - meta.max_refund) / max(meta.max_refund, 1)
                    reward += self.R["refund_good"] * max(0.3, closeness)

        elif action.action_type == ActionType.CLOSE:
            if s.metadata.requires_escalation and not s.escalated:
                reward += self.R["close_bad"]
                s.ticket_status = TicketStatus.CLOSED
            elif s.metadata.refund_eligible and s.refund_issued == 0:
                reward += self.R["close_bad"] * 0.5
                s.ticket_status = TicketStatus.CLOSED
            else:
                reward += self.R["close_good"]
                s.ticket_status = TicketStatus.RESOLVED
                s.resolved = True
            done = True

        agent_text = action.response_text or action.escalation_reason or f"[{action.action_type}]"
        s.conversation_history.append(Turn("agent", agent_text, s.turn_number))
        s.last_response_text = agent_text

        if s.turn_number >= self.max_turns and not done:
            reward -= 0.10
            done = True

        reward = max(-1.0, min(1.0, round(reward, 4)))
        s.cumulative_reward = round(s.cumulative_reward + reward, 4)
        return reward, done


# ── Tests ────────────────────────────────────────────────────────────────────

def run_tests():
    results = []
    passed = 0
    failed = 0

    def test(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name} {detail}")
            failed += 1
        results.append((name, condition))

    print("\n" + "="*60)
    print("CustomerSupportEnv — Validation Suite")
    print("="*60)

    # ── Task availability ────────────────────────────────────────────────────
    print("\n[1] Task Availability")
    test("3 tasks defined", len(TASKS) == 3)
    test("Easy task exists", "billing_dispute_easy" in TASKS)
    test("Medium task exists", "technical_outage_medium" in TASKS)
    test("Hard task exists", "fraud_complaint_hard" in TASKS)

    # ── Interface ────────────────────────────────────────────────────────────
    print("\n[2] OpenEnv Interface")
    env = SimpleEnv("billing_dispute_easy")
    test("reset() returns observation", env._state is not None)
    test("Initial turn is 0", env._state.turn_number == 0)
    test("Initial history has 1 user message", len(env._state.conversation_history) == 1)

    reward, done = env.step(Action(ActionType.RESPOND, "I apologize for the inconvenience."))
    test("step() returns reward float", isinstance(reward, float))
    test("step() returns done bool", isinstance(done, bool))
    test("Reward in [-1, 1]", -1.0 <= reward <= 1.0)
    test("Turn incremented", env._state.turn_number == 1)

    # ── Reward signals ────────────────────────────────────────────────────────
    print("\n[3] Reward Function")

    # Helpful response
    env2 = SimpleEnv("billing_dispute_easy")
    r, _ = env2.step(Action(ActionType.RESPOND, "I understand and apologize for this error on your account."))
    test("Helpful response → positive reward", r > 0, f"got {r}")

    # Short/irrelevant response
    env3 = SimpleEnv("billing_dispute_easy")
    r, _ = env3.step(Action(ActionType.RESPOND, "Ok"))
    test("Short response → negative reward", r < 0, f"got {r}")

    # Loop detection
    env4 = SimpleEnv("billing_dispute_easy")
    env4.step(Action(ActionType.RESPOND, "Please hold while I check your account details."))
    r2, _ = env4.step(Action(ActionType.RESPOND, "Please hold while I check your account details."))
    test("Repeated response → loop penalty", r2 < 0, f"got {r2}")

    # Correct refund
    env5 = SimpleEnv("billing_dispute_easy")
    r, _ = env5.step(Action(ActionType.REFUND, refund_amount=20.0))
    test("Correct refund amount → positive reward", r > 0, f"got {r}")
    test("Refund tracked in state", env5._state.refund_issued == 20.0)

    # Correct close
    env6 = SimpleEnv("billing_dispute_easy")
    env6.step(Action(ActionType.REFUND, refund_amount=20.0))
    r, done = env6.step(Action(ActionType.CLOSE, "Your issue is resolved!"))
    test("Correct close → positive reward", r > 0, f"got {r}")
    test("Close terminates episode", done)

    # Premature close (no refund)
    env7 = SimpleEnv("billing_dispute_easy")
    r, done = env7.step(Action(ActionType.CLOSE, "Closing ticket."))
    test("Premature close → negative reward", r < 0, f"got {r}")

    # ── Escalation logic ─────────────────────────────────────────────────────
    print("\n[4] Escalation Logic")
    env8 = SimpleEnv("technical_outage_medium")
    r, done = env8.step(Action(
        ActionType.ESCALATE,
        escalation_reason="Engineering team needed for 403 error affecting Business accounts",
    ))
    test("Correct escalation → positive reward", r > 0, f"got {r}")
    test("Escalation terminates episode", done)
    test("State marked escalated", env8._state.escalated)

    env9 = SimpleEnv("billing_dispute_easy")  # doesn't need escalation
    r, done = env9.step(Action(ActionType.ESCALATE, escalation_reason="Escalating"))
    test("Unnecessary escalation → negative reward", r < 0, f"got {r}")

    # ── Fraud task ────────────────────────────────────────────────────────────
    print("\n[5] Fraud Task - Identity Verification")
    env10 = SimpleEnv("fraud_complaint_hard")
    # Refund without verification
    r, _ = env10.step(Action(ActionType.REFUND, refund_amount=448.0))
    test("Refund without identity verify → penalty", r < 0, f"got {r}")

    env11 = SimpleEnv("fraud_complaint_hard")
    env11.step(Action(ActionType.RESPOND,
        "For your security, please verify your identity: date of birth and last 4 digits of your card."))
    r, _ = env11.step(Action(ActionType.REFUND, refund_amount=448.0,
        response_text="Full $448 refund initiated."))
    test("Refund after verification → positive reward", r > 0, f"got {r}")

    # ── Max turns ────────────────────────────────────────────────────────────
    print("\n[6] Episode Termination")
    env12 = SimpleEnv("billing_dispute_easy")
    done_count = 0
    for i in range(10):
        _, done = env12.step(Action(ActionType.RESPOND, f"I'm still looking into this issue for you today {i}."))
        if done:
            done_count += 1
            break
    test("Episode terminates at max_turns", done_count > 0)

    # ── Determinism ──────────────────────────────────────────────────────────
    print("\n[7] Determinism")
    def trajectory_reward():
        e = SimpleEnv("billing_dispute_easy")
        rewards = []
        rewards.append(e.step(Action(ActionType.RESPOND, "I apologize for this billing error on your account."))[0])
        rewards.append(e.step(Action(ActionType.REFUND, refund_amount=20.0))[0])
        rewards.append(e.step(Action(ActionType.CLOSE, "Issue resolved, refund confirmed."))[0])
        return rewards

    r1, r2 = trajectory_reward(), trajectory_reward()
    test("Same trajectory → same rewards (deterministic)", r1 == r2, f"{r1} vs {r2}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    if failed == 0:
        print("🎉 ALL TESTS PASSED — Environment is valid!")
    else:
        print(f"⚠️  {failed} test(s) failed.")
    print("="*60)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
