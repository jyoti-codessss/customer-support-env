"""
CustomerSupportEnv — OpenEnv-compliant Customer Support Agent environment.

Implements:
  reset()         → Observation
  step(action)    → (Observation, reward, done, info)
  state()         → EnvironmentState
  grade()         → float (final task score 0.0–1.0)
"""

from __future__ import annotations
import copy
import uuid
from typing import Any, Dict, Optional, Tuple

from env.models import (
    Action, ActionType, Observation, EnvironmentState,
    ConversationTurn, TicketStatus, TicketMetadata,
)
from tasks.task_definitions import TASKS
from graders.task_graders import GRADERS


# ── Reward Constants ─────────────────────────────────────────────────────────

R_HELPFUL_RESPONSE   =  0.05   # relevant, substantive reply
R_EMPATHY            =  0.03   # empathy keywords detected
R_CORRECT_ESCALATION =  0.20   # escalation to right department
R_WRONG_ESCALATION   = -0.10   # escalating when not needed
R_CORRECT_REFUND     =  0.20   # refund within acceptable range
R_WRONG_REFUND       = -0.15   # refund without verification (fraud task)
R_RESOLUTION         =  0.25   # correct close
R_PREMATURE_CLOSE    = -0.20   # closing without resolving
R_LOOP_PENALTY       = -0.08   # repeated identical response
R_IRRELEVANT         = -0.05   # very short / off-topic response

EMPATHY_KEYWORDS = [
    "sorry", "apologize", "understand", "frustrat",
    "inconvenience", "apologies", "help you", "right away"
]


class CustomerSupportEnv:
    """
    Multi-turn customer support simulation environment.

    Usage:
        env = CustomerSupportEnv("billing_dispute_easy")
        obs = env.reset()
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        score = env.grade()
    """

    def __init__(self, task_id: str = "billing_dispute_easy"):
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}"
            )
        self.task_id = task_id
        self._task_def = TASKS[task_id]
        self._state: Optional[EnvironmentState] = None

    # ── Public OpenEnv Interface ─────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment to initial state. Returns first observation."""
        meta: TicketMetadata = copy.deepcopy(self._task_def["metadata"])
        ticket_id = self._task_def["ticket_id"]
        initial_msg = self._task_def["initial_user_message"]

        self._state = EnvironmentState(
            ticket_id=ticket_id,
            ticket_status=TicketStatus.OPEN,
            turn_number=0,
            conversation_history=[
                ConversationTurn(role="user", content=initial_msg, turn=0)
            ],
            metadata=meta,
            cumulative_reward=0.0,
        )
        return self._build_observation(initial_msg)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply agent action. Returns (observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        s = self._state
        s.turn_number += 1
        max_turns = self._task_def["max_turns"]

        reward = 0.0
        done = False
        info: Dict[str, Any] = {"action": action.action_type, "turn": s.turn_number}

        # ── Process action ───────────────────────────────────────────────────

        if action.action_type == ActionType.RESPOND:
            reward += self._handle_respond(action)

        elif action.action_type == ActionType.ESCALATE:
            reward += self._handle_escalate(action)
            done = True  # escalation ends agent's direct handling

        elif action.action_type == ActionType.REFUND:
            reward += self._handle_refund(action)

        elif action.action_type == ActionType.CLOSE:
            reward += self._handle_close(action)
            done = True

        elif action.action_type == ActionType.TRANSFER:
            reward += self._handle_transfer(action)
            done = True

        # ── Add agent turn to history ────────────────────────────────────────
        agent_text = (
            action.response_text or
            action.escalation_reason or
            f"[{action.action_type}]"
        )
        s.conversation_history.append(
            ConversationTurn(role="agent", content=agent_text, turn=s.turn_number)
        )
        s.last_response_text = agent_text

        # ── Max-turn termination ─────────────────────────────────────────────
        if s.turn_number >= max_turns and not done:
            reward -= 0.10  # penalty for not resolving in time
            done = True
            info["terminated"] = "max_turns_reached"

        # ── Accumulate reward ────────────────────────────────────────────────
        reward = round(max(-1.0, min(1.0, reward)), 4)
        s.cumulative_reward = round(s.cumulative_reward + reward, 4)

        # Next user message (simulated — last user message replayed for continuity)
        next_user_msg = self._simulate_user_followup(action, done)
        if next_user_msg and not done:
            s.conversation_history.append(
                ConversationTurn(role="user", content=next_user_msg, turn=s.turn_number + 1)
            )

        info["cumulative_reward"] = s.cumulative_reward
        return self._build_observation(next_user_msg or ""), reward, done, info

    def state(self) -> EnvironmentState:
        """Return the full internal environment state."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def grade(self) -> Tuple[float, Dict[str, Any]]:
        """Run the task grader on final state. Returns (score, breakdown)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        grader = GRADERS[self.task_id]
        return grader(self._state)

    # ── Action Handlers ──────────────────────────────────────────────────────

    def _handle_respond(self, action: Action) -> float:
        reward = 0.0
        text = action.response_text or ""

        # Loop detection
        if (
            self._state.last_response_text and
            text.strip().lower() == self._state.last_response_text.strip().lower()
        ):
            self._state.repeated_response_count += 1
            return R_LOOP_PENALTY

        # Relevance check (very short = likely irrelevant)
        if len(text.strip()) < 30:
            return R_IRRELEVANT

        reward += R_HELPFUL_RESPONSE

        # Empathy bonus
        text_lower = text.lower()
        if any(kw in text_lower for kw in EMPATHY_KEYWORDS):
            reward += R_EMPATHY

        return reward

    def _handle_escalate(self, action: Action) -> float:
        s = self._state
        meta = s.metadata

        if meta.requires_escalation:
            # Check if escalating to correct department
            reason = (action.escalation_reason or "").lower()
            dept = meta.escalation_department or ""
            if dept.lower().split()[0] in reason or dept.lower() in reason:
                s.escalated = True
                s.metadata.escalation_department = dept
                s.ticket_status = TicketStatus.ESCALATED
                return R_CORRECT_ESCALATION
            else:
                # Escalating but to wrong dept / no reason
                s.escalated = True
                s.ticket_status = TicketStatus.ESCALATED
                return R_CORRECT_ESCALATION * 0.5  # partial credit
        else:
            # Escalating when not needed
            s.ticket_status = TicketStatus.ESCALATED
            return R_WRONG_ESCALATION

    def _handle_refund(self, action: Action) -> float:
        s = self._state
        meta = s.metadata
        amount = action.refund_amount or 0.0

        if not meta.refund_eligible:
            return -0.10  # not eligible

        # Fraud task: check identity was verified first
        if s.metadata.issue_category == "fraud":
            verified = self._identity_verified()
            if not verified:
                return R_WRONG_REFUND  # severe penalty

        if amount <= 0:
            return -0.05

        s.refund_issued += amount

        # Score based on proximity to expected refund
        if amount <= meta.max_refund + 2:
            closeness = 1.0 - abs(amount - meta.max_refund) / max(meta.max_refund, 1)
            return R_CORRECT_REFUND * max(0.3, closeness)
        else:
            # Over-refunding (wasteful)
            return R_CORRECT_REFUND * 0.3

    def _handle_close(self, action: Action) -> float:
        s = self._state
        meta = s.metadata

        # Proper resolution requires escalation if needed
        if meta.requires_escalation and not s.escalated:
            s.ticket_status = TicketStatus.CLOSED
            return R_PREMATURE_CLOSE

        # Proper resolution requires refund if eligible and none issued
        if meta.refund_eligible and s.refund_issued == 0:
            s.ticket_status = TicketStatus.CLOSED
            return R_PREMATURE_CLOSE * 0.5

        s.ticket_status = TicketStatus.RESOLVED
        s.resolved = True
        return R_RESOLUTION

    def _handle_transfer(self, action: Action) -> float:
        s = self._state
        meta = s.metadata
        dept = (action.transfer_department or "").lower()

        if meta.requires_escalation and dept:
            s.transferred = True
            s.ticket_status = TicketStatus.ESCALATED
            # Check if correct department
            expected = (meta.escalation_department or "").lower()
            if expected.split()[0] in dept:
                return R_CORRECT_ESCALATION
            return R_CORRECT_ESCALATION * 0.4
        return -0.05

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_observation(self, latest_user_msg: str) -> Observation:
        s = self._state
        return Observation(
            ticket_id=s.ticket_id,
            user_message=latest_user_msg,
            conversation_history=list(s.conversation_history),
            ticket_status=s.ticket_status,
            turn_number=s.turn_number,
            metadata={
                "customer_name": s.metadata.customer_name,
                "account_id": s.metadata.account_id,
                "plan": s.metadata.plan,
                "prior_contacts": s.metadata.prior_contacts,
                "issue_category": s.metadata.issue_category,
            },
        )

    def _identity_verified(self) -> bool:
        """Check if agent verified identity in conversation."""
        verify_keywords = [
            "verify", "confirm your identity", "date of birth", "dob",
            "phone number", "last 4", "security question", "full name"
        ]
        agent_turns = [
            t.content.lower()
            for t in self._state.conversation_history
            if t.role == "agent"
        ]
        combined = " ".join(agent_turns)
        return any(kw in combined for kw in verify_keywords)

    def _simulate_user_followup(self, action: Action, done: bool) -> Optional[str]:
        """Generate a contextual follow-up user message for multi-turn realism."""
        if done:
            return None
        category = self._state.metadata.issue_category
        turn = self._state.turn_number

        if action.action_type == ActionType.RESPOND:
            followups = {
                "billing": [
                    "Okay, when will the refund appear on my statement?",
                    "Thanks for looking into it. Is my plan still correct?",
                    "How do I make sure this doesn't happen again?",
                ],
                "technical": [
                    "I checked the console — it shows a 403 error.",
                    "Our account was created in March 2023.",
                    "The broken URL is app.example.com/dashboard",
                    "Still not working after trying your suggestions.",
                ],
                "fraud": [
                    "My date of birth is March 15, 1988 and last 4 is 4421.",
                    "Please just freeze the account immediately.",
                    "I need those charges reversed as soon as possible.",
                    "What's the status of the investigation?",
                ],
            }
            options = followups.get(category, ["Can you help me further?"])
            return options[min(turn - 1, len(options) - 1)]

        elif action.action_type == ActionType.REFUND:
            return "Thank you. How long does the refund take to process?"

        return None

    # ── String repr ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = self._state.ticket_status if self._state else "uninitialized"
        return f"<CustomerSupportEnv task={self.task_id} status={status}>"
