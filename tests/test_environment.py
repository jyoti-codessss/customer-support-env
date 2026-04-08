"""
tests/test_environment.py — Test suite for CustomerSupportEnv.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from env.environment import CustomerSupportEnv
from env.models import Action, ActionType
from tasks.task_definitions import TASKS
from graders.task_graders import GRADERS


class TestEnvironmentInterface:
    """Test OpenEnv interface compliance."""

    def test_all_tasks_available(self):
        assert len(TASKS) == 3
        assert "billing_dispute_easy" in TASKS
        assert "technical_outage_medium" in TASKS
        assert "fraud_complaint_hard" in TASKS

    def test_all_graders_available(self):
        assert len(GRADERS) == 3
        for task_id in TASKS:
            assert task_id in GRADERS

    def test_reset_returns_observation(self):
        env = CustomerSupportEnv("billing_dispute_easy")
        obs = env.reset()
        assert obs.ticket_id == "TKT-001"
        assert len(obs.user_message) > 0
        assert obs.turn_number == 0
        assert len(obs.conversation_history) == 1

    def test_step_returns_tuple(self):
        env = CustomerSupportEnv("billing_dispute_easy")
        env.reset()
        action = Action(
            action_type=ActionType.RESPOND,
            response_text="I'm sorry to hear about the overcharge, Sarah. Let me look into this right away."
        )
        result = env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_environment_state(self):
        env = CustomerSupportEnv("billing_dispute_easy")
        env.reset()
        state = env.state()
        assert state.ticket_id == "TKT-001"
        assert state.turn_number == 0

    def test_reward_range(self):
        """Reward must be in [-1.0, 1.0] at each step."""
        for task_id in TASKS:
            env = CustomerSupportEnv(task_id)
            env.reset()
            for _ in range(3):
                action = Action(
                    action_type=ActionType.RESPOND,
                    response_text="Thank you for contacting us. I understand your frustration."
                )
                _, reward, done, _ = env.step(action)
                assert -1.0 <= reward <= 1.0, f"Reward {reward} out of range"
                if done:
                    break


class TestBillingTask:
    """Test the easy billing dispute task."""

    def test_correct_resolution(self):
        env = CustomerSupportEnv("billing_dispute_easy")
        env.reset()

        # Empathetic response
        env.step(Action(
            action_type=ActionType.RESPOND,
            response_text="I'm so sorry about this overcharge, Sarah. I can see this was a billing error and will fix it immediately."
        ))
        # Issue correct refund
        env.step(Action(
            action_type=ActionType.REFUND,
            refund_amount=20.00,
            response_text="I've issued a $20 refund. Your Basic plan at $29.99 remains active."
        ))
        # Close ticket
        env.step(Action(
            action_type=ActionType.CLOSE,
            response_text="Your issue has been resolved. The refund will appear in 3-5 business days. Anything else?"
        ))

        score, breakdown = env.grade()
        assert score >= 0.70, f"Expected ≥0.70, got {score}. Breakdown: {breakdown}"
        assert breakdown["refund_correct"] is True
        assert breakdown["ticket_closed"] is True

    def test_no_refund_penalized(self):
        env = CustomerSupportEnv("billing_dispute_easy")
        env.reset()
        # Just close without refunding
        env.step(Action(
            action_type=ActionType.CLOSE,
            response_text="I've noted your concern. Goodbye."
        ))
        score, breakdown = env.grade()
        assert score < 0.50, f"Expected <0.50 without refund, got {score}"

    def test_loop_penalty(self):
        env = CustomerSupportEnv("billing_dispute_easy")
        env.reset()
        identical = Action(
            action_type=ActionType.RESPOND,
            response_text="Please wait while I check your account."
        )
        env.step(identical)
        _, reward2, _, _ = env.step(identical)  # duplicate
        assert reward2 < 0, "Repeated responses should yield negative reward"


class TestTechnicalTask:
    """Test the medium technical outage task."""

    def test_escalation_required(self):
        env = CustomerSupportEnv("technical_outage_medium")
        env.reset()

        env.step(Action(
            action_type=ActionType.RESPOND,
            response_text="I understand your frustration. Can you check the browser console for errors and tell me the URL?"
        ))
        env.step(Action(
            action_type=ActionType.ESCALATE,
            escalation_reason="Engineering team needs to investigate this 403 error affecting Business accounts. Diagnostic: console shows 403, account created 2023.",
            response_text="I'm escalating to our Engineering team who can resolve this permanently."
        ))

        score, breakdown = env.grade()
        assert breakdown["escalated_to_engineering"] is True
        assert score >= 0.55

    def test_premature_close_penalized(self):
        env = CustomerSupportEnv("technical_outage_medium")
        env.reset()
        env.step(Action(
            action_type=ActionType.CLOSE,
            response_text="Please try clearing your cache. Issue closed."
        ))
        score, breakdown = env.grade()
        assert breakdown["no_premature_close"] is False
        assert score < 0.40


class TestFraudTask:
    """Test the hard fraud complaint task."""

    def test_verify_before_refund(self):
        env = CustomerSupportEnv("fraud_complaint_hard")
        env.reset()

        # Verify identity first
        env.step(Action(
            action_type=ActionType.RESPOND,
            response_text="I'm so sorry this happened. For your security, I need to verify your identity. Could you please confirm your date of birth and the last 4 digits of your card?"
        ))
        # Issue full refund
        env.step(Action(
            action_type=ActionType.REFUND,
            refund_amount=448.00,
            response_text="I've initiated a full $448 refund for all fraudulent charges."
        ))
        # Escalate + account lock
        env.step(Action(
            action_type=ActionType.ESCALATE,
            escalation_reason="Fraud & Security team - account compromise, unauthorized charges, email change",
            response_text="I'm locking your account and escalating to our Fraud & Security team. You'll receive an email to recover access."
        ))

        score, breakdown = env.grade()
        assert breakdown["identity_verified_first"] is True
        assert breakdown["full_refund_issued"] is True
        assert breakdown["escalated_to_fraud"] is True
        assert score >= 0.75

    def test_refund_without_verification_penalized(self):
        env = CustomerSupportEnv("fraud_complaint_hard")
        env.reset()
        # Skip verification, go straight to refund
        _, reward, _, _ = env.step(Action(
            action_type=ActionType.REFUND,
            refund_amount=448.00,
            response_text="I'll refund you right away."
        ))
        assert reward < 0, "Refund without verification should be penalized"

    def test_max_turns_termination(self):
        env = CustomerSupportEnv("fraud_complaint_hard")
        env.reset()
        done = False
        turns = 0
        while not done:
            _, _, done, _ = env.step(Action(
                action_type=ActionType.RESPOND,
                response_text="I understand your concern and am investigating."
            ))
            turns += 1
            if turns > 15:
                break
        assert done, "Episode should terminate after max turns"


class TestGraders:
    """Test grader determinism and score boundaries."""

    def test_perfect_score_billing(self):
        env = CustomerSupportEnv("billing_dispute_easy")
        env.reset()
        env.step(Action(action_type=ActionType.RESPOND,
            response_text="I'm so sorry for the overcharge, Sarah. I apologize for this inconvenience. I can see you're on the Basic plan."))
        env.step(Action(action_type=ActionType.REFUND, refund_amount=20.00,
            response_text="I've issued a $20 refund. Your Basic plan at $29.99 is unchanged."))
        env.step(Action(action_type=ActionType.CLOSE,
            response_text="Your billing issue is now resolved. Thank you!"))
        score, _ = env.grade()
        assert score >= 0.85

    def test_score_range(self):
        for task_id in TASKS:
            env = CustomerSupportEnv(task_id)
            env.reset()
            env.step(Action(action_type=ActionType.CLOSE,
                response_text="Closing ticket."))
            score, _ = env.grade()
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] for {task_id}"

    def test_grader_deterministic(self):
        """Same trajectory must produce same score."""
        def run():
            env = CustomerSupportEnv("billing_dispute_easy")
            env.reset()
            env.step(Action(action_type=ActionType.RESPOND,
                response_text="I apologize for this error on your account."))
            env.step(Action(action_type=ActionType.REFUND, refund_amount=20.00,
                response_text="Refund issued for the Basic plan difference."))
            env.step(Action(action_type=ActionType.CLOSE,
                response_text="Your issue is resolved. Thank you!"))
            return env.grade()[0]

        assert run() == run(), "Grader must be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
