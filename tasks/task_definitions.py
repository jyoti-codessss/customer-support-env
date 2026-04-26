"""
Tests for the CustomerSupportEnv environment.
Covers all 5 tasks, graders, and environment lifecycle.
"""

import pytest
from env.environment import CustomerSupportEnv
from env.models import AgentAction, ActionType
from tasks.task_definitions import TASKS
from graders.task_graders import GRADERS


class TestTaskDefinitions:
    def test_all_tasks_available(self):
        assert len(TASKS) == 5

    def test_task_ids_correct(self):
        assert "billing_dispute_easy" in TASKS
        assert "technical_outage_medium" in TASKS
        assert "fraud_complaint_hard" in TASKS
        assert "subscription_cancellation_hard" in TASKS
        assert "vip_account_recovery_expert" in TASKS

    def test_all_tasks_have_required_fields(self):
        for task_id, task in TASKS.items():
            assert "ticket_id" in task, f"{task_id} missing ticket_id"
            assert "initial_user_message" in task, f"{task_id} missing initial_user_message"
            assert "metadata" in task, f"{task_id} missing metadata"
            assert "max_turns" in task, f"{task_id} missing max_turns"


class TestGraders:
    def test_all_graders_available(self):
        assert len(GRADERS) == 5

    def test_all_five_tasks_gradeable(self):
        for task_id in TASKS:
            assert task_id in GRADERS, f"No grader found for {task_id}"


class TestBillingDisputeTask:
    def test_task_metadata(self):
        task = TASKS["billing_dispute_easy"]
        assert task["metadata"].issue_category == "billing"
        assert task["metadata"].refund_eligible is True
        assert task["metadata"].max_refund == 20.00
        assert task["metadata"].requires_escalation is False


class TestTechnicalOutageTask:
    def test_task_metadata(self):
        task = TASKS["technical_outage_medium"]
        assert task["metadata"].issue_category == "technical"
        assert task["metadata"].requires_escalation is True
        assert task["metadata"].escalation_department == "Engineering"


class TestFraudComplaintTask:
    def test_task_metadata(self):
        task = TASKS["fraud_complaint_hard"]
        assert task["metadata"].issue_category == "fraud"
        assert task["metadata"].max_refund == 448.00
        assert task["metadata"].requires_escalation is True
        assert task["metadata"].escalation_department == "Fraud & Security"


class TestSubscriptionCancellationTask:
    def test_task_metadata(self):
        task = TASKS["subscription_cancellation_hard"]
        assert task["metadata"].issue_category == "cancellation"
        assert task["metadata"].max_refund == 499.00
        assert task["metadata"].requires_escalation is False


class TestVIPAccountRecoveryTask:
    def test_task_metadata(self):
        task = TASKS["vip_account_recovery_expert"]
        assert task["metadata"].issue_category == "account_recovery"
        assert task["metadata"].max_refund == 2499.00
        assert task["metadata"].requires_escalation is True
        assert task["metadata"].escalation_department == "VIP Support"


class TestEnvironmentLifecycle:
    def test_environment_reset(self):
        env = CustomerSupportEnv()
        obs = env.reset("billing_dispute_easy")
        assert obs is not None
        assert env.current_task_id == "billing_dispute_easy"

    def test_environment_step(self):
        env = CustomerSupportEnv()
        env.reset("billing_dispute_easy")
        action = AgentAction(
            action_type=ActionType.RESPOND,
            response_text="I can help you with that billing issue."
        )
        obs, reward, done, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)