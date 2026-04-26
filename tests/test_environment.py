"""
Tests for the CustomerSupportEnv environment.
"""
import pytest
from env.environment import CustomerSupportEnv
from env.models import Action, ActionType
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

class TestFraudComplaintTask:
    def test_task_metadata(self):
        task = TASKS["fraud_complaint_hard"]
        assert task["metadata"].issue_category == "fraud"
        assert task["metadata"].max_refund == 448.00

class TestVIPAccountRecoveryTask:
    def test_task_metadata(self):
        task = TASKS["vip_account_recovery_expert"]
        assert task["metadata"].issue_category == "account_recovery"
        assert task["metadata"].requires_escalation is True
