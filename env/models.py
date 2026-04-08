"""
Typed Pydantic models for the Customer Support Agent OpenEnv environment.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Action Space ────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    RESPOND   = "respond"
    ESCALATE  = "escalate"
    REFUND    = "refund"
    CLOSE     = "close"
    TRANSFER  = "transfer"


class Action(BaseModel):
    """Agent action in the customer support environment."""
    action_type: ActionType
    response_text: Optional[str] = Field(
        default=None,
        description="Text sent to the user (required for respond/escalate/close)"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for escalation (required for escalate)"
    )
    refund_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Dollar amount to refund (required for refund)"
    )
    transfer_department: Optional[str] = Field(
        default=None,
        description="Department to transfer to (required for transfer)"
    )

    class Config:
        use_enum_values = True


# ── Observation Space ────────────────────────────────────────────────────────

class TicketStatus(str, Enum):
    OPEN       = "open"
    PENDING    = "pending"
    ESCALATED  = "escalated"
    RESOLVED   = "resolved"
    CLOSED     = "closed"


class ConversationTurn(BaseModel):
    role: str           # "user" | "agent"
    content: str
    turn: int


class TicketMetadata(BaseModel):
    customer_name: str
    account_id: str
    plan: str
    prior_contacts: int = 0
    issue_category: str
    expected_resolution: str   # used by grader, not shown to agent
    refund_eligible: bool = False
    max_refund: float = 0.0
    requires_escalation: bool = False
    escalation_department: Optional[str] = None


class Observation(BaseModel):
    """What the agent sees at each step."""
    ticket_id: str
    user_message: str
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    ticket_status: TicketStatus = TicketStatus.OPEN
    turn_number: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Reward / Step Return ─────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Return value of environment.step()."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ── Environment State ────────────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """Full internal state (superset of observation)."""
    ticket_id: str
    ticket_status: TicketStatus
    turn_number: int
    conversation_history: List[ConversationTurn]
    metadata: TicketMetadata
    cumulative_reward: float = 0.0
    refund_issued: float = 0.0
    escalated: bool = False
    transferred: bool = False
    resolved: bool = False
    repeated_response_count: int = 0
    last_response_text: Optional[str] = None
