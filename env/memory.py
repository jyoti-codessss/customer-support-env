"""
env/memory.py — Conversation Memory System for CustomerSupportEnv.

Stores customer interactions by account_id using simple JSON file storage.
Enables agents to recall previous issues, resolutions, and sentiment history
for smarter, personalized responses.

Features:
  - Sliding window: max 10 memories per customer
  - Sentiment tracking: frustrated, neutral, satisfied
  - Repeat issue detection
  - Works on HuggingFace Spaces Docker deployment (uses /tmp for persistence)
"""

from __future__ import annotations

import json
import os
import time
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


# ── Configuration ────────────────────────────────────────────────────────────

MAX_MEMORIES_PER_CUSTOMER = 10
MEMORY_FILE = os.getenv("MEMORY_STORE_PATH", "/tmp/memory_store.json")

# Windows fallback for local dev
if os.name == "nt" and MEMORY_FILE.startswith("/tmp"):
    MEMORY_FILE = os.path.join(os.environ.get("TEMP", "."), "memory_store.json")


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Interaction:
    """A single stored customer interaction."""
    timestamp: str
    task_id: str
    issue_category: str
    action_types_used: List[str]
    resolution: str           # "resolved", "escalated", "closed", "unresolved"
    score: float              # grading score 0.0–1.0
    sentiment: str            # "frustrated", "neutral", "satisfied"
    summary: str              # brief human-readable summary
    refund_amount: float = 0.0
    escalation_department: Optional[str] = None


@dataclass
class CustomerProfile:
    """Aggregated profile for a customer."""
    account_id: str
    customer_name: str = ""
    plan: str = ""
    total_contacts: int = 0
    interactions: List[Interaction] = field(default_factory=list)
    avg_sentiment_score: float = 0.5   # 0=frustrated, 0.5=neutral, 1=satisfied
    is_repeat_customer: bool = False
    repeat_issue_categories: List[str] = field(default_factory=list)
    last_contact_timestamp: Optional[str] = None


# ── Sentiment Analysis ───────────────────────────────────────────────────────

SENTIMENT_MAP = {
    "frustrated": 0.0,
    "angry": 0.0,
    "upset": 0.1,
    "concerned": 0.3,
    "neutral": 0.5,
    "satisfied": 0.8,
    "happy": 1.0,
    "delighted": 1.0,
}

FRUSTRATED_KEYWORDS = [
    "angry", "furious", "unacceptable", "worst", "terrible", "horrible",
    "ridiculous", "disgusting", "never again", "cancel", "lawsuit",
    "complaint", "fed up", "sick of", "waste of time",
]

SATISFIED_KEYWORDS = [
    "thank you", "thanks", "great", "excellent", "wonderful",
    "appreciate", "helpful", "resolved", "fixed", "perfect",
    "happy", "satisfied", "good job",
]


def detect_sentiment(text: str) -> str:
    """Simple keyword-based sentiment detection."""
    text_lower = text.lower()
    frustrated_count = sum(1 for kw in FRUSTRATED_KEYWORDS if kw in text_lower)
    satisfied_count = sum(1 for kw in SATISFIED_KEYWORDS if kw in text_lower)

    if frustrated_count > satisfied_count:
        return "frustrated"
    if satisfied_count > frustrated_count:
        return "satisfied"
    return "neutral"


def detect_resolution(action_types: List[str], escalated: bool, resolved: bool) -> str:
    """Determine how an interaction was resolved."""
    if escalated:
        return "escalated"
    if resolved:
        return "resolved"
    if "close" in action_types:
        return "closed"
    return "unresolved"


# ── Conversation Memory ──────────────────────────────────────────────────────

class ConversationMemory:
    """
    Thread-safe conversation memory backed by a JSON file.

    Usage:
        memory = ConversationMemory()
        memory.remember("ACC-88421", interaction)
        profile = memory.recall("ACC-88421")
    """

    def __init__(self, store_path: Optional[str] = None):
        self._store_path = store_path or MEMORY_FILE
        self._lock = threading.Lock()
        self._cache: Dict[str, CustomerProfile] = {}
        self._loaded = False

    # ── Public API ───────────────────────────────────────────────────────────

    def remember(self, account_id: str, interaction: Interaction,
                 customer_name: str = "", plan: str = "") -> None:
        """Store a new interaction for a customer."""
        with self._lock:
            self._ensure_loaded()

            if account_id not in self._cache:
                self._cache[account_id] = CustomerProfile(
                    account_id=account_id,
                    customer_name=customer_name,
                    plan=plan,
                )

            profile = self._cache[account_id]
            if customer_name:
                profile.customer_name = customer_name
            if plan:
                profile.plan = plan

            profile.interactions.append(interaction)
            profile.total_contacts += 1
            profile.last_contact_timestamp = interaction.timestamp

            # Sliding window — keep only last N memories
            if len(profile.interactions) > MAX_MEMORIES_PER_CUSTOMER:
                profile.interactions = profile.interactions[-MAX_MEMORIES_PER_CUSTOMER:]

            # Update aggregate stats
            profile.is_repeat_customer = profile.total_contacts > 1
            all_categories = [i.issue_category for i in profile.interactions]
            profile.repeat_issue_categories = [
                cat for cat in set(all_categories)
                if all_categories.count(cat) > 1
            ]
            sentiments = [SENTIMENT_MAP.get(i.sentiment, 0.5) for i in profile.interactions]
            profile.avg_sentiment_score = round(
                sum(sentiments) / max(len(sentiments), 1), 2
            )

            self._save()

    def recall(self, account_id: str) -> Optional[CustomerProfile]:
        """Retrieve a customer's stored profile and history."""
        with self._lock:
            self._ensure_loaded()
            return self._cache.get(account_id)

    def recall_context(self, account_id: str) -> str:
        """
        Build a human-readable context string for the agent.
        Returns empty string if no history exists.
        """
        profile = self.recall(account_id)
        if not profile or not profile.interactions:
            return ""

        lines = [f"CUSTOMER MEMORY — {profile.customer_name} ({account_id}):"]
        lines.append(f"  Total contacts: {profile.total_contacts}")
        lines.append(f"  Avg sentiment: {self._sentiment_label(profile.avg_sentiment_score)}")

        if profile.repeat_issue_categories:
            lines.append(f"  ⚠ Repeat issues: {', '.join(profile.repeat_issue_categories)}")

        # Show last 3 interactions
        recent = profile.interactions[-3:]
        for i, interaction in enumerate(recent):
            ago = self._time_ago(interaction.timestamp)
            lines.append(
                f"  [{ago}] {interaction.issue_category}: "
                f"{interaction.summary} → {interaction.resolution} "
                f"(sentiment: {interaction.sentiment})"
            )
            if interaction.refund_amount > 0:
                lines.append(f"    Refund: ${interaction.refund_amount:.2f}")

        if profile.is_repeat_customer:
            lines.append(f"  🔄 RETURNING CUSTOMER — personalize greeting")
        if profile.avg_sentiment_score < 0.3:
            lines.append(f"  ⚠ PREVIOUSLY FRUSTRATED — proactive empathy needed")

        return "\n".join(lines)

    def has_repeat_issue(self, account_id: str, issue_category: str) -> bool:
        """Check if this customer has had the same issue before."""
        profile = self.recall(account_id)
        if not profile:
            return False
        return any(
            i.issue_category == issue_category
            for i in profile.interactions
        )

    def is_returning(self, account_id: str) -> bool:
        """Check if customer has prior interactions."""
        profile = self.recall(account_id)
        return profile is not None and profile.total_contacts > 0

    def forget(self, account_id: str) -> bool:
        """Delete all memory for a customer. Returns True if found."""
        with self._lock:
            self._ensure_loaded()
            if account_id in self._cache:
                del self._cache[account_id]
                self._save()
                return True
            return False

    def stats(self) -> Dict[str, Any]:
        """Return aggregate memory statistics."""
        with self._lock:
            self._ensure_loaded()
            total_customers = len(self._cache)
            total_interactions = sum(
                p.total_contacts for p in self._cache.values()
            )
            repeat_customers = sum(
                1 for p in self._cache.values() if p.is_repeat_customer
            )
            avg_sentiment = 0.0
            if total_customers > 0:
                avg_sentiment = round(
                    sum(p.avg_sentiment_score for p in self._cache.values()) / total_customers,
                    2,
                )
            return {
                "total_customers_remembered": total_customers,
                "total_interactions_stored": total_interactions,
                "repeat_customers": repeat_customers,
                "avg_sentiment_score": avg_sentiment,
                "avg_sentiment_label": self._sentiment_label(avg_sentiment),
                "memory_file": self._store_path,
                "memory_file_exists": os.path.exists(self._store_path),
            }

    def all_profiles(self) -> Dict[str, Dict]:
        """Return all customer profiles as serializable dicts."""
        with self._lock:
            self._ensure_loaded()
            return {
                account_id: self._profile_to_dict(profile)
                for account_id, profile in self._cache.items()
            }

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load from disk if not yet cached."""
        if self._loaded:
            return
        try:
            if os.path.exists(self._store_path):
                with open(self._store_path, "r") as f:
                    raw = json.load(f)
                for account_id, data in raw.items():
                    interactions = [
                        Interaction(**ix) for ix in data.get("interactions", [])
                    ]
                    self._cache[account_id] = CustomerProfile(
                        account_id=account_id,
                        customer_name=data.get("customer_name", ""),
                        plan=data.get("plan", ""),
                        total_contacts=data.get("total_contacts", 0),
                        interactions=interactions,
                        avg_sentiment_score=data.get("avg_sentiment_score", 0.5),
                        is_repeat_customer=data.get("is_repeat_customer", False),
                        repeat_issue_categories=data.get("repeat_issue_categories", []),
                        last_contact_timestamp=data.get("last_contact_timestamp"),
                    )
        except (json.JSONDecodeError, Exception):
            self._cache = {}
        self._loaded = True

    def _save(self) -> None:
        """Persist cache to disk."""
        try:
            os.makedirs(os.path.dirname(self._store_path) or ".", exist_ok=True)
            data = {
                account_id: self._profile_to_dict(profile)
                for account_id, profile in self._cache.items()
            }
            with open(self._store_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # silently fail on write errors

    def _profile_to_dict(self, profile: CustomerProfile) -> Dict:
        return {
            "account_id": profile.account_id,
            "customer_name": profile.customer_name,
            "plan": profile.plan,
            "total_contacts": profile.total_contacts,
            "interactions": [asdict(ix) for ix in profile.interactions],
            "avg_sentiment_score": profile.avg_sentiment_score,
            "is_repeat_customer": profile.is_repeat_customer,
            "repeat_issue_categories": profile.repeat_issue_categories,
            "last_contact_timestamp": profile.last_contact_timestamp,
        }

    @staticmethod
    def _sentiment_label(score: float) -> str:
        if score >= 0.7:
            return "satisfied"
        if score >= 0.4:
            return "neutral"
        return "frustrated"

    @staticmethod
    def _time_ago(timestamp: str) -> str:
        """Convert ISO timestamp to a human-readable 'X ago' string."""
        try:
            then = time.mktime(time.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ"))
            diff = time.time() - then
            if diff < 60:
                return "just now"
            if diff < 3600:
                return f"{int(diff / 60)}m ago"
            if diff < 86400:
                return f"{int(diff / 3600)}h ago"
            return f"{int(diff / 86400)}d ago"
        except Exception:
            return "recently"


# ── Singleton Instance ───────────────────────────────────────────────────────

_memory_instance: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """Get or create the global ConversationMemory singleton."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance
