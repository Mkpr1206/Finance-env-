"""
server/models.py — Typed Action/Observation for PersonalFinanceEnv.
Subclasses openenv_core Action/Observation as required by the spec.
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import Field

try:
    from openenv_core.env_server.types import Action, Observation
except ImportError:
    from openenv_core.env_server import Action, Observation


class FinanceAction(Action):
    """One agent action in the Personal Finance environment."""
    action_type: str = Field(..., description="categorize|approve|reject|allocate|invest|pay_debt")
    transaction_id: Optional[str] = Field(None, description="Target transaction ID")
    category: Optional[str] = Field(None, description="Expense category for categorize action")
    from_bucket: Optional[str] = Field(None, description="Source budget bucket for allocate")
    to_bucket:   Optional[str] = Field(None, description="Target budget bucket for allocate")
    amount:      Optional[float] = Field(None, description="Dollar amount for invest/pay_debt/allocate")
    rationale:   str = Field("", description="Agent's reasoning (improves reward)")


class FinanceObservation(Observation):
    """Agent's view of the current financial state."""
    day: int = Field(1, description="Current simulation day (1-30)")
    monthly_income: float = Field(3800.0)
    cash_balance: float = Field(0.0)
    savings_balance: float = Field(0.0)
    investment_balance: float = Field(0.0)
    debt_balance: float = Field(0.0)
    savings_rate: float = Field(0.0)
    pending_count: int = Field(0, description="Number of pending transactions")
    uncategorized_count: int = Field(0, description="Number of uncategorized transactions")
    reward: float = Field(0.0, description="Reward from last action")
    done: bool = Field(False, description="Whether episode is complete")
    message: str = Field("", description="Human-readable status message")
