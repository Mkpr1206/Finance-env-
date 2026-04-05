"""
PersonalFinanceEnv — OpenEnv Environment
Real-world task: An AI agent manages a monthly budget, categorizes expenses,
and makes decisions to maximize savings rate over a 30-day simulation.
"""

from __future__ import annotations
import random
from typing import Any
from pydantic import BaseModel, Field
from enum import Enum


# ══════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════

class ExpenseCategory(str, Enum):
    HOUSING       = "housing"
    FOOD          = "food"
    TRANSPORT     = "transport"
    UTILITIES     = "utilities"
    HEALTHCARE    = "healthcare"
    ENTERTAINMENT = "entertainment"
    SHOPPING      = "shopping"
    SAVINGS       = "savings"
    INVESTMENT    = "investment"
    DEBT          = "debt"
    OTHER         = "other"


class ActionType(str, Enum):
    CATEGORIZE = "categorize"
    APPROVE    = "approve"
    REJECT     = "reject"
    ALLOCATE   = "allocate"
    INVEST     = "invest"
    PAY_DEBT   = "pay_debt"


# ══════════════════════════════════════════════════════════════
# Typed Models (OpenEnv spec)
# ══════════════════════════════════════════════════════════════

class Transaction(BaseModel):
    id: str
    description: str
    amount: float
    day: int
    category: ExpenseCategory | None = None
    pending: bool = False
    essential: bool = False


class BudgetBucket(BaseModel):
    name: str
    allocated: float
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        return self.allocated - self.spent

    @property
    def utilization(self) -> float:
        return self.spent / self.allocated if self.allocated > 0 else 0.0


class Action(BaseModel):
    action_type: ActionType
    transaction_id: str | None = None
    category: ExpenseCategory | None = None
    from_bucket: str | None = None
    to_bucket: str | None = None
    amount: float | None = None
    rationale: str = ""


class Observation(BaseModel):
    day: int
    monthly_income: float
    cash_balance: float
    savings_balance: float
    investment_balance: float
    debt_balance: float
    budget_buckets: dict[str, BudgetBucket]
    pending_transactions: list[Transaction]
    recent_transactions: list[Transaction]
    uncategorized_transactions: list[Transaction]
    savings_rate: float
    month_complete: bool = False


class Reward(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    breakdown: dict[str, float] = Field(default_factory=dict)
    reason: str = ""


# ══════════════════════════════════════════════════════════════
# Static transaction data
# ══════════════════════════════════════════════════════════════

_TRANSACTION_TEMPLATES = [
    dict(id="t01", description="Rent payment",          amount=1200.0, day=1,  category="housing",    essential=True,  pending=False),
    dict(id="t02", description="Electricity bill",      amount=85.0,   day=3,  category=None,         essential=True,  pending=False),
    dict(id="t03", description="Grocery store",         amount=140.0,  day=5,  category=None,         essential=True,  pending=False),
    dict(id="t04", description="Internet bill",         amount=60.0,   day=7,  category=None,         essential=True,  pending=False),
    dict(id="t05", description="Doctor visit copay",    amount=30.0,   day=9,  category="healthcare", essential=True,  pending=False),
    dict(id="t06", description="Grocery store",         amount=95.0,   day=14, category=None,         essential=True,  pending=False),
    dict(id="t07", description="Bus pass",              amount=55.0,   day=2,  category="transport",  essential=True,  pending=False),
    dict(id="t08", description="Car insurance",         amount=120.0,  day=1,  category="transport",  essential=True,  pending=False),
    dict(id="t09", description="Netflix subscription",  amount=18.0,   day=6,  category=None,         essential=False, pending=True),
    dict(id="t10", description="Restaurant dinner",     amount=75.0,   day=8,  category=None,         essential=False, pending=True),
    dict(id="t11", description="Amazon purchase",       amount=120.0,  day=11, category=None,         essential=False, pending=True),
    dict(id="t12", description="Gym membership",        amount=45.0,   day=5,  category=None,         essential=False, pending=True),
    dict(id="t13", description="Coffee shop x12",       amount=60.0,   day=15, category=None,         essential=False, pending=False),
    dict(id="t14", description="Online clothing store", amount=185.0,  day=18, category=None,         essential=False, pending=True),
    dict(id="t15", description="Concert tickets",       amount=150.0,  day=20, category=None,         essential=False, pending=True),
    dict(id="t16", description="Pharmacy",              amount=35.0,   day=12, category=None,         essential=True,  pending=False),
    dict(id="t17", description="Credit card minimum",   amount=80.0,   day=15, category="debt",       essential=True,  pending=False),
    dict(id="t18", description="Spotify + Apple Music", amount=25.0,   day=6,  category=None,         essential=False, pending=True),
    dict(id="t19", description="Gas station",           amount=55.0,   day=10, category="transport",  essential=True,  pending=False),
    dict(id="t20", description="Bar tab",               amount=90.0,   day=22, category=None,         essential=False, pending=True),
]


def _fresh_buckets() -> dict[str, BudgetBucket]:
    return {
        "housing":       BudgetBucket(name="housing",       allocated=1300.0),
        "food":          BudgetBucket(name="food",           allocated=400.0),
        "transport":     BudgetBucket(name="transport",      allocated=280.0),
        "utilities":     BudgetBucket(name="utilities",      allocated=180.0),
        "healthcare":    BudgetBucket(name="healthcare",     allocated=100.0),
        "entertainment": BudgetBucket(name="entertainment",  allocated=150.0),
        "shopping":      BudgetBucket(name="shopping",       allocated=150.0),
        "savings":       BudgetBucket(name="savings",        allocated=400.0),
        "debt":          BudgetBucket(name="debt",           allocated=200.0),
        "other":         BudgetBucket(name="other",          allocated=100.0),
    }


# ══════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════

class PersonalFinanceEnv:
    """
    OpenEnv-compliant Personal Finance Manager environment.

    Implements:
      reset()  → Observation
      step()   → (Observation, Reward, done: bool, info: dict)
      state()  → dict

    Tasks:
      1 = Easy   (9 transactions, no shocks)
      2 = Medium (16 transactions, budget pressure)
      3 = Hard   (20 transactions + emergency shock)
    """

    MONTHLY_INCOME  = 3_800.0
    STARTING_CASH   = 2_200.0
    STARTING_DEBT   = 4_500.0
    STARTING_INVEST = 1_800.0

    def __init__(self, task_id: int = 1, seed: int = 42):
        assert task_id in (1, 2, 3), "task_id must be 1, 2, or 3"
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._cfg = {
            1: dict(sl=slice(0, 9),  shock=False),
            2: dict(sl=slice(0, 16), shock=False),
            3: dict(sl=slice(0, 20), shock=True),
        }[task_id]
        self._txns: list[Transaction] = []
        self._buckets: dict[str, BudgetBucket] = {}
        self._cash = self._savings = self._investments = self._debt = 0.0
        self._day = 1
        self._rejected_total = self._invested_this_month = self._extra_debt_paid = 0.0
        self._action_log: list[dict] = []
        self._done = False

    # ── OpenEnv Interface ──────────────────────────────────────

    def reset(self) -> Observation:
        """Reset to start of month. Returns initial Observation."""
        templates = _TRANSACTION_TEMPLATES[self._cfg["sl"]]
        self._txns = [
            Transaction(**{
                **t,
                "category": ExpenseCategory(t["category"]) if t["category"] else None
            })
            for t in templates
        ]
        if self._cfg["shock"]:
            self._txns.append(Transaction(
                id="t_emerg", description="Emergency car repair",
                amount=650.0, day=17, essential=True, pending=True
            ))
        self._buckets = _fresh_buckets()
        self._cash = self.STARTING_CASH + self.MONTHLY_INCOME
        self._savings = 300.0
        self._investments = self.STARTING_INVEST
        self._debt = self.STARTING_DEBT
        self._day = 1
        self._rejected_total = self._invested_this_month = self._extra_debt_paid = 0.0
        self._action_log = []
        self._done = False

        # Auto-book already-categorized non-pending transactions
        for txn in self._txns:
            if not txn.pending and txn.category is not None:
                b = txn.category.value
                if b in self._buckets:
                    self._buckets[b].spent += txn.amount
                self._cash -= txn.amount

        return self._obs()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Apply one action. Returns (Observation, Reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is done — call reset()")

        reward = self._apply(action)
        self._action_log.append({"action": action.model_dump(), "reward": reward.value})
        self._day = min(self._day + 1, 30)

        obs = self._obs()
        remaining = len(obs.pending_transactions) + len(obs.uncategorized_transactions)
        self._done = (remaining == 0 and self._day >= 20) or self._day >= 30

        return obs, reward, self._done, dict(
            day=self._day,
            cash=round(self._cash, 2),
            savings_rate=round(self._savings_rate(), 4),
            remaining_items=remaining,
        )

    def state(self) -> dict[str, Any]:
        """Return full internal state snapshot."""
        return dict(
            task_id=self.task_id,
            day=self._day,
            cash=self._cash,
            savings=self._savings,
            investments=self._investments,
            debt=self._debt,
            savings_rate=self._savings_rate(),
            buckets={k: v.model_dump() for k, v in self._buckets.items()},
            action_log=self._action_log,
            done=self._done,
        )

    # ── Action Dispatch ────────────────────────────────────────

    def _apply(self, action: Action) -> Reward:
        return {
            ActionType.CATEGORIZE: self._categorize,
            ActionType.APPROVE:    self._approve,
            ActionType.REJECT:     self._reject,
            ActionType.ALLOCATE:   self._allocate,
            ActionType.INVEST:     self._invest,
            ActionType.PAY_DEBT:   self._pay_debt,
        }.get(action.action_type, lambda _: Reward(value=-0.05, reason="Unknown action"))(action)

    def _categorize(self, a: Action) -> Reward:
        txn = next((t for t in self._txns
                    if t.id == a.transaction_id and t.category is None and not t.pending), None)
        if txn is None:
            return Reward(value=-0.1, reason="Transaction not found or already categorized")
        if a.category is None:
            return Reward(value=-0.05, reason="No category provided")
        correct = self._guess_category(txn)
        txn.category = a.category
        b = a.category.value
        if b in self._buckets:
            self._buckets[b].spent += txn.amount
        self._cash -= txn.amount
        bd = {
            "accuracy":  0.5 if a.category == correct else -0.1,
            "rationale": 0.1 if len(a.rationale) > 20 else 0.0,
        }
        return Reward(value=round(max(-1.0, min(1.0, sum(bd.values()))), 3),
                      breakdown=bd,
                      reason=f"Labeled as {a.category} (correct: {correct})")

    def _approve(self, a: Action) -> Reward:
        txn = next((t for t in self._txns if t.id == a.transaction_id and t.pending), None)
        if txn is None:
            return Reward(value=-0.1, reason="Pending transaction not found")
        if txn.category is None:
            txn.category = self._guess_category(txn)
        b = txn.category.value
        if b in self._buckets:
            self._buckets[b].spent += txn.amount
        self._cash -= txn.amount
        txn.pending = False
        bd = {}
        if txn.essential:
            bd["essential_ok"] = 0.3
        else:
            bucket = self._buckets.get(b)
            if bucket and bucket.utilization > 0.9:
                bd["over_budget"] = -0.3
            else:
                bd["discretionary"] = 0.1
            if self._cash < 500:
                bd["low_cash"] = -0.2
        bd["rationale"] = 0.1 if len(a.rationale) > 20 else 0.0
        return Reward(value=round(max(-1.0, min(1.0, sum(bd.values()))), 3),
                      breakdown=bd,
                      reason=f"Approved {txn.description} ${txn.amount:.2f}")

    def _reject(self, a: Action) -> Reward:
        txn = next((t for t in self._txns if t.id == a.transaction_id and t.pending), None)
        if txn is None:
            return Reward(value=-0.1, reason="Pending transaction not found")
        txn.pending = False
        self._rejected_total += txn.amount
        bd = {}
        if not txn.essential:
            bd["smart_save"] = 0.4
            if self._cash < 800:
                bd["cash_bonus"] = 0.2
        else:
            bd["essential_blocked"] = -0.5
        bd["rationale"] = 0.1 if len(a.rationale) > 20 else 0.0
        return Reward(value=round(max(-1.0, min(1.0, sum(bd.values()))), 3),
                      breakdown=bd,
                      reason=f"Rejected {txn.description} ${txn.amount:.2f}")

    def _allocate(self, a: Action) -> Reward:
        fb = self._buckets.get(a.from_bucket or "")
        tb = self._buckets.get(a.to_bucket or "")
        if not fb or not tb or not a.amount:
            return Reward(value=-0.1, reason="Invalid allocate params")
        if a.amount > fb.remaining:
            return Reward(value=-0.1, reason="Insufficient remaining in source bucket")
        fb.allocated -= a.amount
        tb.allocated += a.amount
        bd = {
            "reallocation": 0.2,
            "smart":        0.2 if tb.utilization > 0.85 else 0.0,
            "rationale":    0.1 if len(a.rationale) > 20 else 0.0,
        }
        return Reward(value=round(min(1.0, sum(bd.values())), 3),
                      breakdown=bd,
                      reason=f"Moved ${a.amount:.2f}: {a.from_bucket} -> {a.to_bucket}")

    def _invest(self, a: Action) -> Reward:
        amt = a.amount or 0.0
        if amt <= 0:
            return Reward(value=-0.05, reason="Amount must be positive")
        if amt > self._cash - 300:
            return Reward(value=-0.2, reason="Would breach $300 emergency buffer")
        self._cash -= amt
        self._investments += amt
        self._invested_this_month += amt
        bd = {
            "invested":     min(0.5, amt / 500),
            "debt_caution": -0.15 if self._debt > 2000 and amt > 200 else 0.0,
            "rationale":    0.1 if len(a.rationale) > 20 else 0.0,
        }
        return Reward(value=round(max(-1.0, min(1.0, sum(bd.values()))), 3),
                      breakdown=bd, reason=f"Invested ${amt:.2f}")

    def _pay_debt(self, a: Action) -> Reward:
        amt = a.amount or 0.0
        if amt <= 0:
            return Reward(value=-0.05, reason="Amount must be positive")
        if amt > self._cash - 300:
            return Reward(value=-0.2, reason="Would breach emergency buffer")
        actual = min(amt, self._debt)
        self._cash -= actual
        self._debt -= actual
        self._extra_debt_paid += actual
        bd = {
            "debt_reduction": min(0.5, actual / 300),
            "rationale":      0.1 if len(a.rationale) > 20 else 0.0,
        }
        return Reward(value=round(min(1.0, sum(bd.values())), 3),
                      breakdown=bd,
                      reason=f"Paid ${actual:.2f} extra debt. Remaining: ${self._debt:.2f}")

    # ── Helpers ────────────────────────────────────────────────

    def _guess_category(self, txn: Transaction) -> ExpenseCategory:
        d = txn.description.lower()
        if any(w in d for w in ["rent", "mortgage"]):                              return ExpenseCategory.HOUSING
        if any(w in d for w in ["grocery", "restaurant", "coffee", "bar"]):        return ExpenseCategory.FOOD
        if any(w in d for w in ["bus", "metro", "gas", "car", "insurance"]):       return ExpenseCategory.TRANSPORT
        if any(w in d for w in ["electric", "internet", "phone", "utility"]):      return ExpenseCategory.UTILITIES
        if any(w in d for w in ["doctor", "pharmacy", "health", "vitamin"]):       return ExpenseCategory.HEALTHCARE
        if any(w in d for w in ["netflix", "spotify", "concert", "gym", "music"]): return ExpenseCategory.ENTERTAINMENT
        if any(w in d for w in ["amazon", "clothing", "shop", "store"]):           return ExpenseCategory.SHOPPING
        if any(w in d for w in ["credit card", "loan", "debt"]):                   return ExpenseCategory.DEBT
        return ExpenseCategory.OTHER

    def _savings_rate(self) -> float:
        saved = self._savings + self._invested_this_month + self._rejected_total
        return round(max(0.0, min(1.0, saved / self.MONTHLY_INCOME)), 4)

    def _obs(self) -> Observation:
        pending       = [t for t in self._txns if t.pending]
        uncategorized = [t for t in self._txns if t.category is None and not t.pending]
        recent        = [t for t in self._txns if t.category is not None and not t.pending][-10:]
        return Observation(
            day=self._day,
            monthly_income=self.MONTHLY_INCOME,
            cash_balance=round(self._cash, 2),
            savings_balance=round(self._savings, 2),
            investment_balance=round(self._investments, 2),
            debt_balance=round(self._debt, 2),
            budget_buckets=self._buckets,
            pending_transactions=pending,
            recent_transactions=recent,
            uncategorized_transactions=uncategorized,
            savings_rate=self._savings_rate(),
            month_complete=self._done,
        )
