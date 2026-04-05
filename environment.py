"""
server/environment.py — PersonalFinanceEnv core logic
Uses openenv_core.env_server.Environment base class.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any
from openenv_core.env_server import Environment


# ── Action / Observation dataclasses ──────────────────────────

@dataclass
class FinanceAction:
    action_type: str          # categorize | approve | reject | allocate | invest | pay_debt
    transaction_id: str = ""
    category: str = ""
    from_bucket: str = ""
    to_bucket: str = ""
    amount: float = 0.0
    rationale: str = ""


@dataclass
class FinanceObservation:
    day: int
    cash_balance: float
    savings_balance: float
    investment_balance: float
    debt_balance: float
    savings_rate: float
    pending_transactions: list = field(default_factory=list)
    uncategorized_transactions: list = field(default_factory=list)
    budget_buckets: dict = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    info: str = ""


# ── Transaction templates ──────────────────────────────────────

_TEMPLATES = [
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
    dict(id="t18", description="Spotify",               amount=25.0,   day=6,  category=None,         essential=False, pending=True),
    dict(id="t19", description="Gas station",           amount=55.0,   day=10, category="transport",  essential=True,  pending=False),
    dict(id="t20", description="Bar tab",               amount=90.0,   day=22, category=None,         essential=False, pending=True),
]

_TASK_SLICES = {1: slice(0, 9), 2: slice(0, 16), 3: slice(0, 20)}


def _fresh_buckets():
    return {
        "housing":       dict(allocated=1300.0, spent=0.0),
        "food":          dict(allocated=400.0,  spent=0.0),
        "transport":     dict(allocated=280.0,  spent=0.0),
        "utilities":     dict(allocated=180.0,  spent=0.0),
        "healthcare":    dict(allocated=100.0,  spent=0.0),
        "entertainment": dict(allocated=150.0,  spent=0.0),
        "shopping":      dict(allocated=150.0,  spent=0.0),
        "savings":       dict(allocated=400.0,  spent=0.0),
        "debt":          dict(allocated=200.0,  spent=0.0),
        "other":         dict(allocated=100.0,  spent=0.0),
    }


# ── Main Environment class ─────────────────────────────────────

class PersonalFinanceEnv(Environment):

    MONTHLY_INCOME  = 3_800.0
    STARTING_CASH   = 2_200.0
    STARTING_DEBT   = 4_500.0
    STARTING_INVEST = 1_800.0

    def __init__(self, task_id: int = 1, seed: int = 42):
        self.task_id = max(1, min(3, int(task_id)))
        self.seed = seed
        self._txns = []
        self._buckets = {}
        self._cash = self._savings = self._investments = self._debt = 0.0
        self._day = 1
        self._rejected_total = self._invested_this_month = 0.0
        self._done = False

    def reset(self) -> FinanceObservation:
        templates = _TEMPLATES[_TASK_SLICES[self.task_id]]
        self._txns = [dict(t) for t in templates]
        if self.task_id == 3:
            self._txns.append(dict(
                id="t_emerg", description="Emergency car repair",
                amount=650.0, day=17, category=None, essential=True, pending=True
            ))
        self._buckets = _fresh_buckets()
        self._cash = self.STARTING_CASH + self.MONTHLY_INCOME
        self._savings = 300.0
        self._investments = self.STARTING_INVEST
        self._debt = self.STARTING_DEBT
        self._day = 1
        self._rejected_total = self._invested_this_month = 0.0
        self._done = False

        for txn in self._txns:
            if not txn["pending"] and txn["category"]:
                b = txn["category"]
                if b in self._buckets:
                    self._buckets[b]["spent"] += txn["amount"]
                self._cash -= txn["amount"]

        return self._make_obs(0.0)

    def step(self, action: FinanceAction) -> FinanceObservation:
        if self._done:
            return self._make_obs(0.0)

        reward = self._apply(action)
        self._day = min(self._day + 1, 30)

        pending = [t for t in self._txns if t["pending"]]
        uncateg = [t for t in self._txns if not t["category"] and not t["pending"]]
        remaining = len(pending) + len(uncateg)
        self._done = (remaining == 0 and self._day >= 20) or self._day >= 30

        return self._make_obs(reward)

    def state(self) -> dict:
        return dict(
            task_id=self.task_id, day=self._day,
            cash=self._cash, savings=self._savings,
            investments=self._investments, debt=self._debt,
            savings_rate=self._savings_rate(),
            buckets=self._buckets, done=self._done,
        )

    # ── Action handlers ────────────────────────────────────────

    def _apply(self, action: FinanceAction) -> float:
        t = action.action_type
        if t == "categorize":  return self._categorize(action)
        if t == "approve":     return self._approve(action)
        if t == "reject":      return self._reject(action)
        if t == "allocate":    return self._allocate(action)
        if t == "invest":      return self._invest(action)
        if t == "pay_debt":    return self._pay_debt(action)
        return -0.05

    def _categorize(self, a):
        txn = next((t for t in self._txns if t["id"] == a.transaction_id
                    and not t["category"] and not t["pending"]), None)
        if not txn: return -0.1
        correct = self._guess(txn)
        txn["category"] = a.category
        if a.category in self._buckets:
            self._buckets[a.category]["spent"] += txn["amount"]
        self._cash -= txn["amount"]
        base = 0.5 if a.category == correct else -0.1
        return round(max(-1.0, min(1.0, base + (0.1 if len(a.rationale) > 20 else 0))), 3)

    def _approve(self, a):
        txn = next((t for t in self._txns if t["id"] == a.transaction_id and t["pending"]), None)
        if not txn: return -0.1
        if not txn["category"]: txn["category"] = self._guess(txn)
        b = txn["category"]
        if b in self._buckets: self._buckets[b]["spent"] += txn["amount"]
        self._cash -= txn["amount"]
        txn["pending"] = False
        score = 0.3 if txn["essential"] else (
            -0.3 if self._buckets.get(b, {}).get("spent", 0) / max(self._buckets.get(b, {}).get("allocated", 1), 1) > 0.9
            else (-0.2 if self._cash < 500 else 0.1)
        )
        return round(max(-1.0, min(1.0, score + (0.1 if len(a.rationale) > 20 else 0))), 3)

    def _reject(self, a):
        txn = next((t for t in self._txns if t["id"] == a.transaction_id and t["pending"]), None)
        if not txn: return -0.1
        txn["pending"] = False
        self._rejected_total += txn["amount"]
        score = -0.5 if txn["essential"] else (0.4 + (0.2 if self._cash < 800 else 0))
        return round(max(-1.0, min(1.0, score + (0.1 if len(a.rationale) > 20 else 0))), 3)

    def _allocate(self, a):
        fb, tb = self._buckets.get(a.from_bucket), self._buckets.get(a.to_bucket)
        if not fb or not tb or a.amount <= 0: return -0.1
        remaining = fb["allocated"] - fb["spent"]
        if a.amount > remaining: return -0.1
        fb["allocated"] -= a.amount
        tb["allocated"] += a.amount
        smart = tb["spent"] / max(tb["allocated"], 1) > 0.85
        return round(min(1.0, 0.2 + (0.2 if smart else 0) + (0.1 if len(a.rationale) > 20 else 0)), 3)

    def _invest(self, a):
        if a.amount <= 0: return -0.05
        if a.amount > self._cash - 300: return -0.2
        self._cash -= a.amount
        self._investments += a.amount
        self._invested_this_month += a.amount
        score = min(0.5, a.amount / 500) + (-0.15 if self._debt > 2000 and a.amount > 200 else 0)
        return round(max(-1.0, min(1.0, score + (0.1 if len(a.rationale) > 20 else 0))), 3)

    def _pay_debt(self, a):
        if a.amount <= 0: return -0.05
        if a.amount > self._cash - 300: return -0.2
        actual = min(a.amount, self._debt)
        self._cash -= actual
        self._debt -= actual
        return round(min(1.0, min(0.5, actual / 300) + (0.1 if len(a.rationale) > 20 else 0)), 3)

    def _guess(self, txn) -> str:
        d = txn["description"].lower()
        if any(w in d for w in ["rent", "mortgage"]):                              return "housing"
        if any(w in d for w in ["grocery", "restaurant", "coffee", "bar"]):        return "food"
        if any(w in d for w in ["bus", "metro", "gas", "car", "insurance"]):       return "transport"
        if any(w in d for w in ["electric", "internet", "phone", "utility"]):      return "utilities"
        if any(w in d for w in ["doctor", "pharmacy", "health", "vitamin"]):       return "healthcare"
        if any(w in d for w in ["netflix", "spotify", "concert", "gym", "music"]): return "entertainment"
        if any(w in d for w in ["amazon", "clothing", "shop", "store"]):           return "shopping"
        if any(w in d for w in ["credit card", "loan", "debt"]):                   return "debt"
        return "other"

    def _savings_rate(self) -> float:
        saved = self._savings + self._invested_this_month + self._rejected_total
        return round(max(0.0, min(1.0, saved / self.MONTHLY_INCOME)), 4)

    def _make_obs(self, reward: float) -> FinanceObservation:
        pending = [t for t in self._txns if t["pending"]]
        uncateg = [t for t in self._txns if not t["category"] and not t["pending"]]
        remaining = len(pending) + len(uncateg)
        done = (remaining == 0 and self._day >= 20) or self._day >= 30
        return FinanceObservation(
            day=self._day,
            cash_balance=round(self._cash, 2),
            savings_balance=round(self._savings, 2),
            investment_balance=round(self._investments, 2),
            debt_balance=round(self._debt, 2),
            savings_rate=self._savings_rate(),
            pending_transactions=pending,
            uncategorized_transactions=uncateg,
            budget_buckets=self._buckets,
            reward=reward,
            done=done,
            info=f"day={self._day} remaining={remaining}",
        )
