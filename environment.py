from __future__ import annotations
from typing import Tuple


class Action:
    def __init__(self, action_type: str = "noop", **kwargs):
        self.action_type = action_type


class Reward:
    def __init__(self, value: float, reason: str = ""):
        self.value = value
        self.reason = reason


class Observation:
    def __init__(self):
        self.day = 1
        self.monthly_income = 5000.0
        self.cash_balance = 1000.0
        self.savings_balance = 200.0
        self.investment_balance = 0.0
        self.debt_balance = 300.0
        self.savings_rate = 0.2

        self.pending_transactions = []
        self.uncategorized_transactions = []


class PersonalFinanceEnv:

    def __init__(self, task_id: int = 1, seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self.current_step = 0
        self.done = False

    # ✅ REQUIRED
    def reset(self) -> Observation:
        self.current_step = 0
        self.done = False
        return self._obs()

    # ✅ REQUIRED
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        self.current_step += 1

        reward = Reward(0.5, "valid step")

        if self.current_step >= 20:
            self.done = True

        return self._obs(), reward, self.done, {}

    # ✅ REQUIRED
    def state(self) -> dict:
        return {
            "step": self.current_step,
            "savings_rate": 0.2,
            "debt": 300.0,
        }

    # internal
    def _obs(self) -> Observation:
        return Observation()