"""
server/finance_environment.py — PersonalFinanceEnvironment
Subclasses openenv_core.Environment. All reward values strictly in (0.001, 0.999).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv_core.env_server.interfaces import Environment
except ImportError:
    try:
        from openenv_core.env_server import Environment
    except ImportError:
        from openenv_core import Environment

try:
    from server.models import FinanceAction, FinanceObservation
except ImportError:
    from models import FinanceAction, FinanceObservation

from environment import PersonalFinanceEnv, Action as CoreAction, ActionType, ExpenseCategory


def _clamp(v: float) -> float:
    """Clamp to strictly (0.001, 0.999) — never 0.0 or 1.0."""
    return round(max(0.001, min(0.999, float(v))), 4)


class PersonalFinanceEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._env = PersonalFinanceEnv(task_id=1, seed=42)
        self._initialized = False

    def reset(self, task_id: int = 1, seed: int = 42) -> FinanceObservation:
        self._env = PersonalFinanceEnv(task_id=task_id, seed=seed)
        obs = self._env.reset()
        self._initialized = True
        return self._to_obs(obs, reward=0.5, done=False, msg="Episode started")

    def step(self, action: FinanceAction) -> FinanceObservation:
        if not self._initialized:
            self.reset()
        try:
            core_action = self._to_core_action(action)
            obs, reward, done, info = self._env.step(core_action)
            return self._to_obs(obs, reward=_clamp(reward.value), done=done, msg=reward.reason)
        except Exception as e:
            obs = self._env._obs()
            return self._to_obs(obs, reward=0.1, done=False, msg=f"error: {e}")

    @property
    def state(self):
        if not self._initialized:
            return {"status": "not_started"}
        return self._env.state()

    def _to_core_action(self, a: FinanceAction) -> CoreAction:
        return CoreAction(
            action_type=ActionType(a.action_type),
            transaction_id=a.transaction_id,
            category=ExpenseCategory(a.category) if a.category else None,
            from_bucket=a.from_bucket,
            to_bucket=a.to_bucket,
            amount=a.amount,
            rationale=a.rationale or "",
        )

    def _to_obs(self, obs, reward: float, done: bool, msg: str) -> FinanceObservation:
        return FinanceObservation(
            day=obs.day,
            monthly_income=obs.monthly_income,
            cash_balance=obs.cash_balance,
            savings_balance=obs.savings_balance,
            investment_balance=obs.investment_balance,
            debt_balance=obs.debt_balance,
            savings_rate=obs.savings_rate,
            pending_count=len(obs.pending_transactions),
            uncategorized_count=len(obs.uncategorized_transactions),
            reward=_clamp(reward),
            done=done,
            message=msg,
        )
