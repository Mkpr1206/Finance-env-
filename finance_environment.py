"""
server/finance_environment.py — PersonalFinanceEnvironment subclassing openenv_core.Environment
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

# Import the core finance engine (plain Python, no openenv dependency)
from environment import PersonalFinanceEnv, Action as CoreAction, ActionType, ExpenseCategory


class PersonalFinanceEnvironment(Environment):
    """
    OpenEnv-compliant Personal Finance Manager.
    Wraps PersonalFinanceEnv for the openenv_core server framework.
    """

    def __init__(self):
        super().__init__()
        self._env = PersonalFinanceEnv(task_id=1, seed=42)
        self._task_id = 1
        self._initialized = False

    def reset(self, task_id: int = 1, seed: int = 42) -> FinanceObservation:
        self._task_id = task_id
        self._env = PersonalFinanceEnv(task_id=task_id, seed=seed)
        obs = self._env.reset()
        self._initialized = True
        return self._to_obs(obs, reward=0.0, done=False, msg="Episode started")

    def step(self, action: FinanceAction) -> FinanceObservation:
        if not self._initialized:
            self.reset()

        try:
            core_action = self._to_core_action(action)
            obs, reward, done, info = self._env.step(core_action)
            # Clamp reward strictly between -1 and 1 (already is, but be safe)
            reward_val = float(max(-0.999, min(0.999, reward.value)))
            return self._to_obs(obs, reward=reward_val, done=done,
                                msg=reward.reason)
        except Exception as e:
            obs = self._env._obs()
            return self._to_obs(obs, reward=0.001, done=False,
                                msg=f"Action error: {e}")

    @property
    def state(self):
        if not self._initialized:
            return {"status": "not_started"}
        return self._env.state()

    # ── helpers ──────────────────────────────────────────────

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
            reward=reward,
            done=done,
            message=msg,
        )
