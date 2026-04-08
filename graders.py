"""
graders.py — PersonalFinanceEnv task graders.
Scores are ALWAYS strictly between 0.001 and 0.999.
"""

from __future__ import annotations
from environment import PersonalFinanceEnv, Action

PASS_THRESHOLDS = {1: 0.60, 2: 0.55, 3: 0.50}


def grade_task(task_id: int, actions: list[dict], seed: int = 42) -> dict:
    env = PersonalFinanceEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    total_items = len(obs.pending_transactions) + len(obs.uncategorized_transactions)
    rewards: list[float] = []
    errors:  list[str]   = []

    for i, raw in enumerate(actions[:50]):
        try:
            action = Action(**raw)
        except Exception as e:
            errors.append(f"action #{i}: {e}")
            continue
        obs, reward, done, _ = env.step(action)
        rewards.append(reward.value)
        if done:
            break

    state     = env.state()
    final_obs = env._obs()

    avg_reward    = sum(rewards) / len(rewards) if rewards else 0.0
    quality       = max(0.0, min(1.0, (avg_reward + 0.5) / 1.5))
    savings       = min(1.0, state["savings_rate"] / 0.20)
    remaining     = len(final_obs.pending_transactions) + len(final_obs.uncategorized_transactions)
    completion    = 1.0 - (remaining / total_items) if total_items > 0 else 1.0
    debt_progress = min(1.0, (PersonalFinanceEnv.STARTING_DEBT - state["debt"]) / PersonalFinanceEnv.STARTING_DEBT)

    raw_score = quality * 0.40 + savings * 0.30 + completion * 0.20 + debt_progress * 0.10

    # STRICTLY between 0 and 1 — never 0.0, never 1.0
    score = round(max(0.001, min(0.999, raw_score)), 3)

    assert 0.0 < score < 1.0, f"Score {score} not strictly in (0,1)"

    return {
        "task_id":          task_id,
        "score":            score,
        "pass":             score >= PASS_THRESHOLDS[task_id],
        "pass_threshold":   PASS_THRESHOLDS[task_id],
        "savings_rate":     round(state["savings_rate"], 4),
        "debt_remaining":   round(state["debt"], 2),
        "avg_step_reward":  round(avg_reward, 3),
        "completion_rate":  round(completion, 3),
        "steps":            len(rewards),
        "errors":           errors,
        "rewards_per_step": [round(r, 3) for r in rewards],
    }


if __name__ == "__main__":
    dummy = [{"action_type": "pay_debt", "amount": 50.0, "rationale": "test"}]
    for t in [1, 2, 3]:
        r = grade_task(t, dummy)
        assert 0.0 < r["score"] < 1.0, f"FAIL Task {t}: score={r['score']}"
        print(f"Task {t}: score={r['score']}  PASS={r['pass']}  OK")
    print("All scores strictly in (0,1) - ready to submit.")
