"""
graders.py — Deterministic graders for all 3 PersonalFinanceEnv tasks.
Scores are strictly in (0.001, 0.999) — never exactly 0.0 or 1.0.
"""

from __future__ import annotations
from environment import PersonalFinanceEnv, Action

PASS_THRESHOLDS = {1: 0.60, 2: 0.55, 3: 0.50}

print("🔥 NEW GRADER LOADED 🔥")
def grade_task(task_id: int, actions: list[dict], seed: int = 42) -> dict:
    env = PersonalFinanceEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    total_items = len(obs.pending_transactions) + len(obs.uncategorized_transactions)

    rewards: list[float] = []
    errors: list[str] = []

    MAX_STEPS = max(total_items * 4, 30)

    # ── Run environment ───────────────────────────────
    for i, raw in enumerate(actions):
        if i >= MAX_STEPS:
            errors.append(f"Exceeded {MAX_STEPS} max steps")
            break

        try:
            action = Action(**raw)
        except Exception as e:
            errors.append(f"Invalid action #{i}: {e}")
            continue

        obs, reward, done, _ = env.step(action)
        rewards.append(reward.value)

        if done:
            break

    state = env.state()
    final_obs = env._obs()

    # ── SAFE FUNCTION ───────────────────────────────
    def safe(x: float) -> float:
        return max(0.001, min(0.999, x))

    # ── Compute metrics ─────────────────────────────
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    remaining = (
        len(final_obs.pending_transactions)
        + len(final_obs.uncategorized_transactions)
    )

    quality_score = safe((avg_reward + 0.5) / 1.5)
    savings_score = safe(state["savings_rate"] / 0.20)

    if total_items > 0:
        completion = safe(1.0 - (remaining / total_items))
    else:
        completion = 0.999

    debt_progress = safe(
        (PersonalFinanceEnv.STARTING_DEBT - state["debt"])
        / PersonalFinanceEnv.STARTING_DEBT
    )

    # ── Final score ───────────────────────────────
    raw_score = (
        quality_score * 0.40
        + savings_score * 0.30
        + completion * 0.20
        + debt_progress * 0.10
    )

    # HARD GUARANTEE
    score = max(0.001, min(0.999, raw_score))

    # safer formatting (avoid rounding to 1.0)
    score = float(f"{score:.4f}")

    # FINAL SAFETY CHECK
    assert 0.0 < score < 1.0, f"Invalid score: {score}"

    return {
        "task_id": task_id,
        "score": score,
        "pass": score >= PASS_THRESHOLDS[task_id],
        "pass_threshold": PASS_THRESHOLDS[task_id],
        "savings_rate": round(state["savings_rate"], 4),
        "debt_remaining": round(state["debt"], 2),
        "avg_step_reward": round(avg_reward, 3),
        "completion_rate": round(completion, 3),
        "steps": len(rewards),
        "errors": errors,
        "rewards_per_step": [round(r, 3) for r in rewards],
    }


# ── Self-test ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Running grader self-test")
    print("=" * 50)

    dummy_actions = [{"action_type": "noop"}] * 10

    for t in [1, 2, 3]:
        r = grade_task(t, dummy_actions)
        print(f"Task {t}: score={r['score']}")

        assert 0.0 < r["score"] < 1.0

    print("\n✅ All scores strictly within (0,1)")