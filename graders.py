"""
graders.py — Deterministic graders for all 3 PersonalFinanceEnv tasks.
Scores range 0.0–1.0. Pass thresholds: Task 1 ≥ 0.60, Task 2 ≥ 0.55, Task 3 ≥ 0.50
"""

from __future__ import annotations
from environment import PersonalFinanceEnv, Action

PASS_THRESHOLDS = {1: 0.60, 2: 0.55, 3: 0.50}


def grade_task(task_id: int, actions: list[dict], seed: int = 42) -> dict:
    """
    Grade a sequence of actions against one task.

    Args:
        task_id:  1 = easy, 2 = medium, 3 = hard
        actions:  list of dicts matching Action schema
        seed:     RNG seed (keep at 42 for reproducibility)

    Returns dict with keys:
        task_id, score, pass, pass_threshold,
        savings_rate, debt_remaining,
        avg_step_reward, completion_rate,
        steps, errors, rewards_per_step
    """
    env = PersonalFinanceEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    total_items = len(obs.pending_transactions) + len(obs.uncategorized_transactions)
    rewards: list[float] = []
    errors:  list[str]   = []
    MAX_STEPS = max(total_items * 4, 30)

    for i, raw in enumerate(actions):
        if i >= MAX_STEPS:
            errors.append(f"Exceeded {MAX_STEPS} max steps")
            break
        try:
            action = Action(**raw)
        except Exception as e:
            errors.append(f"Invalid action #{i}: {e}")
            continue
        obs, reward, done, info = env.step(action)
        rewards.append(reward.value)
        if done:
            break

    state = env.state()
    final_obs = env._obs()

    # ── Scoring (4 components) ───────────────────────────────
    avg_reward      = sum(rewards) / len(rewards) if rewards else 0.0
    quality_score   = max(0.0, min(1.0, (avg_reward + 0.5) / 1.5))
    savings_score   = min(1.0, state["savings_rate"] / 0.20)   # 20% = perfect
    remaining       = len(final_obs.pending_transactions) + len(final_obs.uncategorized_transactions)
    completion      = 1.0 - (remaining / total_items) if total_items > 0 else 1.0
    debt_progress   = min(1.0, (PersonalFinanceEnv.STARTING_DEBT - state["debt"]) / PersonalFinanceEnv.STARTING_DEBT)

    score = (
        quality_score * 0.40 +
        savings_score * 0.30 +
        completion    * 0.20 +
        debt_progress * 0.10
    )

    return {
        "task_id":         task_id,
        "score":           round(score, 3),
        "pass":            score >= PASS_THRESHOLDS[task_id],
        "pass_threshold":  PASS_THRESHOLDS[task_id],
        "savings_rate":    round(state["savings_rate"], 4),
        "debt_remaining":  round(state["debt"], 2),
        "avg_step_reward": round(avg_reward, 3),
        "completion_rate": round(completion, 3),
        "steps":           len(rewards),
        "errors":          errors,
        "rewards_per_step": [round(r, 3) for r in rewards],
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  PersonalFinanceEnv — Grader Self-Test")
    print("=" * 58)

    t1 = [
        dict(action_type="categorize", transaction_id="t02", category="utilities",
             rationale="Electricity bill is a utility expense"),
        dict(action_type="categorize", transaction_id="t03", category="food",
             rationale="Grocery store is a food expense"),
        dict(action_type="categorize", transaction_id="t04", category="utilities",
             rationale="Internet is a utility bill"),
        dict(action_type="reject", transaction_id="t09",
             rationale="Netflix is discretionary, cutting to save money this month"),
        dict(action_type="approve", transaction_id="t10",
             rationale="One restaurant meal is reasonable within food budget"),
        dict(action_type="reject", transaction_id="t11",
             rationale="Amazon impulse purchase not in this month's budget"),
        dict(action_type="approve", transaction_id="t12",
             rationale="Gym membership supports health, within entertainment budget"),
        dict(action_type="invest", amount=200.0,
             rationale="Investing surplus cash into long-term investments"),
    ]
    r1 = grade_task(1, t1)
    print(f"\nTask 1 (Easy)   {'✓ PASS' if r1['pass'] else '✗ FAIL'}  "
          f"score={r1['score']:.3f}  threshold={r1['pass_threshold']}  "
          f"savings={r1['savings_rate']:.1%}")

    t2 = t1 + [
        dict(action_type="categorize", transaction_id="t06", category="food",
             rationale="Second grocery run this month"),
        dict(action_type="categorize", transaction_id="t13", category="food",
             rationale="Coffee shop is food/beverage spending"),
        dict(action_type="reject", transaction_id="t14",
             rationale="Clothing store over budget this month"),
        dict(action_type="reject", transaction_id="t15",
             rationale="Concert tickets are a luxury we cannot afford now"),
        dict(action_type="categorize", transaction_id="t16", category="healthcare",
             rationale="Pharmacy purchases fall under healthcare"),
        dict(action_type="reject", transaction_id="t18",
             rationale="Already have Netflix rejected, no need for Spotify too"),
        dict(action_type="allocate", from_bucket="entertainment", to_bucket="food",
             amount=50.0, rationale="Entertainment under-budget, food running high"),
        dict(action_type="pay_debt", amount=150.0,
             rationale="Extra debt payment to reduce interest charges"),
    ]
    r2 = grade_task(2, t2)
    print(f"Task 2 (Medium) {'✓ PASS' if r2['pass'] else '✗ FAIL'}  "
          f"score={r2['score']:.3f}  threshold={r2['pass_threshold']}  "
          f"savings={r2['savings_rate']:.1%}")

    t3 = t2 + [
        dict(action_type="reject", transaction_id="t20",
             rationale="Bar tab rejected due to incoming emergency expense"),
        dict(action_type="approve", transaction_id="t_emerg",
             rationale="Emergency car repair is essential and cannot be avoided"),
        dict(action_type="allocate", from_bucket="shopping", to_bucket="other",
             amount=100.0, rationale="Freeing shopping budget to cover emergency"),
        dict(action_type="pay_debt", amount=100.0,
             rationale="Small extra payment even during emergency month"),
    ]
    r3 = grade_task(3, t3)
    print(f"Task 3 (Hard)   {'✓ PASS' if r3['pass'] else '✗ FAIL'}  "
          f"score={r3['score']:.3f}  threshold={r3['pass_threshold']}  "
          f"savings={r3['savings_rate']:.1%}")

    print(f"\nRewards (Task 1): {r1['rewards_per_step']}")
    print(f"Errors:           {r1['errors'] or 'none'}")
    print("=" * 58)
