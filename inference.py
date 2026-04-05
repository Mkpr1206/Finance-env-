"""
inference.py — PersonalFinanceEnv Baseline Inference Script
OpenEnv RL Hackathon Submission

Emits structured stdout logs:
  [START] task=<name> env=personal-finance model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import os
import sys
import json
import traceback

from openai import OpenAI
from environment import (
    PersonalFinanceEnv, Action, ActionType, ExpenseCategory, Observation
)

# ── Environment variables (with required defaults) ────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ENV_NAME     = "personal-finance"
TASK_NAMES   = {1: "budget-easy", 2: "budget-medium", 3: "budget-hard"}
MAX_STEPS    = 35   # well within 20-min runtime


# ══════════════════════════════════════════════════════════════════════════════
# System prompt
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a personal finance AI managing a monthly budget simulation.
Each turn you receive the current financial state and must choose ONE action.

Respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "action_type": "categorize" | "approve" | "reject" | "allocate" | "invest" | "pay_debt",
  "transaction_id": "<id>",         // for categorize / approve / reject
  "category": "<category>",         // for categorize only
  "from_bucket": "<bucket>",        // for allocate only
  "to_bucket": "<bucket>",          // for allocate only
  "amount": <float>,                // for allocate / invest / pay_debt
  "rationale": "<your reasoning>"   // always include
}

Categories: housing | food | transport | utilities | healthcare | entertainment | shopping | savings | investment | debt | other

Priority rules:
1. Categorize uncategorized transactions first (they block progress).
2. Approve essential expenses (rent, utilities, healthcare, minimum debt payments).
3. Reject discretionary items when cash < $800 OR that bucket > 90% utilized.
4. After handling all pending/uncategorized: invest if debt < $3000, else pay_debt.
5. Never breach the $300 emergency cash buffer.
6. Always include a meaningful rationale (it improves your score)."""


def build_prompt(obs: Observation, step: int) -> str:
    buckets = "\n".join(
        f"  {k:15s} alloc=${v.allocated:.0f}  spent=${v.spent:.0f}  "
        f"remaining=${v.remaining:.0f}  ({v.utilization:.0%})"
        for k, v in obs.budget_buckets.items()
    )
    pending = "\n".join(
        f"  [{t.id}] {t.description:<32} ${t.amount:.2f}  essential={t.essential}"
        for t in obs.pending_transactions
    ) or "  (none)"
    uncateg = "\n".join(
        f"  [{t.id}] {t.description:<32} ${t.amount:.2f}"
        for t in obs.uncategorized_transactions
    ) or "  (none)"

    return f"""=== STEP {step} | Day {obs.day}/30 ===

FINANCES:
  Cash:         ${obs.cash_balance:,.2f}
  Savings:      ${obs.savings_balance:,.2f}
  Investments:  ${obs.investment_balance:,.2f}
  Debt:         ${obs.debt_balance:,.2f}
  Income/mo:    ${obs.monthly_income:,.2f}
  Savings Rate: {obs.savings_rate:.1%}

BUDGET BUCKETS:
{buckets}

PENDING (approve or reject):
{pending}

UNCATEGORIZED (must categorize):
{uncateg}

Respond with ONE JSON action object."""


# ══════════════════════════════════════════════════════════════════════════════
# LLM call
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(obs: Observation, step: int) -> tuple[dict, str | None]:
    """
    Ask the LLM for one action.
    Returns (action_dict, error_string_or_None).
    """
    prompt = build_prompt(obs, step)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown fences if the model adds them
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        action_dict = json.loads(raw)
        return action_dict, None

    except json.JSONDecodeError as e:
        return _safe_fallback(obs), f"JSON parse error: {e}"
    except Exception as e:
        return _safe_fallback(obs), f"LLM error: {e}"


def _safe_fallback(obs: Observation) -> dict:
    """Return a safe no-op action when the LLM fails."""
    if obs.pending_transactions:
        t = obs.pending_transactions[0]
        # Approve essentials, reject discretionary
        atype = "approve" if t.essential else "reject"
        return {"action_type": atype, "transaction_id": t.id,
                "rationale": "Fallback: safe default action"}
    if obs.uncategorized_transactions:
        t = obs.uncategorized_transactions[0]
        return {"action_type": "categorize", "transaction_id": t.id,
                "category": "other", "rationale": "Fallback: categorize as other"}
    return {"action_type": "pay_debt", "amount": 50.0,
            "rationale": "Fallback: small debt payment"}


# ══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(task_id: int) -> dict:
    """
    Run one full episode for the given task.
    Emits [START], [STEP]×n, [END] to stdout.
    Returns summary dict.
    """
    task_name = TASK_NAMES[task_id]
    env = PersonalFinanceEnv(task_id=task_id, seed=42)

    rewards: list[float] = []
    last_error: str | None = None
    success = False
    step = 0

    # ── [START] ──────────────────────────────────────────────────────────────
    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset()

        while step < MAX_STEPS:
            step += 1
            remaining = len(obs.pending_transactions) + len(obs.uncategorized_transactions)

            # If nothing left to act on and past day 20, we can close out
            if remaining == 0 and obs.day >= 20:
                # One final invest action to close
                action_dict = {"action_type": "invest", "amount": 100.0,
                               "rationale": "Final surplus investment before close"}
                last_error = None
            else:
                action_dict, last_error = call_llm(obs, step)

            # Build Action (validate)
            try:
                action = Action(**action_dict)
                action_str = json.dumps(action_dict, separators=(",", ":"))
            except Exception as e:
                action_str = str(action_dict)
                last_error = f"Action validation error: {e}"
                action = Action(**_safe_fallback(obs))

            # ── env.step() ────────────────────────────────────────────────────
            obs, reward, done, info = env.step(action)
            rewards.append(reward.value)

            # ── [STEP] ────────────────────────────────────────────────────────
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward.value:.2f} done={'true' if done else 'false'} "
                f"error={'null' if last_error is None else last_error}",
                flush=True
            )

            if done:
                success = True
                break

        # Episode ended cleanly
        success = True

    except Exception as e:
        last_error = f"Episode error: {traceback.format_exc(limit=2)}"
        success = False

    finally:
        # ── [END] ─────────────────────────────────────────────────────────────
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step} rewards={rewards_str}",
            flush=True
        )

    return {
        "task_id":   task_id,
        "task_name": task_name,
        "success":   success,
        "steps":     step,
        "rewards":   rewards,
        "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = []
    for task_id in [1, 2, 3]:
        result = run_episode(task_id)
        results.append(result)

    # Summary to stderr so it doesn't contaminate the structured stdout
    print("\n=== SUMMARY ===", file=sys.stderr)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  [{status}] Task {r['task_id']} ({r['task_name']}): "
            f"steps={r['steps']} avg_reward={r['avg_reward']:.4f}",
            file=sys.stderr
        )
