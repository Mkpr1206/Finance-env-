"""
inference.py — PersonalFinanceEnv Baseline Inference
Emits [START] / [STEP] / [END] to stdout. Never crashes.
"""

import os
import sys
import json
import traceback

from openai import OpenAI
from environment import PersonalFinanceEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "dummy")

if not HF_TOKEN or HF_TOKEN == "dummy":
    print("[WARN] HF_TOKEN not set — running in fallback mode", flush=True)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ENV_NAME   = "personal-finance"
TASK_NAMES = {1: "budget-easy", 2: "budget-medium", 3: "budget-hard"}
MAX_STEPS  = 30
LLM_OK     = True   # set False after first failure to avoid repeated 401/429


SYSTEM_PROMPT = """You manage a personal finance budget.
Respond with ONLY a JSON object — no markdown, no explanation:
{"action_type":"categorize"|"approve"|"reject"|"allocate"|"invest"|"pay_debt",
 "transaction_id":"<id>","category":"<cat>","from_bucket":"<b>","to_bucket":"<b>",
 "amount":<float>,"rationale":"<reason>"}
Categories: housing|food|transport|utilities|healthcare|entertainment|shopping|debt|other
Rules: categorize unlabeled first; approve essentials; reject non-essentials when cash<$800."""


def _fallback(obs) -> dict:
    if obs.pending_transactions:
        t = obs.pending_transactions[0]
        return {"action_type": "approve" if t.essential else "reject",
                "transaction_id": t.id, "rationale": "fallback safe action"}
    if obs.uncategorized_transactions:
        t = obs.uncategorized_transactions[0]
        return {"action_type": "categorize", "transaction_id": t.id,
                "category": "other", "rationale": "fallback categorize"}
    return {"action_type": "pay_debt", "amount": 50.0, "rationale": "fallback debt payment"}


def _call_llm(obs, step: int):
    global LLM_OK
    if not LLM_OK:
        return _fallback(obs), "LLM disabled after prior error"
    prompt = (
        f"Day {obs.day}/30 | Cash=${obs.cash_balance:.0f} | Debt=${obs.debt_balance:.0f}\n"
        f"Pending: {[f'{t.id}:{t.description}(${t.amount})essential={t.essential}' for t in obs.pending_transactions]}\n"
        f"Unlabeled: {[f'{t.id}:{t.description}(${t.amount})' for t in obs.uncategorized_transactions]}\n"
        "Choose ONE action JSON."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": prompt}],
            temperature=0.1, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip()), None
    except Exception as e:
        LLM_OK = False
        return _fallback(obs), f"LLM error: {e}"


def run_episode(task_id: int) -> dict:
    global LLM_OK
    LLM_OK = True   # reset per episode

    env     = PersonalFinanceEnv(task_id=task_id, seed=42)
    rewards = []
    step    = 0
    success = False

    print(f"[START] task={TASK_NAMES[task_id]} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset()
        while step < MAX_STEPS:
            step += 1
            action_dict, err = _call_llm(obs, step)
            try:
                action     = Action(**action_dict)
                action_str = json.dumps(action_dict, separators=(",", ":"))
            except Exception:
                action_dict = _fallback(obs)
                action      = Action(**action_dict)
                action_str  = json.dumps(action_dict, separators=(",", ":"))
                err         = "action validation failed, used fallback"

            obs, reward, done, _ = env.step(action)
            rewards.append(reward.value)

            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward.value:.2f} done={'true' if done else 'false'} "
                f"error={'null' if err is None else err}",
                flush=True
            )
            if done:
                break
        success = True
    except Exception as e:
        print(f"[ERROR] {traceback.format_exc(limit=1)}", flush=True)
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step} rewards={rewards_str}",
            flush=True
        )

    return {"task_id": task_id, "success": success,
            "steps": step, "rewards": rewards}


if __name__ == "__main__":
    for task_id in [1, 2, 3]:
        run_episode(task_id)
