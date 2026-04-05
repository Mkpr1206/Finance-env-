#!/usr/bin/env python3
"""
validate.py — Pre-submission validation script.

Run this before submitting to catch all common failure cases.
Usage: python validate.py
"""

import os
import sys
import json
import importlib

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

errors = []
warnings = []


def check(label: str, condition: bool, error_msg: str = "", warn: bool = False):
    if condition:
        print(f"{PASS} {label}")
    else:
        tag = WARN if warn else FAIL
        print(f"{tag} {label}")
        if error_msg:
            print(f"     → {error_msg}")
        if warn:
            warnings.append(label)
        else:
            errors.append(label)


print("=" * 58)
print("  PersonalFinanceEnv — Pre-Submission Validator")
print("=" * 58)

# ── 1. File structure ─────────────────────────────────────────
print("\n[1/5] File Structure")
required_files = ["inference.py", "environment.py", "graders.py",
                  "openenv.yaml", "Dockerfile", "requirements.txt"]
for f in required_files:
    check(f"File exists: {f}", os.path.exists(f),
          f"Create {f} in root directory")

# ── 2. Environment variables ──────────────────────────────────
print("\n[2/5] Environment Variables")

# Check defaults exist in inference.py
with open("inference.py", "r", encoding="utf-8") as fh:
    src = fh.read()
check("API_BASE_URL has default value",
      'API_BASE_URL = os.getenv("API_BASE_URL",' in src or
      "API_BASE_URL = os.getenv('API_BASE_URL'," in src,
      'inference.py must set API_BASE_URL with a default, e.g. os.getenv("API_BASE_URL", "https://api.openai.com/v1")')
check("MODEL_NAME has default value",
      'MODEL_NAME = os.getenv("MODEL_NAME",' in src or
      "MODEL_NAME = os.getenv('MODEL_NAME'," in src,
      'inference.py must set MODEL_NAME with a default, e.g. os.getenv("MODEL_NAME", "gpt-4.1-mini")')
check("HF_TOKEN is read (no default required)",
      "HF_TOKEN" in src,
      "inference.py must read HF_TOKEN from environment")

hf_token = os.getenv("HF_TOKEN")
check("HF_TOKEN set in current shell", hf_token is not None,
      "export HF_TOKEN=<your-key> before running inference.py", warn=True)

# ── 3. OpenEnv compliance ─────────────────────────────────────
print("\n[3/5] OpenEnv Spec Compliance")
try:
    from environment import PersonalFinanceEnv, Action, Observation, Reward
    check("environment.py imports cleanly", True)

    env = PersonalFinanceEnv(task_id=1, seed=42)
    obs = env.reset()
    check("reset() returns Observation", isinstance(obs, Observation))

    action = Action(action_type="reject", transaction_id="t09",
                    rationale="Test rejection of Netflix subscription")
    result = env.step(action)
    check("step() returns 4-tuple (obs, reward, done, info)", len(result) == 4)
    obs2, rew, done, info = result
    check("Reward value in [-1.0, 1.0]", -1.0 <= rew.value <= 1.0,
          f"Got reward={rew.value}")
    check("done is bool", isinstance(done, bool))

    state = env.state()
    check("state() returns dict", isinstance(state, dict))
    for key in ["task_id", "day", "cash", "savings_rate", "done"]:
        check(f"state() has key '{key}'", key in state)

except Exception as e:
    check("environment.py loads and runs", False, str(e))

# ── 4. Graders ────────────────────────────────────────────────
print("\n[4/5] Task Graders (3 tasks)")
try:
    from graders import grade_task, PASS_THRESHOLDS
    check("graders.py imports cleanly", True)

    for task_id in [1, 2, 3]:
        try:
            # Minimal valid action set
            acts = [
                {"action_type": "reject", "transaction_id": "t09",
                 "rationale": "Non-essential discretionary expense"},
                {"action_type": "invest", "amount": 50.0,
                 "rationale": "Small investment of surplus"},
            ]
            r = grade_task(task_id, acts)
            in_range = 0.0 <= r["score"] <= 1.0
            check(f"Task {task_id} score in [0.0, 1.0]: {r['score']:.3f}", in_range,
                  f"score={r['score']}")
            check(f"Task {task_id} has pass_threshold", "pass_threshold" in r)
        except Exception as e:
            check(f"Task {task_id} grades without error", False, str(e))

except Exception as e:
    check("graders.py loads", False, str(e))

# ── 5. inference.py output format ────────────────────────────
print("\n[5/5] inference.py Output Format")
check("[START] line format present in code",
      "[START]" in src and "task=" in src and "env=" in src and "model=" in src,
      "inference.py must print [START] task=X env=Y model=Z")
check("[STEP] line format present in code",
      "[STEP]" in src and "step=" in src and "reward=" in src and "done=" in src,
      "inference.py must print [STEP] step=N action=X reward=0.00 done=false error=null")
check("[END] line format present in code",
      "[END]" in src and "success=" in src and "rewards=" in src,
      "inference.py must print [END] success=true steps=N rewards=r1,r2,...")
check("OpenAI client used (not raw HTTP)",
      "from openai import OpenAI" in src or "import openai" in src,
      "Must use the openai Python package")
check("flush=True on print statements",
      "flush=True" in src,
      "Add flush=True to print() calls so logs appear in real time")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 58)
if not errors:
    print(f"  ✅ ALL CHECKS PASSED  ({len(warnings)} warning(s))")
    print("  You're ready to submit!")
else:
    print(f"  ❌ {len(errors)} check(s) FAILED, {len(warnings)} warning(s)")
    print("  Fix these before submitting:")
    for e in errors:
        print(f"    • {e}")
if warnings:
    print("  Warnings (non-blocking):")
    for w in warnings:
        print(f"    • {w}")
print("=" * 58)

sys.exit(1 if errors else 0)
