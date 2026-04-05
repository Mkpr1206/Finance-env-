"""
app.py — OpenEnv HTTP Server for PersonalFinanceEnv
Exposes POST /reset, POST /step, GET /state for the evaluator.
Also runs inference.py logic on startup for the hackathon scoring.
"""

import os
import json
import threading
import subprocess
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment import PersonalFinanceEnv, Action, Observation

app = FastAPI(title="PersonalFinanceEnv", version="1.0.0")

# Global env instance (evaluator uses task_id 1 by default)
_env: PersonalFinanceEnv | None = None
_current_task_id: int = 1


# ── Request models ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: int = 42


class StepRequest(BaseModel):
    action: dict


# ── Routes ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "env": "PersonalFinanceEnv", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    global _env, _current_task_id
    task_id = req.task_id if req else 1
    seed = req.seed if req else 42
    _current_task_id = task_id
    _env = PersonalFinanceEnv(task_id=task_id, seed=seed)
    obs = _env.reset()
    return JSONResponse(content=_obs_to_dict(obs))


@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    obs, reward, done, info = _env.step(action)
    return JSONResponse(content={
        "observation": _obs_to_dict(obs),
        "reward": reward.value,
        "done": done,
        "info": info,
    })


@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return JSONResponse(content=_env.state())


# ── Helper ─────────────────────────────────────────────────────

def _obs_to_dict(obs: Observation) -> dict:
    d = obs.model_dump()
    # Convert BudgetBucket objects to dicts
    d["budget_buckets"] = {
        k: {"name": v.name, "allocated": v.allocated, "spent": v.spent,
            "remaining": v.remaining, "utilization": v.utilization}
        for k, v in obs.budget_buckets.items()
    }
    return d


# ── Run inference.py in background on startup ──────────────────

def _run_inference():
    """Run inference.py in background so logs appear in HF Space."""
    try:
        subprocess.run(
            [sys.executable, "inference.py"],
            env=os.environ.copy(),
            timeout=1100,  # under 20min
        )
    except Exception as e:
        print(f"[inference background] {e}", flush=True)


@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=_run_inference, daemon=True)
    t.start()
