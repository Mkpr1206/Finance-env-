"""
server/app.py — OpenEnv HTTP Server for PersonalFinanceEnv
Exposes the standard OpenEnv endpoints:
  POST /reset  → resets env, returns initial observation
  POST /step   → takes one action, returns obs/reward/done/info
  GET  /state  → returns full internal state
"""

from __future__ import annotations
import os
import sys
import json
import threading
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent dir so we can import environment from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import PersonalFinanceEnv, Action, Observation

app = FastAPI(title="PersonalFinanceEnv", version="1.0.0")

_env: PersonalFinanceEnv | None = None


# ── Request schemas ────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: int = 42


class StepRequest(BaseModel):
    action: dict


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "env": "PersonalFinanceEnv", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    global _env
    task_id = req.task_id if req else 1
    seed    = req.seed    if req else 42
    _env = PersonalFinanceEnv(task_id=task_id, seed=seed)
    obs  = _env.reset()
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
        "reward":      reward.value,
        "done":        done,
        "info":        info,
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
    d["budget_buckets"] = {
        k: {
            "name":        v.name,
            "allocated":   v.allocated,
            "spent":       v.spent,
            "remaining":   v.remaining,
            "utilization": v.utilization,
        }
        for k, v in obs.budget_buckets.items()
    }
    return d


# ── Run inference.py in background on startup ──────────────────

def _run_inference():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(
            [sys.executable, os.path.join(root, "inference.py")],
            env=os.environ.copy(),
            timeout=1100,
        )
    except Exception as e:
        print(f"[inference background] {e}", flush=True)


@app.on_event("startup")
def startup_event():
    threading.Thread(target=_run_inference, daemon=True).start()
