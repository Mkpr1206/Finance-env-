"""
server/app.py — PersonalFinanceEnv FastAPI server
Uses openenv_core.env_server.create_app as required by the OpenEnv validator.
Entry point: uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import os
from openenv_core.env_server import create_app
from server.environment import PersonalFinanceEnv, FinanceAction, FinanceObservation

# Instantiate env — task_id can be overridden via env var
task_id = int(os.getenv("TASK_ID", "1"))
seed    = int(os.getenv("SEED", "42"))

env = PersonalFinanceEnv(task_id=task_id, seed=seed)

# create_app wires up /reset, /step, /state, /health automatically
app = create_app(env, FinanceAction, FinanceObservation)
