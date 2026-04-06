from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment import PersonalFinanceEnv

app = FastAPI()

env = PersonalFinanceEnv()

class Action(BaseModel):
    action_type: str
    amount: float | None = None
    transaction_id: str | None = None
    category: str | None = None


@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs}


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action.dict())
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env.state()