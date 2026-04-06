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
    try:
        action_dict = action.dict()

        if action_dict.get("amount") is None:
            action_dict["amount"] = 0.0

        obs, reward, done, info = env.step(action_dict)

        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/state")
def state():
    return env.state()


@app.get("/")
def home():
    return {"status": "Server running", "endpoints": ["/reset", "/step", "/state"]}


import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()