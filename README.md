---
title: Personal Finance Manager
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---
---
title: Personal Finance Manager
emoji: ??
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---
# PersonalFinanceEnv

> **OpenEnv RL Hackathon** Â· Real-world task simulation Â· Personal Budget Management

An AI agent manages a simulated monthly budget: categorizing transactions, approving or rejecting discretionary expenses, reallocating budget categories, and directing money toward savings and debt reduction â€” all to maximize savings rate.

---

## Quick Start

```bash
pip install -r requirements.txt

# Validate everything before submitting
python validate.py

# Run grader self-test (no API key needed)
python graders.py

# Run baseline inference (needs HF_TOKEN)
export HF_TOKEN=sk-...
export API_BASE_URL=https://api.openai.com/v1    # optional, has default
export MODEL_NAME=gpt-4.1-mini                   # optional, has default
python inference.py
```

---

## File Structure

```
.
â”œâ”€â”€ inference.py     â† MAIN ENTRY POINT (required by spec)
â”œâ”€â”€ environment.py   â† PersonalFinanceEnv (OpenEnv compliant)
â”œâ”€â”€ graders.py       â† Deterministic graders for all 3 tasks
â”œâ”€â”€ validate.py      â† Pre-submission checklist
â”œâ”€â”€ openenv.yaml     â† Environment metadata
â”œâ”€â”€ requirements.txt â† pydantic, openai, pyyaml
â”œâ”€â”€ Dockerfile       â† Container definition
â””â”€â”€ README.md        â† This file
```

---

## Environment Overview

### Action Types

| `action_type` | Required fields | What it does |
|---|---|---|
| `categorize` | `transaction_id`, `category` | Label an unlabeled transaction |
| `approve` | `transaction_id` | Book a pending expense |
| `reject` | `transaction_id` | Cancel a pending expense (money saved) |
| `allocate` | `from_bucket`, `to_bucket`, `amount` | Shift budget between categories |
| `invest` | `amount` | Move surplus cash to investments |
| `pay_debt` | `amount` | Make an extra debt principal payment |

### Categories
`housing` Â· `food` Â· `transport` Â· `utilities` Â· `healthcare` Â· `entertainment` Â· `shopping` Â· `savings` Â· `investment` Â· `debt` Â· `other`

---

## Tasks

| Task | Name | Transactions | Special | Threshold |
|---|---|---|---|---|
| 1 | `budget-easy` | 9 | None | 0.60 |
| 2 | `budget-medium` | 16 | Budget pressure | 0.55 |
| 3 | `budget-hard` | 21 | Emergency shock ($650) | 0.50 |

---

## Reward Function

| Action | Reward |
|---|---|
| Correct categorization | +0.50 |
| Wrong categorization | âˆ’0.10 |
| Approve essential expense | +0.30 |
| Reject non-essential (smart) | +0.40 |
| Reject essential expense | **âˆ’0.50** |
| Approve over-budget category | âˆ’0.30 |
| Smart reallocation | +0.20â€“0.40 |
| Invest surplus | up to +0.50 |
| Extra debt payment | up to +0.50 |
| Rationale quality bonus | +0.10 |

### Score Formula
```
score = 0.40 Ã— step_quality
      + 0.30 Ã— savings_rate    (target: 20%)
      + 0.20 Ã— completion      (items handled)
      + 0.10 Ã— debt_progress
```

---

## inference.py Output Format

```
[START] task=budget-easy env=personal-finance model=gpt-4.1-mini
[STEP] step=1 action={"action_type":"reject","transaction_id":"t09",...} reward=0.50 done=false error=null
[STEP] step=2 action={"action_type":"categorize",...} reward=0.60 done=false error=null
[END] success=true steps=12 rewards=0.50,0.60,...
```

---

## Docker

```bash
docker build -t finance-env .
docker run --env HF_TOKEN=$HF_TOKEN finance-env
# Optional overrides:
docker run --env HF_TOKEN=$HF_TOKEN \
           --env API_BASE_URL=https://api.openai.com/v1 \
           --env MODEL_NAME=gpt-4o-mini \
           finance-env
```

---

## Environment Variables

| Variable | Default | Required |
|---|---|---|
| `HF_TOKEN` | â€” | **Yes** |
| `API_BASE_URL` | `https://api.openai.com/v1` | No |
| `MODEL_NAME` | `gpt-4.1-mini` | No |


