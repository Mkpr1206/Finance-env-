FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Validate imports at build time
RUN python -c "from environment import PersonalFinanceEnv; print('Environment OK')"
RUN python -c "from graders import grade_task; print('Graders OK')"

EXPOSE 7860

# Run the FastAPI server (evaluator pings /reset and /step)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
