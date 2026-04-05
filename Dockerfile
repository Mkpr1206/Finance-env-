FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY environment.py graders.py inference.py openenv.yaml README.md ./

# Validate on build (catches import errors early)
RUN python -c "from environment import PersonalFinanceEnv; print('Environment OK')"
RUN python -c "from graders import grade_task; print('Graders OK')"

# Default: run inference (requires HF_TOKEN at runtime)
CMD ["python", "inference.py"]
