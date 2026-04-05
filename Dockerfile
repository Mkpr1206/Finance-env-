FROM python:3.11-slim

WORKDIR /app

# Install uv (fast package manager required by openenv)
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache)
COPY pyproject.toml uv.lock* ./

# Install all dependencies via uv
RUN uv pip install --system --no-cache -r pyproject.toml 2>/dev/null || \
    pip install --no-cache-dir openenv-core>=0.2.0 pydantic>=2.0 openai>=1.0 \
    pyyaml>=6.0 fastapi>=0.100.0 uvicorn>=0.23.0

# Copy source
COPY . .

# Install package itself
RUN pip install --no-cache-dir -e . 2>/dev/null || true

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
