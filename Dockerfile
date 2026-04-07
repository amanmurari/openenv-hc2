FROM python:3.11-slim

LABEL maintainer="OpenEnv Hackathon"
LABEL description="Autonomous Traffic Control – OpenEnv Environment (self-contained)"

WORKDIR /app

# ── Install system deps ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Copy packaging manifest first (layer-cache friendly) ──────────────────
COPY pyproject.toml README.md ./

# ── Install Python deps ────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "openai>=1.0.0" \
    "gradio>=4.0.0" \
    "numpy>=1.24.0" \
    "python-dotenv>=1.0.0" \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.5.0" \
    "requests>=2.31.0" \
    "python-multipart>=0.0.6"

# ── Copy source (self-contained; no root-level deps needed) ───────────────
COPY models.py ./traffic_control/models.py
COPY environment.py ./traffic_control/environment.py
COPY tasks.py ./traffic_control/tasks.py
COPY client.py ./traffic_control/client.py
COPY __init__.py ./traffic_control/__init__.py
COPY inference.py ./traffic_control/inference.py
COPY openenv.yaml ./traffic_control/openenv.yaml
COPY server/ ./traffic_control/server/

# ── Create package anchor so Python treats traffic_control/ as the package ─
RUN echo "" > ./traffic_control/__init__.py || true

# ── Install the package in editable mode ──────────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Runtime config ────────────────────────────────────────────────────────
ENV WORKERS=2
ENV PORT=8000
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sh", "-c", \
     "uvicorn traffic_control.server.app:app \
      --host 0.0.0.0 \
      --port ${PORT} \
      --workers ${WORKERS}"]
