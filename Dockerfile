FROM python:3.11-slim

LABEL maintainer="OpenEnv Hackathon"
LABEL description="Autonomous Traffic Control – OpenEnv Environment (self-contained)"

# ── Install system deps (as root) ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Create user for Hugging Face Spaces (UID 1000) ─────────────────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# ── Copy packaging manifest first (layer-cache friendly) ──────────────────
COPY --chown=user pyproject.toml README.md ./

# Fix the setuptools package lookup to not scan the root filesystem in Docker
RUN sed -i 's|where = \[".."\]|where = \["."\]|g' pyproject.toml || true

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
COPY --chown=user models.py ./traffic_control/models.py
COPY --chown=user environment.py ./traffic_control/environment.py
COPY --chown=user tasks.py ./traffic_control/tasks.py
COPY --chown=user client.py ./traffic_control/client.py
COPY --chown=user __init__.py ./traffic_control/__init__.py
COPY --chown=user inference.py ./traffic_control/inference.py
COPY --chown=user openenv.yaml ./traffic_control/openenv.yaml
COPY --chown=user server/ ./traffic_control/server/

# inference.py must also be at the container root (validator requirement)
COPY --chown=user inference.py ./inference.py

# ── Create package anchor so Python treats traffic_control/ as the package ─
RUN echo "" > ./traffic_control/__init__.py || true

# ── Install the package in editable mode ──────────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Runtime config ────────────────────────────────────────────────────────
ENV PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Single worker is required: Gradio's mount_gradio_app is not fork-safe.
# HF Spaces expects a single healthy process on port 7860.
CMD ["sh", "-c", \
     "uvicorn traffic_control.server.app:app \
      --host 0.0.0.0 \
      --port ${PORT} \
      --workers 1 \
      --timeout-keep-alive 75"]
