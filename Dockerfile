# ── Builder / base stage ──────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System packages required for some pretty_midi / numpy native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first so pip layer is cached independently of source changes
COPY requirements.txt pyproject.toml ./

# Install CPU-only PyTorch (smaller image; swap the index URL for a CUDA wheel if needed)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY configs/ ./configs/
COPY src/     ./src/
COPY api/     ./api/
COPY app.py infer.py train.py ./

# Install the package so that `from configs.config import …` works without sys.path hacks
RUN pip install --no-cache-dir -e .

# Create runtime directories (checkpoints / outputs are mounted at runtime)
RUN mkdir -p checkpoints outputs/midi_examples plots logs

# ── Streamlit target (default) ────────────────────────────────────
FROM base AS streamlit
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]

# ── FastAPI target ────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
