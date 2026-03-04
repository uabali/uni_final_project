# RAG Orchestrator — FastAPI + LangGraph
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Create virtual environment and install dependencies (uv manages under .venv)
RUN uv sync --no-dev

# Copy project source code
COPY src/ src/
COPY api/ api/
COPY config/ config/
COPY main.py ./

# Add .venv/bin to PATH (uv sync creates .venv by default)
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]