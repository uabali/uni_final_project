# RAG Orchestrator — FastAPI + LangGraph
FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları + proje
COPY pyproject.toml uv.lock* ./
COPY src/ src/
COPY api/ api/
COPY main.py ./
COPY config/ config/
RUN pip install uv && uv sync --no-dev

# Port
EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
