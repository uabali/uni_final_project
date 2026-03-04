<div align="center">

# 🤖 Enterprise Agentic RAG & AI Ops Center

**FastAPI · LangGraph · vLLM · Qdrant · Redis · Postgres · Next.js · MCP**

A GPU-accelerated, multi-agent RAG system that runs with a single `docker compose up`.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![vLLM](https://img.shields.io/badge/vLLM-Qwen3--8B--AWQ-orange.svg)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## Overview

This repository is a production-ready **AI Operations Center** prototype featuring:

- **FastAPI Orchestrator** — LangGraph-based multi-agent RAG pipeline with SSE streaming, incremental document ingestion, runtime LLM provider switching, and metrics.
- **vLLM** — OpenAI-compatible local LLM server (Qwen3 / Qwen2.5 family, AWQ quantized).
- **Qdrant** — Multi-tenant vector database with optional strict namespace isolation.
- **Redis + Postgres** — Multi-tier memory (short-term buffer, episodic, entity, conversation summaries) + append-only audit log + departmental metrics.
- **Next.js Frontend** — Modern chat UI with RAG/Web/MCP modes, file upload, department & model selector, agent status panel, and monitoring dashboard.
- **MCP Tooling** — Optional Model Context Protocol integration for filesystem, database, and memory tools.

## Architecture

<div align="center">

![Architecture](docs/architecture.png)

</div>

### Agent Pipeline

```
User Query
  │
  ▼
┌─────────────────────────┐
│   Supervisor Agent      │ ← Routes based on message prefix
│   (LangGraph ReAct)     │
└────────┬────────┬───────┘
         │        │
    ┌────▼──┐  ┌──▼────────┐
    │Research│  │Execution  │ ← [MCP] prefix → Execution
    │ Agent  │  │  Agent    │ ← Otherwise → Research
    └────┬───┘  └──┬────────┘
         │         │
  ┌──────▼──────┐  ├─ read_file
  ├─ search_docs│  ├─ execute_python
  ├─ web_search │  ├─ query_postgres
  └─ calculator │  └─ run_bash
                │
        ┌───────▼────────┐
        │ Hybrid Retrieval│
        │ Vector + BM25   │
        │ + Cross-Encoder │
        │   Reranking     │
        └─────────────────┘
```

---

## 1. Quick Start (Docker Compose)

### 1.1 Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Linux / WSL2 (recommended for GPU passthrough) |
| **Docker** | Docker Engine + Docker Compose v2 |
| **GPU** | NVIDIA GPU with CUDA support (≥ 12 GB VRAM recommended) |
| **Python** | 3.11+ (for local development and scripts) |

### 1.2 Launch All Services

```bash
# Clone and enter the project
cd vLLM_rag

# Core stack (Orchestrator + vLLM + Qdrant + Redis + Postgres)
docker compose up -d

# Full stack with optional services (LiteLLM proxy, SearXNG, etc.)
docker compose --profile full up -d
```

### Default Ports

| Port | Service | Description |
|------|---------|-------------|
| `:8000` | FastAPI Orchestrator | `/chat`, `/ingest`, `/config/llm`, `/metrics/summary` |
| `:8080` | vLLM | OpenAI-compatible LLM endpoint |
| `:6333` | Qdrant | Vector database |
| `:6379` | Redis | Memory cache (short-term + entity) |
| `:5432` | Postgres | Audit log + conversation summaries |
| `:4000` | LiteLLM Proxy | Multi-provider router (optional, `full` profile) |
| `:8888` | SearXNG | Self-hosted search engine (optional) |

### 1.3 Launch Frontend

The frontend runs as a separate Next.js dev server:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.  
The frontend communicates with the backend via `NEXT_PUBLIC_API_URL` (default: `http://localhost:8000`).

### 1.4 Verify Health

```bash
curl http://localhost:8000/health | jq
```

Expected response:

```json
{
  "status": "healthy",
  "vectorstore": true,
  "memory": true,
  "provider": "VllmProvider",
  "model": "Qwen/Qwen3-8B-AWQ",
  "uptime_seconds": 42.5
}
```

---

## 2. Local Development

### 2.1 Install Dependencies

The project uses [`uv`](https://github.com/astral-sh/uv) as package manager:

```bash
# Recommended
uv sync

# Or with pip
pip install .
```

### 2.2 Run Orchestrator Locally

If you prefer to run the orchestrator directly on the host (while infra services stay in containers):

```bash
# Start only infrastructure services
docker compose up -d qdrant redis postgres vllm

# Run the orchestrator
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Default connection URLs (set in `.env`):

```
VLLM_SERVER_URL=http://localhost:8080/v1
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
POSTGRES_URL=postgresql://rag:rag@localhost:5432/rag
```

### 2.3 CLI Mode

For quick testing without the API server:

```bash
python main.py
```

This starts an interactive CLI loop with the full agent pipeline.

---

## 3. Configuration

All configuration is managed through environment variables. Copy the example file to get started:

```bash
cp .env.example .env
```

### Key Environment Variables

<details>
<summary><b>LLM & Providers</b></summary>

| Variable | Default | Description |
|---|---|---|
| `VLLM_SERVER_URL` | — | vLLM OpenAI-compatible endpoint (required) |
| `VLLM_MODEL` | `Qwen/Qwen3-8B-AWQ` | Model name served by vLLM |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_MAX_TOKENS` | `512` | Maximum generation tokens |
| `LLM_TOP_P` | `0.95` | Top-p nucleus sampling |
| `LLM_ENABLE_THINKING` | `false` | Enable Qwen3 thinking mode |
| `OPENAI_API_KEY` | — | OpenAI API key (for provider switching) |
| `LITELLM_BASE_URL` | `http://localhost:4000/v1` | LiteLLM proxy endpoint |

</details>

<details>
<summary><b>Retrieval & Reranking</b></summary>

| Variable | Default | Description |
|---|---|---|
| `RAG_RETRIEVAL_STRATEGY` | `hybrid` | Strategy: `hybrid` / `similarity` / `mmr` / `threshold` / `auto` |
| `RAG_BASE_K` | `8` | Base number of chunks to retrieve |
| `RAG_BM25_WEIGHT` | `0.4` | BM25 weight in hybrid search (RRF) |
| `RAG_CHUNK_SIZE` | `1000` | Document chunk size (characters) |
| `RAG_CHUNK_OVERLAP` | `200` | Chunk overlap |
| `RERANKER_MODEL` | `default` | Reranker model: `default` (BAAI/bge-reranker-base) / `fast` / custom HF model |
| `RERANK_FAST_MODE` | `false` | Skip reranking for simple queries |
| `LOCAL_SEARCH_CONF_THRESHOLD` | `0.35` | Confidence threshold for local search results |

</details>

<details>
<summary><b>Memory</b></summary>

| Variable | Default | Description |
|---|---|---|
| `MEMORY_ENABLED` | `false` | Enable multi-layer memory system |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `MEMORY_SHORT_TERM_TTL` | `7200` | Short-term memory TTL (seconds) |
| `MEMORY_SHORT_TERM_MAX_TURNS` | `20` | Max conversation turns in buffer |
| `SUMMARY_TRIGGER_TURNS` | `20` | Auto-generate summary after N turns |

</details>

<details>
<summary><b>Security & Multi-Tenancy</b></summary>

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | — | Static API key (optional, requires `X-API-Key` header) |
| `JWT_SECRET_KEY` | — | Enable JWT auth (requires `Authorization: Bearer` header) |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `RAG_MULTI_TENANT_STRICT` | `false` | Separate Qdrant collection per department |
| `MEMORY_MULTI_TENANT_STRICT` | `false` | Department-prefixed memory keys |
| `DEFAULT_DEPARTMENT_ID` | `default` | Fallback department for non-JWT requests |

</details>

<details>
<summary><b>Tools & MCP</b></summary>

| Variable | Default | Description |
|---|---|---|
| `TAVILY_API_KEY` | — | Tavily API key for web search |
| `MCP_SERVER_URL` | — | FastMCP server endpoint (optional) |
| `MCP_TIMEOUT` | `30` | MCP tool invocation timeout (seconds) |
| `MCP_ALLOWED_TOOLS` | — | Comma-separated whitelist of MCP tools |

</details>

---

## 4. Security & Multi-Tenancy

### 4.1 Authentication

The orchestrator supports two-layer authentication:

- **API Key** (optional, shared secret):
  - Set `API_KEY` env → all requests must include `X-API-Key` header.
- **JWT** (recommended for production):
  - Set `JWT_SECRET_KEY` and optionally `JWT_ALGORITHM`.
  - `Authorization: Bearer <token>` header becomes mandatory.
  - Expected JWT claims: `user_id` (or `sub`), `department_id` (or `dept`), `role`.

> **Local/Dev mode:** If `JWT_SECRET_KEY` is not set, JWT auth is disabled. Department info can be sent via the `X-Department-ID` header (the frontend does this automatically).

### 4.2 Department-Based RAG Isolation

Two isolation levels are available:

| Mode | Setting | Behavior |
|------|---------|----------|
| **Default** | (default) | Single Qdrant collection with `metadata.department_id` filter per query |
| **Strict** | `RAG_MULTI_TENANT_STRICT=true` | Separate collections per department (`rag_collection_engineering`, `rag_collection_finance`, etc.) |

### 4.3 Memory Isolation

The multi-tier memory system consists of:

| Layer | Backend | Scope |
|-------|---------|-------|
| **Short-term** | Redis | Last N turns, session-based |
| **Entity memory** | Redis hash | Per-user profile data |
| **Episodic memory** | Qdrant | Long-term "memories" |
| **Summaries** | Postgres | Conversation summaries |

With `MEMORY_MULTI_TENANT_STRICT=true`, all memory keys and collections are prefixed with the department ID.

---

## 5. Multi-Agent Orchestration

The backend uses LangGraph to implement a two-tier agent architecture:

### Supervisor Agent
- Analyzes incoming state and routes to the appropriate sub-agent.
- Messages starting with `[MCP]` or `[MCP:...]` → **Execution Agent**.
- All other messages → **Research Agent**.

### Research Agent
- **Tools:** `search_documents`, `web_search`, `calculator` (+ optional MCP tools)
- Full RAG + multi-layer memory + hybrid retrieval + reranking + ReAct loop.

### Execution Agent
- **Tools:** MCP-focused (`read_file`, `list_directory`, `write_file`, `execute_python`, `run_bash`, `query_postgres`, `memory_search`).
- Action-oriented system prompt focusing on tool calls and result reporting.

### Retrieval Pipeline

```
User Query
  ├── Dynamic K calculation (based on query complexity)
  ├── Auto strategy selection (similarity / mmr / hybrid / threshold)
  ├── Vector search (Qdrant, BAAI/bge-m3 embeddings)
  ├── BM25 keyword search (parallel)
  ├── RRF merge (Reciprocal Rank Fusion)
  ├── Cross-encoder reranking (BAAI/bge-reranker-base)
  └── Top-K selection with confidence scoring
```

---

## 6. API Reference

All endpoints are served at `http://localhost:8000`.

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | SSE streaming chat (agentic RAG) |
| `POST` | `/chat/sync` | Synchronous chat (returns full response) |

**Request body:**

```json
{
  "message": "Summarize the Q3 budget deviations",
  "use_rag": true,
  "llm_provider": null
}
```

**SSE events** (`/chat`):
- `event: token` → `{ "text": "..." }` — streaming text fragment
- `event: sources` → `[{ "chunk_id": 1, "source": "report.pdf", "page": "p.5", "snippet": "..." }]`
- `event: done` → `{ "session_id": "...", "latency_ms": 1234.56, "token_count": 128 }`

### Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Ingest documents by file path (incremental, hash-based dedup) |
| `POST` | `/ingest/upload` | Upload files (PDF/TXT) for ingestion |
| `POST` | `/ingest/delete` | Delete ingested documents |

### Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/config/llm` | Current LLM provider and model info |
| `PUT` | `/config/llm` | Switch LLM provider at runtime (vLLM / OpenAI / LiteLLM) |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/metrics/summary` | Department & agent metrics (from audit log) |
| `WS` | `/ws/tasks/{task_id}` | WebSocket for real-time task status updates |

---

## 7. Frontend Features

The frontend is built with **Next.js App Router** (located in `frontend/`):

- **Chat Interface**
  - RAG toggle (local documents vs. plain LLM)
  - "Search Web" mode (`[WEB_ONLY]` prefix)
  - MCP mode with preset commands (filesystem, Postgres, memory search)
  - File upload (PDF/TXT → ingestion pipeline)
  - Source citation panel (RAG chunks and web URLs)
  - Token count & latency badges

- **Department Selector**
  - Dropdown in header and settings
  - Selection maps to `X-Department-ID` header → enables RAG isolation + MCP policy engine

- **Agent Status Panel**
  - Sidebar widget showing real-time agent status (running / completed / error)
  - Powered by `/ws/tasks/{task_id}` WebSocket

- **Monitoring Tab**
  - Per-session metrics (turn count, total tokens, average latency)
  - LangSmith integration guidance
  - Department-level summary cards from `GET /metrics/summary`

---

## 8. Project Structure

```
vLLM_rag/
├── api/
│   ├── app.py              # FastAPI gateway (endpoints, SSE, WebSocket)
│   └── metrics.py          # Metrics API router
├── src/
│   ├── agent.py            # LangGraph agent (Supervisor + Research + Execution)
│   ├── app_orchestrator.py # Application orchestrator (builds the full pipeline)
│   ├── audit.py            # Append-only audit logger (Postgres)
│   ├── config.py           # Centralized configuration
│   ├── context.py          # Request context (JWT claims, department, session)
│   ├── llm.py              # vLLM ChatOpenAI wrapper
│   ├── llm_provider.py     # Multi-provider router (vLLM / OpenAI / LiteLLM)
│   ├── loader.py           # Document loader (PDF, TXT)
│   ├── memory.py           # Multi-layer memory (Redis + Qdrant + Postgres)
│   ├── policy.py           # Department × MCP permission matrix
│   ├── prompting.py        # RAG prompt templates
│   ├── query_translation.py# Multi-query generation
│   ├── reranker.py         # Cross-encoder reranking
│   ├── retriever.py        # Hybrid retrieval (Vector + BM25 + RRF)
│   ├── splitter.py         # Text splitting (recursive / semantic)
│   ├── tasks.py            # Task registry for long-running operations
│   ├── tooling.py          # MCP + Local tool invocation abstraction
│   ├── tools.py            # Agent tools (search_documents, web_search, calculator)
│   ├── tracing.py          # Observability abstraction
│   └── vectorstore.py      # Qdrant vector database operations
├── frontend/               # Next.js chat UI
├── scripts/
│   ├── benchmark.py        # Performance benchmark runner
│   ├── reset_qdrant.py     # Qdrant collection reset utility
│   └── serve_vllm.sh       # vLLM server startup script
├── config/
│   └── litellm-config.yaml # LiteLLM proxy configuration
├── main.py                 # CLI entry point
├── docker-compose.yml      # Full stack deployment
├── Dockerfile              # Orchestrator container
├── pyproject.toml          # Python dependencies
└── .env.example            # Environment variable template
```

---

## 9. Example Workflow

1. **Start the backend:**
   ```bash
   docker compose up -d
   ```

2. **Launch the frontend:**
   ```bash
   cd frontend && npm install && npm run dev
   ```

3. **Open the UI** at `http://localhost:3000`:
   - Enter your API key in Settings (if `API_KEY` is configured).
   - Select your department (e.g., Engineering).

4. **Upload documents** via the sidebar or message composer (PDF/TXT).

5. **Ask questions:**
   - `"Summarize the RAG pipeline from the architecture document."`
   - `"List the files under /data using the MCP filesystem preset."`

6. **Monitor** from Settings → Monitoring:
   - View active session metrics and department-level statistics.

---

## 10. Benchmarking

Run performance benchmarks against the full pipeline:

```bash
python scripts/benchmark.py \
  --dataset benchmarks/test_queries.csv \
  --mode agent \
  --runs 3 \
  --concurrency 1 \
  --use-rerank \
  --strategy auto \
  --output logs/benchmark_results.jsonl
```

The benchmark runner measures:
- End-to-end latency (p50, p95)
- Retrieval time
- LLM generation time
- Time to first token (TTFT, streaming mode)
- GPU utilization stats
- Throughput (requests/second)

---

## 11. Notes & Roadmap

- **Multi-tenant modes** (Qdrant strict + Memory strict) are fully **optional**; defaults are optimized for single-tenant dev environments.
- The **Supervisor** currently routes between Research and Execution agents; the architecture supports additional agents (Analytics, Document, Notification) with minimal changes.
- **Metrics API v1** provides simple aggregates from the Postgres audit table; materialized views and date filters can be added for advanced cost/latency dashboards.
- The **LLM provider** can be switched at runtime via `PUT /config/llm` without restarting the system (vLLM ↔ OpenAI ↔ LiteLLM with automatic fallback chains).

---

<div align="center">

**Built with** 🐍 Python · ⚡ FastAPI · 🦜 LangGraph · 🚀 vLLM · 🔍 Qdrant

</div>
