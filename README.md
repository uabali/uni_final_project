# Agentic RAG Pipeline

> LangGraph ReAct agent + vLLM + Qdrant tabanlı akıllı doküman soru-cevap sistemi.

## Özellikler

- **Agentic RAG**: LangGraph StateGraph ile ReAct pattern — agent kendi tool seçimini yapar
- **Tool Calling**: `search_documents` (yerel), `web_search` (Tavily), `calculator`
- **Hybrid Retrieval**: Vektör + BM25 (RRF fusion), cross-encoder reranking
- **Multi-Provider LLM**: vLLM (lokal), OpenAI, LiteLLM (100+ provider) — runtime değiştirilebilir
- **Multi-Layer Memory**: Redis (short-term) + Qdrant (episodic) + Postgres (summary)
- **Incremental Ingestion**: Hash-based dedup ile doküman ekleme/güncelleme
- **FastAPI Gateway**: SSE streaming + senkron chat + dosya yükleme
- **Next.js Frontend**: Modern chat arayüzü (RAG toggle, markdown render, source chips)
- **Benchmark Runner**: Latency, throughput ve GPU ölçümleri
- **MCP Desteği**: Model Context Protocol ile harici tool entegrasyonu (opsiyonel)

---

## Mimari

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Frontend    │───▶│  FastAPI API  │───▶│  LangGraph Agent│
│  (Next.js)   │    │  Gateway      │    │  (ReAct Pattern) │
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                │
                   ┌────────────────────────────┼────────────────┐
                   │                            │                │
            ┌──────▼──────┐  ┌─────────▼────────┐  ┌───────▼───────┐
            │ search_docs  │  │   web_search      │  │  calculator    │
            │ (Qdrant+BM25)│  │   (Tavily)        │  │  (AST eval)    │
            └──────────────┘  └──────────────────┘  └───────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
  ┌────▼────┐ ┌────▼────┐ ┌───▼────┐
  │ Redis   │ │ Qdrant  │ │Postgres│
  │ Memory  │ │ Vectors │ │Summary │
  └─────────┘ └─────────┘ └────────┘
```

---

## Hızlı Başlangıç

### Gereksinimler

- Python 3.11+
- NVIDIA GPU (CUDA destekli, en az 12GB VRAM önerilir)
- Docker & Docker Compose

### 1. Bağımlılıkları Kur

```bash
# uv ile (önerilen)
uv sync

# veya pip ile
pip install .
```

### 2. Servisleri Başlat

```bash
# Qdrant, Redis, Postgres
docker compose up -d qdrant redis postgres

# vLLM server (ayrı terminal)
./scripts/serve_vllm.sh
```

### 3. Ortam Değişkenlerini Ayarla

```bash
cp .env.example .env
# .env dosyasını düzenleyerek API anahtarlarınızı ekleyin
```

### 4. Çalıştır

```bash
# CLI modu
python main.py

# API Gateway
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Frontend (ayrı terminal)
cd frontend && npm run dev
```

---

## Ortam Değişkenleri

Tüm konfigürasyon `.env` dosyasından okunur. Şablon için `.env.example` dosyasına bakın.

| Değişken | Açıklama | Varsayılan |
|----------|----------|-----------|
| `VLLM_SERVER_URL` | vLLM OpenAI-compat endpoint | `http://localhost:6365/v1` |
| `VLLM_MODEL` | vLLM'de serve edilen model | `Qwen/Qwen3-8B-AWQ` |
| `QDRANT_URL` | Qdrant vector DB adresi | `http://localhost:6333` |
| `REDIS_URL` | Redis memory store | `redis://localhost:6379/0` |
| `MEMORY_ENABLED` | Multi-layer memory aktif? | `true` |
| `RAG_RETRIEVAL_STRATEGY` | Arama stratejisi | `hybrid` |
| `TAVILY_API_KEY` | Web arama API anahtarı | — |

---

## Proje Yapısı

```
├── api/app.py              # FastAPI gateway (SSE streaming, sync, ingestion)
├── main.py                 # CLI giriş noktası
├── src/
│   ├── agent.py            # LangGraph ReAct agent (StateGraph)
│   ├── app_orchestrator.py # Uygulama düzenleyicisi (pipeline build)
│   ├── config.py           # Model/LLM konfigürasyonu
│   ├── llm.py              # vLLM ChatOpenAI wrapper
│   ├── llm_provider.py     # Multi-provider soyutlaması (vLLM/OpenAI/LiteLLM)
│   ├── loader.py           # PDF/TXT doküman yükleme
│   ├── memory.py           # Multi-layer memory (Redis/Qdrant/Postgres)
│   ├── prompting.py        # Prompt template'leri
│   ├── query_translation.py# Multi-query tekniği
│   ├── reranker.py         # Cross-encoder reranking + cache
│   ├── retriever.py        # Hybrid retrieval (BM25 + vektör)
│   ├── splitter.py         # Doküman parçalama
│   ├── tooling.py          # MCP + Local tool invocation
│   ├── tools.py            # Agent tool tanımları
│   ├── tracing.py          # Observability abstraction
│   └── vectorstore.py      # Qdrant vektör DB yönetimi
├── frontend/               # Next.js chat arayüzü
├── config/                 # LiteLLM proxy yapılandırması
├── scripts/
│   ├── serve_vllm.sh       # vLLM server başlatma scripti
│   ├── benchmark.py        # Performans benchmark runner
│   └── reset_qdrant.py     # Qdrant collection sıfırlama
├── docker-compose.yml      # Tüm servislerin orkestrasyonu
├── Dockerfile              # Orchestrator container
└── pyproject.toml          # Python bağımlılıkları
```

---

## Benchmark

```bash
# Agent modu (varsayılan)
python scripts/benchmark.py --dataset benchmarks/benchmark_tr_load.jsonl --runs 3

# Pipeline modu (TTFT ölçümü)
python scripts/benchmark.py --dataset benchmarks/benchmark_tr_load.jsonl --mode pipeline --stream

# Detaylı kullanım
python scripts/benchmark.py --help
```

---

## Docker Compose (Tam Kurulum)

```bash
# Temel servisler (Qdrant, Redis, Postgres)
docker compose up -d

# Tüm servisler dahil (LiteLLM proxy, SearXNG)
docker compose --profile full up -d
```

---

## Lisans

Bu proje kişisel/akademik kullanım içindir.
