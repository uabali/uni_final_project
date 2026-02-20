# RAG Project

This repo contains a Retrieval-Augmented Generation (RAG) pipeline with:
- PDF/TXT ingestion + splitting
- Vector search (Qdrant) + optional BM25
- Optional reranker
- LLM backend: single local vLLM model (Meta-Llama-3.1-8B-Instruct-AWQ-INT4)
- Optional LangSmith tracing
- A separate benchmark runner

---

## 1) Quick Start

### Requirements
- Python 3.11+
- Qdrant running on `http://localhost:6333`

### Install deps
Using uv:
```
uv sync
```
Or pip (from `pyproject.toml`):
```
pip install .
```

### Start Qdrant (Docker)
```
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant
```

---

## 2) Environment (.env)

Keep API keys out of code. Put them in `.env`:
```
OPENAI_API_KEY=sk-...

LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-dev

# LLM backend is now fixed to a single local vLLM model (Meta-Llama-3.1-8B-Instruct-AWQ-INT4),
# so this variable is ignored if set.
# LLM_BACKEND=openai
```

Optional reranker settings:
```
RERANKER_MODEL=default   # default | fast | <hf model>
RERANK_FAST_MODE=false
RERANK_CACHE_TTL=600
RERANK_CACHE_SIZE=100
```

---

## 3) Run the App (normal usage)

```
python main.py
```

This is the interactive RAG app. Use this for everyday work.
If you are NOT benchmarking, you only need `main.py`.

---

## 4) Benchmark Runner

`scripts/benchmark.py` runs the same pipeline over a dataset and writes:
- `logs/benchmark_results.jsonl`
- `logs/benchmark_results_summary.json`

### Example: vLLM (local, fixed backend)
```
python scripts/benchmark.py --dataset data/benchmark.jsonl --runs 3 --warmup 5
```

If you want TTFT (time-to-first-token):
```
python scripts/benchmark.py --dataset data/benchmark.jsonl --runs 3 --warmup 5 --stream
```

GPU comparison example:
```
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py --dataset data/benchmark.jsonl --runs 3 --warmup 5 --gpu-label "GPU0"
CUDA_VISIBLE_DEVICES=1 python scripts/benchmark.py --dataset data/benchmark.jsonl --runs 3 --warmup 5 --gpu-label "GPU1"
```

Important:
- Do NOT run `main.py` while benchmarking (resource conflicts).
- Benchmark needs a dataset (`input` or `question` column).

---

## 5) LangSmith with Local LLM (vLLM)

LangSmith works as long as you use LangChain and have internet access.
Local or cloud LLM does not matter.

Set:
```
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
```

If you do NOT want tracing (PII, offline):
```
LANGCHAIN_TRACING_V2=false
```

---

## 6) BM25 Note

If you see:
```
Could not import rank_bm25
```
make sure `rank-bm25` is installed (now in deps). Re-run:
```
uv sync
```

---

## 7) Project Structure

- `main.py`: interactive RAG app
- `scripts/benchmark.py`: benchmark runner
- `src/loader.py`: PDF/TXT loading
- `src/splitter.py`: recursive/semantic split
- `src/vectorstore.py`: Qdrant vector store
- `src/retriever.py`: hybrid retrieval + rerank
- `src/reranker.py`: cross-encoder reranking + cache
- `src/prompting.py`: shared prompt + context formatting

---

## 8) Typical Flow

1) Start Qdrant
2) Run `python main.py` to confirm everything works
3) Run `python benchmark.py ...` only when you need measurements

