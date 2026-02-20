# Konfigurasyon

Tum ayarlar `.env` dosyasi ile merkezi olarak yonetilir. Kod icinde hardcoded deger yoktur.

## LLM Backend

```ini
LLM_BACKEND=openai   # openai | trendyol | vllm
```

| Deger | Model | Gereksinim | VRAM |
|-------|-------|------------|------|
| `openai` | gpt-4o-mini | OPENAI_API_KEY | 0 (cloud) |
| `vllm` | LLaMA-3.1-8B AWQ INT4 | CUDA GPU | ~5 GB |
| `trendyol` | Trendyol-LLM-8B-T1 | CUDA GPU | ~9 GB |

## API Anahtarlari

```ini
# OpenAI (cloud LLM kullanacaksan)
OPENAI_API_KEY=sk-...

# HuggingFace (model indirmek icin, opsiyonel)
HUGGINGFACEHUB_API_TOKEN=hf_...
```

!!! warning "Guvenlik"
    `.env` dosyasini **asla** git'e commitlemeyin. `.gitignore`'da zaten haric tutulmus olmalidir.

## LangSmith (Tracing)

```ini
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true       # true | false
LANGCHAIN_PROJECT=rag-dev       # LangSmith proje adi
```

Tracing istemiyorsaniz (PII, offline ortam):

```ini
LANGCHAIN_TRACING_V2=false
```

## Reranker Ayarlari

```ini
# Model secimi
RERANKER_MODEL=default           # default | fast | <huggingface-model>

# Hizli mod (basit sorgularda reranking'i atla)
RERANK_FAST_MODE=false           # true | false

# Cache ayarlari
RERANK_CACHE_TTL=600             # saniye (varsayilan: 10 dakika)
RERANK_CACHE_SIZE=100            # maksimum cache girisi
```

| Model | Boyut | Hiz | Accuracy |
|-------|-------|-----|----------|
| `default` (BAAI/bge-reranker-base) | ~400 MB | Normal | Yuksek |
| `fast` (cross-encoder/ms-marco-MiniLM-L-6-v2) | ~80 MB | %40-60 daha hizli | Orta |

## Qdrant

```ini
QDRANT_URL=http://localhost:6333   # Qdrant sunucu adresi
QDRANT_COLLECTION=rag_collection   # Collection adi
QDRANT_STARTUP_TIMEOUT=20          # saniye, servis kalkisi icin bekleme
QDRANT_RETRY_INTERVAL=1            # saniye, retry araligi
```

## Cihaz Secimi (Streamlit)

```ini
RAG_DEVICE=auto    # auto | cpu | cuda
```

`auto` modunda `torch.cuda.is_available()` ile otomatik tespit edilir.

## Tum Degiskenler (Ozet)

| Degisken | Varsayilan | Aciklama |
|----------|-----------|----------|
| `LLM_BACKEND` | `trendyol` | LLM secimi |
| `OPENAI_API_KEY` | — | OpenAI API anahtari |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant adresi |
| `QDRANT_COLLECTION` | `rag_collection` | Qdrant collection adi |
| `QDRANT_STARTUP_TIMEOUT` | `20` | Qdrant kalkisinda maksimum bekleme (sn) |
| `QDRANT_RETRY_INTERVAL` | `1` | Qdrant baglanti retry araligi (sn) |
| `RERANKER_MODEL` | `default` | Cross-encoder modeli |
| `RERANK_FAST_MODE` | `false` | Adaptive reranking skip |
| `RERANK_CACHE_TTL` | `600` | Cache TTL (saniye) |
| `RERANK_CACHE_SIZE` | `100` | Cache boyutu |
| `RAG_DEVICE` | `auto` | Cihaz secimi (Streamlit) |
| `LANGCHAIN_TRACING_V2` | — | LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | — | LangSmith proje adi |
