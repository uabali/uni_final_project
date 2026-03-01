# Projeyi Çalıştırma Rehberi

## Seçenek 1: Docker Compose (Önerilen — Tam Stack)

**Gereksinimler:** Docker, Docker Compose, NVIDIA GPU (vLLM için)

```bash
# Tüm servisleri başlat (vLLM, Qdrant, Redis, Postgres, LiteLLM, MCP, Orchestrator)
docker compose up -d

# Logları izle
docker compose logs -f orchestrator
```

**Portlar:**
- **8000** — API Gateway (FastAPI)
- **8080** — vLLM (LLM inference)
- **6333** — Qdrant (vektör DB)
- **6379** — Redis
- **5432** — Postgres
- **4000** — LiteLLM Proxy
- **8001** — MCP Server
- **8888** — SearXNG

**API testi:**
```bash
curl -X POST http://localhost:8000/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "Merhaba", "use_rag": true}'
```

**Frontend:** Next.js ayrı çalıştırılmalı (aşağıda).

---

## Seçenek 2: Lokal Geliştirme (Docker Olmadan)

### Adım 1: Bağımlılıkları yükle

```bash
uv sync
# veya: pip install -e .
```

### Adım 2: Altyapı servislerini başlat

En azından Qdrant ve Redis gerekli. Postgres opsiyonel (memory summary için):

```bash
# Sadece Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant qdrant/qdrant

# Redis
docker run -d -p 6379:6379 --name redis redis:7

# Postgres (memory summary için)
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
  -v $(pwd)/pg_data:/var/lib/postgresql/data \
  --name postgres postgres:16
```

### Adım 3: vLLM server'ı başlat (ayrı terminal)

```bash
# GPU gerekli
./scripts/serve_vllm.sh

# Varsayılan port: 6365. Docker compose 8080 kullanıyor.
# Lokal için .env'e ekle: VLLM_SERVER_URL=http://localhost:6365/v1
```

### Adım 4: .env dosyası

```bash
# .env
VLLM_SERVER_URL=http://localhost:6365/v1   # veya 8080 (docker vllm)
VLLM_MODEL=Qwen/Qwen3-8B-AWQ
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
POSTGRES_URL=postgresql://rag:rag@localhost:5432/rag
MEMORY_ENABLED=true

# Opsiyonel: Web search (Tavily)
TAVILY_API_KEY=...

# Opsiyonel: LangSmith tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=...
```

### Adım 5: Uygulamayı çalıştır

**CLI (interaktif):**
```bash
python main.py
```

**API (FastAPI):**
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Tarayıcıda: http://localhost:3000

**API URL:** Frontend varsayılan olarak `http://localhost:8000` kullanır. Değiştirmek için:
```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Özet Komutlar

| Mod        | Komut |
|-----------|-------|
| Docker    | `docker compose up -d` |
| CLI       | `python main.py` |
| API       | `uvicorn api.app:app --host 0.0.0.0 --port 8000` |
| Frontend  | `cd frontend && npm run dev` |
| vLLM      | `./scripts/serve_vllm.sh` |

---

## Sorun Giderme

**"vLLM server mode zorunludur"**  
→ vLLM çalışıyor olmalı. `VLLM_SERVER_URL` doğru mu kontrol et.

**"Qdrant bağlantısı kurulamadı"**  
→ Qdrant container/process çalışıyor mu? `curl http://localhost:6333/collections`

**"Redis connection failed"**  
→ Redis çalışıyor mu? `redis-cli ping`

**Docker build hatası**  
→ `uv.lock` var mı? `uv sync` ile lock oluşturun.
