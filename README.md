## Enterprise Agentic RAG & AI Ops Center

**FastAPI + LangGraph + vLLM + Qdrant + Redis + Postgres + Next.js + MCP**

Bu repo, tek bir `docker compose up` ile çalışan, GPU destekli, agentik RAG tabanlı bir **AI Operations Center** prototipidir:

- **FastAPI Orchestrator**: LangGraph tabanlı multi‑agent RAG pipeline, SSE streaming, ingestion, LLM provider switch, metrics.
- **vLLM**: OpenAI‑compatible local LLM server (Qwen3 / Qwen2.5 ailesi).
- **Qdrant**: Multi‑tenant vector DB (isteğe bağlı strict namespace moduyla).
- **Redis + Postgres**: Multi‑tier memory (short‑term, episodic, summaries) + audit log + metrics.
- **Next.js frontend**: Modern chat UI (RAG/web/MCP modları, dosya upload, departman & model seçici, agent durumu, monitoring).
- **MCP Tooling**: Dosya sistemi, Postgres, hafıza vb. tool’lar için MCP entegrasyonu (opsiyonel).

---

## 1. Hızlı Başlangıç (Docker Compose)

### 1.1. Gereksinimler

- **OS**: Linux / WSL2 (GPU passthrough için tavsiye edilen)
- **Docker & Docker Compose**
- **NVIDIA GPU** (CUDA destekli, en az 12 GB VRAM önerilir)
- **Python 3.11+** (lokal geliştirme ve scriptler için)

### 1.2. Servisleri Ayağa Kaldır

Tüm backend bileşenlerini (orchestrator, vLLM, Qdrant, Redis, Postgres, opsiyonel Litellm/SearXNG) aynı anda ayağa kaldırmak için:

```bash
cd vLLM_rag

# Çekirdek stack (orchestrator + vLLM + Qdrant + Redis + Postgres)
docker compose up -d

# Tüm opsiyonel bileşenler ile (Litellm proxy, SearXNG vb.)
docker compose --profile full up -d
```

Varsayılan portlar:

- `:8000` → FastAPI orchestrator (`/chat`, `/ingest`, `/config/llm`, `/metrics/summary`…)
- `:8080` → vLLM OpenAI‑compatible endpoint
- `:6333` → Qdrant
- `:6379` → Redis
- `:5432` → Postgres
- `:4000` → LiteLLM proxy (opsiyonel, `full` profilde)
- `:8888` → SearXNG (opsiyonel)

### 1.3. Frontend’i Çalıştır

Frontend şu an Docker içinde değil; ayrı bir terminalde:

```bash
cd frontend
npm install
npm run dev
```

Varsayılan adres: `http://localhost:3000`  
Frontend, backend ile `NEXT_PUBLIC_API_URL` üzerinden konuşur (default `http://localhost:8000`).

### 1.4. Orchestrator’ı Doğrula

Backend health endpoint’i:

```bash
curl http://localhost:8000/health | jq
```

Her şey doğruysa:

- `status: "healthy"`
- `vectorstore: true`
- `memory: true` (MEMORY_ENABLED=true ise)

---

## 2. Geliştirme (lokal Python ortamı)

### 2.1. Python bağımlılıkları

Projede `uv` kullanımı tavsiye edilir:

```bash
uv sync
# veya
pip install .
```

### 2.2. Orchestrator’ı lokalde çalıştırma

Docker yerine doğrudan lokalde çalıştırmak istersen (Qdrant, Redis, Postgres yine container olabilir):

```bash
# Gerekli servisler (eğer docker-compose'tan ayrı yönetmek istiyorsan)
docker compose up -d qdrant redis postgres vllm

# Orchestrator (FastAPI)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Bu durumda de‑facto topoloji:

- vLLM → `VLLM_SERVER_URL=http://localhost:8080/v1`
- Qdrant → `QDRANT_URL=http://localhost:6333`
- Redis → `REDIS_URL=redis://localhost:6379/0`
- Postgres → `POSTGRES_URL=postgresql://rag:rag@localhost:5432/rag`

---

## 3. Güvenlik ve Çok Tenant Desteği

### 3.1. API Key ve JWT

Orchestrator, iki katmanlı auth destekler:

- **API Key** (opsiyonel, “paylaşılan gizli”):
  - `API_KEY` env set edilirse, tüm istekler `X-API-Key` header’ı ile bu değeri taşımak zorunda.
- **JWT** (prod için tavsiye edilen):
  - `JWT_SECRET_KEY` ve (isteğe bağlı) `JWT_ALGORITHM` (varsayılan `HS256`) set edildiğinde,
  - `Authorization: Bearer <jwt>` header’ı zorunlu hale gelir.
  - JWT payload beklentisi (claims):
    - `user_id` veya `sub`
    - `department_id` (veya `dept`)
    - `role`

Local/dev modda JWT kullanmak istemezsen:

- `JWT_SECRET_KEY` set **etme**,
- Departman bilgisini frontend’in gönderdiği `X-Department-ID` header’ı üzerinden al (Chat UI bunu zaten ayarlıyor).

### 3.2. Departman Bazlı RAG İzolasyonu

İki seviye izolasyon vardır:

- **Default mod** (varsayılan):
  - Tek Qdrant collection (`QDRANT_COLLECTION`, default `rag_collection`).
  - Her chunk’ın `metadata.department_id` alanı set edilir.
  - `search_documents` tool’u, `department_id` üzerinden Qdrant `Filter` ile sadece o departmanın chunk’larını görür.
- **Strict mod**:
  - Env: `RAG_MULTI_TENANT_STRICT=true`
  - Her departman için ayrı collection:
    - `rag_collection_engineering`, `rag_collection_finance` gibi.
  - `RagApp.ingest_paths` ve `delete_paths`, departman bazlı collection’a yazar/siler.
  - `search_documents`, kullanım anındaki `department_id` için doğru collection’ı seçer.

### 3.3. Memory Isolation

Multi‑tier memory şu bileşenlerden oluşur:

- **Short-term** (Redis) → son N turn, session bazlı.
- **Entity memory** (Redis hash) → kullanıcı profili.
- **Episodic memory** (Qdrant) → uzun dönem “anılar”.
- **Summaries** (Postgres) → conversation özetleri.

İki çalışma modu:

- Default:
  - Redis key’leri `session_id` ve `entity_id` üzerinden; `session_id` zaten `department_id:user_id:...` formatında üretildiği için pratikte çakışma riski düşük.
  - Episodic memory tek `MEMORY_EPISODIC_COLLECTION` içinde.
- Strict:
  - Env: `MEMORY_MULTI_TENANT_STRICT=true`
  - Short-term key’leri `"<dept>:<session_id>"`,
  - Entity key’leri `"<dept>:<entity_id>"`,
  - Episodic collection’lar: `memory_collection_<dept>`.

---

## 4. Multi-Agent Orkestrasyon

Backend tarafında LangGraph ile kurulmuş iki katman vardır:

- **ResearchAgent** (`build_research_agent_graph`):
  - Tool set: `search_documents`, `web_search`, `calculator` (+ opsiyonel MCP).
  - RAG + multi‑layer memory + hybrid retrieval + rerank + ReAct döngüsü.
- **Supervisor Agent** (`build_agent_graph`):
  - Gelen state’i analiz eder ve alt‑agent’a yönlendirir:
    - Mesaj `[MCP]` veya `[MCP:...]` ile başlıyorsa → **ExecutionAgent**.
    - Aksi durumda → **ResearchAgent**.
- **ExecutionAgent** (basit v1):
  - Tool set: MCP odaklı (`read_file`, `list_directory`, `write_file`, `execute_python`, `run_bash`, `query_postgres`, `memory_search`).
  - Kısa bir “eylem odaklı” system prompt ile LLM’i sadece tool çağrıları ve sonuç raporlamaya odaklar.

Uygulamanın `/chat` ve `/chat/sync` endpoint’leri her zaman Supervisor üzerinden gider; UI tarafında `[MCP]` prefix’i `MessageInput` tarafından otomatik eklenir (MCP modu açıkken).

---

## 5. Frontend Özellikleri

Frontend, `frontend` klasöründe **Next.js App Router** ile implement edilmiştir:

- **Chat arayüzü**:
  - RAG toggle (yerel dokümanlar vs yalnızca LLM),
  - “İnternette Ara” modu (`[WEB_ONLY]` prefix),
  - MCP modu ve hazır MCP presetleri (dosya alanı, Postgres, hafıza arama),
  - Dosya upload (PDF/TXT, ingestion pipeline’a bağlı),
  - Mesajlara bağlı kaynak listesi (RAG chunk’ları ve web URL’leri),
  - Token & latency badge’leri.
- **Departman seçici**:
  - Header ve Settings altında “Departman” dropdown’ı,
  - Seçim `X-Department-ID` header’ına yansıtılır; backend’de hem RAG izolasyonu hem de MCP PolicyEngine kullanır.
- **Agent Status paneli**:
  - Sidebar’da “Agent Durumu” kutusu,
  - `/ws/tasks/{task_id}` WebSocket akışından gelen status (running/completed/error) + son güncelleme zamanı gösterilir.
- **Monitoring sekmesi**:
  - Aktif sohbet için basit metrikler (turn sayısı, toplam token, ortalama latency),
  - `LANGCHAIN_TRACING_V2` + `LANGCHAIN_API_KEY` ile LangSmith entegrasyonu hakkında açıklama,
  - `GET /metrics/summary` üzerinden departman bazlı özet (istek sayısı, ortalama latency, toplam token) kartları.

---

## 6. Başlıca Endpoint’ler

Tüm endpoint’ler `http://localhost:8000` üzerindedir (orchestrator container’ı).

- **Chat (streaming)**  
  `POST /chat`

  Gövde:

  ```json
  {
    "message": "Q3 bütçe sapmalarını özetle",
    "use_rag": true
  }
  ```

  - SSE event’leri:
    - `event: token` → `{ "text": "..." }`
    - `event: sources` → RAG kaynak listesi
    - `event: done` → `{ "session_id", "latency_ms", "token_count" }`

- **Chat (sync)**  
  `POST /chat/sync` → tek JSON cevap (`answer`, `sources`, `latency_ms`, `used_provider`).

- **Ingestion**  
  - `POST /ingest` → path listesi ile incremental ingest
  - `POST /ingest/upload` → multipart dosya upload + ingest
  - `POST /ingest/delete` → ingest edilen dosyaları sil (Qdrant + registry + disk)

- **LLM Provider Config**  
  - `GET /config/llm` → aktif provider/model
  - `PUT /config/llm` → provider/model/base_url değiştir (vLLM / OpenAI / LiteLLM).

- **Metrics**  
  - `GET /metrics/summary` → Postgres audit tablosundan departman/agent bazlı özet.

- **Health**  
  - `GET /health` → orchestrator durumu.

---

## 7. Örnek Çalıştırma Senaryosu

1. `docker compose up -d` ile tüm backend’i ayağa kaldır.
2. `cd frontend && npm run dev` ile chat UI’yi aç.
3. Tarayıcıda `http://localhost:3000`:
   - Settings’ten `Backend API Anahtarı` gir (eğer `API_KEY` tanımladıysan),
   - Departmanını seç (ör. Engineering).
4. Birkaç PDF yükle (Sidebar’daki “Data Alanı”ndan veya message composer’dan).
5. Chat’te sor:
   - “Bu architecture dokümanına göre RAG pipeline’ı özetle.”
   - “MCP dosya çalışma alanı preset’ini kullanarak /data altındaki dosyaları listele.”
6. Settings → Monitoring sekmesinden:
   - Aktif sohbet metriklerini ve `Son Özeti Yükle` ile departman bazlı istatistikleri izle.

---

## 8. Notlar ve Geliştirme Yönleri

- Qdrant strict multi‑tenant ve Memory strict modları tamamen **opsiyonel**; varsayılan ayarlar tek tenant dev ortam için optimize edilmiştir.
- Supervisor şu anda iki alt‑agent’i (Research + Execution) route ediyor; Analytics/Document/Notification agent’ları için altyapı hazır.
- Metrics API v1, Postgres audit tablosundan basit agregatlar sağlıyor; daha ileri seviye cost/latency dashboard’ları için materialized view’lar ve tarih filtresi eklenebilir.

