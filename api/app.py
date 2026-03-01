"""
FastAPI API Gateway — Gerçek HTTP sunucusu.

Endpoint'ler:
  POST /chat          → SSE streaming ile agentic RAG cevabı
  POST /chat/sync     → Streaming olmadan senkron cevap
  POST /ingest        → Yeni doküman yükleme (incremental)
  PUT  /config/llm    → Runtime LLM provider değişikliği
  GET  /health        → Sistem sağlık kontrolü
  GET  /config/llm    → Mevcut LLM provider bilgisi
"""

from __future__ import annotations

import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langchain_core.messages import HumanMessage

logger = logging.getLogger("rag.api")

# Rate limiter — architecture.pdf: "rate limiting"
limiter = Limiter(key_func=get_remote_address)


def _require_api_key(request: Request) -> None:
    """API_KEY env set ise X-API-Key header zorunlu."""
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != api_key:
        raise HTTPException(status_code=401, detail="Geçersiz veya eksik API anahtarı")

# ── Pydantic Models ───────────────────────────────────────────


class ChatRequest(BaseModel):
    """Chat isteği."""
    message: str = Field(..., min_length=1, max_length=8192, description="Kullanıcı mesajı")
    llm_provider: Optional[str] = Field(None, description="Kullanılacak LLM provider (opsiyonel)")
    use_rag: bool = Field(True, description="RAG modunu etkinleştir/devre dışı bırak")


class ChatResponse(BaseModel):
    """Senkron chat cevabı."""
    session_id: str
    answer: str
    latency_ms: float
    used_provider: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Doküman yükleme isteği (path listesi)."""
    paths: List[str] = Field(..., min_length=1, description="Yüklenecek dosya yolları")


class IngestResponse(BaseModel):
    """Doküman yükleme cevabı."""
    status: str
    ingested: int
    skipped: int
    errors: List[str] = Field(default_factory=list)


class LlmConfigRequest(BaseModel):
    """LLM provider değişikliği isteği."""
    provider: str = Field(..., description="Provider adı: vllm | openai | litellm")
    api_key: Optional[str] = Field(None, description="External provider API key")
    model: Optional[str] = Field(None, description="Kullanılacak model adı")
    base_url: Optional[str] = Field(None, description="Custom endpoint URL")


class LlmConfigResponse(BaseModel):
    """LLM config cevabı."""
    status: str
    provider: str
    model: str


class HealthResponse(BaseModel):
    """Sağlık kontrolü cevabı."""
    status: str
    vectorstore: bool
    memory: bool
    provider: str
    model: str
    uptime_seconds: float


# ── Application State ─────────────────────────────────────────

_app_state: Dict[str, Any] = {}
_start_time: float = 0.0


def _get_rag_app():
    """Singleton RagApp instance'ını döner."""
    app = _app_state.get("rag_app")
    if app is None:
        raise HTTPException(status_code=503, detail="RAG uygulaması henüz başlatılmadı")
    return app


# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uygulama başlangıcında heavy objeleri yükler.
    Kapanışta temizlik yapar.
    """
    global _start_time
    import os
    from dotenv import load_dotenv
    load_dotenv()

    logger.info("RAG uygulaması başlatılıyor...")
    _start_time = time.time()

    try:
        from src.app_orchestrator import RagAppConfig, build_rag_app
        cfg = RagAppConfig()
        rag_app = build_rag_app(cfg)
        _app_state["rag_app"] = rag_app
        logger.info("RAG uygulaması hazır!")
    except Exception as exc:
        logger.error(f"RAG başlatma hatası: {exc}")
        _app_state["rag_app"] = None

    yield

    logger.info("RAG uygulaması kapatılıyor...")
    _app_state.clear()


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG API",
    description="LangGraph ReAct Agent tabanlı RAG sistemi API'si",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper Functions ──────────────────────────────────────────

def _extract_sources(result: dict) -> List[Dict[str, Any]]:
    """Agent sonucundan kaynak bilgilerini çıkarır, tekilleştirir ve chunk snippet'lerini ekler."""
    import re
    from langchain_core.messages import ToolMessage

    sources = []
    seen = set()

    # İlk geçişte chunk metinlerini topla
    chunk_snippets: Dict[int, str] = {}
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            current_chunk_id = None
            current_lines: list[str] = []
            for line in msg.content.splitlines():
                stripped = line.strip()
                chunk_match = re.match(r"\[CHUNK\s+(\d+)\]", stripped)
                if chunk_match:
                    # Önceki chunk'ı kaydet
                    if current_chunk_id is not None and current_lines:
                        chunk_snippets[current_chunk_id] = "\n".join(current_lines).strip()
                    current_chunk_id = int(chunk_match.group(1))
                    # Chunk header'dan sonra gelen metin bu satırda başlayabilir
                    header_end = stripped.find("\n")
                    current_lines = []
                elif current_chunk_id is not None and stripped and not stripped.startswith("["):
                    current_lines.append(stripped)
            # Son chunk'ı da kaydet
            if current_chunk_id is not None and current_lines:
                chunk_snippets[current_chunk_id] = "\n".join(current_lines).strip()

    # İkinci geçişte kaynakları topla
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            for line in msg.content.splitlines():
                line = line.strip()
                if line.startswith("[CHUNK ") and "source=" in line:
                    match = re.match(r"\[CHUNK\s+(\d+)\]\s+source=(\S+)\s+(p\.\S+)", line)
                    if match:
                        chunk_id = int(match.group(1))
                        source_name = match.group(2)
                        page = match.group(3)

                        unique_key = f"{source_name}_{page}"
                        if unique_key not in seen:
                            seen.add(unique_key)
                            snippet = chunk_snippets.get(chunk_id, "")
                            # Snippet'i 300 karakterle sınırla
                            if len(snippet) > 300:
                                snippet = snippet[:300] + "..."
                            sources.append({
                                "chunk_id": chunk_id,
                                "source": source_name,
                                "page": page,
                                "snippet": snippet,
                            })
                elif line.startswith("Source: http"):
                    url = line.replace("Source: ", "").strip()
                    if url not in seen:
                        seen.add(url)
                        sources.append({"type": "web", "url": url})
    return sources


async def _run_agent_sync(rag_app, message: str, session_id: str, provider_name: str | None = None) -> dict:
    """Agent'ı senkron olarak çalıştırır (thread pool'da)."""

    def _invoke():
        return rag_app.run_agent_turn(
            [HumanMessage(content=message)],
            config={
                "configurable": {"session_id": session_id},
                "run_name": "api_chat",
                "tags": ["api", "chat"],
                "metadata": {
                    "session_id": session_id,
                    "llm_provider": provider_name or "",
                },
            },
        )

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _invoke)


async def _run_plain_llm(rag_app, message: str) -> str:
    """RAG olmadan doğrudan LLM'i çalıştırır (normal chatbot modu)."""
    from langchain_core.messages import SystemMessage

    def _invoke():
        system_msg = SystemMessage(content=(
            "Sen yardımcı bir asistansın. Kullanıcının sorularını doğal ve akıcı bir şekilde Türkçe olarak yanıtla. "
            "Bilmediğin konularda dürüst ol ve tahmin yapmak yerine bilmediğini ifade et."
        ))
        return rag_app.llm.invoke([system_msg, HumanMessage(content=message)])

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _invoke)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Sistem sağlık kontrolü."""
    rag_app = _app_state.get("rag_app")
    if rag_app is None:
        return HealthResponse(
            status="initializing",
            vectorstore=False,
            memory=False,
            provider="none",
            model="none",
            uptime_seconds=time.time() - _start_time,
        )

    return HealthResponse(
        status="healthy",
        vectorstore=rag_app.vectorstore is not None,
        memory=rag_app.memory is not None,
        provider=type(rag_app.llm_provider).__name__,
        model=getattr(rag_app.llm_provider, "model", "unknown"),
        uptime_seconds=time.time() - _start_time,
    )


@app.post("/chat/sync", response_model=ChatResponse, tags=["Chat"])
@limiter.limit(os.getenv("RATE_LIMIT_CHAT", "30/minute"))
async def chat_sync(
    request: Request,
    req: ChatRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """Senkron chat endpoint'i. Tam cevabı bekleyip tek seferde döner."""
    _require_api_key(request)
    rag_app = _get_rag_app()
    session_id = x_session_id or str(uuid.uuid4())

    start = time.perf_counter()
    result = await _run_agent_sync(rag_app, req.message, session_id, req.llm_provider)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    final_msg = result["messages"][-1]
    content = getattr(final_msg, "content", "")
    provider_name = getattr(rag_app.llm_provider, "model", "unknown")
    sources = _extract_sources(result)

    return ChatResponse(
        session_id=session_id,
        answer=content,
        latency_ms=round(elapsed_ms, 2),
        used_provider=provider_name,
        sources=sources,
    )


@app.post("/chat", tags=["Chat"])
@limiter.limit(os.getenv("RATE_LIMIT_CHAT", "30/minute"))
async def chat_stream(
    request: Request,
    req: ChatRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """
    SSE streaming chat endpoint'i.
    Token'ları Server-Sent Events olarak canlı gönderir.

    use_rag=true  → Agentic RAG pipeline (search_documents, web_search, vb.)
    use_rag=false → Normal chatbot modu (doğrudan LLM, tool çağrısı yok)

    Event tipleri:
      - ``token``: Tek bir metin parçası
      - ``sources``: Kaynak bilgileri (JSON)
      - ``done``: Akışın bittiğini belirtir
      - ``error``: Hata oluştuğunda
    """
    _require_api_key(request)
    rag_app = _get_rag_app()
    session_id = x_session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            start = time.perf_counter()
            loop = asyncio.get_running_loop()

            if req.use_rag:
                # ── RAG modu: Gerçek stream (LangGraph stream_mode="values") ──
                config = {
                    "configurable": {"session_id": session_id},
                    "run_name": "api_chat_stream",
                    "tags": ["api", "chat", "stream"],
                }
                queue: asyncio.Queue = asyncio.Queue()

                def _produce():
                    for event_type, payload in rag_app.run_agent_stream(
                        [HumanMessage(content=req.message)],
                        config=config,
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))
                    loop.call_soon_threadsafe(queue.put_nowait, (None, None))

                # Run producer in background so consumer can read concurrently
                loop.run_in_executor(None, _produce)

                sources = []
                while True:
                    event_type, payload = await queue.get()
                    if event_type is None:
                        break
                    if event_type == "token":
                        yield f"event: token\ndata: {json.dumps({'text': payload}, ensure_ascii=False)}\n\n"
                    elif event_type == "done" and payload:
                        state = payload.get("state", {})
                        sources = _extract_sources(state)
            else:
                # ── Normal chatbot modu: Doğrudan LLM (stream) ──
                content = await _run_plain_llm(rag_app, req.message)
                content_str = getattr(content, "content", str(content))
                # Karakter bazlı chunk (LLM stream yok)
                chunk_size = 20
                for i in range(0, len(content_str), chunk_size):
                    yield f"event: token\ndata: {json.dumps({'text': content_str[i:i + chunk_size]}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)
                sources = []

            if sources:
                yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'latency_ms': round(elapsed_ms, 2)})}\n\n"

        except Exception as exc:
            logger.error(f"Chat stream hatası: {exc}")
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
        },
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
@limiter.limit(os.getenv("RATE_LIMIT_INGEST", "10/minute"))
async def ingest_documents(request: Request, req: IngestRequest):
    """
    Yeni dokümanları sisteme yükler (incremental).
    Dosya hash'i kontrolü ile aynı dosyanın tekrar yüklenmesini önler.
    """
    _require_api_key(request)
    rag_app = _get_rag_app()

    def _do_ingest():
        return rag_app.ingest_paths(req.paths)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _do_ingest)
        return IngestResponse(**result)
    except AttributeError:
        raise HTTPException(
            status_code=501,
            detail="Incremental ingestion henüz implemente edilmedi"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/upload", response_model=IngestResponse, tags=["Ingestion"])
@limiter.limit(os.getenv("RATE_LIMIT_INGEST", "10/minute"))
async def ingest_upload(request: Request, files: List[UploadFile] = File(...)):
    """
    Dosya yükleyerek doküman ekleme.
    Yüklenen dosyalar data/ dizinine kaydedilir ve indexlenir.
    """
    _require_api_key(request)
    rag_app = _get_rag_app()
    data_dir = Path(rag_app.config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    for upload_file in files:
        dest = data_dir / upload_file.filename
        content = await upload_file.read()
        dest.write_bytes(content)
        saved_paths.append(str(dest))

    loop = asyncio.get_running_loop()

    def _do_ingest():
        return rag_app.ingest_paths(saved_paths)

    try:
        result = await loop.run_in_executor(None, _do_ingest)
        return IngestResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/config/llm", response_model=LlmConfigResponse, tags=["Configuration"])
async def configure_llm(req: LlmConfigRequest):
    """
    Runtime'da LLM provider'ını değiştirir.
    Sistemi yeniden başlatmaya gerek kalmadan farklı bir model/provider kullanılabilir.
    """
    rag_app = _get_rag_app()

    try:
        from src.llm_provider import create_provider_by_name

        new_provider = create_provider_by_name(
            name=req.provider,
            api_key=req.api_key,
            model=req.model,
            base_url=req.base_url,
        )

        # RagApp üzerindeki provider ve LLM'i güncelle
        rag_app.llm_provider = new_provider
        rag_app.llm = new_provider.client

        # Agent graph'ı yeni LLM ile yeniden oluştur
        from src.agent import build_agent_graph
        rag_app.agent = build_agent_graph(new_provider.client, memory=rag_app.memory)

        return LlmConfigResponse(
            status="updated",
            provider=req.provider,
            model=new_provider.model,
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Multi-provider desteği için litellm kurulumu gerekli: pip install litellm"
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/config/llm", tags=["Configuration"])
async def get_llm_config():
    """Mevcut LLM provider bilgisini döner."""
    rag_app = _get_rag_app()
    return {
        "provider": type(rag_app.llm_provider).__name__,
        "model": getattr(rag_app.llm_provider, "model", "unknown"),
    }


# ── Frontend Static Files ────────────────────────────────────
# Frontend Next.js olarak ayrı bir dev server'da çalışır (localhost:3000).
# Production build için Next.js export sonrası burada serve edilebilir.
# Şu an CORS middleware ile cross-origin isteklere izin verilmektedir.

