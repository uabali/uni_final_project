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
# CUDA debug flags — sadece geliştirme için; production'da kapatın (gecikme artırır)
if os.getenv("CUDA_DEBUG", "").lower() in ("1", "true"):
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

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langchain_core.messages import HumanMessage

import jwt
from jwt import PyJWTError

from src.context import (
    RequestContext as AppRequestContext,
    generate_request_ids,
    get_default_department_id,
    set_request_context,
)
from src.audit import get_audit_logger
from src.tasks import upsert_task, get_task_snapshot

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


_JWT_SECRET = os.getenv("JWT_SECRET_KEY", "").strip()
_JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


def _extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth:
        return None
    if not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip() or None


def _decode_jwt_token(token: str) -> dict:
    if not _JWT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="JWT doğrulama etkin ancak JWT_SECRET_KEY tanımlı değil",
        )
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Geçersiz JWT token: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="JWT payload formatı beklenenden farklı")
    return payload


async def get_request_context(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
) -> AppRequestContext:
    """
    Ortak security katmanı:
    - Opsiyonel API key kontrolü (_require_api_key)
    - JWT doğrulama (Authorization: Bearer)
    - RequestContext oluşturma ve global contextvar'a yazma
    """
    _require_api_key(request)

    token = _extract_bearer_token(request)

    # Geliştirme / local senaryolar için:
    # JWT_SECRET_KEY tanımlı değilse JWT zorunlu değildir.
    claims: Dict[str, Any]
    if not _JWT_SECRET:
        # Local / dev mod: departman bilgisini header'dan okumaya çalış
        claims = {}
        user_id = "anonymous"
        header_dept = request.headers.get("X-Department-ID")
        department_id = header_dept.strip() if header_dept else get_default_department_id()
        role = "user"
    else:
        if not token:
            raise HTTPException(status_code=401, detail="Authorization Bearer token eksik")
        claims = _decode_jwt_token(token)
        user_id = str(claims.get("user_id") or claims.get("sub") or "unknown")
        department_id = str(claims.get("department_id") or claims.get("dept") or "").strip()
        role = str(claims.get("role") or "user")
        if not department_id:
            # Departman bilgisi zorunlu, aksi takdirde çoklu-tenant izolasyonu bozulur.
            raise HTTPException(status_code=403, detail="JWT içinde department_id claim'i eksik")

    request_id, session_id = generate_request_ids(
        user_id=user_id,
        department_id=department_id,
        session_id_header=x_session_id,
    )

    correlation_id = request.headers.get("X-Correlation-ID") or request_id

    ctx = AppRequestContext(
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
        department_id=department_id,
        role=role,
        correlation_id=correlation_id,
        claims=claims,
    )
    set_request_context(ctx)

    # Basit istek log'u (fail durumunda ana akışı bozmadan)
    audit = get_audit_logger()
    if audit.is_available:
        audit.log_request(
            context=ctx,
            endpoint=str(request.url.path),
            extra={"method": request.method},
        )

    return ctx


# ── WebSocket Hub (Task Status Streaming) ─────────────────────


@app.websocket("/ws/tasks/{task_id}")
async def task_status_websocket(websocket: WebSocket, task_id: str):
    """
    Uzun süren agent görevleri için WebSocket kanalı.

    Şu an için:
    - task_id genellikle session_id ile aynıdır (X-Session-ID).
    - Sunucu tarafındaki Task registry'de bir değişiklik oldukça client'a
      JSON payload olarak gönderilir.
    """
    await websocket.accept()
    last_version: int | None = None
    try:
        while True:
            snapshot = get_task_snapshot(task_id)
            if snapshot is not None:
                version = int(snapshot.get("version", 0))
                if last_version is None or version != last_version:
                    await websocket.send_text(json.dumps(snapshot, ensure_ascii=False))
                    last_version = version
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.error(f"Task WebSocket hatası ({task_id}): {exc}")
        try:
            await websocket.close()
        except Exception:
            pass

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


class DeleteRequest(BaseModel):
    """Doküman silme isteği."""
    paths: List[str] = Field(..., min_length=1, description="Silinecek dosya yolları veya dosya adları")


class DeleteResponse(BaseModel):
    """Doküman silme cevabı."""
    status: str
    deleted: int
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

# Metrics router
from api.metrics import router as metrics_router  # noqa: E402

app.include_router(metrics_router)


# ── Helper Functions ──────────────────────────────────────────

def _extract_sources(result: dict) -> List[Dict[str, Any]]:
    """Agent sonucundan kaynak bilgilerini çıkarır, tekilleştirir ve chunk snippet'lerini ekler."""
    import re
    from langchain_core.messages import ToolMessage

    sources: List[Dict[str, Any]] = []
    seen = set()

    # Mesajlar üzerinde tek geçiş: her ToolMessage için hem chunk snippet'lerini
    # hem de kaynak metadata'sını aynı anda çıkar.
    for msg in result.get("messages", []):
        if not (isinstance(msg, ToolMessage) and isinstance(msg.content, str)):
            continue

        content = msg.content
        chunk_snippets: Dict[int, str] = {}
        chunk_meta: Dict[int, Dict[str, Any]] = {}
        current_chunk_id: Optional[int] = None
        current_lines: List[str] = []

        for raw_line in content.splitlines():
            line = raw_line.strip()

            # Web kaynak satırları (ToolMessage içindeki "Source: http...")
            if line.startswith("Source: http"):
                url = line.replace("Source: ", "").strip()
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"type": "web", "url": url, "title": url})
                continue

            # CHUNK başlıkları ve opsiyonel source= metadata'sı
            if line.startswith("[CHUNK "):
                # Önceki chunk'ın snippet'ini kaydet
                if current_chunk_id is not None and current_lines:
                    snippet_text = "\n".join(current_lines).strip()
                    if snippet_text:
                        chunk_snippets[current_chunk_id] = snippet_text

                current_chunk_id = None
                current_lines = []

                id_match = re.match(r"\[CHUNK\s+(\d+)\]", line)
                if id_match:
                    cid = int(id_match.group(1))
                    current_chunk_id = cid

                    # Regex: boşluklu dosya adlarını da yakalar (örn: "Python Kitap.pdf")
                    meta_match = re.match(r"\[CHUNK\s+(\d+)\]\s+source=(.+?)\s+(p\.\S+)", line)
                    if meta_match:
                        source_name = meta_match.group(2).strip()
                        page = meta_match.group(3)
                        chunk_meta[cid] = {"source": source_name, "page": page}
                continue

            # CHUNK içeriği (snippet): boş olmayan, yeni CHUNK/metadata olmayan satırlar
            if current_chunk_id is not None and line and not line.startswith("["):
                current_lines.append(line)

        # Son chunk'ı da kaydet
        if current_chunk_id is not None and current_lines:
            snippet_text = "\n".join(current_lines).strip()
            if snippet_text:
                chunk_snippets[current_chunk_id] = snippet_text

        # Bu ToolMessage içindeki doküman kaynaklarını tekilleştirerek ekle
        for chunk_id, meta in chunk_meta.items():
            source_name = meta["source"]
            page = meta["page"]
            unique_key = f"{source_name}_{page}_{chunk_id}"
            if unique_key in seen:
                continue
            seen.add(unique_key)

            snippet = chunk_snippets.get(chunk_id, "")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."

            sources.append({
                "chunk_id": chunk_id,
                "source": source_name,
                "page": page,
                "title": f"[CHUNK {chunk_id}] {source_name} {page}",
                "snippet": snippet,
            })

    return sources


async def _run_agent_sync(rag_app, message: str, session_id: str, provider_name: str | None = None) -> dict:
    """Agent'ı senkron olarak çalıştırır (thread pool'da)."""

    from src.context import get_request_context

    ctx = get_request_context()

    def _invoke():
        configurable = {"session_id": session_id}
        metadata = {
            "session_id": session_id,
            "llm_provider": provider_name or "",
        }
        if ctx is not None:
            configurable.update(
                {
                    "user_id": ctx.user_id,
                    "department_id": ctx.department_id,
                    "role": ctx.role,
                }
            )
            metadata.update(
                {
                    "user_id": ctx.user_id,
                    "department_id": ctx.department_id,
                    "role": ctx.role,
                    "request_id": ctx.request_id,
                }
            )

        return rag_app.run_agent_turn(
            [HumanMessage(content=message)],
            config={
                "configurable": configurable,
                "run_name": "api_chat",
                "tags": ["api", "chat"],
                "metadata": metadata,
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
            "İnternet erişimin yok, bu yüzden hava durumu, güncel haberler, döviz kurları gibi "
            "anlık bilgilere erişemezsin. Bu tür sorularda dürüstçe 'bu bilgiye erişimim yok' de. "
            "Eğer bir konuda kesin emin değilsen, tahmini bilgi vermek yerine emin olmadığını belirt."
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
    ctx: AppRequestContext = Depends(get_request_context),
):
    """Senkron chat endpoint'i. Tam cevabı bekleyip tek seferde döner."""
    rag_app = _get_rag_app()
    session_id = ctx.session_id

    start = time.perf_counter()
    result = await _run_agent_sync(rag_app, req.message, session_id, req.llm_provider)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    final_msg = result["messages"][-1]
    content = getattr(final_msg, "content", "")
    provider_name = getattr(rag_app.llm_provider, "model", "unknown")
    sources = _extract_sources(result)

    audit = get_audit_logger()
    if audit.is_available:
        audit.log_response(
            context=ctx,
            endpoint=str(request.url.path),
            latency_ms=elapsed_ms,
            extra={
                "used_provider": provider_name,
                "has_sources": bool(sources),
            },
        )

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
    ctx: AppRequestContext = Depends(get_request_context),
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
    rag_app = _get_rag_app()
    session_id = ctx.session_id

    # Chat isteği, aynı zamanda uzun sürebilen bir agent görevi olarak da
    # değerlendirilir; Task registry'de "running" olarak işaretleyelim.
    upsert_task(
        task_id=session_id,
        status="running",
        context=ctx,
        meta={"endpoint": "/chat", "use_rag": req.use_rag},
    )

    async def event_generator():
        last_state = None
        try:
            start = time.perf_counter()
            loop = asyncio.get_running_loop()

            if req.use_rag:
                # ── RAG modu: Gerçek stream (LangGraph stream_mode="values") ──
                configurable = {"session_id": session_id}
                metadata: Dict[str, Any] = {
                    "session_id": session_id,
                    "stream": True,
                }
                from src.context import get_request_context as _get_ctx

                current_ctx = _get_ctx()
                if current_ctx is not None:
                    configurable.update(
                        {
                            "user_id": current_ctx.user_id,
                            "department_id": current_ctx.department_id,
                            "role": current_ctx.role,
                        }
                    )
                    metadata.update(
                        {
                            "user_id": current_ctx.user_id,
                            "department_id": current_ctx.department_id,
                            "role": current_ctx.role,
                            "request_id": current_ctx.request_id,
                        }
                    )

                config = {
                    "configurable": configurable,
                    "run_name": "api_chat_stream",
                    "tags": ["api", "chat", "stream"],
                    "metadata": metadata,
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
                accumulated_text = ""
                while True:
                    event_type, payload = await queue.get()
                    if event_type is None:
                        break
                    if event_type == "token":
                        accumulated_text += payload
                        yield f"event: token\ndata: {json.dumps({'text': payload}, ensure_ascii=False)}\n\n"
                    elif event_type == "done" and payload:
                        state = payload.get("state", {})
                        sources = _extract_sources(state)
                        last_state = state
            else:
                # ── Normal chatbot modu: Doğrudan LLM (stream) ──
                content = await _run_plain_llm(rag_app, req.message)
                last_state = {"messages": [content]} if hasattr(content, "type") else None
                content_str = getattr(content, "content", str(content))
                accumulated_text = content_str
                # Karakter bazlı chunk (LLM stream yok)
                chunk_size = 20
                for i in range(0, len(content_str), chunk_size):
                    yield f"event: token\ndata: {json.dumps({'text': content_str[i:i + chunk_size]}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)
                sources = []

            if sources:
                yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

            # Extract real token count from the final state if available
            real_token_count = 0
            if last_state:
                messages = last_state.get("messages", [])
                for msg in reversed(messages):
                    if getattr(msg, "type", "") == "ai":
                        usage = getattr(msg, "usage_metadata", None)
                        if usage and isinstance(usage, dict):
                            real_token_count = usage.get("total_tokens", 0)
                            break
                        
                        resp_meta = getattr(msg, "response_metadata", {})
                        if isinstance(resp_meta, dict):
                            if "token_usage" in resp_meta and isinstance(resp_meta["token_usage"], dict):
                                real_token_count = resp_meta["token_usage"].get("total_tokens", 0)
                                break
                            if "usage" in resp_meta and hasattr(resp_meta["usage"], "total_tokens"):
                                real_token_count = getattr(resp_meta["usage"], "total_tokens")
                                break
                            
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # Fallback to rough calculation if the provider didn't yield token usage
            if real_token_count == 0:
                real_token_count = len(accumulated_text.split()) if accumulated_text else 0
                
            yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'latency_ms': round(elapsed_ms, 2), 'token_count': real_token_count})}\n\n"
            # Task durumunu güncelle
            upsert_task(
                task_id=session_id,
                status="completed",
                context=ctx,
                meta={"latency_ms": round(elapsed_ms, 2), "token_count": real_token_count},
            )
            audit = get_audit_logger()
            if audit.is_available:
                audit.log_response(
                    context=ctx,
                    endpoint=str(request.url.path),
                    latency_ms=elapsed_ms,
                    token_count=real_token_count,
                    extra={"streaming": True},
                )

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
async def ingest_documents(
    request: Request,
    req: IngestRequest,
    ctx: AppRequestContext = Depends(get_request_context),
):
    """
    Yeni dokümanları sisteme yükler (incremental).
    Dosya hash'i kontrolü ile aynı dosyanın tekrar yüklenmesini önler.
    """
    rag_app = _get_rag_app()

    def _do_ingest():
        return rag_app.ingest_paths(req.paths, department_id=ctx.department_id)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _do_ingest)
        resp = IngestResponse(**result)
        audit = get_audit_logger()
        if audit.is_available:
            audit.log_event(
                event_type="ingest_paths",
                context=ctx,
                payload={
                    "paths": req.paths,
                    "status": resp.status,
                    "ingested": resp.ingested,
                    "skipped": resp.skipped,
                },
            )
        return resp
    except AttributeError:
        raise HTTPException(
            status_code=501,
            detail="Incremental ingestion henüz implemente edilmedi"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/upload", response_model=IngestResponse, tags=["Ingestion"])
@limiter.limit(os.getenv("RATE_LIMIT_INGEST", "10/minute"))
async def ingest_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    ctx: AppRequestContext = Depends(get_request_context),
):
    """
    Dosya yükleyerek doküman ekleme.
    Yüklenen dosyalar data/ dizinine kaydedilir ve indexlenir.
    """
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
        return rag_app.ingest_paths(saved_paths, department_id=ctx.department_id)

    try:
        result = await loop.run_in_executor(None, _do_ingest)
        resp = IngestResponse(**result)
        audit = get_audit_logger()
        if audit.is_available:
            audit.log_event(
                event_type="ingest_upload",
                context=ctx,
                payload={
                    "saved_paths": saved_paths,
                    "status": resp.status,
                    "ingested": resp.ingested,
                    "skipped": resp.skipped,
                },
            )
        return resp
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/delete", response_model=DeleteResponse, tags=["Ingestion"])
@limiter.limit(os.getenv("RATE_LIMIT_INGEST", "10/minute"))
async def delete_documents(
    request: Request,
    req: DeleteRequest,
    ctx: AppRequestContext = Depends(get_request_context),
):
    """
    Yüklenmiş dokümanları sistemden siler.

    - Qdrant vectorstore'daki ilgili chunk'ları kaldırır
    - BM25 index'ini günceller
    - Ingestion registry'den hash kaydını temizler
    - Dosyalar data/ dizininden de silinir (varsa)
    """
    rag_app = _get_rag_app()
    data_dir = Path(rag_app.config.data_dir)

    resolved_paths: List[str] = []
    for p in req.paths:
        p_path = Path(p)
        if not p_path.is_absolute():
            resolved_paths.append(str(data_dir / p_path.name))
        else:
            resolved_paths.append(str(p_path))

    # Önce dosyaları diskten silmeyi dene (isteğe bağlı; hata verse bile devam)
    for file_path in resolved_paths:
        try:
            fp = Path(file_path)
            if fp.exists():
                fp.unlink()
        except Exception as exc:
            logger.warning(f"Dosya silinemedi ({file_path}): {exc}")

    def _do_delete():
        return rag_app.delete_paths(resolved_paths, department_id=ctx.department_id)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _do_delete)
        resp = DeleteResponse(**result)
        audit = get_audit_logger()
        if audit.is_available:
            audit.log_event(
                event_type="ingest_delete",
                context=ctx,
                payload={
                    "paths": resolved_paths,
                    "status": resp.status,
                    "deleted": resp.deleted,
                },
            )
        return resp
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/config/llm", response_model=LlmConfigResponse, tags=["Configuration"])
async def configure_llm(
    req: LlmConfigRequest,
    ctx: AppRequestContext = Depends(get_request_context),
):
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

        resp = LlmConfigResponse(
            status="updated",
            provider=req.provider,
            model=new_provider.model,
        )
        audit = get_audit_logger()
        if audit.is_available:
            audit.log_event(
                event_type="llm_config_update",
                context=ctx,
                payload={"provider": req.provider, "model": new_provider.model},
            )
        return resp
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Multi-provider desteği için litellm kurulumu gerekli: pip install litellm"
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/config/llm", tags=["Configuration"])
async def get_llm_config(
    ctx: AppRequestContext = Depends(get_request_context),
):
    """Mevcut LLM provider bilgisini döner."""
    rag_app = _get_rag_app()
    resp = {
        "provider": type(rag_app.llm_provider).__name__,
        "model": getattr(rag_app.llm_provider, "model", "unknown"),
    }
    audit = get_audit_logger()
    if audit.is_available:
        audit.log_event(
            event_type="llm_config_get",
            context=ctx,
            payload={"provider": resp["provider"], "model": resp["model"]},
        )
    return resp


# ── Frontend Static Files ────────────────────────────────────
# Frontend Next.js olarak ayrı bir dev server'da çalışır (localhost:3000).
# Production build için Next.js export sonrası burada serve edilebilir.
# Şu an CORS middleware ile cross-origin isteklere izin verilmektedir.

