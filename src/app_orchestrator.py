from __future__ import annotations

"""
Application-level orchestrator for the RAG stack.

Bu katman, architecture.pdf içindeki "orchestrator" kutusuna denk gelir:
- Ingestion (loader + splitter + vectorstore + BM25)
- Retrieval yapılandırması (BM25, hybrid, rerank)
- LLM provider entegrasyonu
- LangGraph agent graph kurulumunu soyutlar
- Incremental doküman ekleme (ingest_paths)

CLI (`main.py`), FastAPI Gateway (`api/app.py`) ve benchmark (`scripts/benchmark.py`)
bu modül üzerinden pipeline oluşturur.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from . import splitter
from .agent import build_agent_graph
from .context import get_default_department_id
from .loader import load_documents, load_single_document
from .llm_provider import LLMProvider, create_default_provider
from .memory import MemorySystem
from .reranker import create_reranker
from .retriever import build_bm25_retriever
from .tools import register_rag_components, register_recently_ingested, set_mcp_invoker
from .tooling import HybridToolInvoker
from .vectorstore import (
    add_documents_to_collection,
    create_embeddings,
    create_vectorstore,
    delete_from_collection,
    build_collection_name,
    get_vectorstore_for_department,
)

logger = logging.getLogger("rag.orchestrator")


@dataclass
class RagAppConfig:
    """Yüksek seviyeli RAG uygulama konfigürasyonu."""

    data_dir: str = "data"
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333").strip())
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "rag_collection").strip())
    chunk_size: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "200")))
    use_bm25: bool = True
    use_reranker: bool = True
    reranker_device: str = field(default_factory=lambda: os.getenv("RERANKER_DEVICE", "cuda").strip())


# ── Ingestion Hash Registry ──────────────────────────────────────

class IngestionRegistry:
    """
    Dosya hash'lerini takip ederek aynı dosyanın tekrar yüklenmesini önler.
    Hash'ler bellekte tutulur; ileride Redis/Qdrant metadata'ya taşınabilir.
    """

    def __init__(self):
        self._hashes: Dict[str, str] = {}

    def compute_file_hash(self, file_path: str) -> str:
        """Dosyanın SHA-256 hash'ini hesaplar."""
        h = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    def is_already_ingested(self, file_path: str) -> bool:
        """Dosya daha önce yüklendi mi kontrol eder."""
        current_hash = self.compute_file_hash(file_path)
        if not current_hash:
            return False
        stored_hash = self._hashes.get(os.path.abspath(file_path))
        return stored_hash == current_hash

    def mark_ingested(self, file_path: str) -> None:
        """Dosyayı yüklenmiş olarak işaretler."""
        current_hash = self.compute_file_hash(file_path)
        if current_hash:
            self._hashes[os.path.abspath(file_path)] = current_hash

    def remove(self, file_path: str) -> None:
        """Dosyayı kayıttan çıkarır."""
        self._hashes.pop(os.path.abspath(file_path), None)


# ── RagApp ────────────────────────────────────────────────────

@dataclass
class RagApp:
    """
    RAG uygulamasının runtime durumu.

    Agentik kullanım (LangGraph ReAct agent) için API sunar:
    - run_agent_turn(): Agent'ı tek turda çalıştır
    - ingest_paths():   Yeni dokümanları incremental olarak yükle
    """

    config: RagAppConfig
    llm: BaseChatModel
    llm_provider: LLMProvider
    agent: Any
    embeddings: Any
    vectorstore: Any
    bm25_retriever: Optional[Any]
    reranker: Optional[Any]
    docs: list[Any]
    memory: Optional[MemorySystem]
    _ingestion_registry: IngestionRegistry = field(default_factory=IngestionRegistry)

    def run_agent_turn(self, messages: list[Any], config: Optional[dict] = None) -> dict:
        """
        LangGraph agent grafını tek bir adım için çalıştırır.

        `messages` tipik olarak [HumanMessage(...)] listesi olur.
        """
        return self.agent.invoke({"messages": messages}, config=config or {})

    def run_agent_stream(
        self, messages: list[Any], config: Optional[dict] = None
    ):
        """
        LangGraph agent grafını stream modunda çalıştırır.
        stream_mode="values" ile state güncellemeleri alınır; yeni içerik delta olarak yield edilir.
        Son state'ten sources çıkarılır.

        Yields:
            tuple: ("token", content_chunk) veya ("done", {"state": ..., "sources": ...})
        """
        cfg = config or {}
        prev_content = ""
        last_state = None

        for state in self.agent.stream(
            {"messages": messages},
            config=cfg,
            stream_mode="values",
        ):
            last_state = state
            messages_list = state.get("messages", [])
            if messages_list:
                last_msg = messages_list[-1]
                # Yalnızca AI mesajlarındaki textleri stream et, ToolMessage'ları (document chunkları) atla
                if last_msg.type == "ai":
                    content = getattr(last_msg, "content", "") or ""
                    # Tool call veya bos stringleri yield etme
                    if isinstance(content, str) and content and content != prev_content:
                        delta = content[len(prev_content) :]
                        if delta:
                            yield ("token", delta)
                        prev_content = content
                elif last_msg.type != "ai":
                    # Eger ToolMessage veya baska tur isleniyorsa prev_content sifirlansin
                    # Boylece bir sonraki AI mesaji icin delta hesabi bozulmaz
                    prev_content = ""

        if last_state:
            yield ("done", {"state": last_state})

    def ingest_paths(self, paths: List[str], *, department_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Yeni dokümanları incremental olarak sisteme yükler.

        Pipeline: Dosya → Hash kontrolü → Load → Split → Embed → Upsert

        Args:
            paths: Yüklenecek dosya yollarının listesi

        Returns:
            dict: {status, ingested, skipped, errors}
        """
        ingested = 0
        skipped = 0
        errors: List[str] = []

        effective_dept = (department_id or get_default_department_id()).strip() or get_default_department_id()

        for file_path in paths:
            # Dosya var mı kontrol et
            if not os.path.exists(file_path):
                errors.append(f"Dosya bulunamadı: {file_path}")
                continue

            # Idempotency: Hash kontrolü
            if self._ingestion_registry.is_already_ingested(file_path):
                logger.info(f"Dosya zaten yüklü (hash eşleşti), atlaniyor: {file_path}")
                skipped += 1
                continue

            try:
                # 1. Load
                raw_docs = load_single_document(file_path)
                if not raw_docs:
                    errors.append(f"Doküman okunamadı: {file_path}")
                    continue

                # Her dokümanın metadata'sına department_id ekle
                for doc in raw_docs:
                    metadata = getattr(doc, "metadata", {}) or {}
                    metadata["department_id"] = effective_dept
                    setattr(doc, "metadata", metadata)

                # 2. Split
                chunks = splitter.split_documents(
                    raw_docs,
                    method="recursive",
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )

                # Split sonrası oluşan her chunk'ın metadata'sına da department_id taşı
                for ch in chunks:
                    metadata = getattr(ch, "metadata", {}) or {}
                    metadata.setdefault("department_id", effective_dept)
                    setattr(ch, "metadata", metadata)

                if not chunks:
                    errors.append(f"Chunk oluşturulamadı: {file_path}")
                    continue

                # 3. Embed + Upsert
                # Strict multi-tenant modunda her departman icin ayri collection kullan
                target_vs = get_vectorstore_for_department(
                    self.vectorstore,
                    self.config.qdrant_collection,
                    effective_dept,
                )
                add_documents_to_collection(target_vs, chunks)

                # 4. BM25 index'ini güncelle
                self.docs.extend(chunks)
                if self.config.use_bm25:
                    try:
                        self.bm25_retriever = build_bm25_retriever(self.docs)
                        # Tool registry'yi güncelle
                        register_rag_components(
                            vectorstore=self.vectorstore,
                            bm25_retriever=self.bm25_retriever,
                            reranker=self.reranker,
                        )
                    except Exception as exc:
                        logger.warning(f"BM25 index güncellenemedi: {exc}")

                # 5. Hash'i kaydet
                self._ingestion_registry.mark_ingested(file_path)

                # 6. Son yüklenen dosyayı kaydet ("bu belgede" sorguları için)
                register_recently_ingested(os.path.basename(file_path))

                ingested += 1
                logger.info(f"Başarıyla yüklendi: {file_path} ({len(chunks)} chunk)")

            except Exception as exc:
                errors.append(f"Yükleme hatası ({file_path}): {str(exc)}")
                logger.error(f"Ingestion hatası: {file_path} — {exc}")

        status = "completed" if not errors else ("partial" if ingested > 0 else "failed")

        return {
            "status": status,
            "ingested": ingested,
            "skipped": skipped,
            "errors": errors,
        }

    def delete_paths(self, paths: List[str], *, department_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Verilen dosya yollarına ait dokümanları sistemden siler.

        - Qdrant vectorstore'dan ilgili chunk'ları kaldırır
        - Bellekte tutulan docs listesini filtreler
        - BM25 retriever'ı yeniden oluşturur (varsa)
        - IngestionRegistry'den hash kaydını siler
        """
        deleted = 0
        errors: List[str] = []

        effective_dept = (department_id or get_default_department_id()).strip() or get_default_department_id()

        for file_path in paths:
            abs_path = os.path.abspath(file_path)
            try:
                # 1. Vectorstore'dan sil (sadece ilgili departmana ait chunk'lar)
                target_vs = get_vectorstore_for_department(
                    self.vectorstore,
                    self.config.qdrant_collection,
                    effective_dept,
                )
                delete_from_collection(target_vs, abs_path, department_id=effective_dept)

                # 2. docs listesini filtrele (department_id + source eşleşmesine göre)
                self.docs = [
                    d
                    for d in self.docs
                    if not (
                        (getattr(d, "metadata", {}) or {}).get("source") == abs_path
                        and (getattr(d, "metadata", {}) or {}).get("department_id")
                        == effective_dept
                    )
                ]

                # 3. Ingestion registry'den kaldır
                self._ingestion_registry.remove(abs_path)

                deleted += 1
            except Exception as exc:
                logger.error(f"Silme hatası: {abs_path} — {exc}")
                errors.append(f"Silme hatası ({abs_path}): {str(exc)}")

        # 4. BM25 retriever'ı güncelle
        if self.config.use_bm25:
            try:
                self.bm25_retriever = (
                    build_bm25_retriever(self.docs) if self.docs else None
                )
                register_rag_components(
                    vectorstore=self.vectorstore,
                    bm25_retriever=self.bm25_retriever,
                    reranker=self.reranker,
                )
            except Exception as exc:
                logger.warning(f"BM25 index güncellenemedi (delete_paths): {exc}")

        status = "completed" if not errors else ("partial" if deleted > 0 else "failed")
        return {"status": status, "deleted": deleted, "errors": errors}


def build_rag_app(config: Optional[RagAppConfig] = None) -> RagApp:
    """
    Tüm ağır objeleri (embedding, vectorstore, BM25, reranker, LLM, agent graph)
    oluşturup tek bir `RagApp` örneği döner.
    """
    cfg = config or RagAppConfig()

    # 1) Embedding + doküman yükleme
    embeddings = create_embeddings()
    documents = load_documents(data_dir=cfg.data_dir)

    if documents:
        default_dept = get_default_department_id()
        # Tüm başlangıç dokümanlarını varsayılan departman ile etiketle
        for d in documents:
            metadata = getattr(d, "metadata", {}) or {}
            metadata.setdefault("department_id", default_dept)
            setattr(d, "metadata", metadata)

        docs = splitter.split_documents(
            documents,
            method="recursive",
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        # Split edilmiş chunk'lara da department_id taşı
        for ch in docs:
            metadata = getattr(ch, "metadata", {}) or {}
            metadata.setdefault("department_id", default_dept)
            setattr(ch, "metadata", metadata)
    else:
        docs = []

    # 2) Vectorstore (Qdrant)
    # Varsayilan collection adi, strict mod kapaliysa tum departmanlar icin ortak,
    # strict mod aciksa default departman icin departman-spesifik olarak kullanilir.
    default_dept = get_default_department_id()
    collection_name = build_collection_name(cfg.qdrant_collection, default_dept)
    vectorstore = create_vectorstore(
        docs,
        embeddings,
        url=cfg.qdrant_url,
        collection_name=collection_name,
    )

    # 3) BM25 + Reranker
    bm25_retriever = None
    if cfg.use_bm25 and docs:
        try:
            bm25_retriever = build_bm25_retriever(docs)
        except Exception as exc:
            print(f"Uyari: BM25 retriever olusturulamadi: {exc}")
            bm25_retriever = None

    reranker = None
    if cfg.use_reranker:
        try:
            reranker = create_reranker(device=cfg.reranker_device)
        except Exception as exc:
            print(f"Uyari: Reranker olusturulamadi: {exc}")
            reranker = None

    # 4) Tool registry
    register_rag_components(
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )

    # 5) Memory system
    memory_system = None
    if os.getenv("MEMORY_ENABLED", "false").lower() == "true":
        memory_system = MemorySystem.from_env()

    # 6) MCP invoker (opsiyonel) + LLM Provider + Agent graph
    mcp_invoker = None
    if os.getenv("MCP_SERVER_URL", "").strip():
        try:
            mcp_invoker = HybridToolInvoker.from_env()
            set_mcp_invoker(mcp_invoker)
        except Exception as exc:
            logger.warning(f"MCP invoker oluşturulamadı: {exc}")

    provider = create_default_provider()
    llm = provider.client
    agent = build_agent_graph(llm, memory=memory_system)

    # 7) Ingestion registry — mevcut dokümanları kaydet (restart sonrası tekrar ingest önlenir)
    registry = IngestionRegistry()
    seen_paths: set = set()
    for doc in docs:
        src = (getattr(doc, "metadata", None) or {}).get("source")
        if src:
            try:
                abs_path = os.path.abspath(src)
                if abs_path not in seen_paths and os.path.exists(abs_path):
                    registry.mark_ingested(abs_path)
                    seen_paths.add(abs_path)
            except Exception:
                pass

    return RagApp(
        config=cfg,
        llm=llm,
        llm_provider=provider,
        agent=agent,
        embeddings=embeddings,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        docs=docs,
        memory=memory_system,
        _ingestion_registry=registry,
    )
