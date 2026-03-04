from __future__ import annotations

"""
Application-level orchestrator for the RAG stack.

This layer corresponds to the "orchestrator" box in architecture.pdf:
- Ingestion (loader + splitter + vectorstore + BM25)
- Retrieval configuration (BM25, hybrid, rerank)
- LLM provider integration
- LangGraph agent graph setup abstraction
- Incremental document ingestion (ingest_paths)

CLI (`main.py`), FastAPI Gateway (`api/app.py`) and benchmark (`scripts/benchmark.py`)
create their pipelines through this module.
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
    """High-level RAG application configuration."""

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
    Tracks file hashes to prevent re-ingesting the same file.
    Hashes are kept in memory; can be moved to Redis/Qdrant metadata in the future.
    """

    def __init__(self):
        self._hashes: Dict[str, str] = {}

    def compute_file_hash(self, file_path: str) -> str:
        """Computes the SHA-256 hash of a file."""
        h = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    def is_already_ingested(self, file_path: str) -> bool:
        """Checks if the file has already been ingested."""
        current_hash = self.compute_file_hash(file_path)
        if not current_hash:
            return False
        stored_hash = self._hashes.get(os.path.abspath(file_path))
        return stored_hash == current_hash

    def mark_ingested(self, file_path: str) -> None:
        """Marks the file as ingested."""
        current_hash = self.compute_file_hash(file_path)
        if current_hash:
            self._hashes[os.path.abspath(file_path)] = current_hash

    def remove(self, file_path: str) -> None:
        """Removes the file from the registry."""
        self._hashes.pop(os.path.abspath(file_path), None)


# ── RagApp ────────────────────────────────────────────────────

@dataclass
class RagApp:
    """
    Runtime state of the RAG application.

    Provides API for agentic usage (LangGraph ReAct agent):
    - run_agent_turn(): Run the agent for a single turn
    - ingest_paths():   Incrementally ingest new documents
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
        Runs the LangGraph agent graph for a single step.

        `messages` is typically a [HumanMessage(...)] list.
        """
        return self.agent.invoke({"messages": messages}, config=config or {})

    def run_agent_stream(
        self, messages: list[Any], config: Optional[dict] = None
    ):
        """
        Runs the LangGraph agent graph in stream mode.
        Uses stream_mode="values" to receive state updates; yields new content as delta.
        Sources are extracted from the last state.

        Yields:
            tuple: ("token", content_chunk) or ("done", {"state": ..., "sources": ...})
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
                # Only stream text from AI messages, skip ToolMessages (document chunks)
                if last_msg.type == "ai":
                    content = getattr(last_msg, "content", "") or ""
                    # Don't yield tool calls or empty strings
                    if isinstance(content, str) and content and content != prev_content:
                        delta = content[len(prev_content) :]
                        if delta:
                            yield ("token", delta)
                        prev_content = content
                elif last_msg.type != "ai":
                    # If a ToolMessage or other type is being processed, reset prev_content
                    # so that delta calculation for the next AI message is not corrupted
                    prev_content = ""

        if last_state:
            yield ("done", {"state": last_state})

    def ingest_paths(self, paths: List[str], *, department_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Incrementally ingests new documents into the system.

        Pipeline: File → Hash check → Load → Split → Embed → Upsert

        Args:
            paths: List of file paths to ingest

        Returns:
            dict: {status, ingested, skipped, errors}
        """
        ingested = 0
        skipped = 0
        errors: List[str] = []

        effective_dept = (department_id or get_default_department_id()).strip() or get_default_department_id()

        for file_path in paths:
            # Check if file exists
            if not os.path.exists(file_path):
                errors.append(f"File not found: {file_path}")
                continue

            # Idempotency: Hash check
            if self._ingestion_registry.is_already_ingested(file_path):
                logger.info(f"File already ingested (hash matched), skipping: {file_path}")
                skipped += 1
                continue

            try:
                # 1. Load
                raw_docs = load_single_document(file_path)
                if not raw_docs:
                    errors.append(f"Could not read document: {file_path}")
                    continue

                # Add department_id to each document's metadata
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

                # Carry department_id to each chunk's metadata after split
                for ch in chunks:
                    metadata = getattr(ch, "metadata", {}) or {}
                    metadata.setdefault("department_id", effective_dept)
                    setattr(ch, "metadata", metadata)

                if not chunks:
                    errors.append(f"Could not create chunks: {file_path}")
                    continue

                # 3. Embed + Upsert
                # In strict multi-tenant mode, use separate collection per department
                target_vs = get_vectorstore_for_department(
                    self.vectorstore,
                    self.config.qdrant_collection,
                    effective_dept,
                )
                add_documents_to_collection(target_vs, chunks)

                # 4. Update BM25 index
                self.docs.extend(chunks)
                if self.config.use_bm25:
                    try:
                        self.bm25_retriever = build_bm25_retriever(self.docs)
                        # Update tool registry
                        register_rag_components(
                            vectorstore=self.vectorstore,
                            bm25_retriever=self.bm25_retriever,
                            reranker=self.reranker,
                        )
                    except Exception as exc:
                        logger.warning(f"Could not update BM25 index: {exc}")

                # 5. Save hash
                self._ingestion_registry.mark_ingested(file_path)

                # 6. Record last ingested file (for "in this document" queries)
                register_recently_ingested(os.path.basename(file_path))

                ingested += 1
                logger.info(f"Successfully ingested: {file_path} ({len(chunks)} chunks)")

            except Exception as exc:
                errors.append(f"Ingestion error ({file_path}): {str(exc)}")
                logger.error(f"Ingestion error: {file_path} — {exc}")

        status = "completed" if not errors else ("partial" if ingested > 0 else "failed")

        return {
            "status": status,
            "ingested": ingested,
            "skipped": skipped,
            "errors": errors,
        }

    def delete_paths(self, paths: List[str], *, department_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Deletes documents from the system for the given file paths.

        - Removes related chunks from Qdrant vectorstore
        - Filters the in-memory docs list
        - Rebuilds BM25 retriever (if applicable)
        - Removes hash record from IngestionRegistry
        """
        deleted = 0
        errors: List[str] = []

        effective_dept = (department_id or get_default_department_id()).strip() or get_default_department_id()

        for file_path in paths:
            abs_path = os.path.abspath(file_path)
            try:
                # 1. Delete from vectorstore (only chunks for the relevant department)
                target_vs = get_vectorstore_for_department(
                    self.vectorstore,
                    self.config.qdrant_collection,
                    effective_dept,
                )
                delete_from_collection(target_vs, abs_path, department_id=effective_dept)

                # 2. Filter docs list (by department_id + source match)
                self.docs = [
                    d
                    for d in self.docs
                    if not (
                        (getattr(d, "metadata", {}) or {}).get("source") == abs_path
                        and (getattr(d, "metadata", {}) or {}).get("department_id")
                        == effective_dept
                    )
                ]

                # 3. Remove from ingestion registry
                self._ingestion_registry.remove(abs_path)

                deleted += 1
            except Exception as exc:
                logger.error(f"Delete error: {abs_path} — {exc}")
                errors.append(f"Delete error ({abs_path}): {str(exc)}")

        # 4. Update BM25 retriever
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
                logger.warning(f"Could not update BM25 index (delete_paths): {exc}")

        status = "completed" if not errors else ("partial" if deleted > 0 else "failed")
        return {"status": status, "deleted": deleted, "errors": errors}


def build_rag_app(config: Optional[RagAppConfig] = None) -> RagApp:
    """
    Creates all heavy objects (embedding, vectorstore, BM25, reranker, LLM, agent graph)
    and returns a single `RagApp` instance.
    """
    cfg = config or RagAppConfig()

    # 1) Embedding + document loading
    embeddings = create_embeddings()
    documents = load_documents(data_dir=cfg.data_dir)

    if documents:
        default_dept = get_default_department_id()
        # Tag all initial documents with the default department
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
        # Carry department_id to split chunks as well
        for ch in docs:
            metadata = getattr(ch, "metadata", {}) or {}
            metadata.setdefault("department_id", default_dept)
            setattr(ch, "metadata", metadata)
    else:
        docs = []

    # 2) Vectorstore (Qdrant)
    # Default collection name: shared across all departments if strict mode is off,
    # department-specific for default department if strict mode is on.
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
            print(f"Warning: Could not create BM25 retriever: {exc}")
            bm25_retriever = None

    reranker = None
    if cfg.use_reranker:
        try:
            reranker = create_reranker(device=cfg.reranker_device)
        except Exception as exc:
            print(f"Warning: Could not create reranker: {exc}")
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

    # 6) MCP invoker (optional) + LLM Provider + Agent graph
    mcp_invoker = None
    if os.getenv("MCP_SERVER_URL", "").strip():
        try:
            mcp_invoker = HybridToolInvoker.from_env()
            set_mcp_invoker(mcp_invoker)
        except Exception as exc:
            logger.warning(f"Could not create MCP invoker: {exc}")

    provider = create_default_provider()
    llm = provider.client
    agent = build_agent_graph(llm, memory=memory_system)

    # 7) Ingestion registry — register existing documents (prevents re-ingest after restart)
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
