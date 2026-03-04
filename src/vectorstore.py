"""
Vector Database Module - Qdrant Backend

Converts texts to numerical vectors (embeddings) and stores them in Qdrant database.
Supports Incremental Indexing (Add/Delete).
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib
import os
import time
from pathlib import Path


def create_embeddings(model_name="BAAI/bge-m3", device="cuda"):
    """
    Creates the BAAI/bge-m3 (SOTA multilingual) embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _fingerprint_docs(docs) -> str:
    h = hashlib.sha256()
    h.update(f"n={len(docs)}".encode())
    for doc in docs:
        source = str((getattr(doc, "metadata", {}) or {}).get("source", ""))
        page = str((getattr(doc, "metadata", {}) or {}).get("page", ""))
        content = (getattr(doc, "page_content", "") or "")[:300]
        h.update(source.encode("utf-8", errors="ignore"))
        h.update(page.encode("utf-8", errors="ignore"))
        h.update(content.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _fingerprint_file_path(collection_name: str) -> Path:
    base_dir = Path(os.getenv("QDRANT_META_DIR", ".rag_cache"))
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{collection_name}.fingerprint"


def _load_last_fingerprint(collection_name: str) -> str | None:
    file_path = _fingerprint_file_path(collection_name)
    if not file_path.exists():
        return None
    return file_path.read_text(encoding="utf-8").strip() or None


def _save_fingerprint(collection_name: str, fingerprint: str) -> None:
    file_path = _fingerprint_file_path(collection_name)
    file_path.write_text(fingerprint, encoding="utf-8")


def is_multi_tenant_strict() -> bool:
    """
    If RAG_MULTI_TENANT_STRICT=true, use separate Qdrant collection per department.
    Default: false (single collection + department_id payload filter).
    """
    return os.getenv("RAG_MULTI_TENANT_STRICT", "false").strip().lower() == "true"


def _normalize_namespace(name: str) -> str:
    """Converts a department identifier to a safe name for Qdrant collections."""
    import re

    s = (name or "").strip().lower()
    if not s:
        return "default"
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "default"


def build_collection_name(base: str, department_id: str | None) -> str:
    """
    Generates a department-specific collection name from the base collection name.

    If strict mode is off, always returns base.
    """
    if not is_multi_tenant_strict() or not department_id:
        return base
    return f"{base}_{_normalize_namespace(department_id)}"


def _wait_for_qdrant(client: QdrantClient, url: str) -> None:
    """Waits for Qdrant to become ready during startup (handles brief connection errors)."""
    try:
        timeout_s = float(os.getenv("QDRANT_STARTUP_TIMEOUT", "20").strip())
    except Exception:
        timeout_s = 20.0
    try:
        retry_s = float(os.getenv("QDRANT_RETRY_INTERVAL", "1").strip())
    except Exception:
        retry_s = 1.0

    timeout_s = max(0.0, timeout_s)
    retry_s = max(0.2, retry_s)
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None

    while True:
        try:
            client.get_collections()
            return
        except Exception as exc:
            last_exc = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Could not connect to Qdrant: {url}. "
                    "Please check that the service is running and the port is open."
                ) from last_exc
            time.sleep(retry_s)


def create_vectorstore(
    docs, embeddings, url="http://localhost:6333", collection_name="rag_collection"
):
    """
    Creates or loads the Qdrant vector database.

    1. If collection exists -> loads existing data.
    2. If collection doesn't exist and docs are provided -> creates new.
    3. If neither -> returns empty store.
    """
    client = QdrantClient(url=url)
    _wait_for_qdrant(client, url)
    collections = [c.name for c in client.get_collections().collections]

    reindex_mode = os.getenv("QDRANT_AUTO_REINDEX", "smart").strip().lower()
    if reindex_mode not in {"true", "false", "smart"}:
        reindex_mode = "smart"

    current_fp = _fingerprint_docs(docs) if docs else None
    last_fp = _load_last_fingerprint(collection_name) if docs else None

    if collection_name in collections:
        should_reindex = False
        if docs:
            if reindex_mode == "true":
                should_reindex = True
            elif reindex_mode == "smart":
                should_reindex = current_fp != last_fp

        if should_reindex:
            print(
                f"Existing Qdrant collection found ({collection_name}); "
                "document changes detected, re-indexing..."
            )
            try:
                client.delete_collection(collection_name=collection_name)
            except Exception:
                pass
            vectorstore = QdrantVectorStore.from_documents(
                documents=docs,
                embedding=embeddings,
                url=url,
                collection_name=collection_name,
            )
            if current_fp:
                _save_fingerprint(collection_name, current_fp)
            return vectorstore

        print(f"Loading existing Qdrant database: {collection_name}")
        if docs and reindex_mode == "false":
            print(
                "Note: Local document changes will not be reflected in the index "
                "(QDRANT_AUTO_REINDEX=false)."
            )
        if docs and reindex_mode == "smart":
            print("Note: QDRANT_AUTO_REINDEX=smart, no re-indexing if no changes detected.")
        return QdrantVectorStore(
            client=client, collection_name=collection_name, embedding=embeddings
        )

    if docs:
        print(f"Creating new Qdrant database: {collection_name}")
        vectorstore = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            url=url,
            collection_name=collection_name,
        )
        if current_fp:
            _save_fingerprint(collection_name, current_fp)
        return vectorstore

    print("Warning: No collection exists and no documents provided. Returning empty store.")
    return QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embeddings
    )


def add_documents_to_collection(vectorstore, docs):
    """Adds new documents to the existing database (incremental)."""
    if not docs:
        return
    vectorstore.add_documents(docs)
    print(f"{len(docs)} chunks added to vector database.")


def delete_from_collection(vectorstore, file_path, department_id: str | None = None):
    """Deletes all chunks belonging to a specific file from Qdrant.

    If department_id is provided, only payloads for that department are deleted.
    """
    print(f"Deleting: {file_path} (department={department_id or 'ANY'})")
    must_conditions = [
        models.FieldCondition(
            key="source", match=models.MatchValue(value=file_path)
        )
    ]
    if department_id:
        must_conditions.append(
            models.FieldCondition(
                key="department_id",
                match=models.MatchValue(value=department_id),
            )
        )

    info_filter = models.Filter(must=must_conditions)
    vectorstore.client.delete(
        collection_name=vectorstore.collection_name,
        points_selector=models.FilterSelector(filter=info_filter),
    )
    print("Delete operation completed.")


def get_vectorstore_for_department(
    base_vectorstore: QdrantVectorStore,
    base_collection_name: str,
    department_id: str | None,
) -> QdrantVectorStore:
    """
    In strict multi-tenant mode, returns a separate QdrantVectorStore for the given department.

    - strict=false or no department_id → base_vectorstore
    - strict=true → new collection name created from base_collection_name + department_id.
      If this name already matches base_vectorstore.collection_name, returns the same instance.
    """
    if not is_multi_tenant_strict() or not department_id:
        return base_vectorstore

    target_name = build_collection_name(base_collection_name, department_id)
    if target_name == base_vectorstore.collection_name:
        return base_vectorstore

    client = base_vectorstore.client
    embeddings = base_vectorstore.embedding  # type: ignore[attr-defined]
    return QdrantVectorStore(
        client=client,
        collection_name=target_name,
        embedding=embeddings,
    )
