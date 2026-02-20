"""
Vektör Veritabanı Modülü - Qdrant Backend

Metinleri sayısal vektörlere (embedding) çevirir ve Qdrant veritabanına kaydeder.
Incremental Indexing (Ekle/Sil) desteği sunar.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib
import os
from pathlib import Path


def create_embeddings(model_name="BAAI/bge-m3", device="cuda"):
    """
    BAAI/bge-m3 (SOTA multilingual) embedding modelini oluşturur.
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


def create_vectorstore(
    docs, embeddings, url="http://localhost:6333", collection_name="rag_collection"
):
    """
    Qdrant vektör veritabanını oluşturur veya mevcut olanı yükler.

    1. Collection varsa -> mevcut veriyi yükler.
    2. Collection yoksa ve docs varsa -> yeni oluşturur.
    3. İkisi de yoksa -> boş store döner.
    """
    client = QdrantClient(url=url)

    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception:
        collections = []

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
                f"Mevcut Qdrant collection bulundu ({collection_name}); "
                "dokuman degisikligi algilandi, yeniden indexleniyor..."
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

        print(f"Mevcut Qdrant veritabani yukleniyor: {collection_name}")
        if docs and reindex_mode == "false":
            print(
                "Not: Yerel dokuman degisiklikleri indexe yansimaz "
                "(QDRANT_AUTO_REINDEX=false)."
            )
        if docs and reindex_mode == "smart":
            print("Not: QDRANT_AUTO_REINDEX=smart, degisiklik yoksa yeniden indexlenmez.")
        return QdrantVectorStore(
            client=client, collection_name=collection_name, embedding=embeddings
        )

    if docs:
        print(f"Yeni Qdrant veritabani olusturuluyor: {collection_name}")
        vectorstore = QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            url=url,
            collection_name=collection_name,
        )
        if current_fp:
            _save_fingerprint(collection_name, current_fp)
        return vectorstore

    print("Uyari: Collection yok ve dokuman verilmedi. Bos donecek.")
    return QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embeddings
    )


def add_documents_to_collection(vectorstore, docs):
    """Mevcut veritabanına yeni dokümanlar ekler (incremental)."""
    if not docs:
        return
    vectorstore.add_documents(docs)
    print(f"{len(docs)} chunk vektör veritabanına eklendi.")


def delete_from_collection(vectorstore, file_path):
    """Belirli bir dosyaya ait tüm chunk'ları Qdrant'tan siler."""
    print(f"Siliniyor: {file_path}")
    info_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source", match=models.MatchValue(value=file_path)
            )
        ]
    )
    vectorstore.client.delete(
        collection_name=vectorstore.collection_name,
        points_selector=models.FilterSelector(filter=info_filter),
    )
    print("Silme islemi tamamlandi.")
