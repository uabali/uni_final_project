"""
Vektör Veritabanı Modülü - Qdrant Backend (Vectorstore Module)

Bu modül, metinleri sayısal vektörlere (embedding) çevirir ve Qdrant veritabanına kaydeder.
Modern 'langchain-qdrant' paketini kullanır.
Incremental Indexing (Ekle/Sil) desteği sunar.
"""

import os
import json
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models


def create_embeddings(model_name="BAAI/bge-m3", device="cuda"):
    """
    Metinleri vektöre çeviren Embedding modelini oluşturur.
    
    Kullanılan Model: BAAI/bge-m3 (SOTA multilingual model).
    
    Args:
        model_name (str): HuggingFace model adı.
        device (str): Çalışacağı donanım (cuda/cpu).
        
    Returns:
        HuggingFaceEmbeddings: LangChain uyumlu embedding fonksiyonu.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vectorstore(docs, embeddings, url="http://localhost:6333", collection_name="rag_collection"):
    """
    Qdrant vektör veritabanını oluşturur veya yükler.
    
    Mantık:
    1. Eğer collection varsa -> Yükler (Mevcut veriyi korur).
    2. Eğer collection yoksa ve docs varsa -> Yeni oluşturur.
    
    Args:
        docs (list): İndekslenecek doküman parçaları (chunklar).
        embeddings: Embedding fonksiyonu (create_embeddings çıktısı).
        url (str): Qdrant sunucu URL'si (ornegin "http://localhost:6333").
        collection_name (str): Veritabanındaki tablo adı.
        
    Returns:
        QdrantVectorStore: Arama yapılabilir vektör deposu.
    """

    # region agent log
    try:
        with open("/home/uabali/MyCode/GitHub/RAG/.cursor/debug.log", "a") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H1",
                        "location": "src/vectorstore.py:create_vectorstore:entry",
                        "message": "create_vectorstore called",
                        "data": {
                            "has_docs": bool(docs),
                            "collection_name": collection_name,
                            "url": url,
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # endregion

    client = QdrantClient(url=url)
    
    # Mevcut koleksiyonları kontrol et
    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception:
        collections = []

    # region agent log
    try:
        with open("/home/uabali/MyCode/GitHub/RAG/.cursor/debug.log", "a") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H2",
                        "location": "src/vectorstore.py:create_vectorstore:collections",
                        "message": "Collections fetched from server",
                        "data": {
                            "collections": collections,
                            "collection_name": collection_name,
                            "url": url,
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # endregion

    if collection_name in collections:
        print(f"Mevcut Qdrant veritabani yukleniyor: {collection_name}")

        # region agent log
        try:
            with open("/home/uabali/MyCode/GitHub/RAG/.cursor/debug.log", "a") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H3",
                            "location": "src/vectorstore.py:create_vectorstore:existing",
                            "message": "Using existing collection",
                            "data": {
                                "collection_name": collection_name,
                                "url": url,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # endregion

        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
    
    # Yeni oluştur
    if docs:
        print(f"Yeni Qdrant veritabani olusturuluyor: {collection_name}")

        # region agent log
        try:
            with open("/home/uabali/MyCode/GitHub/RAG/.cursor/debug.log", "a") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H4",
                            "location": "src/vectorstore.py:create_vectorstore:new",
                            "message": "Creating new collection from documents",
                            "data": {
                                "collection_name": collection_name,
                                "url": url,
                                "num_docs": len(docs),
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # endregion

        return QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            url=url,
            collection_name=collection_name
        )

    print("Uyari: Collection yok ve dokuman verilmedi. Bos donecek.")

    # region agent log
    try:
        with open("/home/uabali/MyCode/GitHub/RAG/.cursor/debug.log", "a") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H5",
                        "location": "src/vectorstore.py:create_vectorstore:empty",
                        "message": "No docs and no existing collection; returning empty store",
                        "data": {
                            "collection_name": collection_name,
                            "url": url,
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # endregion

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )


# ==========================================
# INCREMENTAL INDEXING FONKSİYONLARI
# ==========================================

def add_documents_to_collection(vectorstore, docs):
    """
    Mevcut veritabanına YENİ dokümanlar ekler (Incremental Add).
    
    Args:
        vectorstore: Aktif QdrantVectorStore objesi.
        docs (list): Eklenecek yeni chunklar.
        
    Kullanım: Streamlit dosya yükleme işlemi sonrası.
    """
    if not docs:
        return
    vectorstore.add_documents(docs)
    print(f"{len(docs)} chunk vektör veritabanına eklendi.")


def delete_from_collection(vectorstore, file_path):
    """
    Belirli bir dosyaya ait TÜM chunkları Qdrant'tan siler (Incremental Delete).
    
    Args:
        vectorstore: Aktif QdrantVectorStore objesi.
        file_path (str): Silinecek dosyanın tam yolu (metadata'daki 'source' alanı).
        
    Kullanım: Streamlit dosya silme işlemi sonrası.
    """
    print(f"Siliniyor: {file_path}")
    
    # Qdrant filtresi: metadata['source'] == file_path olanları bul
    info_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=file_path)
            )
        ]
    )
    
    # Filtreye uyanları sil
    vectorstore.client.delete(
        collection_name=vectorstore.collection_name,
        points_selector=models.FilterSelector(
            filter=info_filter
        )
    )
    print("Silme islemi tamamlandi.")
