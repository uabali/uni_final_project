"""
Re-Ranker Modülü

Bu modül, retrieval sonuçlarını daha güçlü bir modelle (cross-encoder) yeniden sıralar.
Re-ranking, retrieval accuracy'yi %15-25 artırır.

Özellikler:
- Cross-encoder modeli ile reranking
- Skor bazlı yeniden sıralama
- Top-k seçimi
"""

from typing import List, Optional
from langchain_core.documents import Document

# CrossEncoder import (sentence-transformers gerekli)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("Uyarı: CrossEncoder bulunamadı. 'sentence-transformers' paketini yükleyin.")


def create_reranker(
    model_name: str = "BAAI/bge-reranker-base",
    device: str = "cuda"
):
    """
    Cross-encoder reranker modelini oluşturur.
    
    Args:
        model_name: HuggingFace model adı (varsayılan: BAAI/bge-reranker-base)
        device: Çalışacağı donanım (cuda/cpu)
        
    Returns:
        CrossEncoder: Reranker modeli
        
    Not: İlk kullanımda model indirilecek (~400MB).
    """
    if not CROSS_ENCODER_AVAILABLE:
        raise ImportError("CrossEncoder mevcut değil. 'sentence-transformers' yükleyin.")
    
    print(f"Reranker modeli yükleniyor: {model_name}...")
    reranker = CrossEncoder(model_name, device=device)
    print("Reranker hazır.")
    return reranker


def rerank_documents(
    query: str,
    documents: List[Document],
    reranker,
    top_k: Optional[int] = None
) -> List[Document]:
    """
    Dokümanları soruya göre yeniden sıralar (re-ranking).
    
    Bu fonksiyon:
    1. Her dokümanı soruyla birlikte cross-encoder'a verir
    2. Skorlarına göre yeniden sıralar
    3. Top-k kadar en iyi dokümanları döndürür
    
    Args:
        query: Kullanıcı sorusu
        documents: Yeniden sıralanacak dokümanlar listesi
        reranker: CrossEncoder modeli
        top_k: Döndürülecek en iyi doküman sayısı (None = hepsi)
        
    Returns:
        List[Document]: Yeniden sıralanmış dokümanlar (yüksek skorlu önce)
        
    Örnek:
        >>> reranker = create_reranker()
        >>> reranked = rerank_documents("Python nedir?", docs, reranker, top_k=5)
    """
    if not documents:
        return []
    
    if not CROSS_ENCODER_AVAILABLE:
        print("Uyarı: Reranker mevcut değil. Orijinal sıralama kullanılıyor.")
        return documents[:top_k] if top_k else documents
    
    # Her doküman için (query, document) çifti oluştur
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Cross-encoder ile skorları hesapla
    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        print(f"Reranking hatası: {e}. Orijinal sıralama kullanılıyor.")
        return documents[:top_k] if top_k else documents
    
    # Dokümanları skorlarına göre sırala (yüksek skorlu önce)
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Top-k kadar al
    reranked_docs = [doc for _, doc in scored_docs]
    if top_k:
        reranked_docs = reranked_docs[:top_k]
    
    max_score = float(max(scores)) if len(scores) > 0 else 0.0
    print(f"Reranking: {len(documents)} doküman {len(reranked_docs)}'e indirildi (en iyi skor: {max_score:.3f})")
    
    return reranked_docs


def create_rerank_retriever(
    base_retriever,
    query: str,
    reranker,
    top_k: Optional[int] = None,
    rerank_top_n: int = 20
):
    """
    Base retriever'ı reranker ile sarmalar.
    
    Bu fonksiyon:
    1. Base retriever ile daha fazla doküman bulur (rerank_top_n)
    2. Reranker ile yeniden sıralar
    3. Top-k kadar en iyisini döndürür
    
    Args:
        base_retriever: Temel retriever (vectorstore retriever)
        query: Kullanıcı sorusu
        reranker: CrossEncoder modeli
        top_k: Döndürülecek en iyi doküman sayısı
        rerank_top_n: Reranking için alınacak doküman sayısı (top_k'dan fazla olmalı)
        
    Returns:
        List[Document]: Rerank edilmiş dokümanlar
    """
    # 1. Base retriever ile daha fazla doküman al (rerank için)
    if hasattr(base_retriever, 'get_relevant_documents'):
        docs = base_retriever.get_relevant_documents(query)
    else:
        # Callable retriever
        docs = base_retriever(query)
    
    # Rerank için yeterli doküman yoksa, direkt döndür
    if len(docs) <= 1:
        return docs[:top_k] if top_k else docs
    
    # 2. Rerank et
    reranked = rerank_documents(
        query=query,
        documents=docs[:rerank_top_n],  # İlk rerank_top_n kadarını rerank et
        reranker=reranker,
        top_k=top_k
    )
    
    return reranked
