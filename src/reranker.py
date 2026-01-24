"""
Re-Ranker Modülü

Bu modül, retrieval sonuçlarını daha güçlü bir modelle (cross-encoder) yeniden sıralar.
Re-ranking, retrieval accuracy'yi %15-25 artırır.

Özellikler:
- Cross-encoder modeli ile reranking
- Skor bazlı yeniden sıralama
- Top-k seçimi
- TTL Cache ile tekrar eden sorgular için hızlandırma
- Batch size optimizasyonu
"""

import os
import hashlib
from typing import List, Optional, Tuple
from langchain_core.documents import Document

# Cache için cachetools kullan
try:
    from cachetools import TTLCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# CrossEncoder import (sentence-transformers gerekli)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("Uyarı: CrossEncoder bulunamadı. 'sentence-transformers' paketini yükleyin.")

# ============================================================
# RERANK CACHE (TTL Cache - 10 dakika)
# ============================================================
# maxsize: maksimum cache girisi sayisi
# ttl: saniye cinsinden yasam suresi (600 = 10 dakika)
_rerank_cache: Optional[TTLCache] = None

def _get_rerank_cache() -> Optional[TTLCache]:
    """Lazy initialization ile rerank cache'ini dondurur."""
    global _rerank_cache
    if _rerank_cache is None and CACHE_AVAILABLE:
        cache_ttl = int(os.getenv("RERANK_CACHE_TTL", "600"))  # varsayilan 10 dakika
        cache_size = int(os.getenv("RERANK_CACHE_SIZE", "100"))  # varsayilan 100 girdi
        _rerank_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
    return _rerank_cache


def _generate_cache_key(query: str, documents: List[Document], top_k: Optional[int]) -> str:
    """
    Query ve dokuman iceriklerinden benzersiz cache key olusturur.
    
    Key formati: hash(query + sorted(doc_contents) + top_k)
    """
    # Dokuman iceriklerini sirala ve birlestir
    doc_contents = sorted([doc.page_content[:200] for doc in documents])  # ilk 200 karakter yeterli
    content_str = query + "||" + "||".join(doc_contents) + f"||{top_k}"
    
    # MD5 hash ile kisa key olustur
    return hashlib.md5(content_str.encode()).hexdigest()


def get_cache_stats() -> dict:
    """Cache istatistiklerini dondurur (debug icin)."""
    cache = _get_rerank_cache()
    if cache is None:
        return {"available": False}
    return {
        "available": True,
        "size": len(cache),
        "maxsize": cache.maxsize,
        "ttl": cache.ttl
    }


# ============================================================
# RERANKER MODEL SECENEKLERI
# ============================================================
# Yuksek accuracy (varsayilan) - ~400MB, daha yavas
RERANKER_MODEL_DEFAULT = "BAAI/bge-reranker-base"
# Hizli mod - ~80MB, biraz dusuk accuracy ama %40-60 daha hizli
RERANKER_MODEL_FAST = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Model bilgileri (referans icin)
RERANKER_MODELS = {
    "default": {
        "name": RERANKER_MODEL_DEFAULT,
        "size": "~400MB",
        "speed": "normal",
        "accuracy": "high"
    },
    "fast": {
        "name": RERANKER_MODEL_FAST,
        "size": "~80MB",
        "speed": "fast",
        "accuracy": "medium"
    }
}


def get_reranker_model_name(model_name: Optional[str] = None) -> str:
    """
    Kullanilacak reranker model adini belirler.
    
    Oncelik sirasi:
    1. Fonksiyona verilen model_name parametresi
    2. RERANKER_MODEL env degiskeni
    3. Varsayilan model (BAAI/bge-reranker-base)
    
    Env var degerleri:
    - "default" veya "BAAI/bge-reranker-base" -> yuksek accuracy
    - "fast" veya "cross-encoder/ms-marco-MiniLM-L-6-v2" -> hizli mod
    - Baska bir HuggingFace model adi -> o model kullanilir
    
    Args:
        model_name: Opsiyonel model adi (None ise env var veya varsayilan)
        
    Returns:
        str: Kullanilacak model adi
    """
    # 1. Parametre kontrolu
    if model_name is not None:
        # Alias kontrolu
        if model_name.lower() == "fast":
            return RERANKER_MODEL_FAST
        elif model_name.lower() == "default":
            return RERANKER_MODEL_DEFAULT
        return model_name
    
    # 2. Env var kontrolu
    env_model = os.getenv("RERANKER_MODEL")
    if env_model:
        env_model = env_model.strip()
        if env_model.lower() == "fast":
            return RERANKER_MODEL_FAST
        elif env_model.lower() == "default":
            return RERANKER_MODEL_DEFAULT
        return env_model
    
    # 3. Varsayilan
    return RERANKER_MODEL_DEFAULT


def create_reranker(
    model_name: Optional[str] = None,
    device: str = "cuda"
):
    """
    Cross-encoder reranker modelini oluşturur.
    
    Model secimi (oncelik sirasi):
    1. model_name parametresi
    2. RERANKER_MODEL env degiskeni
    3. Varsayilan: BAAI/bge-reranker-base
    
    Desteklenen degerler:
    - "default" veya "BAAI/bge-reranker-base" -> Yuksek accuracy (~400MB)
    - "fast" veya "cross-encoder/ms-marco-MiniLM-L-6-v2" -> Hizli mod (~80MB)
    - Herhangi bir HuggingFace model adi
    
    Args:
        model_name: HuggingFace model adı veya alias ("default", "fast")
        device: Çalışacağı donanım (cuda/cpu)
        
    Returns:
        CrossEncoder: Reranker modeli
        
    Ornekler:
        >>> reranker = create_reranker()  # varsayilan (env var veya default)
        >>> reranker = create_reranker("fast")  # hizli mod
        >>> reranker = create_reranker("BAAI/bge-reranker-large")  # ozel model
    """
    if not CROSS_ENCODER_AVAILABLE:
        raise ImportError("CrossEncoder mevcut değil. 'sentence-transformers' yükleyin.")
    
    # Model adini belirle (env var destegi ile)
    resolved_model = get_reranker_model_name(model_name)
    
    # Model bilgisi goster
    if resolved_model == RERANKER_MODEL_FAST:
        print(f"Reranker modeli yükleniyor: {resolved_model} (FAST mode - ~80MB)")
    else:
        print(f"Reranker modeli yükleniyor: {resolved_model}")
    
    reranker = CrossEncoder(resolved_model, device=device)
    print("Reranker hazır.")
    return reranker


def rerank_documents(
    query: str,
    documents: List[Document],
    reranker,
    top_k: Optional[int] = None,
    batch_size: int = 8,
    use_cache: bool = True
) -> List[Document]:
    """
    Dokümanları soruya göre yeniden sıralar (re-ranking).
    
    Bu fonksiyon:
    1. Cache'de varsa hemen dondurur (TTL cache)
    2. Her dokümanı soruyla birlikte cross-encoder'a verir (batch halinde)
    3. Skorlarına göre yeniden sıralar
    4. Top-k kadar en iyi dokümanları döndürür
    5. Sonucu cache'e kaydeder
    
    Args:
        query: Kullanıcı sorusu
        documents: Yeniden sıralanacak dokümanlar listesi
        reranker: CrossEncoder modeli
        top_k: Döndürülecek en iyi doküman sayısı (None = hepsi)
        batch_size: Cross-encoder batch boyutu (GPU memory optimizasyonu)
        use_cache: Cache kullanimi (varsayilan: True)
        
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
    
    # ============================================================
    # CACHE LOOKUP
    # ============================================================
    cache = _get_rerank_cache() if use_cache else None
    cache_key = None
    
    if cache is not None:
        cache_key = _generate_cache_key(query, documents, top_k)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            print(f"Reranking: Cache hit! ({len(cached_result)} doküman)")
            return cached_result
    
    # ============================================================
    # RERANKING (with batch_size optimization)
    # ============================================================
    # Her doküman için (query, document) çifti oluştur
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Cross-encoder ile skorları hesapla (batch_size ile)
    try:
        scores = reranker.predict(pairs, batch_size=batch_size)
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
    
    # ============================================================
    # CACHE STORE
    # ============================================================
    if cache is not None and cache_key is not None:
        cache[cache_key] = reranked_docs
    
    return reranked_docs


def create_rerank_retriever(
    base_retriever,
    query: str,
    reranker,
    top_k: Optional[int] = None,
    rerank_top_n: int = 20,
    batch_size: int = 8,
    use_cache: bool = True
):
    """
    Base retriever'ı reranker ile sarmalar.
    
    Bu fonksiyon:
    1. Base retriever ile daha fazla doküman bulur (rerank_top_n)
    2. Reranker ile yeniden sıralar (cache ve batch_size ile optimize)
    3. Top-k kadar en iyisini döndürür
    
    Args:
        base_retriever: Temel retriever (vectorstore retriever)
        query: Kullanıcı sorusu
        reranker: CrossEncoder modeli
        top_k: Döndürülecek en iyi doküman sayısı
        rerank_top_n: Reranking için alınacak doküman sayısı (top_k'dan fazla olmalı)
        batch_size: Cross-encoder batch boyutu (GPU memory optimizasyonu)
        use_cache: Cache kullanimi (varsayilan: True)
        
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
    
    # 2. Rerank et (cache ve batch_size ile)
    reranked = rerank_documents(
        query=query,
        documents=docs[:rerank_top_n],  # İlk rerank_top_n kadarını rerank et
        reranker=reranker,
        top_k=top_k,
        batch_size=batch_size,
        use_cache=use_cache
    )
    
    return reranked
