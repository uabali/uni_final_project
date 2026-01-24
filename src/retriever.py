"""
Retriever Modülü (Gelişmiş Arama)

Bu modül, kullanıcının sorusuna (Query) en uygun doküman parçalarını bulmaktan sorumludur.
Akıllı stratejiler kullanarak arama kalitesini artırır.

Özellikler:
- Auto Strategy: Soru tipine göre (nasıl, kaç, nedir) otomatik strateji seçimi.
- Hybrid Search: Vektör + Kelime Bazlı (BM25) arama.
- Dynamic K: Soru karmaşıklığına göre getirilecek parça sayısını ayarlar.
- Multi-query: Bir soruyu birden fazla şekilde ifade ederek arama yapar (daha iyi retrieval).
- Re-ranking: Cross-encoder ile sonuçları yeniden sıralar (%15-25 daha iyi accuracy).
- Adaptive Reranking: Basit sorgular için reranking'i atlayarak latency azaltır.
"""

import os
from typing import Dict, List, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


# ============================================================
# 0️⃣ ADAPTIVE RERANKING (AKILLI SKIP)
# ============================================================
def should_skip_rerank(question: str, fast_mode: bool = False) -> bool:
    """
    Reranking'in atlanip atlanmayacagini belirler (latency optimizasyonu).
    
    Skip Kosullari:
    1. fast_mode=True ise (kullanici hiz istiyorsa)
    2. Soru cok kisa (< 4 kelime) VE basit keyword sorgusu ise
    3. RERANK_FAST_MODE env var True ise
    
    Args:
        question: Kullanici sorusu
        fast_mode: Hizli mod aktif mi (varsayilan: False)
        
    Returns:
        bool: True ise reranking atlanir
        
    Ornekler:
        - "Python" -> True (tek kelime, skip)
        - "API nedir" -> True (2 kelime, basit, skip)
        - "Python'da async nasil calisir ve neden onemlidir" -> False (karmasik, rerank)
    """
    # Env var kontrolu (global override)
    env_fast_mode = os.getenv("RERANK_FAST_MODE", "false").lower() == "true"
    if fast_mode or env_fast_mode:
        return True
    
    # Soru analizi
    q = question.strip().lower()
    words = q.split()
    word_count = len(words)
    
    # Cok kisa sorgular (< 4 kelime)
    if word_count < 4:
        # Ama karmasiklik gostergesi varsa rerank yap
        complexity_indicators = ["nasil", "neden", "hangi", "ne zaman", "arasindaki fark"]
        has_complexity = any(ind in q for ind in complexity_indicators)
        if not has_complexity:
            return True  # Basit kisa sorgu -> skip
    
    # Tek kelime sorgulari her zaman skip
    if word_count == 1:
        return True
    
    return False  # Karmasik sorgu -> rerank yap


def get_rerank_decision(question: str, use_rerank: bool, reranker, fast_mode: bool = False) -> bool:
    """
    Reranking yapilip yapilmayacagina karar verir.
    
    Args:
        question: Kullanici sorusu
        use_rerank: Kullanici tercihi (True/False)
        reranker: Reranker modeli (None ise zaten yapamaz)
        fast_mode: Hizli mod aktif mi
        
    Returns:
        bool: True ise reranking yapilacak
    """
    # Temel kontroller
    if not use_rerank or reranker is None:
        return False
    
    # Adaptive skip kontrolu
    if should_skip_rerank(question, fast_mode):
        print(f"Reranking: Skipped (basit sorgu veya fast_mode)")
        return False
    
    return True


# ============================================================
# 1️⃣ BM25 BUILDER (UYGULAMA BAŞLANGICINDA 1 KEZ ÇAĞRILIR)
# ============================================================
def build_bm25_retriever(docs, k=6):
    """
    BM25 (Kelime bazlı arama) indeksini oluşturur.
    Bunu her sorguda DEĞİL, uygulama başında 1 kez yapmak performans için kritiktir.
    
    Args:
        docs (list): Tüm doküman parçaları.
        
    Returns:
        BM25Retriever: Hazır BM25 arama motoru.
    """
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25


# ============================================================
# 2️⃣ DYNAMIC K (SORU KARMAŞIKLIĞINA GÖRE PARÇA SAYISI)
# ============================================================
def calculate_dynamic_k(question: str, base_k: int = 6, max_k: int = 12) -> int:
    """
    Sorunun karmaşıklığına göre 'k' (getirilecek chunk sayısı) değerini hesaplar.
    Karmaşık sorular ("ve", "neden", "nasıl") daha fazla bağlama ihtiyaç duyar.
    
    Args:
        question (str): Kullanıcı sorusu.
        base_k (int): Temel chunk sayısı (varsayılan: 6).
        
    Returns:
        int: Hesaplanan k değeri.
    """
    q = question.lower()
    indicators = ["ve", "neden", "nasil", "hangi", "ne zaman", "kim"]
    score = sum(1 for x in indicators if x in q)

    if score >= 2:
        return min(base_k + 4, max_k)  # Çok karmaşık -> +4 chunk
    elif score == 1:
        return min(base_k + 2, max_k)  # Orta -> +2 chunk
    return base_k  # Basit -> varsayılan


# ============================================================
# 3️⃣ AUTO STRATEGY SELECTION (OTOMATİK STRATEJİ SEÇİMİ)
# ============================================================
def auto_select_strategy(question: str) -> str:
    """
    Sorunun türüne göre en iyi arama stratejisini seçer.
    
    Kurallar:
    - Sayısal/Kesin bilgi ("kaç", "süre", "dakika") -> Hybrid (BM25 + Vektör)
    - Açıklayıcı ("neden", "nasıl") -> MMR (Çeşitlilik odaklı)
    - Diğerleri -> Similarity (Hızlı, benzerlik odaklı)
    """
    q = question.lower()

    # Sayısal / kesin sorular
    if any(x in q for x in ["kac", "sure", "ne zaman", "dakika"]):
        return "hybrid"     # Sayısal / Kesin bilgi

    # Açıklayıcı sorular
    if any(x in q for x in ["neden", "nasil"]):
        return "mmr"        # Açıklayıcı / Çeşitlilik

    # Kullanim alanlari / nerelerde kullanilir gibi genis kapsamli sorular icin
    # zayif eslesmeleri elemek uzere threshold kullan
    if any(x in q for x in ["kullanim alanlari", "hangi projelerde", "nerelerde kullanilir", "nerede kullanilir"]):
        return "threshold"  # Skor esigine gore filtreleme

    return "similarity"     # Hızlı varsayılan


# ============================================================
# 3️⃣ HYBRID MERGE (RRF - WEIGHTED)
# ============================================================
def _doc_key(doc: Document) -> Tuple[str, str]:
    """Dokumanlari benzersizlestirmek icin anahtar."""
    source = str(doc.metadata.get("source", ""))
    content = doc.page_content[:200]
    return (source, content)


def _rrf_merge(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    bm25_weight: float,
    top_k: int,
    rrf_k: int = 60
) -> List[Document]:
    """
    RRF (Reciprocal Rank Fusion) ile iki listeyi birlestirir.
    Her liste icin skor: weight * (1 / (rrf_k + rank))
    """
    scores: Dict[Tuple[str, str], float] = {}
    doc_map: Dict[Tuple[str, str], Document] = {}

    vector_weight = 1.0 - bm25_weight

    for rank, doc in enumerate(vector_docs, start=1):
        key = _doc_key(doc)
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + (vector_weight / (rrf_k + rank))

    for rank, doc in enumerate(bm25_docs, start=1):
        key = _doc_key(doc)
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + (bm25_weight / (rrf_k + rank))

    # Skorlara gore sirala
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged_docs = [doc_map[key] for key, _ in ranked]
    return merged_docs[:top_k]


class HybridRetriever:
    """
    Vektor ve BM25 retriever'larini RRF ile birlestiren custom retriever.
    LangChain 1.x ile uyumlu (EnsembleRetriever yerine).
    """

    def __init__(self, vector_retriever, bm25_retriever, bm25_weight: float = 0.3, top_k: int = 6):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.bm25_weight = bm25_weight
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        return _rrf_merge(vector_docs, bm25_docs, self.bm25_weight, self.top_k)

    def __call__(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


# ============================================================
# 4️⃣ HYBRID RETRIEVER (VEKTÖR + BM25 BİRLEŞİMİ)
# ============================================================
def create_hybrid_retriever(
    vectorstore,
    bm25_retriever,
    k,
    fetch_k,
    lambda_mult,
    bm25_weight
):
    """
    Hybrid Retriever oluşturur: Hem anlamsal (vektör) hem de kelime (BM25) araması yapar.
    Sonuçları ağırlıklandırarak (RRF) birleştirir.
    """
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )

    bm25_retriever.k = k

    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        bm25_weight=bm25_weight,
        top_k=k
    )


# ============================================================
# 5️⃣ MAIN: CREATE RETRIEVER (ANA FONKSİYON)
# ============================================================
def create_retriever(
    vectorstore,
    question,
    bm25_retriever=None,
    strategy="auto",
    base_k=6,
    fetch_k=20,
    lambda_mult=0.7,
    score_threshold=0.75,
    metadata_filter=None,
    bm25_weight=0.3,
    use_multi_query=False,
    llm=None,
    num_queries=3,
    use_rerank=False,
    reranker=None,
    rerank_top_n=20,
    fast_mode=False
):
    """
    Tüm ayarları yaparak nihai Retriever objesini oluşturur.
    Her kullanıcı sorgusunda çağrılır ve dinamik olarak yapılandırılır.
    
    Args:
        vectorstore: Qdrant veritabanı.
        question (str): Kullanıcı sorusu.
        strategy (str): "auto", "mmr", "similarity", "hybrid".
        base_k (int): Getirilecek temel chunk sayısı.
        use_multi_query (bool): Multi-query tekniğini kullan (varsayılan: False).
        llm: LLM modeli (multi-query için gerekli).
        num_queries (int): Multi-query için alternatif soru sayısı (varsayılan: 3).
        use_rerank (bool): Re-ranking kullan (varsayılan: False).
        reranker: CrossEncoder modeli (rerank için gerekli).
        rerank_top_n (int): Reranking için alınacak doküman sayısı (varsayılan: 20).
        fast_mode (bool): Hizli mod - basit sorgularda reranking atlanir (varsayilan: False).
        
    Returns:
        Retriever veya Callable: LangChain tarafından kullanılabilir arama motoru.
    """
    # Adaptive reranking karari
    should_rerank = get_rerank_decision(question, use_rerank, reranker, fast_mode)
    
    # Multi-query kullanılıyorsa
    if use_multi_query and llm:
        from src.query_translation import create_multi_query_retriever
        base_retriever = create_multi_query_retriever(
            vectorstore=vectorstore,
            question=question,
            llm=llm,
            num_queries=num_queries,
            bm25_retriever=bm25_retriever,
            strategy=strategy,
            base_k=base_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
            bm25_weight=bm25_weight
        )
        
        # Multi-query + Rerank kombinasyonu (adaptive skip ile)
        if should_rerank:
            from src.reranker import create_rerank_retriever
            return lambda q: create_rerank_retriever(
                base_retriever=base_retriever,
                query=q,
                reranker=reranker,
                top_k=base_k,
                rerank_top_n=rerank_top_n
            )
        
        return base_retriever
    
    # 1. Dinamik K Hesapla
    k = calculate_dynamic_k(question, base_k)

    # 2. Strateji Belirle
    if strategy == "auto":
        strategy = auto_select_strategy(question)

    # Rerank kullanılıyorsa, daha fazla doküman al (rerank için)
    if should_rerank:
        search_k = max(rerank_top_n, k * 2)  # Rerank için daha fazla al
    else:
        search_k = k

    search_kwargs = {"k": search_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    # --- STRATEJİ UYGULAMA ---

    # SIMILARITY (En hızlı, basit benzerlik)
    if strategy == "similarity":
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    # MMR (Max Marginal Relevance - Çeşitlilik)
    elif strategy == "mmr":
        search_kwargs.update({
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        })
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
    # THRESHOLD (Skor Eşiği - Gürültü Filtreleme)
    elif strategy == "threshold":
        search_kwargs["score_threshold"] = score_threshold
        base_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs
        )
    # HYBRID (Vektör + BM25)
    elif strategy == "hybrid" and bm25_retriever:
        base_retriever = create_hybrid_retriever(
            vectorstore,
            bm25_retriever,
            search_k,
            fetch_k,
            lambda_mult,
            bm25_weight
        )
    # FALLBACK (Varsayılan olarak MMR kullanılır)
    else:
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": search_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
        )
    
    # Rerank kullanılıyorsa, base retriever'ı rerank ile sarmala (adaptive skip ile)
    if should_rerank:
        from src.reranker import create_rerank_retriever
        
        def rerank_wrapper(query: str):
            return create_rerank_retriever(
                base_retriever=base_retriever,
                query=query,
                reranker=reranker,
                top_k=k,  # Final k değeri
                rerank_top_n=rerank_top_n
            )
        
        return rerank_wrapper
    
    return base_retriever
