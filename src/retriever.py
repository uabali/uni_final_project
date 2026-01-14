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
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


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
    Sonuçları ağırlıklandırarak (Ensemble) birleştirir.
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

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[1 - bm25_weight, bm25_weight]
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
    rerank_top_n=20
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
        
    Returns:
        Retriever veya Callable: LangChain tarafından kullanılabilir arama motoru.
    """
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
        
        # Multi-query + Rerank kombinasyonu
        if use_rerank and reranker:
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
    if use_rerank and reranker:
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
    
    # Rerank kullanılıyorsa, base retriever'ı rerank ile sarmala
    if use_rerank and reranker:
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
