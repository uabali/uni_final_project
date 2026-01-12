"""
Retriever Modülü (Gelişmiş Arama)

Bu modül, kullanıcının sorusuna (Query) en uygun doküman parçalarını bulmaktan sorumludur.
Akıllı stratejiler kullanarak arama kalitesini artırır.

Özellikler:
- Auto Strategy: Soru tipine göre (nasıl, kaç, nedir) otomatik strateji seçimi.
- Hybrid Search: Vektör + Kelime Bazlı (BM25) arama.
- Dynamic K: Soru karmaşıklığına göre getirilecek parça sayısını ayarlar.
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

    if any(x in q for x in ["kac", "sure", "ne zaman", "dakika"]):
        return "hybrid"     # Sayısal / Kesin bilgi
    elif any(x in q for x in ["neden", "nasil"]):
        return "mmr"        # Açıklayıcı / Çeşitlilik
    else:
        return "similarity" # Hızlı varsayılan


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
    bm25_weight=0.3
):
    """
    Tüm ayarları yaparak nihai Retriever objesini oluşturur.
    Her kullanıcı sorgusunda çağrılır ve dinamik olarak yapılandırılır.
    
    Args:
        vectorstore: Qdrant veritabanı.
        question (str): Kullanıcı sorusu.
        strategy (str): "auto", "mmr", "similarity", "hybrid".
        base_k (int): Getirilecek temel chunk sayısı.
        
    Returns:
        Retriever: LangChain tarafından kullanılabilir arama motoru.
    """
    # 1. Dinamik K Hesapla
    k = calculate_dynamic_k(question, base_k)

    # 2. Strateji Belirle
    if strategy == "auto":
        strategy = auto_select_strategy(question)

    search_kwargs = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    # --- STRATEJİ UYGULAMA ---

    # SIMILARITY (En hızlı, basit benzerlik)
    if strategy == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    # MMR (Max Marginal Relevance - Çeşitlilik)
    # fetch_k kadar aday bulur, içinden en çeşitli k tanesini seçer.
    if strategy == "mmr":
        search_kwargs.update({
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        })
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )

    # THRESHOLD (Skor Eşiği - Gürültü Filtreleme)
    # Sadece belli bir benzerlik puanının üzerindekileri getirir.
    if strategy == "threshold":
        search_kwargs["score_threshold"] = score_threshold
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs
        )

    # HYBRID (Vektör + BM25)
    if strategy == "hybrid" and bm25_retriever:
        return create_hybrid_retriever(
            vectorstore,
            bm25_retriever,
            k,
            fetch_k,
            lambda_mult,
            bm25_weight
        )

    # FALLBACK (Varsayılan olarak MMR kullanılır)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    )
