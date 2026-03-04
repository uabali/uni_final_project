"""
Retriever Module (Advanced Search)

This module is responsible for finding the most relevant document chunks
for the user's query. It improves search quality using smart strategies.

Features:
- Auto Strategy: Automatic strategy selection based on question type (how, how many, what is).
- Hybrid Search: Vector + Keyword-based (BM25) search.
- Dynamic K: Adjusts the number of chunks to retrieve based on question complexity.
- Multi-query: Searches by rephrasing a question in multiple ways (better retrieval).
- Re-ranking: Re-ranks results using a cross-encoder (15-25% better accuracy).
- Adaptive Reranking: Skips reranking for simple queries to reduce latency.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def run_retriever(retriever, query: str) -> List[Document]:
    """
    Runs different retriever interfaces from a single place.

    Support order:
    1) invoke(query)
    2) get_relevant_documents(query)
    3) callable(query)
    """
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
        return docs if isinstance(docs, list) else []

    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query)
        return docs if isinstance(docs, list) else []

    if callable(retriever):
        docs = retriever(query)
        return docs if isinstance(docs, list) else []

    return []


# ============================================================
# 0. RERANK DECISION
# ============================================================
def get_rerank_decision(question: str, use_rerank: bool, reranker, fast_mode: bool = False) -> bool:
    if not use_rerank or reranker is None:
        return False

    env_fast = os.getenv("RERANK_FAST_MODE", "false").lower() == "true"
    if fast_mode or env_fast:
        return False

    return True


# ============================================================
# 1. BM25 BUILDER (CALLED ONCE AT APPLICATION STARTUP)
# ============================================================
def build_bm25_retriever(docs, k=6):
    """
    Builds the BM25 (keyword-based search) index.
    Doing this once at startup (not per query) is critical for performance.
    
    Args:
        docs (list): All document chunks.
        
    Returns:
        BM25Retriever: Ready BM25 search engine.
    """
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25


# ============================================================
# 2. DYNAMIC K (CHUNK COUNT BASED ON QUESTION COMPLEXITY)
# ============================================================
def calculate_dynamic_k(question: str, base_k: int = 8, max_k: int = 15) -> int:
    q = question.lower()
    indicators = ["ve", "neden", "nasil", "hangi", "ne zaman", "kim", "arasindaki fark",
                   "karsilastir", "and", "how", "why", "which", "compare", "difference"]
    score = sum(1 for x in indicators if x in q)

    if score >= 2:
        return min(base_k + 4, max_k)
    elif score == 1:
        return min(base_k + 2, max_k)
    return base_k


# ============================================================
# 3. AUTO STRATEGY SELECTION
# ============================================================
def auto_select_strategy(question: str) -> str:
    """
    Selects the best search strategy based on the question type.
    
    Rules:
    - Numeric/exact info ("how many", "duration", "minutes") -> Hybrid (BM25 + Vector)
    - Explanatory ("why", "how") -> MMR (Diversity-focused)
    - Others -> Similarity (Fast, similarity-focused)
    """
    q = question.lower()

    # Numeric / exact queries
    if any(x in q for x in ["kac", "sure", "ne zaman", "dakika", "how many", "how long", "when", "duration"]):
        return "hybrid"     # Numeric / Exact info

    # Explanatory queries
    if any(x in q for x in ["neden", "nasil", "why", "how"]):
        return "mmr"        # Explanatory / Diversity

    # Use cases / broad scope queries — use threshold to filter weak matches
    if any(x in q for x in ["kullanim alanlari", "hangi projelerde", "nerelerde kullanilir",
                              "nerede kullanilir", "use cases", "applications", "where is it used"]):
        return "threshold"  # Score threshold filtering

    return "similarity"     # Fast default


# ============================================================
# 3. HYBRID MERGE (RRF - WEIGHTED)
# ============================================================
def _doc_key(doc: Document) -> Tuple[str, str]:
    """Key for deduplicating documents."""
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
    Merges two lists using RRF (Reciprocal Rank Fusion).
    Score for each list: weight * (1 / (rrf_k + rank))
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

    # Sort by scores
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged_docs = [doc_map[key] for key, _ in ranked]
    return merged_docs[:top_k]


class HybridRetriever:
    """
    Custom retriever that merges vector and BM25 retrievers using RRF.
    Compatible with LangChain 1.x (replaces EnsembleRetriever).
    """

    def __init__(self, vector_retriever, bm25_retriever, bm25_weight: float = 0.3, top_k: int = 6):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.bm25_weight = bm25_weight
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            vec_future = executor.submit(run_retriever, self.vector_retriever, query)
            bm25_future = executor.submit(run_retriever, self.bm25_retriever, query)
            vector_docs = vec_future.result()
            bm25_docs = bm25_future.result()
        return _rrf_merge(vector_docs, bm25_docs, self.bm25_weight, self.top_k)

    def __call__(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


# ============================================================
# 4. HYBRID RETRIEVER (VECTOR + BM25 COMBINATION)
# ============================================================
def create_hybrid_retriever(
    vectorstore,
    bm25_retriever,
    k,
    fetch_k,
    lambda_mult,
    bm25_weight,
):
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    bm25_retriever.k = max(k, int(k * 1.5))

    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        bm25_weight=bm25_weight,
        top_k=k,
    )


# ============================================================
# 5. MAIN: CREATE RETRIEVER (MAIN FUNCTION)
# ============================================================
def create_retriever(
    vectorstore,
    question,
    bm25_retriever=None,
    strategy="auto",
    base_k=8,
    fetch_k=30,
    lambda_mult=0.6,
    score_threshold=0.70,
    metadata_filter=None,
    bm25_weight=0.4,
    use_multi_query=False,
    llm=None,
    num_queries=3,
    use_rerank=False,
    reranker=None,
    rerank_top_n=20,
    fast_mode=False,
):
    """
    Configures and returns the final Retriever object with all settings applied.
    Called for each user query and dynamically configured.
    
    Args:
        vectorstore: Qdrant database.
        question (str): User query.
        strategy (str): "auto", "mmr", "similarity", "hybrid".
        base_k (int): Base number of chunks to retrieve.
        use_multi_query (bool): Use multi-query technique (default: False).
        llm: LLM model (required for multi-query).
        num_queries (int): Number of alternative queries for multi-query (default: 3).
        use_rerank (bool): Use re-ranking (default: False).
        reranker: CrossEncoder model (required for rerank).
        rerank_top_n (int): Number of documents to fetch for reranking (default: 20).
        fast_mode (bool): Fast mode — skip reranking for simple queries (default: False).
        
    Returns:
        Retriever or Callable: Search engine usable by LangChain.
    """
    # Adaptive reranking decision
    should_rerank = get_rerank_decision(question, use_rerank, reranker, fast_mode)
    
    # If multi-query is being used
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
        
        # Multi-query + Rerank combination (with adaptive skip)
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
    
    # 1. Calculate Dynamic K
    k = calculate_dynamic_k(question, base_k)

    # 2. Determine Strategy
    if strategy == "auto":
        strategy = auto_select_strategy(question)

    # If rerank is used, fetch more documents (for reranking)
    if should_rerank:
        search_k = max(rerank_top_n, k * 2)  # Fetch more for rerank
    else:
        search_k = k

    search_kwargs = {"k": search_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    # --- STRATEGY APPLICATION ---

    # SIMILARITY (Fastest, simple similarity)
    if strategy == "similarity":
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    # MMR (Max Marginal Relevance - Diversity)
    elif strategy == "mmr":
        search_kwargs.update({
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        })
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
    # THRESHOLD (Score Threshold - Noise Filtering)
    elif strategy == "threshold":
        search_kwargs["score_threshold"] = score_threshold
        base_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs
        )
    # HYBRID (Vector + BM25)
    elif strategy == "hybrid" and bm25_retriever:
        base_retriever = create_hybrid_retriever(
            vectorstore,
            bm25_retriever,
            search_k,
            fetch_k,
            lambda_mult,
            bm25_weight
        )
    # FALLBACK (Default: MMR)
    else:
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": search_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
        )
    
    # If rerank is used, wrap base retriever with rerank (with adaptive skip)
    if should_rerank:
        from src.reranker import create_rerank_retriever
        
        def rerank_wrapper(query: str):
            return create_rerank_retriever(
                base_retriever=base_retriever,
                query=query,
                reranker=reranker,
                top_k=k,  # Final k value
                rerank_top_n=rerank_top_n
            )
        
        return rerank_wrapper
    
    return base_retriever
