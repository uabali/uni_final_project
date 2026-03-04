"""
Re-Ranker Module

This module re-ranks retrieval results using a stronger model (cross-encoder).
Re-ranking improves retrieval accuracy by 15-25%.

Features:
- Cross-encoder model reranking
- Score-based re-ordering
- Top-k selection
- TTL Cache for speedup on repeated queries
- Batch size optimization
"""

import os
import hashlib
from typing import List, Optional, Tuple
from langchain_core.documents import Document

# Use cachetools for cache
try:
    from cachetools import TTLCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# CrossEncoder import (requires sentence-transformers)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("Warning: CrossEncoder not found. Install 'sentence-transformers' package.")

# ============================================================
# RERANK CACHE (TTL Cache - 10 minutes)
# ============================================================
# maxsize: maximum number of cache entries
# ttl: time-to-live in seconds (600 = 10 minutes)
_rerank_cache: Optional[TTLCache] = None

def _get_rerank_cache() -> Optional[TTLCache]:
    """Returns the rerank cache with lazy initialization."""
    global _rerank_cache
    if _rerank_cache is None and CACHE_AVAILABLE:
        cache_ttl = int(os.getenv("RERANK_CACHE_TTL", "600"))  # default 10 minutes
        cache_size = int(os.getenv("RERANK_CACHE_SIZE", "100"))  # default 100 entries
        _rerank_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
    return _rerank_cache


def _generate_cache_key(query: str, documents: List[Document], top_k: Optional[int]) -> str:
    """
    Generates a unique cache key from query and document contents.
    
    Key format: hash(query + sorted(doc_contents) + top_k)
    """
    # Sort and merge document contents
    doc_contents = sorted([doc.page_content[:200] for doc in documents])  # first 200 chars is sufficient
    content_str = query + "||" + "||".join(doc_contents) + f"||{top_k}"
    
    # Create a short key using MD5 hash
    return hashlib.md5(content_str.encode()).hexdigest()


def get_cache_stats() -> dict:
    """Returns cache statistics (for debugging)."""
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
# RERANKER MODEL OPTIONS
# ============================================================
# High accuracy (default) - ~400MB, slower
RERANKER_MODEL_DEFAULT = "BAAI/bge-reranker-base"
# Fast mode - ~80MB, slightly lower accuracy but 40-60% faster
RERANKER_MODEL_FAST = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Model information (for reference)
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
    Determines the reranker model name to use.
    
    Priority order:
    1. model_name parameter passed to function
    2. RERANKER_MODEL env variable
    3. Default model (BAAI/bge-reranker-base)
    
    Env var values:
    - "default" or "BAAI/bge-reranker-base" -> high accuracy
    - "fast" or "cross-encoder/ms-marco-MiniLM-L-6-v2" -> fast mode
    - Any other HuggingFace model name -> that model is used
    
    Args:
        model_name: Optional model name (if None, uses env var or default)
        
    Returns:
        str: Model name to use
    """
    # 1. Parameter check
    if model_name is not None:
        # Alias check
        if model_name.lower() == "fast":
            return RERANKER_MODEL_FAST
        elif model_name.lower() == "default":
            return RERANKER_MODEL_DEFAULT
        return model_name
    
    # 2. Env var check
    env_model = os.getenv("RERANKER_MODEL")
    if env_model:
        env_model = env_model.strip()
        if env_model.lower() == "fast":
            return RERANKER_MODEL_FAST
        elif env_model.lower() == "default":
            return RERANKER_MODEL_DEFAULT
        return env_model
    
    # 3. Default
    return RERANKER_MODEL_DEFAULT


def create_reranker(
    model_name: Optional[str] = None,
    device: str = "cuda"
):
    """
    Creates the cross-encoder reranker model.
    
    Model selection (priority order):
    1. model_name parameter
    2. RERANKER_MODEL env variable
    3. Default: BAAI/bge-reranker-base
    
    Supported values:
    - "default" or "BAAI/bge-reranker-base" -> High accuracy (~400MB)
    - "fast" or "cross-encoder/ms-marco-MiniLM-L-6-v2" -> Fast mode (~80MB)
    - Any HuggingFace model name
    
    Args:
        model_name: HuggingFace model name or alias ("default", "fast")
        device: Hardware to run on (cuda/cpu)
        
    Returns:
        CrossEncoder: Reranker model
        
    Examples:
        >>> reranker = create_reranker()  # default (env var or default)
        >>> reranker = create_reranker("fast")  # fast mode
        >>> reranker = create_reranker("BAAI/bge-reranker-large")  # custom model
    """
    if not CROSS_ENCODER_AVAILABLE:
        raise ImportError("CrossEncoder not available. Install 'sentence-transformers'.")
    
    # Determine model name (with env var support)
    resolved_model = get_reranker_model_name(model_name)
    
    # Show model info
    if resolved_model == RERANKER_MODEL_FAST:
        print(f"Loading reranker model: {resolved_model} (FAST mode - ~80MB)")
    else:
        print(f"Loading reranker model: {resolved_model}")
    
    reranker = CrossEncoder(resolved_model, device=device)
    print("Reranker ready.")
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
    Re-ranks documents based on the query (re-ranking).
    
    This function:
    1. Returns immediately if found in cache (TTL cache)
    2. Feeds each document with the query to the cross-encoder (in batches)
    3. Re-orders by scores
    4. Returns the top-k best documents
    5. Saves the result to cache
    
    Args:
        query: User query
        documents: List of documents to re-rank
        reranker: CrossEncoder model
        top_k: Number of best documents to return (None = all)
        batch_size: Cross-encoder batch size (GPU memory optimization)
        use_cache: Use cache (default: True)
        
    Returns:
        List[Document]: Re-ranked documents (highest score first)
        
    Example:
        >>> reranker = create_reranker()
        >>> reranked = rerank_documents("What is Python?", docs, reranker, top_k=5)
    """
    if not documents:
        return []

    verbose = os.getenv("RERANK_VERBOSE", "false").lower() == "true"
    
    if not CROSS_ENCODER_AVAILABLE:
        print("Warning: Reranker not available. Using original ordering.")
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
            if verbose:
                print(f"Reranking: Cache hit! ({len(cached_result)} documents)")
            return cached_result
    
    # ============================================================
    # RERANKING (with batch_size optimization)
    # ============================================================
    # Create (query, document) pairs for each document
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Calculate scores with cross-encoder (using batch_size)
    try:
        scores = reranker.predict(pairs, batch_size=batch_size)
    except Exception as e:
        print(f"Reranking error: {e}. Using original ordering.")
        return documents[:top_k] if top_k else documents
    
    # Sort documents by scores (highest score first)
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take top-k and write scores to metadata
    reranked_docs = []
    for score, doc in scored_docs[:top_k] if top_k else scored_docs:
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)
    
    max_score = float(max(scores)) if len(scores) > 0 else 0.0
    if verbose:
        print(
            f"Reranking: {len(documents)} documents reduced to {len(reranked_docs)} "
            f"(best score: {max_score:.3f})"
        )
    
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
    Wraps the base retriever with a reranker.
    
    This function:
    1. Finds more documents with the base retriever (rerank_top_n)
    2. Re-ranks with the reranker (optimized with cache and batch_size)
    3. Returns the top-k best results
    
    Args:
        base_retriever: Base retriever (vectorstore retriever)
        query: User query
        reranker: CrossEncoder model
        top_k: Number of best documents to return
        rerank_top_n: Number of documents to fetch for reranking (should be more than top_k)
        batch_size: Cross-encoder batch size (GPU memory optimization)
        use_cache: Use cache (default: True)
        
    Returns:
        List[Document]: Re-ranked documents
    """
    # 1. Get more documents with base retriever (for reranking)
    if hasattr(base_retriever, "invoke"):
        docs = base_retriever.invoke(query)
    elif hasattr(base_retriever, "get_relevant_documents"):
        docs = base_retriever.get_relevant_documents(query)
    elif callable(base_retriever):
        docs = base_retriever(query)
    else:
        docs = []
    
    # Not enough documents for reranking — return directly
    if len(docs) <= 1:
        return docs[:top_k] if top_k else docs
    
    # 2. Rerank (with cache and batch_size)
    reranked = rerank_documents(
        query=query,
        documents=docs[:rerank_top_n],  # Rerank only the first rerank_top_n
        reranker=reranker,
        top_k=top_k,
        batch_size=batch_size,
        use_cache=use_cache
    )
    
    return reranked
