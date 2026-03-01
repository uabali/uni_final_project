"""
Agent tool tanımları.

Her tool, LangGraph agent'ının çağırabileceği bağımsız bir fonksiyondur.
Tool'lar agent tarafından seçilir; agent ReAct döngüsünde hangi tool'u
çağıracağına, sonucu gözlemledikten sonra tekrar mı deneyeceğine karar verir.
"""

from __future__ import annotations

import os
import re

from langchain_core.tools import tool
from cachetools import TTLCache


# ── RAG Search ──────────────────────────────────────────────

_rag_components: dict = {}
_SEARCH_CACHE: TTLCache | None = None
_WEB_CACHE: TTLCache | None = None


def _get_search_cache() -> TTLCache:
    global _SEARCH_CACHE
    if _SEARCH_CACHE is None:
        ttl = int(os.getenv("TOOL_CACHE_TTL", "300"))
        size = int(os.getenv("TOOL_CACHE_SIZE", "128"))
        _SEARCH_CACHE = TTLCache(maxsize=size, ttl=ttl)
    return _SEARCH_CACHE


def _get_web_cache() -> TTLCache:
    global _WEB_CACHE
    if _WEB_CACHE is None:
        ttl = int(os.getenv("TOOL_CACHE_TTL", "300"))
        size = int(os.getenv("TOOL_CACHE_SIZE", "128"))
        _WEB_CACHE = TTLCache(maxsize=size, ttl=ttl)
    return _WEB_CACHE


def register_rag_components(
    vectorstore,
    bm25_retriever,
    reranker,
) -> None:
    """
    Uygulama başlangıcında çağrılır; RAG tool'unun ihtiyaç duyduğu
    heavy objeleri (vectorstore, bm25, reranker) bir kez oluşturup
    buraya kaydeder.  Tool fonksiyonu her çağrıda bunları kullanır.
    """
    _rag_components["vectorstore"] = vectorstore
    _rag_components["bm25_retriever"] = bm25_retriever
    _rag_components["reranker"] = reranker


_SIMPLE_QUERY_TERMS = [
    "kimdir", "kim", "nedir", "ne demek", "tanimi",
    "what is", "who is", "define",
]

_QUERY_STOPWORDS = {
    "ve", "ile", "icin", "bu", "su", "o", "da", "de", "mi", "mu", "midir",
    "the", "is", "are", "what", "who", "where", "when", "how", "in", "on", "of",
}


def _is_simple_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in _SIMPLE_QUERY_TERMS) and len(q.split()) <= 8


def _tokenize_query_terms(query: str) -> list[str]:
    terms = re.findall(r"\w+", query.lower())
    return [t for t in terms if len(t) >= 3 and t not in _QUERY_STOPWORDS]


def _estimate_local_confidence(query: str, docs: list) -> float:
    """
    Retrieval sonuclarinin sorguya ne kadar alakali oldugunu 0-1 arasinda tahmin eder.

    Karar sirasi:
      1. Reranker skoru varsa (metadata.rerank_score) — en guvenilir sinyal.
      2. Yoksa term-overlap heuristigi — fallback.
    """
    if not docs:
        return 0.0

    # ── 1. Reranker skor bazli confidence ────────────────────
    top_scores = [
        doc.metadata.get("rerank_score")
        for doc in docs[:3]
        if hasattr(doc, "metadata") and doc.metadata.get("rerank_score") is not None
    ]
    if top_scores:
        best = max(top_scores)
        rerank_threshold = float(os.getenv("LOCAL_SEARCH_RERANK_THRESHOLD", "0.15"))
        if best >= rerank_threshold:
            return min(1.0, best / rerank_threshold * 0.5 + 0.5)
        return min(0.35, best / rerank_threshold * 0.35)

    # ── 2. Term-overlap fallback ─────────────────────────────
    query_terms = _tokenize_query_terms(query)
    if not query_terms:
        return 0.0

    inspected = docs[:3]
    joined = " ".join((getattr(doc, "page_content", "") or "").lower() for doc in inspected)
    if not joined.strip():
        return 0.0

    overlap = sum(1 for term in query_terms if term in joined)
    coverage = overlap / max(len(query_terms), 1)

    if len(query_terms) <= 4:
        return coverage
    return min(1.0, coverage * 1.15)


def _format_chunked_context(docs: list, query: str) -> str:
    env_max_chunks = int(os.getenv("SEARCH_TOOL_MAX_CHUNKS", "6"))
    env_max_chars = int(os.getenv("SEARCH_TOOL_MAX_CHARS_PER_CHUNK", "1200"))

    if _is_simple_query(query):
        max_chunks = min(env_max_chunks, 3)
        max_chunk_chars = min(env_max_chars, 800)
    else:
        max_chunks = env_max_chunks
        max_chunk_chars = env_max_chars

    selected = docs[:max_chunks]

    parts: list[str] = [
        f"Local document search results (query: {query})",
        "Use the following chunks as source material. Cite them in your answer as [CHUNK N].",
    ]

    for idx, doc in enumerate(selected, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", "unknown"))
        source_name = os.path.basename(source) if source else "unknown"
        page = metadata.get("page")
        page_txt = f"p.{page + 1}" if isinstance(page, int) else "p.?"
        snippet = (getattr(doc, "page_content", "") or "").strip()
        if len(snippet) > max_chunk_chars:
            snippet = snippet[:max_chunk_chars] + " ..."

        parts.append(
            f"[CHUNK {idx}] source={source_name} {page_txt}\n"
            f"{snippet}"
        )

    return "\n\n".join(parts)


@tool
def search_documents(query: str) -> str:
    """Search the local document database (PDFs, TXT files) for relevant information.
    Use this tool FIRST for any knowledge question about people, technical topics,
    lecture notes, or concepts. The query can be in Turkish or English."""

    from src.retriever import create_retriever, run_retriever
    vs = _rag_components.get("vectorstore")
    if vs is None:
        return "ERROR: Vectorstore is not initialized yet."

    cache_key = query.strip().lower()
    search_cache = _get_search_cache()
    cached = search_cache.get(cache_key)
    if cached is not None:
        return cached

    retrieval_strategy = os.getenv("RAG_RETRIEVAL_STRATEGY", "hybrid")

    retriever = create_retriever(
        vectorstore=vs,
        question=query,
        bm25_retriever=_rag_components.get("bm25_retriever"),
        strategy=retrieval_strategy,
        base_k=int(os.getenv("RAG_BASE_K", "8")),
        bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", "0.4")),
        use_rerank=(
            _rag_components.get("reranker") is not None
            and os.getenv("RAG_USE_RERANK", "true").lower() == "true"
        ),
        reranker=_rag_components.get("reranker"),
        rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "20")),
    )

    docs = run_retriever(retriever, query)

    if not docs:
        return (
            "[LOCAL_SEARCH_STATUS]: none\n"
            "[LOCAL_SEARCH_CONFIDENCE]: 0.00\n"
            "No relevant information found in local documents for this query."
        )

    confidence = _estimate_local_confidence(query, docs)
    threshold = float(os.getenv("LOCAL_SEARCH_CONF_THRESHOLD", "0.35"))
    status = "high" if confidence >= threshold else "low"

    context = _format_chunked_context(docs, query)
    output = (
        f"[LOCAL_SEARCH_STATUS]: {status}\n"
        f"[LOCAL_SEARCH_CONFIDENCE]: {confidence:.2f}\n\n"
        f"{context}"
    )

    search_cache[cache_key] = output
    return output


# ── Web Search (Tavily) ────────────────────────────────────

_NEWS_TERMS = {"haber", "son dakika", "guncel", "gelisme", "secim", "deprem"}


def _detect_topic(query: str) -> str:
    q = query.lower()
    if any(t in q for t in _NEWS_TERMS):
        return "news"
    return "general"


def _clean_snippet(text: str, max_chars: int) -> str:
    import re
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


@tool
def web_search(query: str) -> str:
    """Search the internet for real-time or up-to-date information.
    Use this tool ONLY when search_documents returned no results OR the question
    requires live data (weather, news, exchange rates, sports scores, current events)."""

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return (
            "ERROR: TAVILY_API_KEY environment variable is not set. "
            "Web search is unavailable."
        )

    cache_key = query.strip().lower()
    web_cache = _get_web_cache()
    cached = web_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        max_results = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
        snippet_max_chars = int(os.getenv("WEB_SEARCH_SNIPPET_MAX_CHARS", "160"))
        compact_mode = os.getenv("WEB_SEARCH_COMPACT_MODE", "true").lower() == "true"
        search_depth = os.getenv("WEB_SEARCH_DEPTH", "basic")
        topic = _detect_topic(query)

        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
            topic=topic,
            search_depth=search_depth,
        )

        answer = (response.get("answer") or "").strip()
        results = response.get("results", [])

        if not results and not answer:
            return "No results found from web search."

        parts: list[str] = [f"Web search results for: {query}"]

        if answer:
            parts.append(f"[Summary]: {answer}")

        for idx, r in enumerate(results, 1):
            title = r.get("title", "")
            raw_content = r.get("content", "") or ""
            content = _clean_snippet(raw_content, snippet_max_chars)
            url = r.get("url", "")
            if compact_mode:
                if url:
                    parts.append(f"Source: {url}")
            else:
                parts.append(f"[Result {idx}] {title}\n{content}\nSource: {url}")

        output = "\n\n".join(parts)
        web_cache[cache_key] = output
        return output

    except ImportError:
        return "ERROR: 'tavily-python' package is not installed. Run: pip install tavily-python"
    except Exception as exc:
        return f"Web search error: {exc}"


# ── Calculator ──────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.
    Examples: '2 + 2', '15 * 3.14', '(100 / 4) ** 2'.
    Only arithmetic operators (+, -, *, /, **) and parentheses are allowed."""
    import ast

    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return f"Security error: expression contains disallowed characters: {expression}"

    try:
        tree = ast.parse(expression, mode="eval")
        # Yalnızca sabitler ve temel aritmetik operatörlere izin ver
        _SAFE_NODES = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
            ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
        )
        for node in ast.walk(tree):
            if not isinstance(node, _SAFE_NODES):
                return f"Security error: disallowed expression node: {type(node).__name__}"
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except SyntaxError:
        return f"Syntax error in expression: {expression}"
    except Exception as exc:
        return f"Calculation error: {exc}"


# ── MCP Tool Wrappers (architecture.pdf) ────────────────────

_MCP_INVOKER = None


def set_mcp_invoker(invoker):
    """MCP HybridToolInvoker'ı kaydeder (app başlangıcında)."""
    global _MCP_INVOKER
    _MCP_INVOKER = invoker


def _create_mcp_tools():
    """MCP etkinse LangChain tool listesi döner."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    if _MCP_INVOKER is None:
        return []

    def _make_invoker(tool_name: str):
        inv = _MCP_INVOKER  # closure capture

        def _invoke(**kwargs):
            try:
                result = inv.invoke(tool_name, kwargs)
                return str(result) if result is not None else ""
            except Exception as exc:
                return f"MCP tool hatası ({tool_name}): {exc}"
        return _invoke

    class ReadFileArgs(BaseModel):
        path: str = Field(description="Dosya yolu (/data volume'dan)")

    class ListDirArgs(BaseModel):
        path: str = Field(default=".", description="Listelenecek dizin")

    class WriteFileArgs(BaseModel):
        path: str = Field(description="Dosya yolu")
        content: str = Field(description="Yazılacak içerik")

    class ExecutePythonArgs(BaseModel):
        code: str = Field(description="Çalıştırılacak Python kodu")

    class RunBashArgs(BaseModel):
        command: str = Field(description="Çalıştırılacak shell komutu")

    class QueryPostgresArgs(BaseModel):
        query: str = Field(description="Read-only SQL sorgusu")

    class MemorySearchArgs(BaseModel):
        query: str = Field(description="Hafızada aranacak sorgu")

    specs = [
        ("read_file", "Dosya içeriğini oku (/data volume).", ReadFileArgs),
        ("list_directory", "Dizindeki dosyaları listele.", ListDirArgs),
        ("write_file", "Dosyaya içerik yaz.", WriteFileArgs),
        ("execute_python", "Python kodu çalıştır (sandbox, 15s timeout).", ExecutePythonArgs),
        ("run_bash", "Shell komutu çalıştır (kısıtlı).", RunBashArgs),
        ("query_postgres", "PostgreSQL'de read-only SQL çalıştır.", QueryPostgresArgs),
        ("memory_search", "Agent hafızasında semantik arama.", MemorySearchArgs),
    ]

    return [
        StructuredTool.from_function(
            func=_make_invoker(name),
            name=name,
            description=desc,
            args_schema=schema,
        )
        for name, desc, schema in specs
    ]


def get_all_tools():
    """Local + MCP tool'ların birleşik listesi."""
    base = [search_documents, web_search, calculator]
    mcp = _create_mcp_tools()
    return base + mcp


# ── Tool registry ──────────────────────────────────────────

ALL_TOOLS = [search_documents, web_search, calculator]
