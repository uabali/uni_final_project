"""
Agent tool definitions.

Each tool is an independent function that the LangGraph agent can call.
Tools are selected by the agent; in the ReAct loop, the agent decides which tool
to call and whether to retry after observing the result.
"""

from __future__ import annotations

import os
import re
import time

from langchain_core.tools import tool
from cachetools import TTLCache

from src.context import get_request_context
from src.audit import get_audit_logger
from src.vectorstore import get_vectorstore_for_department, build_collection_name


# ── RAG Search ──────────────────────────────────────────────

# Track recently ingested files — for "in this document" type queries
_recently_ingested_files: list[dict] = []  # [{"source": "...", "timestamp": ...}]


def register_recently_ingested(source_name: str) -> None:
    """Registers a newly ingested file (called after ingest)."""
    _recently_ingested_files.append({
        "source": source_name,
        "timestamp": time.time(),
    })
    # Keep only the last 20 files
    if len(_recently_ingested_files) > 20:
        _recently_ingested_files.pop(0)


def _extract_file_reference(query: str) -> str | None:
    """
    Extracts a file reference from the query.
    
    Supported patterns:
    - "in this document", "in this file", "in this PDF" → last ingested file
    - "in architecture.pdf" → direct file name
    - "[Recently ingested files: X.pdf]" → from metadata
    - Turkish patterns: "bu belgede", "bu dosyada", "bu PDF'de"
    """
    ql = query.lower()
    
    # 1. Direct file name reference ("about architecture.pdf", "in X.pdf")
    file_match = re.search(r'([\w\-\.]+\.(?:pdf|txt|docx|md))', ql)
    if file_match:
        return file_match.group(1)
    
    # 2. "Recently ingested files" metadata (Turkish: "Sisteme yeni yuklenen dosyalar")
    upload_match = re.search(r'\[(?:recently ingested files?|sisteme yeni y[uü]klenen dosyalar?):\s*([^\]]+)\]', ql)
    if upload_match:
        files_str = upload_match.group(1).strip()
        # Get the first file
        first_file = files_str.split(',')[0].strip()
        return first_file if first_file else None
    
    # 3. "this document", "this file", "this PDF" → reference to last ingested file
    deictic_patterns = [
        r'this\s+document', r'this\s+file', r'this\s+pdf',
        r'uploaded\s+document', r'uploaded\s+file',
        r'bu\s+belge', r'bu\s+dosya', r'bu\s+pdf',
        r'bu\s+dokuman', r'yuklen\w*\s+belge', r'yuklen\w*\s+dosya',
    ]
    if any(re.search(p, ql) for p in deictic_patterns):
        if _recently_ingested_files:
            return _recently_ingested_files[-1]["source"]
    
    return None


def _filter_docs_by_source(docs: list, source_name: str) -> list:
    """
    Filters document list by a specific source file.
    Prioritizes chunks matching source_name.
    Returns the original list if no match is found.
    """
    if not source_name or not docs:
        return docs
    
    source_lower = source_name.lower()
    matched = []
    unmatched = []
    
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        doc_source = str(metadata.get("source", "")).lower()
        doc_basename = os.path.basename(doc_source).lower()
        
        if source_lower in doc_source or source_lower in doc_basename:
            matched.append(doc)
        else:
            unmatched.append(doc)
    
    if matched:
        # Matched chunks first, then others (but limited)
        return matched + unmatched[:2]
    
    # No match — return original list
    return docs


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
    Called at application startup; creates the heavy objects needed by
    the RAG tool (vectorstore, bm25, reranker) once and registers them here.
    The tool function uses these on every call.
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
    # Turkish question words and frequently used general words
    "nedir", "kimdir", "hangi", "nasil", "neden", "kac", "ne", "var", "yok",
}


def _is_simple_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in _SIMPLE_QUERY_TERMS) and len(q.split()) <= 8


def _tokenize_query_terms(query: str) -> list[str]:
    terms = re.findall(r"\w+", query.lower())
    return [t for t in terms if len(t) >= 3 and t not in _QUERY_STOPWORDS]


def _estimate_local_confidence(query: str, docs: list) -> float:
    """
    Estimates how relevant the retrieval results are to the query (0-1).

    Decision order:
      1. Reranker score if available (metadata.rerank_score) — most reliable signal.
      2. Otherwise term-overlap heuristic — fallback.
    """
    if not docs:
        return 0.0

    # ── 1. Reranker score-based confidence ────────────────────
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
    vs_base = _rag_components.get("vectorstore")
    if vs_base is None:
        return "ERROR: Vectorstore is not initialized yet."

    cache_key = query.strip().lower()
    search_cache = _get_search_cache()
    cached = search_cache.get(cache_key)
    if cached is not None:
        return cached

    # Detect file reference
    file_ref = _extract_file_reference(query)

    # Prepare Qdrant filter for department-based namespace isolation
    ctx = get_request_context()
    metadata_filter = None
    if ctx is not None and getattr(ctx, "department_id", None):
        try:
            from qdrant_client.http import models as qmodels

            metadata_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="department_id",
                        match=qmodels.MatchValue(value=ctx.department_id),
                    )
                ]
            )
        except Exception:
            # If Qdrant client is unavailable or filter creation fails, silently skip
            metadata_filter = None

    # In strict multi-tenant mode, use department-based collection
    if ctx is not None and getattr(ctx, "department_id", None):
        base_collection = os.getenv("QDRANT_COLLECTION", "rag_collection").strip()
        vs = get_vectorstore_for_department(
            vs_base,
            base_collection,
            ctx.department_id,
        )
    else:
        vs = vs_base

    retrieval_strategy = os.getenv("RAG_RETRIEVAL_STRATEGY", "hybrid")

    retriever = create_retriever(
        vectorstore=vs,
        question=query,
        bm25_retriever=_rag_components.get("bm25_retriever"),
        strategy=retrieval_strategy,
        base_k=int(os.getenv("RAG_BASE_K", "8")),
        bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", "0.4")),
        metadata_filter=metadata_filter,
        use_rerank=(
            _rag_components.get("reranker") is not None
            and os.getenv("RAG_USE_RERANK", "true").lower() == "true"
        ),
        reranker=_rag_components.get("reranker"),
        rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "20")),
    )

    docs = run_retriever(retriever, query)

    # If file reference exists, filter chunks by that file
    if file_ref and docs:
        docs = _filter_docs_by_source(docs, file_ref)

    if not docs:
        audit = get_audit_logger()
        if audit.is_available:
            audit.log_rag_retrieval(
                context=ctx,
                query=query,
                status="none",
                confidence=0.0,
                num_docs=0,
                extra={"file_ref": file_ref},
            )
        return (
            "[LOCAL_SEARCH_STATUS]: none\n"
            "[LOCAL_SEARCH_CONFIDENCE]: 0.00\n"
            "No relevant information found in local documents for this query."
        )

    confidence = _estimate_local_confidence(query, docs)
    
    # If file filter was applied and matched chunks exist, boost confidence
    if file_ref:
        matched_count = sum(
            1 for doc in docs
            if file_ref.lower() in str(getattr(doc, "metadata", {}).get("source", "")).lower()
        )
        if matched_count > 0:
            confidence = max(confidence, 0.60)  # Minimum 0.60 if file matches
    
    threshold = float(os.getenv("LOCAL_SEARCH_CONF_THRESHOLD", "0.35"))
    status = "high" if confidence >= threshold else "low"

    context = _format_chunked_context(docs, query)
    
    # Add file filter info
    filter_info = ""
    if file_ref:
        filter_info = f"[DOCUMENT_FILTER]: {file_ref}\n"
    
    output = (
        f"[LOCAL_SEARCH_STATUS]: {status}\n"
        f"[LOCAL_SEARCH_CONFIDENCE]: {confidence:.2f}\n"
        f"{filter_info}\n"
        f"{context}"
    )

    audit = get_audit_logger()
    if audit.is_available:
        audit.log_rag_retrieval(
            context=ctx,
            query=query,
            status=status,
            confidence=confidence,
            num_docs=len(docs),
            extra={"file_ref": file_ref},
        )

    search_cache[cache_key] = output
    return output


# ── Web Search (Tavily) ────────────────────────────────────

_NEWS_TERMS = {"haber", "son dakika", "guncel", "gelisme", "secim", "deprem",
               "news", "breaking", "latest", "election", "earthquake"}


def _detect_topic(query: str) -> str:
    q = query.lower()
    if any(t in q for t in _NEWS_TERMS):
        return "news"
    return "general"


def _clean_snippet(text: str, max_chars: int) -> str:
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
        # Allow only constants and basic arithmetic operators
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


# ── MCP Tool Wrappers (Dynamic Discovery) ────────────────────

_MCP_INVOKER = None
_MCP_TOOLS_CACHE: list | None = None


def set_mcp_invoker(invoker):
    """Registers the MCP HybridToolInvoker (at app startup)."""
    global _MCP_INVOKER, _MCP_TOOLS_CACHE
    _MCP_INVOKER = invoker
    _MCP_TOOLS_CACHE = None  # Reset cache when invoker changes


# Tool categories for better organization
TOOL_CATEGORIES = {
    "knowledge": {
        "description": "Information retrieval and search",
        "tools": ["search_documents", "web_search"],
    },
    "computation": {
        "description": "Mathematical calculations and data processing",
        "tools": ["calculator"],
    },
    "file_operations": {
        "description": "File system operations",
        "tools": ["read_file", "write_file", "list_directory"],
    },
    "execution": {
        "description": "Code and command execution",
        "tools": ["execute_python", "run_bash"],
    },
    "database": {
        "description": "Database operations",
        "tools": ["query_postgres", "memory_search"],
    },
}


def get_tool_category(tool_name: str) -> str:
    """Get category for a tool, returns 'other' if not found."""
    for category, info in TOOL_CATEGORIES.items():
        if tool_name in info["tools"]:
            return category
    return "other"


def _create_mcp_tools():
    """Returns LangChain tool list if MCP is enabled (dynamic discovery)."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    global _MCP_TOOLS_CACHE

    if _MCP_INVOKER is None:
        return []

    # Return cached tools if available
    if _MCP_TOOLS_CACHE is not None:
        return _MCP_TOOLS_CACHE

    try:
        # Dynamic tool discovery from MCP server
        available_tools = _MCP_INVOKER.list_tools()

        if not available_tools:
            return []

        # Fallback schema definitions for common tools
        tool_schemas = {
            "read_file": {"args": {"path": str}, "desc": "Read file content from /data volume"},
            "list_directory": {"args": {"path": str}, "desc": "List files in a directory"},
            "write_file": {"args": {"path": str, "content": str}, "desc": "Write content to a file"},
            "execute_python": {"args": {"code": str}, "desc": "Execute Python code (sandboxed, 15s timeout)"},
            "run_bash": {"args": {"command": str}, "desc": "Run shell command (restricted)"},
            "query_postgres": {"args": {"query": str}, "desc": "Execute read-only SQL query"},
            "memory_search": {"args": {"query": str}, "desc": "Search in agent memory"},
        }

        mcp_tools = []

        for tool_info in available_tools:
            # Handle both dict and object formats
            if isinstance(tool_info, dict):
                tool_name = tool_info.get("name", "")
                tool_desc = tool_info.get("description", "")
            else:
                tool_name = getattr(tool_info, "name", "")
                tool_desc = getattr(tool_info, "description", "")

            if not tool_name:
                continue

            # Get schema or use fallback
            schema_info = tool_schemas.get(tool_name, {"args": {}, "desc": tool_desc})

            def _make_invoker_fn(name: str):
                inv = _MCP_INVOKER

                def _invoke(**kwargs):
                    try:
                        result = inv.invoke(name, kwargs)
                        return str(result) if result is not None else ""
                    except Exception as exc:
                        return f"MCP tool error ({name}): {exc}"

                return _invoke

            # Dynamically create args schema
            class DynamicArgs(BaseModel):
                pass

            args_fields = {}
            for arg_name, arg_type in schema_info["args"].items():
                args_fields[arg_name] = Field(default=..., description=schema_info["desc"], type=arg_type)

            if args_fields:
                DynamicArgs = type("DynamicArgs", (BaseModel,), args_fields)

            mcp_tools.append(
                StructuredTool.from_function(
                    func=_make_invoker_fn(tool_name),
                    name=tool_name,
                    description=schema_info["desc"],
                    args_schema=DynamicArgs if args_fields else None,
                )
            )

        _MCP_TOOLS_CACHE = mcp_tools
        return mcp_tools

    except Exception as e:
        # If dynamic discovery fails, return empty list
        print(f"MCP dynamic discovery error: {e}")
        return []


def refresh_mcp_tools():
    """Force refresh of MCP tools cache."""
    global _MCP_TOOLS_CACHE
    _MCP_TOOLS_CACHE = None
    return _create_mcp_tools()


def get_mcp_tools():
    """Get only MCP tools."""
    return _create_mcp_tools()


def get_all_tools():
    """Combined list of local + MCP tools."""
    base = [search_documents, web_search, calculator]
    mcp = _create_mcp_tools()
    return base + mcp


def get_tools_by_category(category: str | None = None) -> list:
    """Get tools filtered by category. If category is None, return all."""
    all_tools = get_all_tools()

    if category is None:
        return all_tools

    category_tools = TOOL_CATEGORIES.get(category, {}).get("tools", [])
    return [t for t in all_tools if t.name in category_tools]


def get_tool_descriptions() -> str:
    """Get formatted tool descriptions for LLM prompt."""
    tools = get_all_tools()
    lines = ["Available tools:"]

    # Group by category
    for category, info in TOOL_CATEGORIES.items():
        lines.append(f"\n## {category.upper()}: {info['description']}")
        for tool in tools:
            if tool.name in info["tools"]:
                lines.append(f"- {tool.name}: {tool.description}")

    # Add tools not in any category
    categorized = set()
    for info in TOOL_CATEGORIES.values():
        categorized.update(info["tools"])

    other_tools = [t for t in tools if t.name not in categorized]
    if other_tools:
        lines.append("\n## OTHER:")
        for tool in other_tools:
            lines.append(f"- {tool.name}: {tool.description}")

    return "\n".join(lines)


# ── Tool registry ──────────────────────────────────────────

ALL_TOOLS = [search_documents, web_search, calculator]
