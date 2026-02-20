"""
Agentic RAG — CLI entry point.

Başlatma sırası:
  1. Embedding modeli, vectorstore, BM25, reranker gibi heavy objeleri yükler.
  2. Bu objeleri tool registry'ye kaydeder (src/tools.register_rag_components).
  3. LLM'i oluşturur ve LangGraph ReAct agent graph'ını derler.
  4. Kullanıcıdan sorgu alıp agent'a yönlendirir; agent kendi tool seçimini yapar.
"""

import os
import re
import sys
import time

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage

from src import splitter
from src.agent import build_agent_graph
from src.llm import create_llm
from src.loader import load_documents
from src.reranker import create_reranker
from src.retriever import build_bm25_retriever
from src.tools import register_rag_components
from src.vectorstore import create_embeddings, create_vectorstore


def _classify_query(query: str) -> str:
    from src.agent import _is_live_info_query, _is_pure_math

    q = query.strip().lower()
    if not q:
        return "empty"
    if q in {"merhaba", "selam", "hello", "hi"}:
        return "greeting"
    if _is_pure_math(q):
        return "math"
    if _is_live_info_query(q):
        return "live_info"
    return "knowledge"


def main():
    load_dotenv()
    print("--- Agentic RAG Pipeline Initializing ---")

    # ── 1. Embedding + docs + vectorstore ───────────────────
    embeddings = create_embeddings()

    documents = load_documents()
    if documents:
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        docs = splitter.split_documents(
            documents,
            method="recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        print(f"Documents split into {len(docs)} chunks.")
    else:
        print("Warning: No documents found. Continuing with empty vectorstore.")
        docs = []

    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "rag_collection").strip()
        vectorstore = create_vectorstore(
            docs,
            embeddings,
            url=qdrant_url,
            collection_name=qdrant_collection,
        )
        print(f"Qdrant: {qdrant_url} | Collection: {qdrant_collection}")
    except Exception as exc:
        print(f"Error during vectorstore initialization: {exc}")
        sys.exit(1)

    # ── 2. BM25 + Reranker ──────────────────────────────────
    bm25_retriever = None
    if docs:
        try:
            bm25_retriever = build_bm25_retriever(docs)
            print("BM25 retriever built.")
        except Exception as exc:
            print(f"BM25 build failed (hybrid disabled): {exc}")

    reranker = None
    try:
        reranker = create_reranker(device="cuda")
        print("Reranker loaded.")
    except Exception as exc:
        print(f"Reranker load failed (reranking disabled): {exc}")

    # ── 3. Tool registry'ye kaydet ──────────────────────────
    register_rag_components(
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )
    print("RAG components registered as agent tools.")

    # ── 4. LLM + Agent graph ────────────────────────────────
    llm = create_llm()
    vllm_server_url = os.getenv("VLLM_SERVER_URL", "").strip()
    print(f"LLM backend: Qwen3-8B-AWQ (vLLM server: {vllm_server_url})")

    agent = build_agent_graph(llm)
    print("\n--- Agentic RAG Ready (ReAct Pattern) ---")
    print("Tools: search_documents | web_search | calculator")
    print("(Type 'exit' to quit)\n")

    # ── 5. CLI loop ─────────────────────────────────────────
    from langchain_core.messages import HumanMessage

    turn_index = 0
    while True:
        try:
            query = input("Kullanici: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "cikis"):
                break
            turn_index += 1

            print("Agent: ", end="", flush=True)
            start_ts = time.perf_counter()
            query_type = _classify_query(query)

            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={
                    "run_name": "agent_turn",
                    "tags": ["cli", "agentic-rag", "vllm"],
                    "metadata": {
                        "turn_index": turn_index,
                        "query_type": query_type,
                        "query_length": len(query),
                    },
                },
            )

            final_msg = result["messages"][-1]
            print(final_msg.content)

            # Deterministik kaynak çıktısı:
            # ToolMessage içindeki CHUNK satırlarını topla, final cevapta cite edilenleri öncelikle yazdır.
            source_lines: list[str] = []
            source_by_chunk_id: dict[int, str] = {}
            web_source_lines: list[str] = []
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                    for line in msg.content.splitlines():
                        line = line.strip()
                        if line.startswith("[CHUNK ") and "source=" in line:
                            source_lines.append(line)
                            match = re.match(r"\[CHUNK\s+(\d+)\]", line)
                            if match:
                                chunk_id = int(match.group(1))
                                source_by_chunk_id.setdefault(chunk_id, line)
                        if line.startswith("Source: http"):
                            web_source_lines.append(line.replace("Source: ", ""))

            final_text = final_msg.content if isinstance(final_msg.content, str) else ""
            final_has_source_section = ("Kaynaklar:" in final_text) or ("Web Kaynaklari:" in final_text)

            # Model zaten kendi kaynak bölümünü yazdıysa tekrar basma (duplicate önleme).
            if not final_has_source_section:
                cited_chunk_ids = [
                    int(m.group(1))
                    for m in re.finditer(r"\[CHUNK\s+(\d+)\]", final_text)
                ]

                selected_source_lines: list[str] = []
                if cited_chunk_ids and source_by_chunk_id:
                    for chunk_id in cited_chunk_ids:
                        line = source_by_chunk_id.get(chunk_id)
                        if line:
                            selected_source_lines.append(line)

                if selected_source_lines:
                    print("Kaynaklar:")
                    for line in list(dict.fromkeys(selected_source_lines)):
                        print(f"- {line}")
                elif web_source_lines:
                    print("Web Kaynaklari:")
                    for line in list(dict.fromkeys(web_source_lines))[:5]:
                        print(f"- {line}")

            elapsed = time.perf_counter() - start_ts
            print(f"[latency: {elapsed:.2f}s]")

        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"\nQuery error: {exc}")


if __name__ == "__main__":
    main()
