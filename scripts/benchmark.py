"""
Benchmark runner for RAG pipeline.
Measures latency, throughput, and (optional) GPU stats per query.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_documents
from src.llm import create_llm
from src.prompting import build_prompt, format_docs
from src.reranker import create_reranker
from src.retriever import build_bm25_retriever, create_retriever
from src.splitter import split_documents
from src.vectorstore import create_embeddings, create_vectorstore


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def _get_gpu_stats() -> List[Dict[str, Any]]:
    """Best-effort GPU stats via nvidia-smi (returns empty list if unavailable)."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "utilization": float(parts[2]),
                "memory_used": float(parts[3]),
                "memory_total": float(parts[4]),
                "power_draw": float(parts[5]),
                "temperature": float(parts[6]),
            }
        )
    return gpus


def _guess_input_key(fieldnames: Iterable[str]) -> Optional[str]:
    candidates = ["input", "question", "query", "prompt"]
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def _load_dataset(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            input_key = _guess_input_key(reader.fieldnames)
            if input_key is None:
                raise ValueError("Dataset must include an 'input' or 'question' column.")
            for row in reader:
                rows.append(row)

    if limit:
        return rows[:limit]
    return rows


def _run_retriever(retriever, query: str):
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if callable(retriever):
        return retriever(query)
    raise TypeError(f"Unsupported retriever type: {type(retriever).__name__}")


def _get_llm(_backend: str):
    """
    Tek backend: Meta-Llama-3.1-8B-Instruct-AWQ-INT4 (vLLM local).

    backend parametresi sadece CLI uyumluluğu için korunur.
    """
    return create_llm()


def _get_llm_name(llm) -> str:
    for attr in ["model", "model_name", "model_id"]:
        if hasattr(llm, attr):
            return str(getattr(llm, attr))
    return type(llm).__name__


def _approx_tokens(text: str) -> int:
    # Simple heuristic: ~4 chars per token
    return max(1, int(len(text) / 4))


def build_pipeline(
    split_method: str,
    use_multi_query: bool,
    use_rerank: bool,
    fast_mode: bool,
    backend: str,
):
    embeddings = create_embeddings()
    documents = load_documents()

    if documents:
        if split_method == "semantic":
            docs = split_documents(documents, method="semantic", embeddings=embeddings)
        else:
            docs = split_documents(
                documents,
                method="recursive",
                chunk_size=600,
                chunk_overlap=100,
            )
    else:
        docs = []

    vectorstore = create_vectorstore(docs, embeddings)
    bm25_retriever = build_bm25_retriever(docs) if docs else None

    reranker = None
    if use_rerank:
        reranker = create_reranker(device="cuda")

    llm = _get_llm(backend)
    prompt = build_prompt()
    chain = prompt | llm | StrOutputParser()

    return {
        "llm": llm,
        "chain": chain,
        "docs": docs,
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever,
        "reranker": reranker,
        "use_multi_query": use_multi_query,
        "use_rerank": use_rerank,
        "fast_mode": fast_mode,
    }


def run_benchmark(args):
    load_dotenv()
    dataset = _load_dataset(args.dataset, limit=args.limit)
    if not dataset:
        raise ValueError("Dataset is empty.")

    backend = args.backend or "local-llama3.1-awq"
    pipeline = build_pipeline(
        split_method=args.split_method,
        use_multi_query=args.use_multi_query,
        use_rerank=args.use_rerank,
        fast_mode=args.fast_mode,
        backend=backend,
    )

    llm = pipeline["llm"]
    chain = pipeline["chain"]
    bm25_retriever = pipeline["bm25_retriever"]
    vectorstore = pipeline["vectorstore"]

    run_id = str(uuid.uuid4())
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Warmup
    for i in range(args.warmup):
        q = dataset[i % len(dataset)]
        question = q.get("input") or q.get("question") or q.get("query")
        if not question:
            continue
        retriever = create_retriever(
            vectorstore=vectorstore,
            question=question,
            bm25_retriever=bm25_retriever,
            strategy=args.strategy,
            base_k=args.base_k,
            fetch_k=args.fetch_k,
            lambda_mult=args.lambda_mult,
            score_threshold=args.score_threshold,
            bm25_weight=args.bm25_weight,
            use_multi_query=args.use_multi_query,
            llm=llm if args.use_multi_query else None,
            num_queries=args.num_queries,
            use_rerank=args.use_rerank,
            reranker=pipeline["reranker"],
            rerank_top_n=args.rerank_top_n,
            fast_mode=args.fast_mode,
        )
        docs = _run_retriever(retriever, question)
        context = format_docs(docs)
        _ = chain.invoke({"context": context, "question": question})

    results = []
    base_run_config = {
        "tags": [
            f"backend:{backend}",
            f"strategy:{args.strategy}",
            f"rerank:{args.use_rerank}",
            f"multi_query:{args.use_multi_query}",
        ],
        "metadata": {
            "run_id": run_id,
            "backend": backend,
            "llm_name": _get_llm_name(llm),
            "gpu_label": args.gpu_label or "",
            "fast_mode": args.fast_mode,
        },
    }

    for r in range(args.runs):
        for row_idx, row in enumerate(dataset):
            question = row.get("input") or row.get("question") or row.get("query")
            if not question:
                continue

            retriever = create_retriever(
                vectorstore=vectorstore,
                question=question,
                bm25_retriever=bm25_retriever,
                strategy=args.strategy,
                base_k=args.base_k,
                fetch_k=args.fetch_k,
                lambda_mult=args.lambda_mult,
                score_threshold=args.score_threshold,
                bm25_weight=args.bm25_weight,
                use_multi_query=args.use_multi_query,
                llm=llm if args.use_multi_query else None,
                num_queries=args.num_queries,
                use_rerank=args.use_rerank,
                reranker=pipeline["reranker"],
                rerank_top_n=args.rerank_top_n,
                fast_mode=args.fast_mode,
            )

            t0 = time.perf_counter()
            t_retr_start = time.perf_counter()
            docs = _run_retriever(retriever, question)
            t_retr_end = time.perf_counter()
            context = format_docs(docs)

            ttft_ms = None
            output_text = ""
            t_llm_start = time.perf_counter()
            run_config = {
                "tags": base_run_config["tags"],
                "metadata": {
                    **base_run_config["metadata"],
                    "run_index": r,
                    "row_index": row_idx,
                },
            }

            if args.stream:
                chunks = []
                first_chunk_time = None
                for chunk in chain.stream({"context": context, "question": question}, config=run_config):
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    chunks.append(chunk)
                output_text = "".join(chunks)
                if first_chunk_time is not None:
                    ttft_ms = (first_chunk_time - t_llm_start) * 1000.0
            else:
                output_text = chain.invoke({"context": context, "question": question}, config=run_config)

            t_llm_end = time.perf_counter()
            t1 = time.perf_counter()

            gpu_stats = []
            if args.gpu_sample == "per_query":
                gpu_stats = _get_gpu_stats()

            result = {
                "run_id": run_id,
                "timestamp": _now_iso(),
                "run_index": r,
                "row_index": row_idx,
                "question": question,
                "expected": row.get("output") or row.get("answer") or "",
                "answer": output_text,
                "backend": backend,
                "llm_name": _get_llm_name(llm),
                "context_chars": len(context),
                "context_docs": len(docs),
                "approx_prompt_tokens": _approx_tokens(context) + _approx_tokens(question),
                "approx_output_tokens": _approx_tokens(output_text),
                "retrieval_ms": (t_retr_end - t_retr_start) * 1000.0,
                "llm_ms": (t_llm_end - t_llm_start) * 1000.0,
                "total_ms": (t1 - t0) * 1000.0,
                "ttft_ms": ttft_ms,
                "gpu_label": args.gpu_label or "",
                "gpu_stats": gpu_stats,
            }

            results.append(result)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    total_ms = [r["total_ms"] for r in results]
    llm_ms = [r["llm_ms"] for r in results]
    retr_ms = [r["retrieval_ms"] for r in results]
    ttft_vals = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]

    summary = {
        "run_id": run_id,
        "timestamp": _now_iso(),
        "backend": backend,
        "llm_name": _get_llm_name(llm),
        "dataset": args.dataset,
        "count": len(results),
        "p50_total_ms": _percentile(total_ms, 0.50),
        "p95_total_ms": _percentile(total_ms, 0.95),
        "p50_llm_ms": _percentile(llm_ms, 0.50),
        "p95_llm_ms": _percentile(llm_ms, 0.95),
        "p50_retrieval_ms": _percentile(retr_ms, 0.50),
        "p95_retrieval_ms": _percentile(retr_ms, 0.95),
        "p50_ttft_ms": _percentile(ttft_vals, 0.50) if ttft_vals else None,
        "p95_ttft_ms": _percentile(ttft_vals, 0.95) if ttft_vals else None,
        "gpu_label": args.gpu_label or "",
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        },
    }

    if args.gpu_sample == "per_run":
        summary["gpu_stats"] = _get_gpu_stats()

    summary_path = os.path.splitext(output_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Benchmark results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Benchmark Runner")
    parser.add_argument("--dataset", required=True, help="CSV or JSONL dataset path")
    parser.add_argument("--output", default="logs/benchmark_results.jsonl", help="Output JSONL path")
    parser.add_argument("--runs", type=int, default=1, help="Number of full dataset passes")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup queries")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size")
    parser.add_argument(
        "--backend",
        default=None,
        help="Artık yalnızca local-llama3.1-awq backend kullanılır (parametre görmezden gelinir).",
    )
    parser.add_argument("--split-method", choices=["recursive", "semantic"], default="recursive")
    parser.add_argument("--use-multi-query", action="store_true")
    parser.add_argument("--use-rerank", action="store_true")
    parser.add_argument("--fast-mode", action="store_true", help="Skip reranking on simple queries")
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--base-k", type=int, default=6)
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--lambda-mult", type=float, default=0.7)
    parser.add_argument("--score-threshold", type=float, default=0.75)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--strategy", default="auto", choices=["auto", "mmr", "similarity", "hybrid", "threshold"])
    parser.add_argument("--stream", action="store_true", help="Measure TTFT via streaming")
    parser.add_argument("--gpu-sample", choices=["none", "per_query", "per_run"], default="per_run")
    parser.add_argument("--gpu-label", default="", help="Label for GPU (e.g. A100-80GB)")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
