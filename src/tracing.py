from __future__ import annotations

"""
Tracing / observability abstraction.

architecture.pdf'teki LangSmith/observability katmanina karsi gelir.
Varsayilan olarak no-op'tur; LangSmith veya farkli bir backend eklenebilir.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Protocol


class Tracer(Protocol):
    def trace_node_start(self, name: str, state: Dict[str, Any]) -> None:  # pragma: no cover - interface
        ...

    def trace_node_end(self, name: str, result: Dict[str, Any]) -> None:  # pragma: no cover - interface
        ...

    def trace_llm_call(self, model: str, tokens_in: int, tokens_out: int, latency_ms: float) -> None:  # pragma: no cover - interface
        ...


@dataclass
class NoopTracer(Tracer):
    def trace_node_start(self, name: str, state: Dict[str, Any]) -> None:
        _ = (name, state)

    def trace_node_end(self, name: str, result: Dict[str, Any]) -> None:
        _ = (name, result)

    def trace_llm_call(self, model: str, tokens_in: int, tokens_out: int, latency_ms: float) -> None:
        _ = (model, tokens_in, tokens_out, latency_ms)


def get_tracer() -> Tracer:
    """
    Simdilik sadece NoopTracer dondurur.
    Ileride LANGCHAIN_TRACING_V2 veya ozel bir backend'e gore degistirilebilir.
    """
    _ = os.getenv("LANGCHAIN_TRACING_V2", "")
    return NoopTracer()

