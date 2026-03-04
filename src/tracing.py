from __future__ import annotations

"""
Tracing / observability abstraction.

Corresponds to the LangSmith/observability layer in architecture.pdf.
No-op by default; LangSmith or a different backend can be added.
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
    Returns NoopTracer for now.
    Can be changed in the future based on LANGCHAIN_TRACING_V2 or a custom backend.
    """
    _ = os.getenv("LANGCHAIN_TRACING_V2", "")
    return NoopTracer()
