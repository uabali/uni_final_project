from __future__ import annotations

"""
Request context utilities.

This module collects information coming from the HTTP layer (FastAPI):
- department_id
- user_id
- role
- session_id
- request_id / correlation_id

in a single place and provides a thread-local-like structure via contextvars
so that agent / tool / memory layers can access it.

Purpose:
- Parse JWT once at the API layer
- Access request context safely and testably in lower layers (RAG, tools, memory, audit)
  without using global state.
"""

import contextvars
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RequestContext:
    """Identity and authorization information for a single HTTP request."""

    request_id: str
    session_id: str
    user_id: str
    department_id: str
    role: str
    correlation_id: str
    claims: Dict[str, Any]


_REQUEST_CONTEXT_VAR: contextvars.ContextVar[Optional[RequestContext]] = (
    contextvars.ContextVar("rag_request_context", default=None)
)


def set_request_context(ctx: RequestContext) -> None:
    """Sets the current request context."""
    _REQUEST_CONTEXT_VAR.set(ctx)


def get_request_context() -> Optional[RequestContext]:
    """Returns the current request context (or None if not set)."""
    return _REQUEST_CONTEXT_VAR.get()


def get_default_department_id() -> str:
    """
    Default department_id value.

    Used in non-JWT scenarios or background jobs.
    """
    return os.getenv("DEFAULT_DEPARTMENT_ID", "default").strip() or "default"


def generate_request_ids(
    *,
    user_id: Optional[str] = None,
    department_id: Optional[str] = None,
    session_id_header: Optional[str] = None,
) -> tuple[str, str]:
    """
    Helper for request + session ID generation.

    - request_id: Fully random UUID (for tracing).
    - session_id: Uses the header value if provided; otherwise generates from
      department_id + user_id + random UUID combination.
    """
    request_id = str(uuid.uuid4())

    if session_id_header:
        return request_id, session_id_header

    dept = (department_id or get_default_department_id()).strip() or "default"
    user = (user_id or "anonymous").strip() or "anonymous"
    random_part = uuid.uuid4().hex[:12]
    session_id = f"{dept}:{user}:{random_part}"

    return request_id, session_id
