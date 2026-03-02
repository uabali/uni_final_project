from __future__ import annotations

"""
Request context utilities.

Bu modül, HTTP katmanından (FastAPI) gelen:
- department_id
- user_id
- role
- session_id
- request_id / correlation_id

gibi bilgileri tek bir yerde toplar ve agent / tool / memory katmanlarının
erişebilmesi için contextvars üzerinden thread-local benzeri bir yapı sağlar.

Amaç:
- API katmanında bir kez JWT parse etmek
- Aşağı katmanlarda (RAG, tools, memory, audit) global state kullanmadan
  güvenli ve test edilebilir bir şekilde request bağlamına erişebilmek.
"""

import contextvars
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RequestContext:
    """Tek bir HTTP isteğinin kimlik ve yetki bilgileri."""

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
    """Geçerli request context'ini ayarlar."""
    _REQUEST_CONTEXT_VAR.set(ctx)


def get_request_context() -> Optional[RequestContext]:
    """Geçerli request context'ini döner (yoksa None)."""
    return _REQUEST_CONTEXT_VAR.get()


def get_default_department_id() -> str:
    """
    Varsayılan department_id değeri.

    JWT olmayan senaryolarda veya background job'larda kullanılır.
    """
    return os.getenv("DEFAULT_DEPARTMENT_ID", "default").strip() or "default"


def generate_request_ids(
    *,
    user_id: Optional[str] = None,
    department_id: Optional[str] = None,
    session_id_header: Optional[str] = None,
) -> tuple[str, str]:
    """
    Request + session ID üretimi için yardımcı.

    - request_id: Tamamen rastgele UUID (izleme için).
    - session_id: Eğer header'da verilmişse onu kullanır; yoksa
      department_id + user_id + rastgele UUID kombinasyonundan üretir.
    """
    request_id = str(uuid.uuid4())

    if session_id_header:
        return request_id, session_id_header

    dept = (department_id or get_default_department_id()).strip() or "default"
    user = (user_id or "anonymous").strip() or "anonymous"
    random_part = uuid.uuid4().hex[:12]
    session_id = f"{dept}:{user}:{random_part}"

    return request_id, session_id

