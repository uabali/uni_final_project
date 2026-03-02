from __future__ import annotations

"""
Append-only audit logging for agent operations, tool calls and data access.

Bu modül, architecture dokümanındaki "Audit Log" kutusunu uygular:
- Her önemli eylem (istek, tool çağrısı, RAG retrieval, hafıza yazımı, cevap)
  Postgres'teki append-only bir tabloya JSON payload olarak kaydedilir.
- Postgres yoksa gracefully no-op moda düşer (sistemi bloklamaz).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .context import RequestContext

logger = logging.getLogger("rag.audit")


_AUDIT_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    event_type      VARCHAR(100) NOT NULL,
    request_id      VARCHAR(255),
    session_id      VARCHAR(255),
    user_id         VARCHAR(255),
    department_id   VARCHAR(255),
    role            VARCHAR(100),
    payload         JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_audit_created
    ON audit_events(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_session
    ON audit_events(session_id);

CREATE INDEX IF NOT EXISTS idx_audit_department
    ON audit_events(department_id);
"""


@dataclass
class AuditLogger:
    """
    Basit, append-only audit logger.

    - Postgres bağlantısı kurulamazsa `_available = False` olur ve tüm log_* çağrıları no-op'tur.
    - Uygulama kapanışında close() çağrılabilir; ancak bağlantı sızmasını engellemek için
      connection autocommit modunda ve hafif kullanılır.
    """

    postgres_url: Optional[str] = None
    _conn: Any = field(default=None, init=False, repr=False)
    _available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._init_connection()

    def _init_connection(self) -> None:
        url = self.postgres_url or os.getenv(
            "POSTGRES_URL", "postgresql://rag:rag@localhost:5432/rag"
        )
        if not url:
            logger.info("Postgres URL tanımlı değil — AuditLogger no-op modda")
            return

        try:
            import psycopg2

            self._conn = psycopg2.connect(url)
            self._conn.autocommit = True
            with self._conn.cursor() as cur:
                cur.execute(_AUDIT_TABLE_DDL)
            self._available = True
            logger.info("AuditLogger Postgres bağlantısı hazır")
        except ImportError:
            logger.warning(
                "psycopg2 kurulu değil. AuditLogger no-op modda çalışacak. "
                "Kurulum: pip install psycopg2-binary"
            )
        except Exception as exc:
            logger.warning(f"AuditLogger Postgres bağlantısı kurulamadı: {exc}. Audit no-op modda.")

    @property
    def is_available(self) -> bool:
        return self._available

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._available = False

    # ---- Core logging API -------------------------------------------------

    def log_event(
        self,
        *,
        event_type: str,
        context: Optional[RequestContext],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Genel amaçlı event log fonksiyonu."""
        if not self._available or not self._conn:
            return

        ctx = context
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_events
                        (event_type, request_id, session_id, user_id, department_id, role, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        event_type,
                        getattr(ctx, "request_id", None) if ctx else None,
                        getattr(ctx, "session_id", None) if ctx else None,
                        getattr(ctx, "user_id", None) if ctx else None,
                        getattr(ctx, "department_id", None) if ctx else None,
                        getattr(ctx, "role", None) if ctx else None,
                        json.dumps(payload or {}, ensure_ascii=False),
                    ),
                )
        except Exception as exc:
            # Audit sistemi asla ana akışı bozmaz; sadece log yazar.
            logger.error(f"Audit log hatası ({event_type}): {exc}")

    # ---- Convenience wrappers ---------------------------------------------

    def log_request(
        self,
        *,
        context: Optional[RequestContext],
        endpoint: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_event(
            event_type="request",
            context=context,
            payload={
                "endpoint": endpoint,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(extra or {}),
            },
        )

    def log_response(
        self,
        *,
        context: Optional[RequestContext],
        endpoint: str,
        latency_ms: float,
        token_count: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "endpoint": endpoint,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if token_count is not None:
            payload["token_count"] = token_count
        if extra:
            payload.update(extra)

        self.log_event(
            event_type="response",
            context=context,
            payload=payload,
        )

    def log_tool_call(
        self,
        *,
        context: Optional[RequestContext],
        tool_name: str,
        success: bool,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "tool_name": tool_name,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            payload["error"] = error
        if extra:
            payload.update(extra)

        self.log_event(
            event_type="tool_call",
            context=context,
            payload=payload,
        )

    def log_rag_retrieval(
        self,
        *,
        context: Optional[RequestContext],
        query: str,
        status: str,
        confidence: float,
        num_docs: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "query": query,
            "status": status,
            "confidence": confidence,
            "num_docs": num_docs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)

        self.log_event(
            event_type="rag_retrieval",
            context=context,
            payload=payload,
        )


_AUDIT_LOGGER: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Lazy-init edilmiş global AuditLogger instance'ı döner.

    Postgres yoksa veya bağlantı kurulamazsa, dönen logger'ın `is_available`
    alanı False olur ve tüm log çağrıları no-op çalışır.
    """
    global _AUDIT_LOGGER
    if _AUDIT_LOGGER is None:
        _AUDIT_LOGGER = AuditLogger()
    return _AUDIT_LOGGER

