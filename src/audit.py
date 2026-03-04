from __future__ import annotations

"""
Append-only audit logging for agent operations, tool calls and data access.

This module implements the "Audit Log" box from the architecture document:
- Every significant action (request, tool call, RAG retrieval, memory write, response)
  is recorded as a JSON payload in an append-only Postgres table.
- Gracefully falls back to no-op mode if Postgres is unavailable (does not block the system).
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
    Simple, append-only audit logger.

    - If Postgres connection cannot be established, `_available = False` and all log_* calls are no-op.
    - close() can be called on application shutdown; however, the connection uses autocommit mode
      and is used lightly to prevent connection leaks.
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
            logger.info("Postgres URL not defined — AuditLogger in no-op mode")
            return

        try:
            import psycopg2

            self._conn = psycopg2.connect(url)
            self._conn.autocommit = True
            with self._conn.cursor() as cur:
                cur.execute(_AUDIT_TABLE_DDL)
            self._available = True
            logger.info("AuditLogger Postgres connection ready")
        except ImportError:
            logger.warning(
                "psycopg2 is not installed. AuditLogger will run in no-op mode. "
                "Install: pip install psycopg2-binary"
            )
        except Exception as exc:
            logger.warning(f"AuditLogger Postgres connection failed: {exc}. Audit in no-op mode.")

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
        """General-purpose event log function."""
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
            # Audit system never disrupts the main flow; only logs the error.
            logger.error(f"Audit log error ({event_type}): {exc}")

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
    Returns the lazy-initialized global AuditLogger instance.

    If Postgres is unavailable or connection fails, the returned logger's
    `is_available` field will be False and all log calls will be no-op.
    """
    global _AUDIT_LOGGER
    if _AUDIT_LOGGER is None:
        _AUDIT_LOGGER = AuditLogger()
    return _AUDIT_LOGGER
