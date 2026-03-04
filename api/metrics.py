from __future__ import annotations

"""
Simple metrics API: request count, latency, and token summaries by department and agent.

Data source: audit_events table populated by src/audit.py.
"""

import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/metrics", tags=["Metrics"])


def _get_connection():
    import psycopg2

    url = os.getenv("POSTGRES_URL", "postgresql://rag:rag@localhost:5432/rag")
    return psycopg2.connect(url)


@router.get("/summary")
def metrics_summary() -> Dict[str, Any]:
    """
    Simple summary:
    - Per department: total requests, average latency, total tokens
    - Per agent type: total requests
    """
    try:
        conn = _get_connection()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not connect to Postgres: {exc}")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    department_id,
                    COUNT(*) AS total_requests,
                    AVG( (payload->>'latency_ms')::float ) AS avg_latency_ms,
                    SUM( COALESCE((payload->>'token_count')::bigint, 0) ) AS total_tokens
                FROM audit_events
                WHERE event_type = 'response'
                GROUP BY department_id
                ORDER BY department_id
                """
            )
            dept_rows = cur.fetchall()

            cur.execute(
                """
                SELECT
                    (payload->>'agent') AS agent_name,
                    COUNT(*) AS total_requests
                FROM audit_events
                WHERE event_type = 'response'
                  AND payload ? 'agent'
                GROUP BY agent_name
                ORDER BY agent_name
                """
            )
            agent_rows = cur.fetchall()

        departments: List[Dict[str, Any]] = []
        for row in dept_rows:
            dept, total_requests, avg_latency_ms, total_tokens = row
            departments.append(
                {
                    "department_id": dept or "unknown",
                    "total_requests": int(total_requests or 0),
                    "avg_latency_ms": float(avg_latency_ms or 0.0),
                    "total_tokens": int(total_tokens or 0),
                }
            )

        agents: List[Dict[str, Any]] = []
        for row in agent_rows:
            agent_name, total_requests = row
            agents.append(
                {
                    "agent": agent_name or "unknown",
                    "total_requests": int(total_requests or 0),
                }
            )

        return {"departments": departments, "agents": agents}
    finally:
        try:
            conn.close()
        except Exception:
            pass
