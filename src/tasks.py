from __future__ import annotations

"""
Simple Task registry + tracking infrastructure.

Purpose:
- Represent long-running agent tasks (especially in multi-agent orchestration scenarios)
  in a single place.
- Allow the WebSocket hub to periodically read this registry and relay status updates
  to the frontend.

Note:
- Currently uses an in-memory dictionary; task information is lost on process restart.
  Redis/Postgres-based persistence can be added when needed.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .context import RequestContext


@dataclass
class TaskInfo:
    task_id: str
    status: str  # pending | running | completed | error
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    department_id: Optional[str] = None
    role: Optional[str] = None
    progress: Optional[float] = None  # Estimated progress between 0.0–1.0
    last_error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    meta: Dict[str, Any] = field(default_factory=dict)
    version: int = 0  # Monotonic counter for change detection on WebSocket side


_TASKS: Dict[str, TaskInfo] = {}


def upsert_task(
    task_id: str,
    *,
    status: Optional[str] = None,
    context: Optional[RequestContext] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> TaskInfo:
    """
    Adds a new entry to the task registry or updates an existing one.
    """
    now = datetime.now(timezone.utc)
    existing = _TASKS.get(task_id)

    if existing is None:
        existing = TaskInfo(
            task_id=task_id,
            status=status or "pending",
            session_id=getattr(context, "session_id", None) if context else None,
            user_id=getattr(context, "user_id", None) if context else None,
            department_id=getattr(context, "department_id", None) if context else None,
            role=getattr(context, "role", None) if context else None,
            progress=progress,
            last_error=error,
            meta=meta or {},
        )
        _TASKS[task_id] = existing
    else:
        if status is not None:
            existing.status = status
        if progress is not None:
            existing.progress = progress
        if error is not None:
            existing.last_error = error
        if meta:
            existing.meta.update(meta)

    existing.updated_at = now
    existing.version += 1
    return existing


def get_task_snapshot(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Returns the TaskInfo object as a JSON-serializable dict (or None).
    """
    task = _TASKS.get(task_id)
    if task is None:
        return None
    data = asdict(task)
    # Convert datetime fields to ISO strings
    data["created_at"] = task.created_at.isoformat()
    data["updated_at"] = task.updated_at.isoformat()
    return data
