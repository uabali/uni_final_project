from __future__ import annotations

"""
Basit Task registry + izleme altyapısı.

Amaç:
- Uzun sürebilen agent görevlerini (özellikle çok-ajan orkestrasyon senaryolarında)
  tek bir yerde temsil etmek.
- WebSocket hub'ın bu registry'yi periyodik olarak okuyarak durum güncellemelerini
  frontend'e iletebilmesine imkan vermek.

Not:
- Şimdilik sadece bellek içi (in-memory) bir sözlük kullanıyor; proses restart'ında
  task bilgisi kaybolur. Gerektiğinde Redis/Postgres tabanlı persistance eklenebilir.
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
    progress: Optional[float] = None  # 0.0–1.0 arası tahmini ilerleme
    last_error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    meta: Dict[str, Any] = field(default_factory=dict)
    version: int = 0  # WebSocket tarafında değişiklik algılama için monotonic sayaç


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
    Task registry'ye yeni kayıt ekler veya mevcut kaydı günceller.
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
    TaskInfo nesnesini JSON-serializable dict olarak döner (veya None).
    """
    task = _TASKS.get(task_id)
    if task is None:
        return None
    data = asdict(task)
    # datetime alanlarını ISO string'e çevir
    data["created_at"] = task.created_at.isoformat()
    data["updated_at"] = task.updated_at.isoformat()
    return data

