from __future__ import annotations

"""
Multi-layer memory service.

architecture.pdf'teki memory katmanini kod seviyesinde temsil eder:
- Short-term buffer (Redis, session bazli sliding window)
- Long-term episodic memory (Qdrant "memory" collection)
- Entity memory (Redis hash)
- Summary store (Postgres — uzun konusmalarin ozetleri)

CLI senaryosunda bile session_id kavramini koruyarak ileride HTTP/GUI
adapter'larinin ayni API'yi kullanmasini hedefler.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import redis
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from .vectorstore import create_embeddings

logger = logging.getLogger("rag.memory")


@dataclass
class MemoryConfig:
    """Memory konfigürasyonu. ENV değişkenleri runtime'da okunur (dotenv sonrası)."""
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    short_term_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("MEMORY_SHORT_TERM_TTL", "7200")))
    short_term_max_turns: int = field(default_factory=lambda: int(os.getenv("MEMORY_SHORT_TERM_MAX_TURNS", "20")))
    entity_prefix: str = field(default_factory=lambda: os.getenv("MEMORY_ENTITY_PREFIX", "entity:"))
    episodic_collection: str = field(default_factory=lambda: os.getenv("MEMORY_EPISODIC_COLLECTION", "memory_collection"))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333").strip())
    # Postgres summary store
    postgres_url: str = field(default_factory=lambda: os.getenv("POSTGRES_URL", "postgresql://rag:rag@localhost:5432/rag"))
    summary_trigger_turns: int = field(default_factory=lambda: int(os.getenv("SUMMARY_TRIGGER_TURNS", "20")))


class ShortTermMemory:
    """
    Redis tabanli sliding window:
    - key: session_id
    - value: JSON serialized BaseMessage listesi
    """

    def __init__(self, client: redis.Redis, cfg: MemoryConfig):
        self._client = client
        self._cfg = cfg

    def load(self, session_id: str) -> Sequence[BaseMessage]:
        raw = self._client.get(session_id)
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            return []
        # Şu an için sadece text içerik taşıyan mesajları geri getiriyoruz.
        from langchain_core.messages import HumanMessage, AIMessage

        messages: List[BaseMessage] = []
        for item in data:
            role = item.get("role")
            content = item.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
        return messages[-self._cfg.short_term_max_turns :]

    def save(self, session_id: str, history: Sequence[BaseMessage]) -> None:
        serializable: List[Dict[str, Any]] = []
        for msg in history[-self._cfg.short_term_max_turns :]:
            role = "other"
            if msg.type == "human":
                role = "human"
            elif msg.type == "ai":
                role = "ai"
            serializable.append({"role": role, "content": msg.content})
        self._client.setex(
            session_id,
            self._cfg.short_term_ttl_seconds,
            json.dumps(serializable, ensure_ascii=False),
        )

    def get_turn_count(self, session_id: str) -> int:
        """Session'daki mesaj sayısını döner."""
        raw = self._client.get(session_id)
        if not raw:
            return 0
        try:
            data = json.loads(raw)
            return len(data)
        except Exception:
            return 0


class EntityMemory:
    """
    Kullanici/varlik bazli kalici bilgiler icin Redis hash:
    - key: f\"{entity_prefix}{entity_id}\"
    - field: arbitrary
    """

    def __init__(self, client: redis.Redis, cfg: MemoryConfig):
        self._client = client
        self._cfg = cfg

    def load(self, entity_id: str) -> Dict[str, Any]:
        key = f"{self._cfg.entity_prefix}{entity_id}"
        raw = self._client.hgetall(key)
        result: Dict[str, Any] = {}
        for field, value in raw.items():
            try:
                result[field.decode("utf-8")] = json.loads(value)
            except Exception:
                result[field.decode("utf-8")] = value.decode("utf-8")
        return result

    def save(self, entity_id: str, data: Dict[str, Any]) -> None:
        key = f"{self._cfg.entity_prefix}{entity_id}"
        mapping = {
            k: json.dumps(v, ensure_ascii=False) for k, v in data.items()
        }
        if mapping:
            self._client.hset(key, mapping=mapping)


class EpisodicMemory:
    """
    Uzun donem episodik hafiza:
    - Qdrant icinde ayri bir collection
    - Her entry bir Document olarak saklanir.
    """

    def __init__(self, cfg: MemoryConfig):
        self._cfg = cfg
        self._embeddings = create_embeddings()
        # Collection yoksa once olustur, sonra baglan.
        self._ensure_collection_exists()
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore

        client = QdrantClient(url=self._cfg.qdrant_url)
        self._vectorstore = QdrantVectorStore(
            client=client,
            collection_name=self._cfg.episodic_collection,
            embedding=self._embeddings,
        )

    def _ensure_collection_exists(self) -> None:
        """Qdrant'ta episodic collection yoksa bos olusturur."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(url=self._cfg.qdrant_url)
        collections = [c.name for c in client.get_collections().collections]
        if self._cfg.episodic_collection not in collections:
            # Embedding boyutunu tespit et
            sample = self._embeddings.embed_query("test")
            dim = len(sample)
            client.create_collection(
                collection_name=self._cfg.episodic_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"Episodic memory collection olusturuldu: {self._cfg.episodic_collection}")

    def search(self, query: str, k: int = 5) -> List[Document]:
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def add_episode(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        doc = Document(page_content=text, metadata=metadata or {})
        self._vectorstore.add_documents([doc])


# ── Summary Store (Postgres) ─────────────────────────────────

_SUMMARY_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id              SERIAL PRIMARY KEY,
    session_id      VARCHAR(255) NOT NULL,
    summary         TEXT NOT NULL,
    turn_range      VARCHAR(100),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_summaries_session
    ON conversation_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_summaries_created
    ON conversation_summaries(created_at DESC);
"""


class SummaryStore:
    """
    Postgres tabanlı konuşma özet deposu.

    Uzun konuşmalarda:
      1. Short-term buffer dolduğunda (20+ mesaj) otomatik olarak
         bir LLM çağrısı ile özet oluşturulur.
      2. Özet Postgres'e kaydedilir.
      3. Yeni konuşma turlarında ilgili özetler retrieval sırasında
         context'e eklenir.

    Postgres bağlantısı yoksa gracefully degrade eder (no-op).
    """

    def __init__(self, postgres_url: str | None = None):
        self._postgres_url = postgres_url
        self._conn = None
        self._available = False
        self._init_connection()

    def _init_connection(self) -> None:
        """Postgres bağlantısını başlatır ve tabloyu oluşturur."""
        if not self._postgres_url:
            logger.info("Postgres URL tanımlı değil — SummaryStore no-op modda")
            return

        try:
            import psycopg2
            self._conn = psycopg2.connect(self._postgres_url)
            self._conn.autocommit = True
            with self._conn.cursor() as cur:
                cur.execute(_SUMMARY_TABLE_DDL)
            self._available = True
            logger.info("SummaryStore Postgres bağlantısı hazır")
        except ImportError:
            logger.warning(
                "psycopg2 kurulu değil. SummaryStore no-op modda çalışacak. "
                "Kurulum: pip install psycopg2-binary"
            )
        except Exception as exc:
            logger.warning(f"Postgres bağlantısı kurulamadı: {exc}. SummaryStore no-op modda.")

    @property
    def is_available(self) -> bool:
        return self._available

    def save_summary(
        self,
        session_id: str,
        summary: str,
        turn_range: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Konuşma özetini Postgres'e kaydeder."""
        if not self._available or not self._conn:
            logger.debug(f"SummaryStore no-op: save_summary({session_id})")
            return

        try:
            meta_json = json.dumps(metadata or {}, ensure_ascii=False)
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_summaries
                        (session_id, summary, turn_range, metadata)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (session_id, summary, turn_range, meta_json),
                )
            logger.info(f"Özet kaydedildi: session={session_id}, turns={turn_range}")
        except Exception as exc:
            logger.error(f"Özet kaydetme hatası: {exc}")

    def load_summaries(
        self,
        session_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Session'a ait en son özetleri yükler."""
        if not self._available or not self._conn:
            return []

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT summary, turn_range, created_at, metadata
                    FROM conversation_summaries
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
                rows = cur.fetchall()

            return [
                {
                    "summary": row[0],
                    "turn_range": row[1],
                    "created_at": row[2].isoformat() if row[2] else None,
                    "metadata": row[3] or {},
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error(f"Özet yükleme hatası: {exc}")
            return []

    def delete_session_summaries(self, session_id: str) -> int:
        """Session'a ait tüm özetleri siler."""
        if not self._available or not self._conn:
            return 0

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_summaries WHERE session_id = %s",
                    (session_id,),
                )
                return cur.rowcount
        except Exception as exc:
            logger.error(f"Özet silme hatası: {exc}")
            return 0

    def close(self) -> None:
        """Postgres bağlantısını kapatır."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._available = False


# ── Summary Generator ─────────────────────────────────────────

SUMMARY_PROMPT = """Aşağıdaki konuşma geçmişinin kısa bir özetini oluştur.
Önemli bilgileri, kararları ve bağlamı koru. Türkçe yaz.
Maksimum 3-4 cümle.

Konuşma geçmişi:
{conversation}

Özet:"""


def generate_summary_if_needed(
    memory: "MemorySystem",
    *,
    session_id: str,
    messages: Sequence[BaseMessage],
    llm: Any = None,
) -> Optional[str]:
    """
    Short-term buffer dolduğunda (trigger_turns mesajdan fazla) otomatik
    olarak LLM ile özet oluşturur ve Postgres'e kaydeder.

    Args:
        memory: MemorySystem instance
        session_id: Mevcut session ID
        messages: Güncel mesaj listesi
        llm: Özet oluşturmak için kullanılacak LLM (opsiyonel)

    Returns:
        Oluşturulan özet metni veya None
    """
    turn_count = len(messages)
    trigger = memory.cfg.summary_trigger_turns

    if turn_count < trigger:
        return None

    if not memory.summaries.is_available:
        logger.debug("SummaryStore mevcut değil, özet atlanıyor")
        return None

    # Konuşmayı text'e dönüştür
    conversation_lines = []
    for msg in messages:
        prefix = "Kullanıcı" if msg.type == "human" else "Asistan"
        content = (getattr(msg, "content", "") or "")[:300]
        conversation_lines.append(f"{prefix}: {content}")
    conversation_text = "\n".join(conversation_lines)

    # LLM ile özet oluştur
    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage
            prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
            result = llm.invoke([HumanMessage(content=prompt)])
            summary = getattr(result, "content", str(result))
        except Exception as exc:
            logger.warning(f"LLM ile özet oluşturulamadı: {exc}")
            # Fallback: basit metin özeti
            summary = f"Konuşma ({turn_count} mesaj): {conversation_text[:200]}..."
    else:
        # LLM yoksa basit truncation
        summary = f"Konuşma ({turn_count} mesaj): {conversation_text[:300]}..."

    # Postgres'e kaydet
    turn_range = f"1-{turn_count}"
    memory.summaries.save_summary(
        session_id=session_id,
        summary=summary,
        turn_range=turn_range,
        metadata={"turn_count": turn_count, "auto_generated": True},
    )

    return summary


# ── MemorySystem ──────────────────────────────────────────────

@dataclass
class MemorySystem:
    cfg: MemoryConfig
    short_term: ShortTermMemory
    episodic: EpisodicMemory
    entity: EntityMemory
    summaries: SummaryStore

    @classmethod
    def from_env(cls, cfg: Optional[MemoryConfig] = None) -> "MemorySystem":
        cfg = cfg or MemoryConfig()
        redis_client = redis.Redis.from_url(cfg.redis_url, socket_connect_timeout=5)
        redis_client.ping()  # Fail-fast: baglanti sorununu erken yakala
        short_term = ShortTermMemory(redis_client, cfg)
        entity = EntityMemory(redis_client, cfg)
        episodic = EpisodicMemory(cfg)
        summaries = SummaryStore(postgres_url=cfg.postgres_url)
        return cls(cfg=cfg, short_term=short_term, episodic=episodic, entity=entity, summaries=summaries)


def inject_memory_context(
    memory: MemorySystem,
    *,
    session_id: str,
    entity_id: Optional[str],
    user_query: str,
    messages: Sequence[BaseMessage],
) -> Sequence[BaseMessage]:
    """
    Short-term buffer + episodik + entity + summary bilgilerini system prompt'a enjekte eder.
    """
    short_history = memory.short_term.load(session_id)
    episodic_docs = memory.episodic.search(user_query, k=3)
    entity_data = memory.entity.load(entity_id) if entity_id else {}

    # Postgres'ten ilgili özetleri yükle
    past_summaries = memory.summaries.load_summaries(session_id, limit=3)

    context_lines: List[str] = []

    # Geçmiş konuşma özetleri (uzun dönem)
    if past_summaries:
        context_lines.append("Gecmis konusma ozetleri:")
        for s in past_summaries:
            context_lines.append(f"- [{s.get('turn_range', '?')}] {s['summary']}")

    if short_history:
        context_lines.append("\nKisa donem hafiza (son konusmalar):")
        for msg in short_history[-5:]:
            prefix = "KULLANICI" if msg.type == "human" else "ASISTAN"
            context_lines.append(f"- {prefix}: {msg.content}")
    if episodic_docs:
        context_lines.append("\nGecmis episodik anilar:")
        for doc in episodic_docs:
            context_lines.append(f"- {doc.page_content[:120]}...")
    if entity_data:
        context_lines.append("\nVarlik bilgileri:")
        for k, v in entity_data.items():
            context_lines.append(f"- {k}: {v}")

    if not context_lines:
        return messages

    from langchain_core.messages import SystemMessage

    injected = SystemMessage(
        content="Hafiza baglamin:\n" + "\n".join(context_lines),
    )
    return [injected, *messages]


ENTITY_EXTRACT_PROMPT = """Konusmadan kullanici hakkinda kalici bilgileri cikar.
Ornek: meslek, ilgi alanlari, tercihler, ad, yasadigi yer vb.
Sadece acikca soylenen veya cikarilabilecek gercekleri JSON olarak dondur.
Format: {"key": "value", ...} — key'ler kucuk harf, tire ile (ornek: favorite_language).
Eger cikarilacak bilgi yoksa bos obje {} dondur.

Konusma:
{conversation}

JSON:"""


def _extract_entity_facts(messages: Sequence[BaseMessage], llm: Any) -> Dict[str, Any]:
    """Konusmadan kullanici hakkinda entity fact'leri cikarir."""
    lines = []
    for msg in messages:
        prefix = "Kullanici" if msg.type == "human" else "Asistan"
        content = (getattr(msg, "content", "") or "")[:500]
        lines.append(f"{prefix}: {content}")
    text = "\n".join(lines)
    if not text.strip():
        return {}

    try:
        from langchain_core.messages import HumanMessage
        prompt = ENTITY_EXTRACT_PROMPT.format(conversation=text)
        result = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(result, "content", "") or ""
        # JSON blok cikar
        import re
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            import json
            return json.loads(match.group())
    except Exception as exc:
        logger.debug(f"Entity extraction hatasi: {exc}")
    return {}


def update_memory_after_response(
    memory: MemorySystem,
    *,
    session_id: str,
    entity_id: Optional[str],
    messages: Sequence[BaseMessage],
    llm: Any = None,
) -> None:
    """
    Yeni mesajlari short-term buffer'a yazar.
    Short-term buffer doluysa otomatik özet oluşturur.
    Entity memory: LLM ile konusmadan kullanici bilgileri cikarilip Redis'e yazilir.
    """
    memory.short_term.save(session_id, messages)

    # Otomatik özet tetikleme
    generate_summary_if_needed(
        memory,
        session_id=session_id,
        messages=messages,
        llm=llm,
    )

    # Entity memory guncelleme (PDF: her konusmada)
    effective_entity_id = entity_id or session_id
    if effective_entity_id and llm is not None:
        try:
            new_facts = _extract_entity_facts(messages, llm)
            if new_facts:
                existing = memory.entity.load(effective_entity_id)
                merged = {**existing, **new_facts}
                memory.entity.save(effective_entity_id, merged)
                logger.debug(f"Entity guncellendi: {effective_entity_id}, keys={list(new_facts.keys())}")
        except Exception as exc:
            logger.warning(f"Entity memory guncelleme hatasi: {exc}")
