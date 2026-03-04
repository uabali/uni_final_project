from __future__ import annotations

"""
Multi-layer memory service.

Represents the memory layer from architecture.pdf at the code level:
- Short-term buffer (Redis, session-based sliding window)
- Long-term episodic memory (Qdrant "memory" collection)
- Entity memory (Redis hash)
- Summary store (Postgres — summaries of long conversations)

Preserves the session_id concept even in CLI scenarios so that future
HTTP/GUI adapters can use the same API.
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
from .context import get_request_context, get_default_department_id

logger = logging.getLogger("rag.memory")


@dataclass
class MemoryConfig:
    """Memory configuration. ENV variables are read at runtime (after dotenv)."""
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    short_term_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("MEMORY_SHORT_TERM_TTL", "7200")))
    short_term_max_turns: int = field(default_factory=lambda: int(os.getenv("MEMORY_SHORT_TERM_MAX_TURNS", "20")))
    entity_prefix: str = field(default_factory=lambda: os.getenv("MEMORY_ENTITY_PREFIX", "entity:"))
    episodic_collection: str = field(default_factory=lambda: os.getenv("MEMORY_EPISODIC_COLLECTION", "memory_collection"))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333").strip())
    # Postgres summary store
    postgres_url: str = field(default_factory=lambda: os.getenv("POSTGRES_URL", "postgresql://rag:rag@localhost:5432/rag"))
    summary_trigger_turns: int = field(default_factory=lambda: int(os.getenv("SUMMARY_TRIGGER_TURNS", "20")))


def is_memory_multi_tenant_strict() -> bool:
    """
    If MEMORY_MULTI_TENANT_STRICT=true, use department-based key/collection
    in the memory (Redis + episodic) layer.
    """
    return os.getenv("MEMORY_MULTI_TENANT_STRICT", "false").strip().lower() == "true"


def _effective_department_id() -> str:
    ctx = get_request_context()
    if ctx is not None and getattr(ctx, "department_id", None):
        return ctx.department_id
    return get_default_department_id()


def _with_department_prefix(key: str) -> str:
    """
    In strict mode, prefixes Redis keys with department.
    """
    if not is_memory_multi_tenant_strict():
        return key
    dept = _effective_department_id()
    return f"{dept}:{key}"


class ShortTermMemory:
    """
    Redis-based sliding window:
    - key: session_id
    - value: JSON-serialized BaseMessage list
    """

    def __init__(self, client: redis.Redis, cfg: MemoryConfig):
        self._client = client
        self._cfg = cfg

    def load(self, session_id: str) -> Sequence[BaseMessage]:
        key = _with_department_prefix(session_id)
        raw = self._client.get(key)
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            return []
        # Currently we only return messages carrying text content.
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
        key = _with_department_prefix(session_id)
        serializable: List[Dict[str, Any]] = []
        for msg in history[-self._cfg.short_term_max_turns :]:
            role = "other"
            if msg.type == "human":
                role = "human"
            elif msg.type == "ai":
                role = "ai"
            serializable.append({"role": role, "content": msg.content})
        self._client.setex(
            key,
            self._cfg.short_term_ttl_seconds,
            json.dumps(serializable, ensure_ascii=False),
        )

    def get_turn_count(self, session_id: str) -> int:
        """Returns the number of messages in the session."""
        key = _with_department_prefix(session_id)
        raw = self._client.get(key)
        if not raw:
            return 0
        try:
            data = json.loads(raw)
            return len(data)
        except Exception:
            return 0


class EntityMemory:
    """
    Redis hash for persistent per-user/entity information:
    - key: f\"{entity_prefix}{entity_id}\"
    - field: arbitrary
    """

    def __init__(self, client: redis.Redis, cfg: MemoryConfig):
        self._client = client
        self._cfg = cfg

    def load(self, entity_id: str) -> Dict[str, Any]:
        prefixed_id = _with_department_prefix(entity_id) if is_memory_multi_tenant_strict() else entity_id
        key = f"{self._cfg.entity_prefix}{prefixed_id}"
        raw = self._client.hgetall(key)
        result: Dict[str, Any] = {}
        for field, value in raw.items():
            try:
                result[field.decode("utf-8")] = json.loads(value)
            except Exception:
                result[field.decode("utf-8")] = value.decode("utf-8")
        return result

    def save(self, entity_id: str, data: Dict[str, Any]) -> None:
        prefixed_id = _with_department_prefix(entity_id) if is_memory_multi_tenant_strict() else entity_id
        key = f"{self._cfg.entity_prefix}{prefixed_id}"
        mapping = {
            k: json.dumps(v, ensure_ascii=False) for k, v in data.items()
        }
        if mapping:
            self._client.hset(key, mapping=mapping)


class EpisodicMemory:
    """
    Long-term episodic memory:
    - Stored in a separate Qdrant collection
    - Each entry is stored as a Document.
    """

    def __init__(self, cfg: MemoryConfig):
        self._cfg = cfg
        self._embeddings = create_embeddings()
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=self._cfg.qdrant_url)
        # Ensure default collection exists at least once
        self._ensure_collection_exists(self._cfg.episodic_collection)

    def _ensure_collection_exists(self, collection_name: str) -> None:
        """Creates an empty episodic collection in Qdrant if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            # Detect embedding dimension
            sample = self._embeddings.embed_query("test")
            dim = len(sample)
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"Episodic memory collection created: {collection_name}")

    def _get_vectorstore_for_current_department(self):
        from langchain_qdrant import QdrantVectorStore

        base = self._cfg.episodic_collection
        if not is_memory_multi_tenant_strict():
            collection_name = base
        else:
            dept = _effective_department_id()
            # Simple normalization for memory: same function logic
            import re

            s = re.sub(r"[^a-zA-Z0-9]+", "_", dept).strip("_").lower() or "default"
            collection_name = f"{base}_{s}"

        self._ensure_collection_exists(collection_name)

        return QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings,
        )

    def search(self, query: str, k: int = 5) -> List[Document]:
        vs = self._get_vectorstore_for_current_department()
        retriever = vs.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def add_episode(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        vs = self._get_vectorstore_for_current_department()
        doc = Document(page_content=text, metadata=metadata or {})
        vs.add_documents([doc])


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
    Postgres-based conversation summary store.

    For long conversations:
      1. When the short-term buffer fills up (20+ messages), a summary is
         automatically generated via an LLM call.
      2. The summary is saved to Postgres.
      3. In new conversation turns, relevant summaries are added to the
         context during retrieval.

    Gracefully degrades if Postgres connection is unavailable (no-op).
    """

    def __init__(self, postgres_url: str | None = None):
        self._postgres_url = postgres_url
        self._conn = None
        self._available = False
        self._init_connection()

    def _init_connection(self) -> None:
        """Initializes Postgres connection and creates the table."""
        if not self._postgres_url:
            logger.info("Postgres URL not defined — SummaryStore in no-op mode")
            return

        try:
            import psycopg2
            self._conn = psycopg2.connect(self._postgres_url)
            self._conn.autocommit = True
            with self._conn.cursor() as cur:
                cur.execute(_SUMMARY_TABLE_DDL)
            self._available = True
            logger.info("SummaryStore Postgres connection ready")
        except ImportError:
            logger.warning(
                "psycopg2 not installed. SummaryStore will run in no-op mode. "
                "Install: pip install psycopg2-binary"
            )
        except Exception as exc:
            logger.warning(f"Postgres connection failed: {exc}. SummaryStore in no-op mode.")

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
        """Saves a conversation summary to Postgres."""
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
            logger.info(f"Summary saved: session={session_id}, turns={turn_range}")
        except Exception as exc:
            logger.error(f"Summary save error: {exc}")

    def load_summaries(
        self,
        session_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Loads the most recent summaries for the session."""
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
            logger.error(f"Summary load error: {exc}")
            return []

    def delete_session_summaries(self, session_id: str) -> int:
        """Deletes all summaries for the session."""
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
            logger.error(f"Summary delete error: {exc}")
            return 0

    def close(self) -> None:
        """Closes the Postgres connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._available = False


# ── Summary Generator ─────────────────────────────────────────

SUMMARY_PROMPT = """Create a brief summary of the following conversation history.
Preserve important information, decisions, and context.
Maximum 3-4 sentences.

Conversation history:
{conversation}

Summary:"""


def generate_summary_if_needed(
    memory: "MemorySystem",
    *,
    session_id: str,
    messages: Sequence[BaseMessage],
    llm: Any = None,
) -> Optional[str]:
    """
    Automatically generates a summary via LLM when the short-term buffer
    fills up (more than trigger_turns messages) and saves it to Postgres.

    Args:
        memory: MemorySystem instance
        session_id: Current session ID
        messages: Current message list
        llm: LLM to use for summary generation (optional)

    Returns:
        Generated summary text or None
    """
    turn_count = len(messages)
    trigger = memory.cfg.summary_trigger_turns

    if turn_count < trigger:
        return None

    if not memory.summaries.is_available:
        logger.debug("SummaryStore not available, skipping summary")
        return None

    # Convert conversation to text
    conversation_lines = []
    for msg in messages:
        prefix = "User" if msg.type == "human" else "Assistant"
        content = (getattr(msg, "content", "") or "")[:300]
        conversation_lines.append(f"{prefix}: {content}")
    conversation_text = "\n".join(conversation_lines)

    # Generate summary with LLM
    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage
            prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
            result = llm.invoke([HumanMessage(content=prompt)])
            summary = getattr(result, "content", str(result))
        except Exception as exc:
            logger.warning(f"Failed to generate summary with LLM: {exc}")
            # Fallback: simple text summary
            summary = f"Conversation ({turn_count} messages): {conversation_text[:200]}..."
    else:
        # No LLM available, simple truncation
        summary = f"Conversation ({turn_count} messages): {conversation_text[:300]}..."

    # Save to Postgres
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
        redis_client.ping()  # Fail-fast: catch connection issues early
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
    Injects short-term buffer + episodic + entity + summary information into the system prompt.
    """
    short_history = memory.short_term.load(session_id)
    episodic_docs = memory.episodic.search(user_query, k=3)
    entity_data = memory.entity.load(entity_id) if entity_id else {}

    # Load relevant summaries from Postgres
    past_summaries = memory.summaries.load_summaries(session_id, limit=3)

    context_lines: List[str] = []

    # Past conversation summaries (long-term)
    if past_summaries:
        context_lines.append("Past conversation summaries:")
        for s in past_summaries:
            context_lines.append(f"- [{s.get('turn_range', '?')}] {s['summary']}")

    if short_history:
        context_lines.append("\nShort-term memory (recent conversations):")
        for msg in short_history[-5:]:
            prefix = "USER" if msg.type == "human" else "ASSISTANT"
            context_lines.append(f"- {prefix}: {msg.content}")
    if episodic_docs:
        context_lines.append("\nPast episodic memories:")
        for doc in episodic_docs:
            context_lines.append(f"- {doc.page_content[:120]}...")
    if entity_data:
        context_lines.append("\nEntity information:")
        for k, v in entity_data.items():
            context_lines.append(f"- {k}: {v}")

    if not context_lines:
        return messages

    from langchain_core.messages import SystemMessage

    injected = SystemMessage(
        content="Memory context:\n" + "\n".join(context_lines),
    )
    return [injected, *messages]


ENTITY_EXTRACT_PROMPT = """Extract persistent information about the user from the conversation.
Examples: profession, interests, preferences, name, location, etc.
Return only explicitly stated or clearly inferable facts as JSON.
Format: {{"key": "value", ...}} — keys should be lowercase with hyphens (e.g. favorite_language).
If there is no information to extract, return an empty object {{}}.

Conversation:
{conversation}

JSON:"""


def _extract_entity_facts(messages: Sequence[BaseMessage], llm: Any) -> Dict[str, Any]:
    """Extracts entity facts about the user from the conversation."""
    lines = []
    for msg in messages:
        prefix = "User" if msg.type == "human" else "Assistant"
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
        # Extract JSON block
        import re
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            import json
            return json.loads(match.group())
    except Exception as exc:
        logger.debug(f"Entity extraction error: {exc}")
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
    Writes new messages to the short-term buffer.
    Automatically generates a summary if the short-term buffer is full.
    Entity memory: extracts user information from the conversation via LLM and writes to Redis.
    """
    memory.short_term.save(session_id, messages)

    # Automatic summary trigger
    generate_summary_if_needed(
        memory,
        session_id=session_id,
        messages=messages,
        llm=llm,
    )

    # Entity memory update (every conversation)
    effective_entity_id = entity_id or session_id
    if effective_entity_id and llm is not None:
        try:
            new_facts = _extract_entity_facts(messages, llm)
            if new_facts:
                existing = memory.entity.load(effective_entity_id)
                merged = {**existing, **new_facts}
                memory.entity.save(effective_entity_id, merged)
                logger.debug(f"Entity updated: {effective_entity_id}, keys={list(new_facts.keys())}")
        except Exception as exc:
            logger.warning(f"Entity memory update error: {exc}")
