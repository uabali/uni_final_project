"""
Agentic RAG — LangGraph ReAct agent.

ReAct döngüsü:
  1. REASON  — LLM sorguyu analiz eder, hangi tool'u çağıracağına karar verir.
  2. ACT     — Seçilen tool çalıştırılır (search_documents, web_search, calculator).
  3. OBSERVE — Tool çıktısı state'e yazılır; LLM sonucu değerlendirir.
  4. REPEAT  — Yeterli bilgi yoksa agent döngüye geri döner (max iterasyon limiti ile).

Kullanılan yapı: LangGraph StateGraph + ToolNode.
Model, tool'ları bind ederek hangi tool'u çağıracağını kendi seçer
(Qwen3-8B-AWQ, ChatOpenAI üzerinden tool-calling desteği).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.tools import ALL_TOOLS

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_WEB_SUMMARY_RE = re.compile(r"\[Summary\]:\s*(.*?)(?:\n\n\[Result|\n\nSource:|\Z)", re.DOTALL)
_WEB_SOURCE_RE = re.compile(r"^Source:\s*(https?://\S+)", re.MULTILINE)
_LOCAL_STATUS_RE = re.compile(r"\[LOCAL_SEARCH_STATUS\]:\s*(\w+)")

# ── Agent State ─────────────────────────────────────────────

MAX_ITERATIONS = 3
_TOOL_NAMES = {tool.name for tool in ALL_TOOLS}

SYSTEM_PROMPT = """\
You are a RAG (Retrieval-Augmented Generation) assistant with access to three tools.

## TOOL SELECTION RULES (order matters)

1. **search_documents** — Default FIRST step for knowledge questions \
(people, technical topics, lecture notes, definitions, concepts).
2. **web_search** — Use when:
   - search_documents is insufficient (low confidence / no relevant info), OR
   - the question explicitly requires live information \
(weather, breaking news, live scores, exchange rates, prayer times).
3. **calculator** — Use for arithmetic or mathematical expressions.
4. **No tool** — For simple greetings or small talk, reply directly without calling any tool.

## ANSWERING AFTER TOOL RESULTS

- **Language**: Turkish, ASCII-only (no ş, ç, ğ, ı, ö, ü).
- **Length**: Prefer 2-4 clear sentences. Use 1-2 bullet points only when the question asks for a list or comparison; otherwise write in prose. Do not repeat the same idea or add filler (e.g. "Ozetle...", "Sonuc olarak...").
- **Citation**: Keep short. search_documents: "Kaynaklar: [CHUNK N] dosya.pdf p.X" (only chunks you used). web_search: "Web Kaynaklari: url1, url2" (max 3 URLs).
- **No hallucination**: If the tool result does not contain the answer, respond exactly: "Bilgi bulunamadi."
- **Stop**: End as soon as the answer and citation are complete. No closing summaries or sign-offs.

## IMPORTANT

- Do NOT answer from training data. ALWAYS use a tool first.
- If local retrieval is insufficient, prefer web_search before saying "Bilgi bulunamadi."
- NEVER fabricate sources. Keep technical terms as-is (API, HTTP, Scrum, etc.).
"""


class AgentState(TypedDict):
    """Agent'ın taşıdığı state. messages listesi tüm konuşma geçmişini tutar."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def _extract_text_tool_calls(content: str) -> list[dict]:
    """vLLM parser kaçırırsa text içinden tool_call çıkarır."""
    if not content:
        return []

    calls: list[dict] = []

    # Format 1: <tool_call>{...}</tool_call>
    for payload in _TOOL_CALL_RE.findall(content):
        try:
            raw = json.loads(payload)
        except Exception:
            continue
        name = raw.get("name")
        args = raw.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        if name in _TOOL_NAMES and isinstance(args, dict):
            calls.append(
                {
                    "name": name,
                    "args": args,
                    "id": f"manual-tool-{uuid.uuid4().hex[:12]}",
                    "type": "tool_call",
                }
            )

    if calls:
        return calls

    # Format 2: web_search + JSON args (iki satır formatı)
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if lines and lines[0] in _TOOL_NAMES and len(lines) > 1:
        try:
            args = json.loads("\n".join(lines[1:]))
            if isinstance(args, dict):
                return [
                    {
                        "name": lines[0],
                        "args": args,
                        "id": f"manual-tool-{uuid.uuid4().hex[:12]}",
                        "type": "tool_call",
                    }
                ]
        except Exception:
            pass

    return []


_LIVE_PHRASES = [
    "hava durumu", "sicaklik", "weather",
    "canli skor", "mac sonucu", "mac skoru",
    "dolar kuru", "euro kuru", "altin fiyat", "bitcoin fiyat", "ethereum fiyat",
    "borsa endeks", "bist endeks", "faiz orani",
    "ramazan ne zaman", "imsak vakti", "iftar vakti", "iftar saati",
    "namaz vakti", "hicri takvim", "bayram ne zaman",
    "son dakika", "deprem oldu",
]

_LIVE_COMBO_TERMS = [
    ("bugun", {"hava", "sicaklik", "skor", "mac", "kur", "dolar", "euro", "altin", "fiyat"}),
    ("yarin", {"hava", "sicaklik", "mac", "iftar", "imsak"}),
    ("su an", {"kur", "dolar", "euro", "altin", "fiyat", "skor", "sicaklik"}),
    ("simdi", {"kur", "dolar", "euro", "altin", "fiyat", "skor", "sicaklik"}),
    ("guncel", {"kur", "dolar", "euro", "altin", "fiyat", "haber", "skor", "hava"}),
]


def _is_live_info_query(query: str) -> bool:
    q = query.lower()
    if any(phrase in q for phrase in _LIVE_PHRASES):
        return True
    for anchor, companions in _LIVE_COMBO_TERMS:
        if anchor in q and any(c in q for c in companions):
            return True
    return False


def _is_pure_math(query: str) -> str | None:
    """Sorgu tamamen matematik ifadesiyse ifadeyi dondurur, degilse None."""
    if not query:
        return None
    stripped = query.strip()
    allowed = set("0123456789+-*/.() eE")
    if all(ch in allowed for ch in stripped) and any(ch.isdigit() for ch in stripped):
        ops = set("+-*/()")
        if any(ch in ops for ch in stripped):
            return stripped
    return None


def _build_forced_tool_call(query: str) -> dict | None:
    q = (query or "").strip()
    if not q:
        return None

    if q.lower() in {"merhaba", "selam", "hello", "hi"}:
        return None

    if _is_live_info_query(q):
        return {"name": "web_search", "args": {"query": q}}

    expr = _is_pure_math(q)
    if expr:
        return {"name": "calculator", "args": {"expression": expr}}

    return {"name": "search_documents", "args": {"query": q}}


def _extract_local_search_status(tool_output: str) -> str | None:
    if not tool_output:
        return None
    match = _LOCAL_STATUS_RE.search(tool_output)
    if not match:
        return None
    return match.group(1).lower()


def _get_latest_user_query(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def _build_fast_web_answer(tool_output: str) -> str | None:
    """Tavily summary + URL'lerden LLM'siz hizli cevap uretir."""
    if not tool_output:
        return None

    summary_match = _WEB_SUMMARY_RE.search(tool_output)
    summary = summary_match.group(1).strip() if summary_match else ""
    if not summary:
        return None
    if len(summary) > 400:
        summary = summary[:400].rstrip() + "..."

    urls = list(dict.fromkeys(_WEB_SOURCE_RE.findall(tool_output)))[:3]

    lines: list[str] = [summary]

    if urls:
        lines.append("\nWeb Kaynaklari:")
        for url in urls:
            lines.append(f"- {url}")

    return "\n".join(lines)


# ── Graph Builder ───────────────────────────────────────────


def build_agent_graph(llm):
    """
    Verilen LLM'i ve tool listesini kullanarak derlenmiş bir LangGraph
    ReAct agent döndürür.

    Args:
        llm: Tool-calling destekleyen LangChain LLM (bind_tools uyumlu).

    Returns:
        CompiledGraph: .invoke() veya .stream() ile çalıştırılabilir graph.
    """
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # ── Node: agent (Reason) ────────────────────────────────
    def agent_node(state: AgentState) -> dict:
        messages = list(state["messages"])

        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

        # Ilk adimda deterministic tool secimi: RAG oncelikli ve daha stabil.
        if (
            len(messages) == 2
            and isinstance(messages[1], HumanMessage)
            and not any(getattr(m, "tool_calls", None) for m in messages)
        ):
            forced_call = _build_forced_tool_call(messages[1].content)
            if forced_call is not None:
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": forced_call["name"],
                                    "args": forced_call["args"],
                                    "id": f"forced-tool-{uuid.uuid4().hex[:12]}",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                }

        last_message = messages[-1]

        # Deterministic fallback:
        # search_documents sonucu yetersizse (low/none), LLM kararini beklemeden web_search'e gec.
        enable_local_web_fallback = os.getenv("AGENT_ENABLE_LOCAL_WEB_FALLBACK", "true").lower() == "true"
        if (
            enable_local_web_fallback
            and isinstance(last_message, ToolMessage)
            and getattr(last_message, "name", "") == "search_documents"
        ):
            local_status = _extract_local_search_status(
                last_message.content if isinstance(last_message.content, str) else ""
            )
            if local_status in {"low", "none"}:
                user_query = _get_latest_user_query(messages)
                if user_query:
                    return {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "web_search",
                                        "args": {"query": user_query},
                                        "id": f"forced-tool-{uuid.uuid4().hex[:12]}",
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        ]
                    }

        # Fast web finalization:
        # web_search sonucu yeterliyse ikinci LLM turunu atlayip gecikmeyi dusur.
        fast_web_finalize = os.getenv("AGENT_FAST_WEB_FINALIZE", "true").lower() == "true"
        if fast_web_finalize and isinstance(last_message, ToolMessage) and getattr(last_message, "name", "") == "web_search":
            fast_answer = _build_fast_web_answer(last_message.content if isinstance(last_message.content, str) else "")
            if fast_answer:
                return {"messages": [AIMessage(content=fast_answer)]}

        response = llm_with_tools.invoke(
            messages,
            config={
                "run_name": "agent_reason_step",
                "tags": ["agent", "reason"],
                "metadata": {"message_count": len(messages), "tool_count": len(ALL_TOOLS)},
            },
        )

        if response.content:
            response.content = _THINK_RE.sub("", response.content).strip()

        if not getattr(response, "tool_calls", None):
            parsed_calls = _extract_text_tool_calls(response.content or "")
            if parsed_calls:
                response = AIMessage(content="", tool_calls=parsed_calls)

        return {"messages": [response]}

    # ── Node: tools (Act) ───────────────────────────────────
    tool_node = ToolNode(tools=ALL_TOOLS)

    # ── Router (Observe → Repeat or End) ────────────────────
    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]

        if hasattr(last, "tool_calls") and last.tool_calls:
            tool_call_count = sum(
                1 for m in state["messages"] if hasattr(m, "tool_calls") and m.tool_calls
            )
            if tool_call_count >= MAX_ITERATIONS:
                return "end"
            return "tools"

        return "end"

    # ── Graph wiring ────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    graph.add_edge("tools", "agent")

    return graph.compile()
