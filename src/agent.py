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
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.memory import MemorySystem, inject_memory_context, update_memory_after_response
from src.tools import ALL_TOOLS, get_all_tools
from src.tracing import get_tracer

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_WEB_SUMMARY_RE = re.compile(r"\[Summary\]:\s*(.*?)(?:\n\n\[Result|\n\nSource:|\Z)", re.DOTALL)
_WEB_SOURCE_RE = re.compile(r"^Source:\s*(https?://\S+)", re.MULTILINE)
_LOCAL_STATUS_RE = re.compile(r"\[LOCAL_SEARCH_STATUS\]:\s*(\w+)")

# ── Agent State ─────────────────────────────────────────────

MAX_ITERATIONS = 3

# Chat / konuşma patternleri — tool çağrısı gerektirmeyen sorgular
_CHAT_PATTERNS = {
    "emin misin", "emin misiniz", "tesekkurler", "tesekkur ederim", "sagol",
    "tamam", "anladim", "peki", "oldu", "ok", "evet", "hayir",
    "neden", "nasil yani", "ne demek", "baska", "devam et", "devam",
    "bir daha soyle", "tekrar et", "ozetler misin", "ozetle",
    "iyi gunler", "gorusuruz", "hosca kal", "bye",
}

SYSTEM_PROMPT = """You are a RAG (Retrieval-Augmented Generation) assistant.

TOOLS (priority order)
- search_documents: Varsayılan ilk adım. Ders notları, PDF içerikleri, kavramlar, tanımlar, kişi/olay vb. tüm bilgi soruları için önce bunu kullan.
- web_search: Sadece search_documents yeterli değilse (low/none) veya soru canlı veri gerektiriyorsa (hava durumu, döviz, haber, skor, vb.) kullan.
- calculator: Matematiksel ifadeler ve hesaplamalar için.
- Hiçbir tool: Selamlaşma, küçük konuşma veya takip soruları ("emin misin", "devam", "??" vb.) için doğrudan cevap ver.
- MCP tool'ları: Kullanıcı mesajında [MCP] veya [MCP:... ] öneki varsa, dosya sistemi / Postgres / hafıza gibi MCP araçlarını kullanmaya özellikle istekli ol.

LOCAL DOC CHUNKS
- search_documents çıktısındaki [CHUNK N] bölümlerini önce alaka açısından değerlendir.
- Genel/soyut bir soruysa ve chunk'lar konu dışı görünüyorsa, genel bilgiden cevap ver ve chunk'ları alıntılama.
- [DOCUMENT_FILTER] varsa, kullanıcının belirli bir dosyayı sorduğu anlamına gelir; mümkün olduğunca o dosyadan gelen chunk'lara odaklan.

ANSWER STYLE
- Dil: Türkçe, doğru karakterlerle (ş, ç, ğ, ı, ö, ü).
- Uzunluk: Tercihen 2–4 net cümle, gerekiyorsa kısa madde işaretleri.
- Atıf: Sadece gerçekten kullandığın chunk'lardan yararlandıysan, cevabın EN SONUNDA tek satırda yaz. Örnek:
  - Yerel dokümanlar: "Kaynaklar: [CHUNK 1] architecture.pdf p.3"
  - Web: "Web Kaynakları: url1, url2"
- Sadece sohbet, kod veya genel bilgi cevapları için asla kaynak yazma.
- Eğer belirli bir doküman sorulmuşsa ama tool sonucu cevabı içermiyorsa, tam olarak şöyle yanıtla: "Bilgi bulunamadı."

FOLLOW-UPS
- "emin misin", "neden", "biraz daha açıkla", "??" gibi takip sorularında önceki mesajları bağlam olarak kullan, "hangi konuda?" diye sorma.
- Bu tip takipler için ekstra tool çağırman çoğu durumda gereksizdir; önce mevcut bağlamdan açıklamayı dene.

Güvenilir olmayan veya konu dışı chunk'lara dayanarak cevap uydurma; gerekirse genel bilgiden kısa bir cevap ver veya "Bilgi bulunamadı." de."""


class AgentState(TypedDict):
    """Agent'ın taşıdığı state. messages listesi tüm konuşma geçmişini tutar."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def _get_intent_from_messages(messages: Sequence[BaseMessage]) -> str:
    """[INTENT]: ... system mesajindan intent'i ceker; yoksa knowledge kabul edilir."""
    for m in messages:
        if isinstance(m, SystemMessage) and m.content.startswith("[INTENT]:"):
            _, _, rest = m.content.partition(":")
            intent = (rest or "").strip()
            return intent or "knowledge"
    return "knowledge"


def _extract_text_tool_calls(content: str, tool_names: set) -> list[dict]:
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
        if name in tool_names and isinstance(args, dict):
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
    if lines and lines[0] in tool_names and len(lines) > 1:
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


def _is_chat_query(query: str) -> bool:
    """Konuşma/chat sorusu mu kontrol eder."""
    q = query.strip().lower()
    return q in _CHAT_PATTERNS


def _build_forced_tool_call(query: str) -> dict | None:
    raw = (query or "").strip()
    if not raw:
        return None

    # Özel frontend modu: [WEB_ONLY] prefix'i varsa her durumda web_search çağır
    WEB_ONLY_PREFIX = "[WEB_ONLY]"
    web_only = False
    if raw.startswith(WEB_ONLY_PREFIX):
        web_only = True
        q = raw[len(WEB_ONLY_PREFIX):].strip()
    else:
        q = raw

    if not q:
        return None

    # Eger query uzunlugu cok kisaysa veya sadece noktalama isareti ise tool cagirma
    if len(q) < 3 or all(c in "?!.,;- " for c in q):
        return None

    ql = q.lower()

    # Selamlama veya chat patternleri → tool çağırma
    if ql in {"merhaba", "selam", "hello", "hi"} or ql in _CHAT_PATTERNS:
        return None

    if web_only:
        return {"name": "web_search", "args": {"query": q}}

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


def build_research_agent_graph(llm, memory: MemorySystem | None = None, tools: list | None = None):
    """
    Verilen LLM'i ve tool listesini kullanarak derlenmiş bir LangGraph
    ReAct agent döndürür.

    Args:
        llm: Tool-calling destekleyen LangChain LLM (bind_tools uyumlu).
        tools: Kullanılacak tool listesi. None ise get_all_tools() (local + MCP).

    Returns:
        CompiledGraph: .invoke() veya .stream() ile çalıştırılabilir graph.
    """
    tool_list = tools if tools is not None else get_all_tools()
    _tool_names = {t.name for t in tool_list}
    llm_with_tools = llm.bind_tools(tool_list)

    # ── Entry Node: intent classifier / initial routing ─────
    tracer = get_tracer()

    def entry_node(state: AgentState) -> dict:
        # Su anda intent sonucu sadece metadata olarak tutuluyor; ileride
        # Memory / RAG node'larina routing icin kullanilabilir.
        messages = list(state["messages"])
        if not messages:
            return {"messages": []}
        last = messages[-1]
        if isinstance(last, HumanMessage):
            intent = "knowledge"
            q = last.content.strip().lower()
            if q in {"merhaba", "selam", "hello", "hi"}:
                intent = "greeting"
            elif q in _CHAT_PATTERNS:
                intent = "chat"
            elif _is_pure_math(q):
                intent = "math"
            elif _is_live_info_query(q):
                intent = "live_info"
            # Intent bilgisini SystemMessage icine encode ederek tasiyoruz.
            intent_system = SystemMessage(content=f"[INTENT]: {intent}")
            result = {"messages": [intent_system, last]}
            tracer.trace_node_end("entry", {"intent": intent})
            return result
        return {"messages": messages}

    # ── RAG Node: local retrieval augmentation ───────────────
    def rag_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
        # Memory sistemi varsa, hafiza baglamini system prompt olarak enjekte et.
        if memory is None:
            return {"messages": list(state["messages"])}

        messages = list(state["messages"])
        user_query = _get_latest_user_query(messages)
        configurable = (config or {}).get("configurable", {}) or {}
        session_id = configurable.get("session_id") or os.getenv("MEMORY_SESSION_ID", "cli-session")
        entity_id = configurable.get("entity_id") or os.getenv("MEMORY_ENTITY_ID")
        tracer.trace_node_start(
            "memory",
            {"session_id": session_id, "entity_id": entity_id, "query": user_query},
        )
        enriched = inject_memory_context(
            memory,
            session_id=session_id,
            entity_id=entity_id,
            user_query=user_query,
            messages=messages,
        )
        tracer.trace_node_end("memory", {"injected_messages": len(enriched)})
        return {"messages": list(enriched)}

    # ── Chat Node: saf konusma / greeting ─────────────────────
    def chat_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
        """
        Intent'i greeting/chat olan durumlarda RAG ve tool'lar ile ugrasmak yerine
        dogrudan LLM'den kisa bir cevap aliriz.
        PDF: Chat cevabi da memory'e yazilmalidir.
        """
        messages = list(state["messages"])

        has_system_prompt = any(
            isinstance(m, SystemMessage) and "RAG" in m.content for m in messages
        )
        if not has_system_prompt:
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

        tracer.trace_node_start("chat", {"message_count": len(messages)})
        response = llm.invoke(
            messages,
            config={
                "run_name": "agent_chat_turn",
                "tags": ["agent", "chat"],
                "metadata": {"message_count": len(messages)},
            },
        )
        tracer.trace_node_end("chat", {})

        # Memory'e yaz (PDF: Response Node benzeri)
        if memory is not None:
            configurable = (config or {}).get("configurable", {}) or {}
            session_id = configurable.get("session_id") or os.getenv("MEMORY_SESSION_ID", "cli-session")
            entity_id = configurable.get("entity_id") or os.getenv("MEMORY_ENTITY_ID")
            messages_with_response = list(messages) + [response]
            update_memory_after_response(
                memory,
                session_id=session_id,
                entity_id=entity_id,
                messages=messages_with_response,
                llm=llm,
            )

        return {"messages": [response]}

    # ── ReAct Agent Node: Reason ────────────────────────────
    def react_agent_node(state: AgentState) -> dict:
        messages = list(state["messages"])

        has_system_prompt = any(
            isinstance(m, SystemMessage) and "RAG" in m.content
            for m in messages
        )
        if not has_system_prompt:
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

        # Ilk adimda deterministic tool secimi: RAG oncelikli ve daha stabil.
        has_tool_calls = any(getattr(m, "tool_calls", None) for m in messages)
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if (
            not has_tool_calls
            and len(human_messages) == 1
            and not any(isinstance(m, ToolMessage) for m in messages)
        ):
            forced_call = _build_forced_tool_call(human_messages[0].content)
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
        # search_documents sonucu yetersizse (none), LLM kararini beklemeden web_search'e gec.
        enable_local_web_fallback = os.getenv("AGENT_ENABLE_LOCAL_WEB_FALLBACK", "true").lower() == "true"
        if (
            enable_local_web_fallback
            and isinstance(last_message, ToolMessage)
            and getattr(last_message, "name", "") == "search_documents"
        ):
            local_status = _extract_local_search_status(
                last_message.content if isinstance(last_message.content, str) else ""
            )
            if local_status in {"none"}:
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

        tracer.trace_node_start("react_agent", {"message_count": len(messages)})

        response = llm_with_tools.invoke(
            messages,
            config={
                "run_name": "agent_reason_step",
                "tags": ["agent", "reason"],
                "metadata": {"message_count": len(messages), "tool_count": len(tool_list)},
            },
        )

        tracer.trace_node_end("react_agent", {"has_tool_calls": bool(getattr(response, "tool_calls", None))})

        if response.content:
            response.content = _THINK_RE.sub("", response.content).strip()

        if not getattr(response, "tool_calls", None):
            parsed_calls = _extract_text_tool_calls(response.content or "", _tool_names)
            if parsed_calls:
                response = AIMessage(content="", tool_calls=parsed_calls)

        return {"messages": [response]}

    # ── Node: tools (Act) ───────────────────────────────────
    tool_node = ToolNode(tools=tool_list)

    # ── Router: decide next node (tools / response / end) ───
    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]

        if hasattr(last, "tool_calls") and last.tool_calls:
            tool_call_count = sum(
                1 for m in state["messages"] if hasattr(m, "tool_calls") and m.tool_calls
            )
            if tool_call_count >= MAX_ITERATIONS:
                return "response"
            return "tools"

        # Tool cagrisi yoksa artik Response Node'a gec.
        return "response"

    # ── Response Node: final synthesis (LLM or fast web) ────
    def response_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
        messages = list(state["messages"])
        if memory is not None:
            configurable = (config or {}).get("configurable", {}) or {}
            session_id = configurable.get("session_id") or os.getenv("MEMORY_SESSION_ID", "cli-session")
            entity_id = configurable.get("entity_id") or os.getenv("MEMORY_ENTITY_ID")
            update_memory_after_response(
                memory,
                session_id=session_id,
                entity_id=entity_id,
                messages=messages,
                llm=llm,
            )
            tracer.trace_node_end("response", {"session_id": session_id, "message_count": len(messages)})
        return {"messages": []}  # Yeni mesaj yok; side-effect (memory) zaten yapildi.

    # ── Router: intent → hangi ana path? ────────────────────
    def route_after_entry(state: AgentState) -> str:
        """
        entry node'un ekledigi [INTENT] system mesajina gore hangi node'a
        gidecegimize karar verir.

        - greeting/chat  → chat
        - math/live_info → react_agent (dogrudan tool + LLM, RAG/memory yok)
        - knowledge      → rag (RAG + memory, sonra react_agent)
        """
        messages = list(state["messages"])
        intent = _get_intent_from_messages(messages)

        if intent in {"greeting", "chat"}:
            return "chat"
        if intent in {"math", "live_info"}:
            return "react_agent"
        # default: knowledge
        return "rag"

    # ── Graph wiring ────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("entry", entry_node)
    graph.add_node("rag", rag_node)
    graph.add_node("chat", chat_node)
    graph.add_node("react_agent", react_agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("entry")

    # Intent temelli ilk branching
    graph.add_conditional_edges(
        "entry",
        route_after_entry,
        {"chat": "chat", "react_agent": "react_agent", "rag": "rag"},
    )
    graph.add_edge("rag", "react_agent")

    graph.add_conditional_edges(
        "react_agent",
        should_continue,
        {"tools": "tools", "response": "response"},
    )

    graph.add_edge("tools", "react_agent")
    graph.add_edge("response", END)
    graph.add_edge("chat", END)

    return graph.compile()


class SupervisorState(TypedDict):
    """Supervisor grafi icin state; su an sadece mesaj listesini tasir."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_agent_graph(llm, memory: MemorySystem | None = None, tools: list | None = None):
    """
    Supervisor agent: intent routing + sub-agent delegation.

    - ResearchAgent: RAG + web_search + calculator (bilgi edinme)
    - ExecutionAgent: MCP tabanlı eylemler (dosya sistemi, Postgres, hafıza, vb.)
    """

    research_graph = build_research_agent_graph(llm, memory=memory, tools=tools)

    # ExecutionAgent için MCP odaklı tool listesi
    exec_tool_names = {
        "read_file",
        "list_directory",
        "write_file",
        "execute_python",
        "run_bash",
        "query_postgres",
        "memory_search",
    }
    all_tools = get_all_tools()
    exec_tools = [t for t in all_tools if t.name in exec_tool_names]

    exec_llm_with_tools = llm.bind_tools(exec_tools)
    tracer = get_tracer()

    def execution_agent_node(
        state: SupervisorState,
        config: RunnableConfig | None = None,
    ) -> dict:
        """
        Basit ExecutionAgent:
        - Sadece MCP tabanlı tool'ları bağlar.
        - Aynı ReAct benzeri döngüyü uygular (tool_calls → ToolNode → tekrar).
        """
        messages = list(state["messages"])

        # Kısa, eylem odaklı system prompt
        exec_system = SystemMessage(
            content=(
                "You are an Execution Agent responsible for performing actions via MCP tools.\n"
                "Use the available tools to inspect files, run read-only queries and execute safe operations.\n"
                "Always explain what you did and what you found. Do not invent results."
            )
        )
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system:
            messages.insert(0, exec_system)

        tracer.trace_node_start("execution_agent", {"message_count": len(messages)})
        response = exec_llm_with_tools.invoke(
            messages,
            config={
                "run_name": "execution_agent_step",
                "tags": ["agent", "execution"],
            },
        )
        tracer.trace_node_end(
            "execution_agent",
            {"has_tool_calls": bool(getattr(response, "tool_calls", None))},
        )

        return {"messages": [response]}

    graph = StateGraph(SupervisorState)

    def supervisor_entry(state: SupervisorState) -> dict:
        return {"messages": list(state["messages"])}

    def route_to_agent(state: SupervisorState) -> str:
        """
        Basit routing:
        - Kullanıcı mesajı [MCP] veya [MCP:...] ile başlıyorsa → ExecutionAgent
        - Aksi durumda → ResearchAgent
        """
        messages = list(state["messages"])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                text = (msg.content or "").strip()
                if text.startswith("[MCP]") or text.startswith("[MCP:"):
                    return "execution"
                break
        return "research"

    def research_agent_node(
        state: SupervisorState,
        config: RunnableConfig | None = None,
    ) -> dict:
        sub_state = {"messages": list(state["messages"])}
        result = research_graph.invoke(sub_state, config=config or {})
        return {"messages": result.get("messages", [])}

    graph.add_node("supervisor_entry", supervisor_entry)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("execution_agent", execution_agent_node)

    graph.set_entry_point("supervisor_entry")
    graph.add_conditional_edges(
        "supervisor_entry",
        route_to_agent,
        {"research": "research_agent", "execution": "execution_agent"},
    )
    graph.add_edge("research_agent", END)
    graph.add_edge("execution_agent", END)

    return graph.compile()
