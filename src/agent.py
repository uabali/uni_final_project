"""
Agentic RAG — LangGraph ReAct agent.

ReAct loop:
  1. REASON  — LLM analyzes the query and decides which tool to call.
  2. ACT     — The selected tool is executed (search_documents, web_search, calculator).
  3. OBSERVE — Tool output is written to state; LLM evaluates the result.
  4. REPEAT  — If there is not enough information, the agent loops back (with max iteration limit).

Architecture: LangGraph StateGraph + ToolNode.
The model binds tools and chooses which tool to call on its own
(Qwen3-8B-AWQ, tool-calling support via ChatOpenAI).
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

# Chat / conversation patterns — queries that do not require tool calls
_CHAT_PATTERNS = {
    "thanks", "thank you", "thx", "ty",
    "ok", "okay", "got it", "understood", "alright",
    "yes", "no", "nope", "yep", "sure",
    "why", "how so", "what do you mean", "anything else", "continue",
    "say that again", "repeat", "can you summarize", "summarize",
    "good day", "goodbye", "see you", "bye",
    "emin misin", "emin misiniz", "tesekkurler", "tesekkur ederim", "sagol",
    "tamam", "anladim", "peki", "oldu", "evet", "hayir",
    "neden", "nasil yani", "ne demek", "baska", "devam et", "devam",
    "bir daha soyle", "tekrar et", "ozetler misin", "ozetle",
    "iyi gunler", "gorusuruz", "hosca kal",
}

SYSTEM_PROMPT = """You are a RAG (Retrieval-Augmented Generation) assistant.

TOOLS (priority order)
- search_documents: Default first step. Use this first for all knowledge questions about lecture notes, PDF content, concepts, definitions, people/events, etc.
- web_search: Use only when search_documents is insufficient (low/none) or the question requires live data (weather, exchange rates, news, scores, etc.).
- calculator: For mathematical expressions and calculations.
- MCP tools: For file system, Postgres, memory, etc. — use when the user uses [MCP] prefix or when file/database operations are needed.

LOCAL DOC CHUNKS
- First evaluate [CHUNK N] sections from search_documents output for relevance.
- If the query is general/abstract and the chunks seem off-topic, answer from general knowledge and do not cite chunks.
- If [DOCUMENT_FILTER] is present, it means the user is asking about a specific file; focus on chunks from that file as much as possible.

ANSWER STYLE
- Language: Answer in the same language as the user's query.
- Length: Preferably 2–4 clear sentences; use short bullet points if necessary.
- Citation: Only if you actually used chunks, write citations at the VERY END of your answer in a single line. Example:
  - Local documents: "Sources: [CHUNK 1] architecture.pdf p.3"
  - Web: "Web Sources: url1, url2"
- Never write sources for pure chat, code, or general knowledge answers.
- If a specific document was asked about but the tool result does not contain the answer, respond exactly with: "No relevant information found."

FOLLOW-UPS
- For follow-up questions like "are you sure", "why", "explain more", "??", use previous messages as context; do not ask "about which topic?".
- For such follow-ups, calling extra tools is usually unnecessary; try to explain from the existing context first.

Do not fabricate answers based on unreliable or off-topic chunks; if necessary, give a short answer from general knowledge or say "No relevant information found.\""""

# LLM-based Intent Classification Prompt
INTENT_CLASSIFIER_PROMPT = """Given the user's message, classify the intent and select the most appropriate tool to use.

Available tools:
- search_documents: For questions about documents, PDFs, concepts, definitions, people, events,
- web_search technical topics: For real-time info, weather, news, exchange rates, sports scores, current events
- calculator: For mathematical expressions and calculations
- MCP tools (read_file, write_file, list_directory, query_postgres, memory_search, etc.): For file operations, database queries, memory access
- NO_TOOL: For greetings, small talk, acknowledgments, follow-up questions

User message: {message}

Previous conversation context: {context}

Respond with ONLY a JSON object (no other text):
{{"intent": "knowledge|math|live_info|file_operation|db_query|memory|greeting|chat|no_tool", "tool": "tool_name or null", "reason": "brief explanation"}}

Examples:
- "Hello how are you" → {{"intent": "greeting", "tool": null, "reason": "greeting message"}}
- "What does this PDF say" → {{"intent": "knowledge", "tool": "search_documents", "reason": "document question"}}
- "What's the weather today" → {{"intent": "live_info", "tool": "web_search", "reason": "weather is live info"}}
- "Calculate 5 + 3 * 2" → {{"intent": "math", "tool": "calculator", "reason": "mathematical expression"}}
- "[MCP] List files in /data folder" → {{"intent": "file_operation", "tool": "list_directory", "reason": "directory listing request"}}
- "Query users in the database" → {{"intent": "db_query", "tool": "query_postgres", "reason": "database query request"}}
- "What did we discuss in old chats" → {{"intent": "memory", "tool": "memory_search", "reason": "memory search request"}}
- "Continue" → {{"intent": "chat", "tool": null, "reason": "follow-up question"}}"""


class AgentState(TypedDict):
    """Agent state. The messages list holds the entire conversation history."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def _get_intent_from_messages(messages: Sequence[BaseMessage]) -> str:
    """Extracts intent from [INTENT]: ... system message; defaults to knowledge."""
    for m in messages:
        if isinstance(m, SystemMessage) and m.content.startswith("[INTENT]:"):
            _, _, rest = m.content.partition(":")
            intent = (rest or "").strip()
            return intent or "knowledge"
    return "knowledge"


def _extract_text_tool_calls(content: str, tool_names: set) -> list[dict]:
    """Extracts tool_call from text if vLLM parser misses it."""
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

    # Format 2: web_search + JSON args (two-line format)
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
    "weather", "temperature", "forecast",
    "live score", "match result", "match score",
    "dollar rate", "euro rate", "gold price", "bitcoin price", "ethereum price",
    "stock index", "interest rate",
    "breaking news", "earthquake",
    "hava durumu", "sicaklik",
    "canli skor", "mac sonucu", "mac skoru",
    "dolar kuru", "euro kuru", "altin fiyat", "bitcoin fiyat", "ethereum fiyat",
    "borsa endeks", "bist endeks", "faiz orani",
    "son dakika", "deprem oldu",
]

_LIVE_COMBO_TERMS = [
    ("today", {"weather", "temperature", "score", "match", "rate", "dollar", "euro", "gold", "price"}),
    ("tomorrow", {"weather", "temperature", "match", "forecast"}),
    ("right now", {"rate", "dollar", "euro", "gold", "price", "score", "temperature"}),
    ("current", {"rate", "dollar", "euro", "gold", "price", "news", "score", "weather"}),
    ("bugun", {"hava", "sicaklik", "skor", "mac", "kur", "dolar", "euro", "altin", "fiyat"}),
    ("yarin", {"hava", "sicaklik", "mac"}),
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
    """Returns the expression if the query is purely mathematical, otherwise None."""
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
    """Checks whether this is a conversational/chat query."""
    q = query.strip().lower()
    return q in _CHAT_PATTERNS


def _build_forced_tool_call(query: str) -> dict | None:
    raw = (query or "").strip()
    if not raw:
        return None

    # Special frontend mode: if [WEB_ONLY] prefix is present, always call web_search
    WEB_ONLY_PREFIX = "[WEB_ONLY]"
    web_only = False
    if raw.startswith(WEB_ONLY_PREFIX):
        web_only = True
        q = raw[len(WEB_ONLY_PREFIX):].strip()
    else:
        q = raw

    if not q:
        return None

    # If query length is too short or only punctuation, don't call a tool
    if len(q) < 3 or all(c in "?!.,;- " for c in q):
        return None

    ql = q.lower()

    # Greetings or chat patterns → do not call a tool
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
    """Produces a fast answer from Tavily summary + URLs without LLM."""
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
        lines.append("\nWeb Sources:")
        for url in urls:
            lines.append(f"- {url}")

    return "\n".join(lines)


# ── Graph Builder ───────────────────────────────────────────


def build_research_agent_graph(llm, memory: MemorySystem | None = None, tools: list | None = None):
    """
    Builds and returns a compiled LangGraph ReAct agent using the given LLM
    and tool list.

    Args:
        llm: LangChain LLM with tool-calling support (bind_tools compatible).
        tools: Tool list to use. If None, uses get_all_tools() (local + MCP).

    Returns:
        CompiledGraph: Graph that can be run with .invoke() or .stream().
    """
    tool_list = tools if tools is not None else get_all_tools()
    _tool_names = {t.name for t in tool_list}
    llm_with_tools = llm.bind_tools(tool_list)

    # ── Entry Node: intent classifier / initial routing ─────
    tracer = get_tracer()

    def entry_node(state: AgentState) -> dict:
        # Currently intent result is only kept as metadata; later it can be
        # used for routing to Memory / RAG nodes.
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
            # Encode intent info inside a SystemMessage for transport.
            intent_system = SystemMessage(content=f"[INTENT]: {intent}")
            result = {"messages": [intent_system, last]}
            tracer.trace_node_end("entry", {"intent": intent})
            return result
        return {"messages": messages}

    # ── RAG Node: local retrieval augmentation ───────────────
    def rag_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
        # If memory system exists, inject memory context as system prompt.
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

    # ── Chat Node: pure conversation / greeting ───────────────
    def chat_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
        """
        For greeting/chat intents, get a short answer directly from LLM
        without going through RAG and tools.
        Chat responses should also be written to memory.
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

        # Write to memory
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

        # First step: deterministic tool selection — RAG-first and more stable.
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
        # If search_documents result is insufficient (none), fall back to web_search without waiting for LLM decision.
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
        # If web_search result is sufficient, skip second LLM turn to reduce latency.
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

        # No tool calls — proceed to Response Node.
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
        return {"messages": []}  # No new message; side-effect (memory) already done.

    # ── Router: intent → which main path? ────────────────────
    def route_after_entry(state: AgentState) -> str:
        """
        Decides which node to go to based on the [INTENT] system message
        added by the entry node.

        - greeting/chat  → chat
        - math/live_info → react_agent (direct tool + LLM, no RAG/memory)
        - knowledge      → rag (RAG + memory, then react_agent)
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

    # Intent-based initial branching
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
    """State for the supervisor graph; currently only carries the message list."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_agent_graph(llm, memory: MemorySystem | None = None, tools: list | None = None):
    """
    Supervisor agent: intent routing + sub-agent delegation.

    - ResearchAgent: RAG + web_search + calculator (information retrieval)
    - ExecutionAgent: MCP-based actions (file system, Postgres, memory, etc.)
    """

    research_graph = build_research_agent_graph(llm, memory=memory, tools=tools)

    # MCP-focused tool list for ExecutionAgent
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
        Simple ExecutionAgent:
        - Only binds MCP-based tools.
        - Applies the same ReAct-like loop (tool_calls → ToolNode → repeat).
        """
        messages = list(state["messages"])

        # Short, action-oriented system prompt
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
        Simple routing:
        - If user message starts with [MCP] or [MCP:...] → ExecutionAgent
        - Otherwise → ResearchAgent
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
