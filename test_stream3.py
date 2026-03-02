import asyncio
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.language_models import FakeListChatModel

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def my_node(state, config=None):
    res = AIMessageChunk(content="Final Output")
    return {"messages": [res]}

def test_graph():
    graph = StateGraph(AgentState)
    graph.add_node("node", my_node)
    graph.set_entry_point("node")
    graph.add_edge("node", END)
    compiled = graph.compile()
    
    for mode, payload in compiled.stream({"messages": [HumanMessage(content="Hi")]}, stream_mode=["messages", "values"]):
        if mode == "messages":
            msg, meta = payload
            print("MSG EVENT:", msg.type, repr(msg.content))
        elif mode == "values":
            print("VALUE EVENT!")

test_graph()
