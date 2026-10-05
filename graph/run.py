"""Shared helper for invoking a compiled employee agent graph and extracting
its final answer plus which tools ran.

Used by both app/web.py's POST /api/ask and agent/a2a_server.py's A2A
executor, so the invoke-and-extract logic lives in exactly one place.
"""

from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# Which underlying "agent" a tool call belongs to, purely for surfacing a
# human-readable label in the UI (there is one real LangGraph agent node;
# this just names the capability behind each tool).
TOOL_AGENT_LABELS = {
    "policy_search": "RAG Agent",
    "get_employee": "Employee Agent",
    "search_employee": "Employee Agent",
    "list_employees": "Employee Agent",
    "delegate_employee_write": "Employee Agent (Java)",
}


def agent_label_for_tool(tool_name: str) -> str:
    return TOOL_AGENT_LABELS.get(tool_name, "Agent")


@dataclass
class AgentRunResult:
    answer: str
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (tool name, result)


def run_agent_graph(
    graph, question: str, history: list[BaseMessage] | None = None
) -> AgentRunResult:
    result = graph.invoke({"messages": [*(history or []), HumanMessage(content=question)]})
    messages = result["messages"]

    tool_calls = [
        (message.name or "tool", message.content)
        for message in messages
        if isinstance(message, ToolMessage)
    ]
    final_message = messages[-1]
    answer = final_message.content if isinstance(final_message, AIMessage) else str(final_message.content)

    return AgentRunResult(answer=answer, tool_calls=tool_calls)


def stream_agent_graph(
    graph, question: str, history: list[BaseMessage] | None = None
) -> Iterator[dict]:
    """Streams progress events while the agent graph runs, for a live
    "what's it doing" indicator: a tool_start event as soon as the agent
    decides to call a tool (before it runs), a tool_end event once it
    returns, and a final event with the answer and all tool_calls -- the
    same information run_agent_graph returns in one shot, exposed as it
    happens instead of only after the whole run completes.
    """
    messages = [*(history or []), HumanMessage(content=question)]
    tool_calls: list[tuple[str, str]] = []
    answer = ""

    for chunk in graph.stream({"messages": messages}, stream_mode="updates"):
        for node, delta in chunk.items():
            for message in delta.get("messages", []):
                if node == "agent" and isinstance(message, AIMessage):
                    if message.tool_calls:
                        for call in message.tool_calls:
                            yield {
                                "type": "tool_start",
                                "agent": agent_label_for_tool(call["name"]),
                                "tool": call["name"],
                                "args": call.get("args", {}),
                            }
                    elif message.content:
                        answer = message.content
                elif node == "verify" and isinstance(message, AIMessage) and message.content:
                    answer = message.content
                elif node == "tools" and isinstance(message, ToolMessage):
                    name = message.name or "tool"
                    tool_calls.append((name, message.content))
                    yield {
                        "type": "tool_end",
                        "agent": agent_label_for_tool(name),
                        "tool": name,
                        "result": message.content,
                    }

    yield {"type": "final", "answer": answer, "tool_calls": tool_calls}
