"""Shared helper for invoking a compiled employee agent graph and extracting
its final answer plus which tools ran.

Used by both app/web.py's POST /api/ask and agent/a2a_server.py's A2A
executor, so the invoke-and-extract logic lives in exactly one place.
"""

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


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
