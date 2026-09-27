"""Shared state passed between LangGraph nodes."""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class EmployeeGraphState(TypedDict):
    question: str
    chat_history: Annotated[list[BaseMessage], add_messages]
    documents: list[Document]
    answer: str


class EmployeeAgentState(TypedDict):
    """State for the tool-calling employee agent graph (graph/employee_agent_graph.py)."""

    messages: Annotated[list[BaseMessage], add_messages]