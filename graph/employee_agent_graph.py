"""Builds the tool-calling employee agent graph: agent <-> tools loop.

Extends the existing graph package with a second, additive graph. Does not
modify or replace graph/employee_graph.py (the plain retrieve -> generate
RAG graph), which keeps working exactly as before.
"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph.nodes import make_agent_node
from graph.state import EmployeeAgentState
from rag.retriever import Retriever
from services.employee_service import EmployeeServiceClient
from tools.a2a_tools import make_java_agent_delegate_tool
from tools.employee_tools import make_employee_tools
from tools.policy_tools import make_policy_search_tool


def build_employee_agent_graph(
    llm: BaseChatModel,
    retriever: Retriever,
    employee_service: EmployeeServiceClient,
    java_agent_url: str,
):
    tools = [
        make_policy_search_tool(retriever),
        *make_employee_tools(employee_service),
        make_java_agent_delegate_tool(java_agent_url),
    ]

    graph = StateGraph(EmployeeAgentState)
    graph.add_node("agent", make_agent_node(llm, tools))
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
