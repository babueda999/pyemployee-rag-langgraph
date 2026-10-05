"""Builds the tool-calling employee agent graph: agent <-> tools loop.

Extends the existing graph package with a second, additive graph. Does not
modify or replace graph/employee_graph.py (the plain retrieve -> generate
RAG graph), which keeps working exactly as before.
"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph.nodes import make_agent_node, make_agent_verify_node
from graph.state import EmployeeAgentState
from rag.prompts import agent_system_prompt_for_role
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
    delegated_role: str = "MANAGER",
):
    tools = [
        make_policy_search_tool(retriever),
        *make_employee_tools(employee_service),
        make_java_agent_delegate_tool(java_agent_url, role=delegated_role),
    ]

    graph = StateGraph(EmployeeAgentState)
    graph.add_node(
        "agent", make_agent_node(llm, tools, agent_system_prompt_for_role(delegated_role))
    )
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("verify", make_agent_verify_node(llm))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "verify"})
    graph.add_edge("tools", "agent")
    graph.add_edge("verify", END)

    return graph.compile()
