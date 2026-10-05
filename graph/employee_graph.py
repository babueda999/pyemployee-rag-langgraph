"""Builds the compiled LangGraph for employee Q&A: retrieve -> generate -> verify."""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from graph.nodes import make_generate_node, make_retrieve_node, make_verify_node
from graph.state import EmployeeGraphState
from rag.retriever import Retriever


def build_employee_graph(llm: BaseChatModel, retriever: Retriever):
    graph = StateGraph(EmployeeGraphState)
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("generate", make_generate_node(llm))
    graph.add_node("verify", make_verify_node(llm))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()
