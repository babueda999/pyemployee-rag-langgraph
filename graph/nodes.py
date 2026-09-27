"""LangGraph node functions for the employee RAG graph."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph.state import EmployeeAgentState, EmployeeGraphState
from rag.prompts import AGENT_SYSTEM_PROMPT, ANSWER_PROMPT, format_docs
from rag.retriever import Retriever


def make_retrieve_node(retriever: Retriever):
    def retrieve_node(state: EmployeeGraphState) -> dict:
        documents = retriever.retrieve(state["question"])
        return {"documents": documents}

    return retrieve_node


def make_generate_node(llm: BaseChatModel):
    def generate_node(state: EmployeeGraphState) -> dict:
        prompt_value = ANSWER_PROMPT.invoke(
            {
                "context": format_docs(state["documents"]),
                "question": state["question"],
            }
        )
        response = llm.invoke(prompt_value)
        return {
            "answer": response.content,
            "chat_history": [
                HumanMessage(content=state["question"]),
                AIMessage(content=response.content),
            ],
        }

    return generate_node


def make_agent_node(llm: BaseChatModel, tools: list):
    """Builds the agent node for the tool-calling employee agent graph.

    The system prompt is prepended for each LLM call but never persisted
    back into state, so it appears exactly once per call without
    accumulating duplicates across the agent<->tools loop.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: EmployeeAgentState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node
