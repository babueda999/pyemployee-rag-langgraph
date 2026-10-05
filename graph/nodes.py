"""LangGraph node functions for the employee RAG graph."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from graph.state import EmployeeAgentState, EmployeeGraphState
from rag.prompts import (
    AGENT_SYSTEM_PROMPT,
    ANSWER_PROMPT,
    CORRECTION_PROMPT,
    GROUNDEDNESS_PROMPT,
    UNGROUNDED_CAVEAT,
    format_docs,
)
from rag.retriever import Retriever


def make_retrieve_node(retriever: Retriever):
    def retrieve_node(state: EmployeeGraphState) -> dict:
        history = state.get("chat_history") or []
        # A bare follow-up ("what about 7 years?") retrieves poorly on its
        # own -- fold in the last couple of prior questions so the vector
        # search still has the topic keywords (e.g. "vacation days").
        recent_questions = [m.content for m in history if isinstance(m, HumanMessage)][-2:]
        query = "\n".join([*recent_questions, state["question"]])
        documents = retriever.retrieve(query)
        return {"documents": documents}

    return retrieve_node


def make_generate_node(llm: BaseChatModel):
    def generate_node(state: EmployeeGraphState) -> dict:
        prompt_value = ANSWER_PROMPT.invoke(
            {
                "context": format_docs(state["documents"]),
                "question": state["question"],
                "chat_history": state.get("chat_history") or [],
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


def _check_grounded(llm: BaseChatModel, evidence: str, answer: str) -> tuple[bool, str]:
    """LLM-as-judge: is every claim in `answer` supported by `evidence`?"""
    verdict = llm.invoke(
        GROUNDEDNESS_PROMPT.invoke({"evidence": evidence, "answer": answer})
    ).content.strip()
    grounded = verdict.upper().startswith("GROUNDED") and not verdict.upper().startswith(
        "UNGROUNDED"
    )
    return grounded, verdict


def _rewrite_grounded(llm: BaseChatModel, evidence: str, answer: str, verdict: str) -> str:
    return llm.invoke(
        CORRECTION_PROMPT.invoke({"evidence": evidence, "answer": answer, "verdict": verdict})
    ).content


def make_verify_node(llm: BaseChatModel):
    """Groundedness check for the RAG graph: judges generate's answer against
    the retrieved documents, rewrites once if ungrounded, and prepends a
    caveat if the rewrite is still not supported."""

    def verify_node(state: EmployeeGraphState) -> dict:
        answer = state["answer"]
        evidence = format_docs(state.get("documents") or [])

        grounded, verdict = _check_grounded(llm, evidence, answer)
        if grounded:
            return {"grounded": True}

        corrected = _rewrite_grounded(llm, evidence, answer, verdict)
        grounded, _ = _check_grounded(llm, evidence, corrected)
        if not grounded:
            corrected = f"{UNGROUNDED_CAVEAT}\n\n{corrected}"
        return {"answer": corrected, "grounded": grounded}

    return verify_node


def make_agent_verify_node(llm: BaseChatModel):
    """Groundedness check for the agent graph: judges the final answer
    against the tool outputs. No tool outputs means nothing to check
    against, so the answer passes through. An ungrounded answer is rewritten
    once; the corrected message is appended so it becomes the final answer."""

    def verify_node(state: EmployeeAgentState) -> dict:
        messages = state["messages"]
        final = messages[-1]
        tool_outputs = [m.content for m in messages if isinstance(m, ToolMessage)]
        if not isinstance(final, AIMessage) or not tool_outputs:
            return {"messages": []}

        evidence = "\n\n".join(tool_outputs)
        grounded, verdict = _check_grounded(llm, evidence, final.content)
        if grounded:
            return {"messages": []}

        corrected = _rewrite_grounded(llm, evidence, final.content, verdict)
        grounded, _ = _check_grounded(llm, evidence, corrected)
        if not grounded:
            corrected = f"{UNGROUNDED_CAVEAT}\n\n{corrected}"
        return {"messages": [AIMessage(content=corrected)]}

    return verify_node


def make_agent_node(llm: BaseChatModel, tools: list, system_prompt: str = AGENT_SYSTEM_PROMPT):
    """Builds the agent node for the tool-calling employee agent graph.

    The system prompt is prepended for each LLM call but never persisted
    back into state, so it appears exactly once per call without
    accumulating duplicates across the agent<->tools loop.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: EmployeeAgentState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node
