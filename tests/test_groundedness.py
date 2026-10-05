"""Offline tests for the groundedness verify nodes: the RAG graph's
retrieve -> generate -> verify flow and the agent graph's verify node,
using the same scripted fake chat model as the other agent tests."""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from graph.employee_agent_graph import build_employee_agent_graph
from graph.employee_graph import build_employee_graph
from rag.prompts import UNGROUNDED_CAVEAT
from tests.test_employee_agent_graph import FakeRetriever, ScriptedChatModel, tool_call
from tests.test_employee_tools import FakeEmployeeService


def _build_rag_graph(responses):
    llm = ScriptedChatModel(responses=responses)
    retriever = FakeRetriever(
        documents=[
            Document(page_content="Full-time employees accrue 15 PTO days per year.", metadata={})
        ]
    )
    return build_employee_graph(llm, retriever)


def test_rag_grounded_answer_passes_through():
    graph = _build_rag_graph(
        [
            AIMessage(content="Full-time employees accrue 15 PTO days per year."),
            AIMessage(content="GROUNDED"),
        ]
    )
    result = graph.invoke(
        {"question": "How much PTO?", "chat_history": [], "documents": [], "answer": ""}
    )
    assert result["grounded"] is True
    assert "15 PTO days" in result["answer"]


def test_rag_ungrounded_answer_gets_rewritten():
    graph = _build_rag_graph(
        [
            AIMessage(content="Employees accrue 99 PTO days per year."),
            AIMessage(content="UNGROUNDED: '99 days' is not in the sources"),
            AIMessage(content="Full-time employees accrue 15 PTO days per year."),
            AIMessage(content="GROUNDED"),
        ]
    )
    result = graph.invoke(
        {"question": "How much PTO?", "chat_history": [], "documents": [], "answer": ""}
    )
    assert "15 PTO days" in result["answer"]
    assert result["grounded"] is True


def test_rag_still_ungrounded_gets_caveat():
    graph = _build_rag_graph(
        [
            AIMessage(content="Employees accrue 99 PTO days per year."),
            AIMessage(content="UNGROUNDED: '99 days' is not in the sources"),
            AIMessage(content="Employees accrue 50 PTO days per year."),
            AIMessage(content="UNGROUNDED: '50 days' is still not in the sources"),
        ]
    )
    result = graph.invoke(
        {"question": "How much PTO?", "chat_history": [], "documents": [], "answer": ""}
    )
    assert result["answer"].startswith(UNGROUNDED_CAVEAT)
    assert result["grounded"] is False


def test_agent_ungrounded_answer_gets_rewritten():
    llm = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("get_employee", {"employee_id": 1}, "call_1")],
            ),
            AIMessage(content="Employee 1 works in Marketing."),
            AIMessage(content="UNGROUNDED: tool said Engineering, not Marketing"),
            AIMessage(content="Employee 1 is in Engineering."),
            AIMessage(content="GROUNDED"),
        ]
    )
    employee = {
        "id": 1,
        "firstName": "Ada",
        "lastName": "Lovelace",
        "email": "ada@example.com",
        "department": "Engineering",
        "salary": 100000,
    }
    employee_service = FakeEmployeeService(employees=[employee])

    graph = build_employee_agent_graph(
        llm, FakeRetriever(), employee_service, "http://localhost:8080/a2a"
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="What department is employee 1 in?")]}
    )
    assert "Engineering" in result["messages"][-1].content
    assert "Marketing" not in result["messages"][-1].content


def test_agent_answer_without_tool_calls_passes_verify():
    llm = ScriptedChatModel(
        responses=[
            AIMessage(content="Hello! How can I help you today?"),
        ]
    )
    graph = build_employee_agent_graph(
        llm, FakeRetriever(), FakeEmployeeService(employees=[]), "http://localhost:8080/a2a"
    )
    result = graph.invoke({"messages": [HumanMessage(content="Hi")]})
    assert result["messages"][-1].content == "Hello! How can I help you today?"
