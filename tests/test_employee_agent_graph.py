"""Verifies the agent<->tools loop routes to the right tool(s) and produces
a final answer, using a scripted fake chat model (no API key/network
needed) so these tests stay offline like the rest of the suite."""

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from graph.employee_agent_graph import build_employee_agent_graph
from tests.test_employee_tools import FakeEmployeeService


class FakeRetriever:
    def __init__(self, documents=None):
        self.documents = documents or [
            Document(page_content="Remote work is allowed 2 days/week.", metadata={"source": "policy.txt"})
        ]
        self.calls = []

    def retrieve(self, question: str):
        self.calls.append(question)
        return self.documents


class ScriptedChatModel(BaseChatModel):
    """Returns each message in `responses` in order, one per invoke() call."""

    responses: list

    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self.responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_policy_only_question_calls_only_policy_search():
    llm = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("policy_search", {"query": "vacation policy"}, "call_1")],
            ),
            AIMessage(content="Employees get 15-30 vacation days depending on tenure."),
        ]
    )
    retriever = FakeRetriever()
    employee_service = FakeEmployeeService(employees=[])

    graph = build_employee_agent_graph(llm, retriever, employee_service)
    result = graph.invoke({"messages": [HumanMessage(content="What is the vacation policy?")]})

    assert retriever.calls == ["vacation policy"]
    assert "vacation days" in result["messages"][-1].content


def test_employee_only_question_calls_only_employee_tool():
    llm = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("get_employee", {"employee_id": 1}, "call_1")],
            ),
            AIMessage(content="Employee 1 is in Engineering."),
        ]
    )
    retriever = FakeRetriever()
    employee = {
        "id": 1,
        "firstName": "Ada",
        "lastName": "Lovelace",
        "email": "ada@example.com",
        "department": "Engineering",
        "salary": 100000,
    }
    employee_service = FakeEmployeeService(employees=[employee])

    graph = build_employee_agent_graph(llm, retriever, employee_service)
    result = graph.invoke(
        {"messages": [HumanMessage(content="What department does employee 1 belong to?")]}
    )

    assert retriever.calls == []
    assert "Engineering" in result["messages"][-1].content


def test_combined_question_calls_both_tools_before_final_answer():
    llm = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("get_employee", {"employee_id": 1}, "call_1")],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("policy_search", {"query": "remote work eligibility"}, "call_2")
                ],
            ),
            AIMessage(content="Employee 1 (Engineering) is eligible for remote work."),
        ]
    )
    retriever = FakeRetriever(
        documents=[Document(page_content="Engineering may work remote 2 days/week.", metadata={})]
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

    graph = build_employee_agent_graph(llm, retriever, employee_service)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Is employee 1 eligible for remote work?")]}
    )

    assert retriever.calls == ["remote work eligibility"]
    assert "eligible for remote work" in result["messages"][-1].content


def test_unavailable_employee_reported_without_hallucinating():
    llm = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("get_employee", {"employee_id": 999}, "call_1")],
            ),
            AIMessage(content="No employee found with id 999, so I can't answer that."),
        ]
    )
    retriever = FakeRetriever()
    employee_service = FakeEmployeeService(employees=[])

    graph = build_employee_agent_graph(llm, retriever, employee_service)
    result = graph.invoke(
        {"messages": [HumanMessage(content="What department does employee 999 belong to?")]}
    )

    assert "No employee found with id 999" in result["messages"][-1].content
