"""Offline tests for the Python->Java A2A delegate tool. Mocks the A2A
transport call itself (tools.a2a_tools._send_to_java_agent), the same way
test_employee_tools.py mocks the HTTP layer with FakeEmployeeService --
no running Java agent or network access required.
"""

import asyncio

import tools.a2a_tools as a2a_tools


def run(tool, tool_input):
    return asyncio.run(tool.ainvoke(tool_input))


def make_tool(monkeypatch, reply="Employee 7's salary is now $88,000.", error=None):
    async def fake_send(base_url, text):
        if error:
            raise error
        fake_send.calls.append((base_url, text))
        return reply

    fake_send.calls = []
    monkeypatch.setattr(a2a_tools, "_send_to_java_agent", fake_send)
    return a2a_tools.make_java_agent_delegate_tool("http://localhost:8080/a2a"), fake_send


def test_rejects_empty_request(monkeypatch):
    tool, fake_send = make_tool(monkeypatch)

    result = run(tool, {"request": "   "})

    assert "Invalid request" in result
    assert fake_send.calls == []


def test_refuses_delete_without_calling_java_agent(monkeypatch):
    tool, fake_send = make_tool(monkeypatch)

    result = run(tool, {"request": "delete employee 7"})

    assert "does not perform deletions" in result
    assert fake_send.calls == []


def test_forwards_update_request_and_returns_reply(monkeypatch):
    tool, fake_send = make_tool(monkeypatch, reply="Employee 7's department is now Sales.")

    result = run(tool, {"request": "update employee 7's department to Sales"})

    assert result == "Employee 7's department is now Sales."
    assert fake_send.calls == [
        ("http://localhost:8080/a2a", "update employee 7's department to Sales")
    ]


def test_reports_unavailable_agent_without_raising(monkeypatch):
    tool, _ = make_tool(monkeypatch, error=a2a_tools.DelegateAgentError("connection refused"))

    result = run(tool, {"request": "give employee 7 a 10% raise"})

    assert "unavailable" in result.lower()
