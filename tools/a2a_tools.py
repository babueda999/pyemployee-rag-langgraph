"""A2A client tool: delegates employee *write* requests to the Java
EmployeeAgent (employee-servicves-main), which has the update/adjust-salary
tools this agent deliberately doesn't (see tools/employee_tools.py).

Runs delegated calls as a fixed MANAGER role and never mentions deletion —
MANAGER cannot delete per the Java side's AuthorizationGuardrail, and this
tool adds a second, independent guard by simply never offering that action.
"""

import asyncio

import httpx
from langchain_core.tools import tool

from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.helpers.proto_helpers import get_message_text, get_text_parts, new_text_message
from a2a.types.a2a_pb2 import Role, SendMessageRequest

DELEGATED_ROLE = "MANAGER"


class DelegateAgentError(Exception):
    """Raised when the Java agent is unreachable or returns no reply."""


async def _send_to_java_agent(base_url: str, text: str) -> str:
    # httpx's 5s default is too tight for a call that's backed by an LLM
    # round trip on the Java side; supply our own client with more headroom
    # rather than hanging forever on a dead agent.
    factory = ClientFactory(
        ClientConfig(streaming=False, httpx_client=httpx.AsyncClient(timeout=30.0))
    )
    client = await factory.create_from_url(base_url)
    try:
        message = new_text_message(text, role=Role.ROLE_USER)
        message.metadata.update({"authRole": DELEGATED_ROLE})
        request = SendMessageRequest(message=message)

        async for response in client.send_message(request):
            if response.HasField("message"):
                return get_message_text(response.message)
            if response.HasField("task"):
                if response.task.status.HasField("message"):
                    return get_message_text(response.task.status.message)
                # The Java EmployeeAgent replies via a completed Task with the
                # answer in its final artifact (the SDK's documented pattern:
                # AgentEmitter.addArtifact(...) + .complete()), not a status
                # message.
                artifact_texts = [
                    text
                    for artifact in response.task.artifacts
                    for text in get_text_parts(artifact.parts)
                ]
                if artifact_texts:
                    return "\n".join(artifact_texts)
            if response.HasField("status_update") and response.status_update.status.HasField("message"):
                return get_message_text(response.status_update.status.message)

        raise DelegateAgentError("Java agent returned no reply.")
    finally:
        await client.close()


def make_java_agent_delegate_tool(base_url: str):
    @tool
    def delegate_employee_write(request: str) -> str:
        """Delegate an employee record UPDATE or SALARY ADJUSTMENT to the
        Java Employee Agent (e.g. "update employee 3's department to Sales",
        "give employee 7 a 10% raise"). Runs as a MANAGER-level request. Do
        NOT use this for deletion — the Java agent will refuse it at that
        role, and this tool must never be used to attempt one. Do not use
        this for read-only lookups; use get_employee/search_employee/
        list_employees instead."""
        request = (request or "").strip()
        if not request:
            return "Invalid request: must not be empty."
        if "delete" in request.lower():
            return "This tool does not perform deletions. Refusing to forward the request."

        try:
            return asyncio.run(_send_to_java_agent(base_url, request))
        except DelegateAgentError as exc:
            return f"Employee agent is currently unavailable: {exc}"
        except Exception as exc:  # noqa: BLE001 - network/transport errors from the A2A client
            return f"Employee agent is currently unavailable: {exc}"

    return delegate_employee_write
