"""Exposes the compiled employee agent graph as an A2A server.

Mounted directly onto the existing FastAPI app (app/web.py) via
add_a2a_routes_to_fastapi, so it rides the same uvicorn process/port instead
of standing up a second server. Reuses graph.run.run_agent_graph — the exact
same invoke-and-extract logic POST /api/ask uses — so behavior over A2A
matches behavior over the REST API.
"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Role,
)
from a2a.helpers.proto_helpers import new_text_message
from a2a.utils.constants import TransportProtocol

from graph.run import run_agent_graph

AGENT_NAME = "employee-rag-agent"


def _build_agent_card(self_url: str) -> AgentCard:
    return AgentCard(
        name=AGENT_NAME,
        description=(
            "Answers company policy questions from local documents (RAG) and "
            "looks up employee records (read-only) via the Employee REST API."
        ),
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=self_url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version="1.0.0",
            )
        ],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="policy_search",
                name="Policy search",
                description="Answer questions about company policy documents (vacation, benefits, remote work, etc.).",
                tags=["policy", "rag"],
                examples=["What is the remote work policy?"],
            ),
            AgentSkill(
                id="employee_lookup",
                name="Employee lookup (read-only)",
                description="Look up or search employee records by id, name, or department. Cannot create, update, or delete.",
                tags=["employee", "read-only"],
                examples=["Find employee 101", "List employees in Engineering"],
            ),
        ],
    )


class GraphAgentExecutor(AgentExecutor):
    """Bridges A2A requests to the compiled LangGraph employee agent graph."""

    def __init__(self, graph):
        self._graph = graph

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = context.get_user_input()
        run_result = run_agent_graph(self._graph, question)
        await event_queue.enqueue_event(
            new_text_message(
                run_result.answer,
                role=Role.ROLE_AGENT,
                context_id=context.context_id,
                task_id=context.task_id,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Requests are handled synchronously to completion in execute(); there
        # is no in-flight work to cancel.
        pass


def mount_a2a_routes(app, graph, self_url: str) -> None:
    """Mounts the A2A agent-card and JSON-RPC routes for `graph` onto `app`
    at the paths implied by `self_url` (e.g. http://host:port/a2a).

    JSON-RPC only, deliberately — create_rest_routes() also always adds a
    Mount('/{tenant}', ...) catch-all for multi-tenant REST support, which
    (unprefixed, at the top of app.routes) swallows every other single-
    segment path on this app, e.g. /api/ask. Not needed: JSON-RPC is the
    default/primary A2A transport and is all tools/a2a_tools.py's client
    uses.
    """
    agent_card = _build_agent_card(self_url)
    request_handler = DefaultRequestHandler(
        agent_executor=GraphAgentExecutor(graph),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card, card_url="/a2a/.well-known/agent-card.json"),
        jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url="/a2a"),
    )
