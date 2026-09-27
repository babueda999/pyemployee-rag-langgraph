"""FastAPI entry point: single home page with navigation between an
Employees browser and the combined RAG + Employee-API chat agent.

Reuses the existing vector store / retriever (app.main), the employee
agent graph (graph.employee_agent_graph, tools/, services/), and the LLM
provider fallback (app.agent_main) — no logic is duplicated here.

Run with:

    uvicorn app.web:app --reload --port 8000
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.agent_main import get_agent_llm
from app.main import get_or_build_vector_store
from agent.a2a_server import mount_a2a_routes
from graph.employee_agent_graph import build_employee_agent_graph
from graph.run import run_agent_graph
from rag.retriever import get_retriever
from services.employee_service import (
    EmployeeApiError,
    EmployeeApiUnavailableError,
    EmployeeNotFoundError,
    get_employee_service,
)

app = FastAPI(title="Employee Assistant")

STATIC_DIR = Path(__file__).parent / "static"

documents_dir = os.getenv("DOCUMENTS_DIR", "documents")
persist_dir = os.getenv("VECTOR_STORE_DIR", ".vector_store")
java_agent_url = os.getenv("EMPLOYEE_AGENT_A2A_URL", "http://localhost:8080/a2a")

_vector_store = get_or_build_vector_store(documents_dir, persist_dir)
_retriever = get_retriever(_vector_store)
_employee_service = get_employee_service()
_llm = get_agent_llm()
_graph = build_employee_agent_graph(_llm, _retriever, _employee_service, java_agent_url)

mount_a2a_routes(app, _graph, self_url=os.getenv("SELF_A2A_URL", "http://localhost:8000/a2a"))


class HistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []


class ToolStep(BaseModel):
    tool: str
    result: str


class AskResponse(BaseModel):
    answer: str
    steps: list[ToolStep]


def _service_error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, EmployeeNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, EmployeeApiUnavailableError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, EmployeeApiError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected error.")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    history = [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in request.history
    ]
    run_result = run_agent_graph(_graph, question, history)
    steps = [ToolStep(tool=name, result=result) for name, result in run_result.tool_calls]

    return AskResponse(answer=run_result.answer, steps=steps)


@app.get("/api/employees")
def list_employees(department: str | None = None):
    try:
        employees = _employee_service.list_employees()
    except Exception as exc:  # noqa: BLE001 - translated to a clean HTTP error below
        raise _service_error_to_http(exc) from exc

    if department:
        department_lower = department.strip().lower()
        employees = [e for e in employees if (e.get("department") or "").lower() == department_lower]
    return employees


@app.get("/api/employees/search")
def search_employees(name: str):
    if not name.strip():
        raise HTTPException(status_code=400, detail="name must not be empty.")
    try:
        return _employee_service.search_employees(name)
    except Exception as exc:  # noqa: BLE001
        raise _service_error_to_http(exc) from exc


@app.get("/api/employees/{employee_id}")
def get_employee(employee_id: int):
    try:
        return _employee_service.get_employee(employee_id)
    except Exception as exc:  # noqa: BLE001
        raise _service_error_to_http(exc) from exc
