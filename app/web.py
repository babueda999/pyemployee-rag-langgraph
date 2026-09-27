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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from app.agent_main import get_agent_llm
from app.main import get_or_build_vector_store
from graph.employee_agent_graph import build_employee_agent_graph
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

_vector_store = get_or_build_vector_store(documents_dir, persist_dir)
_retriever = get_retriever(_vector_store)
_employee_service = get_employee_service()
_llm = get_agent_llm()
_graph = build_employee_agent_graph(_llm, _retriever, _employee_service)


class AskRequest(BaseModel):
    question: str


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

    result = _graph.invoke({"messages": [HumanMessage(content=question)]})
    messages = result["messages"]

    steps = [
        ToolStep(tool=message.name or "tool", result=message.content)
        for message in messages
        if isinstance(message, ToolMessage)
    ]
    final_message = messages[-1]
    answer = final_message.content if isinstance(final_message, AIMessage) else str(final_message.content)

    return AskResponse(answer=answer, steps=steps)


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
