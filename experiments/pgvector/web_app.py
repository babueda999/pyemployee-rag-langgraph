"""FastAPI frontend for the pgvector RAG experiment.

Serves a single-page chat UI and a small JSON API on top of the existing
retriever + LangGraph pipeline (experiments/pgvector/retriever.py, graph.py).

Run with:

    uvicorn experiments.pgvector.web_app:app --reload --port 8000
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from experiments.pgvector.graph import build_graph
from experiments.pgvector.retriever import similarity_search

app = FastAPI(title="Employee Policy Assistant")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_graph = build_graph()


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    chunk_id: int
    content: str
    similarity: float
    metadata: dict


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()

    chunks = similarity_search(question, top_k=3)
    result = _graph.invoke({"question": question, "context": "", "answer": ""})

    return AskResponse(
        answer=result["answer"],
        sources=[
            Source(chunk_id=row[0], content=row[1], similarity=float(row[3]), metadata=row[2] or {})
            for row in chunks
        ],
    )
