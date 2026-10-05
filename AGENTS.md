# AGENTS.md

Guidance for coding agents working in this repository. The always-on Devin
rules in `.devin/rules/pyemployee-rag-langgraph.md` carry the full detail —
this file is the quick orientation.

## What this repo is

Two things built on one codebase:

1. **RAG pipeline** — answers policy questions from `documents/` using a
   numpy-backed `SimpleVectorStore`, `Retriever`, and a LangGraph
   `retrieve -> generate -> verify` graph (`graph/employee_graph.py`).
2. **Employee agent** — a tool-calling LangGraph agent on top of that
   pipeline (`graph/employee_agent_graph.py`) plus an Employee REST API
   client (`services/employee_service.py`).

## Layout

- `ingestion/` — document loading, chunking, embeddings, vector store
- `rag/` — retriever and prompt templates (do NOT duplicate these)
- `graph/` — LangGraph state machines and nodes (both graphs + verify nodes)
- `tools/` — agent tools (policy search wraps `Retriever`, employee tools wrap
  `EmployeeServiceClient`, a2a delegation)
- `services/` — Employee REST API client (fixed base URL only, never arbitrary)
- `agent/` — A2A server exposing the agent
- `app/` — entry points: `main.py` (RAG CLI), `agent_main.py` (agent CLI),
  `web.py` + `static/` (FastAPI web UI)
- `evals/` — offline eval harness (real LLM calls, NOT part of pytest)
- `mcp/` — Docker-wrapper MCP servers (github, jira)
- `experiments/` — standalone pgvector experiment; keep its Postgres deps
  out of the main suite

## Setup

```
pip install -r requirements.txt
pip install pytest uvicorn   # not in requirements.txt
```

Copy/edit `.env`: `EMBEDDINGS_PROVIDER=openai` + `OPENAI_API_KEY` are the
current working combo. `ANTHROPIC_API_KEY` is optional (agent falls back to
`gpt-4o-mini`).

## Run

- RAG CLI: `python -m app.main`
- Agent CLI: `python -m app.agent_main`
- Web UI: `uvicorn app.web:app --reload --port 8000`
- Evals: `python -m evals.run_eval [-v]` (needs an LLM key)

## Test / verify

```
python -m pytest tests/
```

Never bare `pytest` — it collects `experiments/pgvector/database_test.py`,
which needs psycopg + a live Postgres and fails at collection. Agent tests are
fully offline (mocked `requests`, scripted fake `BaseChatModel`); keep them
that way.

## Conventions

- Reuse `rag.retriever.Retriever` and `rag.prompts.format_docs` for any new
  retrieval — no second vector store.
- Tools must turn service errors into string results, never raise.
- `.vector_store/` is a disposable gitignored cache — delete it after
  changing `EMBEDDINGS_PROVIDER` (old vectors keep stale dimensionality).
- `.env` and `.vector_store/` are never committed.

## Git workflow

Work on `dev` -> PR -> `main`. A GitHub Action merges `main` back into `dev`
on every `main` push — never hand-merge `main` into `dev`. See `plan.md`.

## Cloud environment

Snapshot blueprint `snapshot-blueprint-6a230441c0914e2ab5789c100d0e6424`
preinstalls deps and prebuilds `.vector_store/`. `OPENAI_API_KEY` is an org
secret; `ANTHROPIC_API_KEY` is not set. The Spring Boot Employee API
(`localhost:8080`) is unreachable from cloud — employee tools degrade
gracefully there.
