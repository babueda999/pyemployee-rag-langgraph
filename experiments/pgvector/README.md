# pgvector experiment

A scratch exploration of a Postgres/pgvector-backed store (HuggingFace
embeddings for ingestion, OpenAI for generation), kept separate from the
main app in `ingestion/`, `graph/`, and `rag/`, which uses a local numpy
vector store instead.

Not wired into `requirements.txt` or the test suite. Extra deps used here:
`psycopg`, `langchain-huggingface`, `langchain-community`, `langchain-openai`, `mcp`.

```
docker compose -f experiments/pgvector/docker-compose.yml up -d
python -m experiments.pgvector.vector_store   # create table + ingest PDF
python -m experiments.pgvector.graph          # ask a sample question
```

## MCP server

`mcp_server.py` exposes this pipeline to Claude Code (or any MCP client) as
two tools:

- `search_employee_policies(query, top_k)` — raw pgvector similarity search,
  no LLM generation. Good for checking retrieval quality on its own.
- `ask_employee_policy_question(question)` — full LangGraph retrieve+generate
  pipeline (pgvector retrieval, OpenAI `gpt-4o-mini` generation). Requires
  `OPENAI_API_KEY` in `.env`.

Registered as a project-scoped server in `.mcp.json` at the repo root. Claude
Code needs a restart (or `/mcp` reconnect) and a one-time approval prompt to
pick it up. To run it standalone for debugging: `python experiments/pgvector/mcp_server.py`.
