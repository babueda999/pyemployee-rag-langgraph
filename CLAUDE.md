# pyemployee-rag-langgraph

## Two things live here

1. **RAG pipeline** (`app/main.py`, `ingestion/`, `rag/`, `graph/employee_graph.py`) — answers
   policy questions from local documents (`documents/`). This is the original, working
   implementation. **Do not duplicate or replace it.** Any new capability that needs retrieval
   must reuse `rag.retriever.Retriever` and `rag.prompts.format_docs` as-is.
2. **Employee agent** (`app/agent_main.py`, `services/`, `tools/`, `graph/employee_agent_graph.py`)
   — a tool-calling LangGraph agent built *on top of* the RAG pipeline, plus an Employee REST API
   client, that can answer policy questions, employee questions, or both combined.

Full architecture/rationale: `C:\Users\eanbb\.claude\plans\cuddly-beaming-puppy.md`.

## RAG pipeline

`SimpleVectorStore` (`ingestion/vector_store.py`, numpy cosine similarity) + `get_embeddings()`
(`ingestion/embeddings.py`, local hashing embedder by default; `EMBEDDINGS_PROVIDER=openai`/`voyage`
for real embeddings) → `Retriever` (`rag/retriever.py`) → `graph/employee_graph.py`:
`START -> retrieve -> generate -> END` using `ChatAnthropic`. State: `EmployeeGraphState`
(`graph/state.py`). Run: `python -m app.main`.

## Employee agent

```
START -> agent (LLM with tools bound) -> [tool_calls?] -> tools -> agent -> ... -> END
```

- `graph/employee_agent_graph.py` — `build_employee_agent_graph(llm, retriever, employee_service)`.
  Uses `langgraph.prebuilt.ToolNode` + `tools_condition`. State: `EmployeeAgentState`
  (`graph/state.py`, `messages` only).
- `graph/nodes.py::make_agent_node` — binds tools, prepends `rag.prompts.AGENT_SYSTEM_PROMPT`
  per call (not persisted into state, so it never duplicates across the tool loop).
- `tools/policy_tools.py::make_policy_search_tool(retriever)` — wraps the **existing**
  `Retriever.retrieve()`. No second vector store.
- `tools/employee_tools.py::make_employee_tools(service)` — `get_employee`, `search_employee`,
  `list_employees`. Each validates its own input and turns service errors into a plain string
  result (never raises), so the agent reports "not found" / "unavailable" instead of hallucinating.
- `services/employee_service.py::EmployeeServiceClient` — talks to the Spring Boot Employee API.
  Only ever calls a fixed, env-configured base URL + known path templates — never an arbitrary
  caller-supplied URL. Run the CLI: `python -m app.agent_main`.
- `app.agent_main::get_agent_llm()` — picks `ChatAnthropic` if `ANTHROPIC_API_KEY` is set, else
  falls back to `ChatOpenAI` (`AGENT_OPENAI_MODEL`, default `gpt-4o-mini`) using `OPENAI_API_KEY`.

## Web frontend

`app/web.py` (FastAPI) + `app/static/index.html` (single page, tab nav, no build step) — one home
page with two tabs, reusing everything above (no separate logic):

- **Employees** tab — browse/search/filter employees. Calls `GET /api/employees`,
  `/api/employees/search?name=`, proxied straight through `EmployeeServiceClient` (avoids CORS,
  single origin for the browser).
- **Assistant** tab — chat UI over `POST /api/ask`, which invokes the same
  `build_employee_agent_graph` graph as the CLI and returns the answer plus which tools ran
  ("Tools used" panel).

Run: `uvicorn app.web:app --reload --port 8000`, open `http://127.0.0.1:8000/`. This replaces the
older `experiments/pgvector/web_app.py` UI for day-to-day use (that experiment's own separate
Postgres-backed RAG is untouched, just no longer the "front door").

Gotcha hit once: `section[hidden]` needs an explicit `display: none` rule in the page CSS because
the `section { display: flex; ... }` rule (author stylesheet) otherwise beats the browser's default
`[hidden]` UA rule.

Gotcha hit once: `.vector_store/` is a disposable, gitignored cache — if `EMBEDDINGS_PROVIDER`
changes (e.g. local→openai) after a store was already built, the persisted vectors keep the old
dimensionality and `similarity_search` throws a numpy shape-mismatch. Fix: delete `.vector_store/`
and let `get_or_build_vector_store` rebuild it under the current provider.

### Employee REST API contract (sibling repo `employee-servicves-main`, port 8080 by default)

- `GET /api/employees` → `EmployeeResponse[]` (`id, firstName, lastName, email, department, salary`).
  No server-side filter params — `list_employees` filters by department locally.
- `GET /api/employees/{id}` → one `EmployeeResponse`, 404 JSON error if missing.
- `GET /api/employees/search?name=X` → array, case-insensitive substring match on first/last name.
- There's also a separate Java-side agent at `POST /api/agent` with its own OpenAI tool loop and
  role-based guardrails (`AgentController`/`EmployeeAgent`). The Python agent here does **not** call
  that endpoint — it talks to the plain CRUD endpoints above directly.

### Env vars (new, optional, in `.env`)

- `EMPLOYEE_API_BASE_URL` (default `http://localhost:8080`)
- `EMPLOYEE_API_TIMEOUT_SECONDS` (default `5`)

## MCP servers (`mcp/`)

Claude-Code-facing MCP integrations live in their own top-level `mcp/` folder — kept separate
from the app code the same way `employee-servicves-main` sits as its own sibling project, rather
than mixed into `app/`/`services/`. Registered in `.mcp.json` (+ `.claude/settings.local.json`).
Each is a thin Python wrapper that loads `.env`, resolves credentials, and `subprocess.run`s the
matching official/community Docker image over stdio — secrets are forwarded to the container by
reference (`-e NAME` with no value) so they never appear in argv/process listings.

- `mcp/github_mcp_server.py` — `ghcr.io/github/github-mcp-server`. Token: `GITHUB_PERSONAL_ACCESS_TOKEN`
  (checked first) or `GITHUB_PAT` (this user's existing env var), either in `.env` or the OS env.
- `mcp/jira_mcp_server.py` — `ghcr.io/sooperset/mcp-atlassian` (Jira + Confluence). Reads this
  user's existing `JIRA_SITE_URL`/`JIRA_EMAIL`/`JIRA_API_TOKEN` (or the tool's own
  `JIRA_URL`/`JIRA_USERNAME`/`JIRA_API_TOKEN` names, or `.env`) and maps them to what
  mcp-atlassian expects.
- Not moved here: `experiments/pgvector/mcp_server.py` — that one exposes the pgvector
  *experiment's* own retrieval/answer tools and is tightly coupled to that package's imports, so
  it stays alongside the code it wraps rather than in the generic `mcp/` folder.
- New MCP servers only load at Claude Code session startup — restart/reconnect after adding one.

## Git / GitHub workflow

Repo: https://github.com/babueda999/pyemployee-rag-langgraph. Branches: `dev` (work here) ->
PR -> `main` (protected target). `.github/workflows/sync-dev-with-main.yml` runs on every push
to `main` and merges `main` back into `dev` automatically, so `dev` never drifts after a merge —
don't hand-merge `main` into `dev`, let the workflow do it.

Gotcha hit once: the local `GITHUB_PAT` env var is the `employee-services-mcp` fine-grained
token (name predates this repo, originally scoped only to `employee-servicves-main`) — not the
similarly-named `GithubToken` also present in the account. Don't assume the token matching a
generic name is the one actually in use; check `github-authentication-token-expiration` in a
response header (or just test a write) against the token list at
https://github.com/settings/personal-access-tokens to confirm which one it is. Currently scoped
to `babueda999/employee-services` and `babueda999/pyemployee-rag-langgraph` with Contents,
Pull requests, and Workflows set to read/write. Editing a fine-grained token's scopes requires a
sudo-mode email re-verification in the browser — can't be done via the API.

Open items: branch protection on `main` isn't set up yet (needs the `Administration` scope added
to the token first); the repo is public, not private; `GithubToken` was left scoped to this repo
with Contents read/write from debugging and is unused — see `plan.md`.

## Testing

`pytest` from the project root. Existing tests (`test_chunker.py`, `test_vector_store.py`) must
keep passing untouched. New agent tests (`test_employee_service.py`, `test_employee_tools.py`,
`test_employee_agent_graph.py`) are fully offline — no API keys or running services required —
using mocked `requests` calls and a scripted fake `BaseChatModel`.
