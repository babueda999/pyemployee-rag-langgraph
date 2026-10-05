---
name: verify
description: Run this project's tests and checks after making code changes. Use before committing, when asked to check that nothing broke, or when the user asks whether tests pass.
---

# Verify

Run from the repo root:

```
python -m pytest tests/ -x -q
```

Rules:

- Always scope to `tests/`. Bare `pytest` collects
  `experiments/pgvector/database_test.py`, which requires `psycopg` and a
  live Postgres and fails at collection — that failure is environmental,
  not a code regression.
- All tests in `tests/` must run offline. Do not add tests that need real
  API keys, network, or the Employee REST API — mock `requests` or use a
  scripted fake `BaseChatModel` like `tests/test_employee_agent_graph.py`.
- `python -m evals.run_eval` is NOT part of this check — it makes real LLM
  calls. Only run it when the user explicitly asks.
- If `.vector_store/` dimensionality errors appear after an
  `EMBEDDINGS_PROVIDER` change, delete `.vector_store/` and let it rebuild.
