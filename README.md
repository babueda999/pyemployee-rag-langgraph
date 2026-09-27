# pyemployee-rag-langgraph

A small RAG (retrieval-augmented generation) assistant over employee/HR
documents, orchestrated with LangGraph and answered by Claude.

## Architecture

- `ingestion/` — load documents (`document_loader.py`), split into chunks
  (`chunker.py`), embed them (`embeddings.py`), and store/search vectors
  (`vector_store.py`, a small numpy-backed store — no chromadb/faiss
  dependency).
- `rag/` — retrieval (`retriever.py`), prompt templates (`prompts.py`), and
  the retrieve+generate chain (`rag_chain.py`).
- `graph/` — a LangGraph state machine (`employee_graph.py`,`nodes.py`,
  `state.py`) that runs `retrieve -> generate`.
- `app/main.py` — CLI: ingests `documents/` on first run, then answers
  questions interactively.

## Setup

```
pip install -r requirements.txt
```

Edit `.env` and set `ANTHROPIC_API_KEY`. Embeddings default to a local,
dependency-free hashing embedder (`EMBEDDINGS_PROVIDER=local`) so the app
runs fully offline aside from the Claude call. For better retrieval quality,
install `langchain-openai` or `langchain-voyageai` and set
`EMBEDDINGS_PROVIDER=openai` or `=voyage` with the matching API key.

## Run

Drop `.txt`, `.md`, or `.pdf` files into `documents/` (a sample PTO policy
is included), then:

```
python -m app.main
```

The first run ingests `documents/` into `.vector_store/`; subsequent runs
reuse it. Delete `.vector_store/` to force re-ingestion.

## Tests

```
pytest
```
