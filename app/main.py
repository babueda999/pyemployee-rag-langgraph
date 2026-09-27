"""CLI entry point: ingest employee documents, then answer questions via LangGraph."""

import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from graph.employee_graph import build_employee_graph
from ingestion.chunker import chunk_documents
from ingestion.document_loader import load_documents
from ingestion.embeddings import get_embeddings
from ingestion.vector_store import SimpleVectorStore, build_vector_store, load_vector_store
from rag.retriever import get_retriever


def get_or_build_vector_store(documents_dir: str, persist_dir: str) -> SimpleVectorStore:
    embeddings = get_embeddings()

    if SimpleVectorStore.exists(persist_dir):
        store = load_vector_store(persist_dir, embeddings)
        print(f"Loaded existing vector store from {persist_dir} ({len(store)} chunks).")
        return store

    print(f"No vector store found at {persist_dir}. Ingesting documents from {documents_dir}...")
    documents = load_documents(documents_dir)
    if not documents:
        print(
            f"No documents found in {documents_dir}. Add .txt/.md/.pdf files there and rerun.",
            file=sys.stderr,
        )
    chunks = chunk_documents(documents)
    store = build_vector_store(chunks, embeddings, persist_dir)
    print(f"Ingested {len(documents)} document(s) into {len(chunks)} chunks.")
    return store


def main() -> None:
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Add it to .env before running.", file=sys.stderr)
        sys.exit(1)

    documents_dir = os.getenv("DOCUMENTS_DIR", "documents")
    persist_dir = os.getenv("VECTOR_STORE_DIR", ".vector_store")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

    vector_store = get_or_build_vector_store(documents_dir, persist_dir)
    retriever = get_retriever(vector_store)
    llm = ChatAnthropic(model=model, temperature=0)
    app = build_employee_graph(llm, retriever)

    print("\nEmployee RAG assistant ready. Type a question, or 'exit' to quit.")
    chat_history = []
    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = app.invoke({"question": question, "chat_history": chat_history})
        chat_history = result["chat_history"]
        print(f"\n{result['answer']}")
        sources = {doc.metadata.get("source", "unknown") for doc in result["documents"]}
        if sources:
            print(f"\nSources: {', '.join(sorted(sources))}")


if __name__ == "__main__":
    main()
