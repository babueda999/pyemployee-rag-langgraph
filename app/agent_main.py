"""CLI entry point: the tool-calling employee agent (RAG + Employee API).

Reuses the existing vector store / retriever setup from app.main so the
underlying RAG pipeline is not duplicated. app/main.py is untouched and
still runs the original RAG-only flow.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.main import get_or_build_vector_store
from graph.employee_agent_graph import build_employee_agent_graph
from rag.retriever import get_retriever
from services.employee_service import get_employee_service


def get_agent_llm():
    """Prefers Anthropic (matches app/main.py); falls back to OpenAI when
    only OPENAI_API_KEY is configured."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        return ChatAnthropic(model=model, temperature=0)

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        model = os.getenv("AGENT_OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)

    print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env before running.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    load_dotenv()

    documents_dir = os.getenv("DOCUMENTS_DIR", "documents")
    persist_dir = os.getenv("VECTOR_STORE_DIR", ".vector_store")

    vector_store = get_or_build_vector_store(documents_dir, persist_dir)
    retriever = get_retriever(vector_store)
    employee_service = get_employee_service()
    llm = get_agent_llm()
    app = build_employee_agent_graph(llm, retriever, employee_service)

    print("\nEmployee agent ready (policy + employee lookups). Type a question, or 'exit' to quit.")
    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = app.invoke({"messages": [HumanMessage(content=question)]})
        print(f"\n{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
