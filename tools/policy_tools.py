"""Exposes the existing RAG retriever as an agent tool.

Reuses rag.retriever.Retriever and rag.prompts.format_docs as-is — no new
vector store, embeddings, or retrieval logic.
"""

from langchain_core.tools import tool

from rag.prompts import format_docs
from rag.retriever import Retriever

MAX_QUERY_LENGTH = 500


def make_policy_search_tool(retriever: Retriever):
    @tool
    def policy_search(query: str) -> str:
        """Search the company policy documents (vacation, sick leave,
        benefits, remote work, etc.) for passages relevant to the query.
        Use this for any question about company policy."""
        query = (query or "").strip()
        if not query:
            return "Invalid query: must not be empty."
        if len(query) > MAX_QUERY_LENGTH:
            return f"Invalid query: must be {MAX_QUERY_LENGTH} characters or fewer."

        documents = retriever.retrieve(query)
        return format_docs(documents)

    return policy_search
