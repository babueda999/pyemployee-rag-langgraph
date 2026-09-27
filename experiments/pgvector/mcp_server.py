"""MCP server exposing the pgvector RAG experiment's retrieval and Q&A as tools.

Run directly (stdio transport) or launch via Claude Code's MCP config:

    python -m experiments.pgvector.mcp_server
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from mcp.server.mcpserver import MCPServer

from experiments.pgvector.retriever import similarity_search

mcp = MCPServer("employee-rag-pgvector")


@mcp.tool()
def search_employee_policies(query: str, top_k: int = 3) -> list[dict]:
    """Search the pgvector-backed employee policy chunks for the passages most
    similar to the query. Returns raw retrieval results (no LLM generation) so
    retrieval quality can be inspected on its own."""
    results = similarity_search(query, top_k=top_k)
    return [
        {
            "chunk_id": row[0],
            "content": row[1],
            "metadata": row[2],
            "similarity": float(row[3]),
        }
        for row in results
    ]


@mcp.tool()
def ask_employee_policy_question(question: str) -> str:
    """Run the full LangGraph retrieve+generate pipeline (pgvector retrieval,
    OpenAI generation) and return the answer. Requires OPENAI_API_KEY."""
    from experiments.pgvector.graph import build_graph

    graph = build_graph()
    result = graph.invoke({"question": question, "context": "", "answer": ""})
    return result["answer"]


if __name__ == "__main__":
    mcp.run()
