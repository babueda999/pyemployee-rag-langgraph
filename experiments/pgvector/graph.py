from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from experiments.pgvector.retriever import similarity_search


class RAGState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve_documents(state: RAGState):
    question = state["question"]

    results = similarity_search(
        question,
        top_k=3,
    )

    context_parts = []

    for result in results:
        chunk_id = result[0]
        content = result[1]
        similarity = result[3]

        context_parts.append(
            f"""
Chunk ID: {chunk_id}
Similarity: {similarity:.4f}

{content}
"""
        )

    context = "\n\n".join(context_parts)

    return {
        "context": context
    }


def generate_answer(state: RAGState):

    question = state["question"]
    context = state["context"]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = f"""
You are an employee policy assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:

"I don't have enough information in the employee policy
documents to answer that question."

Do not invent or assume information.

Context:
--------------------
{context}
--------------------

Question:
{question}

Provide a clear and concise answer.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def build_graph():

    graph = StateGraph(RAGState)

    graph.add_node(
        "retrieve",
        retrieve_documents
    )

    graph.add_node(
        "generate_answer",
        generate_answer
    )

    graph.add_edge(
        START,
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "generate_answer"
    )

    graph.add_edge(
        "generate_answer",
        END
    )

    return graph.compile()


if __name__ == "__main__":

    graph = build_graph()

    question = "What is the vacation policy for employees?"

    result = graph.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
        }
    )

    print("\n======================================")
    print("EMPLOYEE RAG ANSWER")
    print("======================================")

    print(f"\nQuestion:")
    print(result["question"])

    print(f"\nAnswer:")
    print(result["answer"])

    print("\n======================================")
