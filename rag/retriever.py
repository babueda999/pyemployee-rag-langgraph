"""Wraps the vector store behind a simple retrieval interface."""

from langchain_core.documents import Document

from ingestion.vector_store import SimpleVectorStore


class Retriever:
    def __init__(self, vector_store: SimpleVectorStore, k: int = 4):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, question: str) -> list[Document]:
        return self.vector_store.similarity_search(question, k=self.k)


def get_retriever(vector_store: SimpleVectorStore, k: int = 4) -> Retriever:
    return Retriever(vector_store, k=k)
