"""A minimal numpy-backed vector store with cosine similarity search.

Avoids a chromadb/faiss dependency so the project installs cleanly on any
Python version. Fine for the small document sets this sample targets; swap
in a real vector database for production scale.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class SimpleVectorStore:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self._vectors: np.ndarray | None = None
        self._documents: list[Document] = []

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        vectors = np.array(
            self.embeddings.embed_documents([doc.page_content for doc in documents]),
            dtype=np.float32,
        )
        self._documents.extend(documents)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        if self._vectors is None or len(self._documents) == 0:
            return []
        query_vector = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        scores = self._vectors @ query_vector
        top_k = min(k, len(self._documents))
        top_indices = np.argsort(-scores)[:top_k]
        return [self._documents[i] for i in top_indices]

    def persist(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(path / "vectors.npy", self._vectors)
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"count": len(self._documents)}, f)

    @classmethod
    def load(cls, directory: str | Path, embeddings: Embeddings) -> "SimpleVectorStore":
        path = Path(directory)
        store = cls(embeddings)
        vectors_path = path / "vectors.npy"
        documents_path = path / "documents.pkl"
        if vectors_path.exists() and documents_path.exists():
            store._vectors = np.load(vectors_path)
            with open(documents_path, "rb") as f:
                store._documents = pickle.load(f)
        return store

    @staticmethod
    def exists(directory: str | Path) -> bool:
        path = Path(directory)
        return (path / "vectors.npy").exists() and (path / "documents.pkl").exists()

    def __len__(self) -> int:
        return len(self._documents)


def build_vector_store(
    documents: list[Document], embeddings: Embeddings, persist_directory: str | Path
) -> SimpleVectorStore:
    store = SimpleVectorStore(embeddings)
    store.add_documents(documents)
    store.persist(persist_directory)
    return store


def load_vector_store(persist_directory: str | Path, embeddings: Embeddings) -> SimpleVectorStore:
    return SimpleVectorStore.load(persist_directory, embeddings)
