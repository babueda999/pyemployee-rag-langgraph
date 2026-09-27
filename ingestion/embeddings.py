"""Embedding provider selection.

Defaults to a local, dependency-free hashing embedder so the project runs
offline with no API key. Set EMBEDDINGS_PROVIDER=openai or =voyage (plus the
matching API key) for real embeddings once those packages are installed.
"""

import hashlib
import os

import numpy as np
from langchain_core.embeddings import Embeddings

DEFAULT_DIMENSIONS = 384


class LocalHashingEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embedder.

    Not semantically strong, but requires no model download or API key,
    which keeps the sample runnable anywhere. Swap in a real provider for
    production-quality retrieval.
    """

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings() -> Embeddings:
    provider = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    if provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(model="voyage-3")

    if provider != "local":
        raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {provider}")

    return LocalHashingEmbeddings()
