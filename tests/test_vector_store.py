import tempfile
from pathlib import Path

from langchain_core.documents import Document

from ingestion.embeddings import LocalHashingEmbeddings
from ingestion.vector_store import SimpleVectorStore, load_vector_store


def test_similarity_search_returns_most_relevant_document():
    embeddings = LocalHashingEmbeddings(dimensions=64)
    store = SimpleVectorStore(embeddings)
    store.add_documents(
        [
            Document(page_content="paid time off vacation policy", metadata={"source": "pto.txt"}),
            Document(page_content="quarterly sales revenue figures", metadata={"source": "sales.txt"}),
        ]
    )

    results = store.similarity_search("vacation days", k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "pto.txt"


def test_persist_and_load_roundtrip():
    embeddings = LocalHashingEmbeddings(dimensions=32)
    store = SimpleVectorStore(embeddings)
    store.add_documents([Document(page_content="hello world", metadata={"source": "a.txt"})])

    with tempfile.TemporaryDirectory() as tmp_dir:
        persist_dir = Path(tmp_dir) / "store"
        store.persist(persist_dir)

        assert SimpleVectorStore.exists(persist_dir)
        loaded = load_vector_store(persist_dir, embeddings)

    assert len(loaded) == 1
