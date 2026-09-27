from langchain_core.documents import Document

from ingestion.chunker import chunk_documents


def test_chunk_documents_splits_long_text():
    long_text = "word " * 500
    doc = Document(page_content=long_text, metadata={"source": "test.txt"})

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "test.txt" for chunk in chunks)


def test_chunk_documents_keeps_short_text_whole():
    doc = Document(page_content="short text", metadata={"source": "test.txt"})

    chunks = chunk_documents([doc], chunk_size=800, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"
