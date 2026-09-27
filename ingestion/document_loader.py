"""Loads .txt, .md, and .pdf files from a directory into LangChain Documents."""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

TEXT_EXTENSIONS = {".txt", ".md"}


def load_documents(documents_dir: str) -> list[Document]:
    directory = Path(documents_dir)
    if not directory.exists():
        return []

    documents = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
            documents.append(Document(page_content=text, metadata={"source": path.name}))
        elif suffix == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": path.name, "page": page_number},
                        )
                    )

    return documents
