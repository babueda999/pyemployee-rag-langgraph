from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("documents/Employee_Policies_RAG_Source.pdf")


def load_and_split_documents():
    print(f"Loading PDF: {PDF_PATH}")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"PDF pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


if __name__ == "__main__":
    chunks = load_and_split_documents()

    for index, chunk in enumerate(chunks[:5]):
        print("\n==============================")
        print(f"CHUNK {index + 1}")
        print("==============================")
        print(chunk.page_content[:500])
