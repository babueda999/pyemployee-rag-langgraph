from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model():
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded!")

    return embeddings


if __name__ == "__main__":
    embeddings = create_embedding_model()

    text = "What is the vacation policy for employees?"

    vector = embeddings.embed_query(text)

    print(f"Text: {text}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 10 values: {vector[:10]}")
