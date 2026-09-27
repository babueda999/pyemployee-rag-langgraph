import psycopg

from experiments.pgvector.embeddings import create_embedding_model


DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "employee_rag"
DB_USER = "raguser"
DB_PASSWORD = "ragpassword"


def get_connection():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def similarity_search(query, top_k=3):

    print(f"\nQuery: {query}")

    # Load embedding model
    embeddings = create_embedding_model()

    # Convert user query into a 384-dimensional vector
    query_vector = embeddings.embed_query(query)

    conn = get_connection()

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (
                query_vector,
                query_vector,
                top_k,
            ),
        )

        results = cur.fetchall()

    conn.close()

    return results


if __name__ == "__main__":

    results = similarity_search(
        "What is the vacation policy for employees?",
        top_k=3,
    )

    print("\n======================================")
    print("TOP MATCHES")
    print("======================================")

    for rank, result in enumerate(results, start=1):

        chunk_id = result[0]
        content = result[1]
        metadata = result[2]
        similarity = result[3]

        print(f"\n--- Result {rank} ---")
        print(f"Chunk ID: {chunk_id}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Metadata: {metadata}")
        print(f"Content:\n{content[:700]}")
