import json

import psycopg

from experiments.pgvector.document_loader import load_and_split_documents
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


def create_table():
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding VECTOR(384)
            );
        """)

    conn.commit()
    conn.close()

    print("document_chunks table created!")


def insert_chunks():
    print("Loading documents...")

    chunks = load_and_split_documents()

    print(f"Chunks loaded: {len(chunks)}")

    print("Loading embedding model...")

    embeddings = create_embedding_model()

    conn = get_connection()

    with conn.cursor() as cur:

        # Clear existing data so we don't create duplicates
        cur.execute("DELETE FROM document_chunks")

        for index, chunk in enumerate(chunks, start=1):

            vector = embeddings.embed_query(chunk.page_content)

            cur.execute(
                """
                INSERT INTO document_chunks
                (content, metadata, embedding)
                VALUES (%s, %s, %s)
                """,
               (
                chunk.page_content,
                json.dumps(chunk.metadata),
                vector,
               ),
            )

            print(f"Inserted chunk {index}/{len(chunks)}")

    conn.commit()
    conn.close()

    print("All chunks inserted successfully!")


if __name__ == "__main__":

    print("======================================")
    print("Employee RAG Vector Store")
    print("======================================")

    create_table()

    insert_chunks()

    print("======================================")
    print("Vector store setup completed!")
    print("======================================")
