import psycopg

DATABASE_URL = "postgresql://raguser:ragpassword@localhost:5433/employee_rag"

connection = psycopg.connect(DATABASE_URL)

with connection.cursor() as cursor:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vector_test (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(3)
        );
    """)

    records = [
        ("Employee vacation policy", "[0.10,0.20,0.30]"),
        ("Employee sick leave policy", "[0.15,0.22,0.28]"),
        ("Remote work policy", "[0.42,0.11,0.19]"),
        ("Parental leave policy", "[0.33,0.27,0.44]"),
        ("Expense reimbursement policy", "[0.51,0.09,0.37]"),
        ("Code of conduct policy", "[0.08,0.61,0.23]"),
        ("Overtime and compensation policy", "[0.29,0.35,0.12]"),
        ("Health and safety policy", "[0.44,0.18,0.55]"),
        ("Equipment and IT usage policy", "[0.19,0.48,0.31]"),
        ("Anti-harassment policy", "[0.37,0.26,0.14]"),
        ("Performance review policy", "[0.22,0.39,0.46]"),
    ]

    cursor.executemany(
        """
        INSERT INTO vector_test (content, embedding)
        VALUES (%s, %s);
        """,
        records,
    )

    cursor.execute("""
        SELECT id, content, embedding
        FROM vector_test;
    """)

    for row in cursor.fetchall():
        print(row)

connection.commit()
connection.close()

print("Python → PostgreSQL → pgvector test successful!")