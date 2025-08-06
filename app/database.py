import psycopg2
from psycopg2.extras import RealDictCursor
import os

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=RealDictCursor)

def store_document_metadata(url: str, name: str) -> int:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO documents (url, name) VALUES (%s, %s) RETURNING id", (url, name))
            return cur.fetchone()["id"]

def store_clause(document_id: int, clause_text: str, embedding: list):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO clauses (document_id, clause_text, embedding_vector) VALUES (%s, %s, %s)",
                        (document_id, clause_text, embedding.tolist()))