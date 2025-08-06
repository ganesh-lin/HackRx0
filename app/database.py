import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_db_connection():
    """Establish PostgreSQL connection."""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=RealDictCursor)
        logging.info("Connected to PostgreSQL database")
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to database: {str(e)}")
        raise

def store_document_metadata(url: str, name: str) -> int:
    """Store document metadata in PostgreSQL."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO documents (url, name) VALUES (%s, %s) RETURNING id", (url, name))
                doc_id = cur.fetchone()["id"]
                conn.commit()
                logging.info(f"Stored document metadata: {name}, ID: {doc_id}")
                return doc_id
    except Exception as e:
        logging.error(f"Failed to store document metadata: {str(e)}")
        raise

def store_clause(document_id: int, clause_text: str, embedding: list):
    """Store clause and its embedding in PostgreSQL."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO clauses (document_id, clause_text, embedding_vector) VALUES (%s, %s, %s)",
                            (document_id, clause_text, embedding.tolist()))
                conn.commit()
                logging.info(f"Stored clause for document ID: {document_id}")
    except Exception as e:
        logging.error(f"Failed to store clause: {str(e)}")
        raise