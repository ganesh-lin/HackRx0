import psycopg2
from psycopg2.extras import RealDictCursor, Json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import hashlib
from contextlib import contextmanager

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.connection_pool = None
        self._initialize_schema()
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _initialize_schema(self):
        """Initialize database schema if not exists."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Create documents table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS documents (
                            id SERIAL PRIMARY KEY,
                            url TEXT NOT NULL,
                            name VARCHAR(255) NOT NULL,
                            file_type VARCHAR(50),
                            file_size BIGINT,
                            content_hash VARCHAR(64),
                            processing_status VARCHAR(50) DEFAULT 'pending',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB
                        )
                    """)
                    
                    # Create chunks table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS document_chunks (
                            id SERIAL PRIMARY KEY,
                            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                            chunk_index INTEGER NOT NULL,
                            chunk_text TEXT NOT NULL,
                            chunk_tokens INTEGER,
                            start_position INTEGER,
                            end_position INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB
                        )
                    """)
                    
                    # Create clauses table (legacy compatibility)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS clauses (
                            id SERIAL PRIMARY KEY,
                            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                            clause_text TEXT NOT NULL,
                            embedding_vector JSON,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create query logs table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS query_logs (
                            id SERIAL PRIMARY KEY,
                            query_text TEXT NOT NULL,
                            parsed_query JSONB,
                            document_id INTEGER REFERENCES documents(id),
                            response_text TEXT,
                            processing_time FLOAT,
                            confidence_score FLOAT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB
                        )
                    """)
                    
                    # Create performance metrics table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS performance_metrics (
                            id SERIAL PRIMARY KEY,
                            metric_type VARCHAR(100) NOT NULL,
                            metric_value FLOAT NOT NULL,
                            document_id INTEGER REFERENCES documents(id),
                            query_id INTEGER REFERENCES query_logs(id),
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB
                        )
                    """)
                    
                    # Create indexes for better performance
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_clauses_document_id ON clauses(document_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_query_logs_document_id ON query_logs(document_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type)")
                    
                    conn.commit()
                    logging.info("Database schema initialized successfully")
                    
        except Exception as e:
            logging.error(f"Failed to initialize database schema: {str(e)}")
            raise
    
    def store_document_metadata(self, url: str, name: str, file_type: str = None, 
                              file_size: int = None, content: str = None,
                              metadata: Dict[str, Any] = None) -> int:
        """Store document metadata with enhanced information."""
        try:
            # Generate content hash if content is provided
            content_hash = None
            if content:
                content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Check if document already exists
                    if content_hash:
                        cur.execute("SELECT id FROM documents WHERE content_hash = %s", (content_hash,))
                        existing = cur.fetchone()
                        if existing:
                            logging.info(f"Document already exists with ID: {existing['id']}")
                            return existing['id']
                    
                    # Insert new document
                    cur.execute("""
                        INSERT INTO documents (url, name, file_type, file_size, content_hash, 
                                             processing_status, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """, (url, name, file_type, file_size, content_hash, 'processing', Json(metadata or {})))
                    
                    doc_id = cur.fetchone()['id']
                    conn.commit()
                    
                    logging.info(f"Stored document metadata: {name}, ID: {doc_id}")
                    return doc_id
                    
        except Exception as e:
            logging.error(f"Failed to store document metadata: {str(e)}")
            raise
    
    def store_document_chunks(self, document_id: int, chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks in the database."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Clear existing chunks for this document
                    cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                    
                    # Insert new chunks
                    for chunk in chunks:
                        cur.execute("""
                            INSERT INTO document_chunks 
                            (document_id, chunk_index, chunk_text, chunk_tokens, 
                             start_position, end_position, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            document_id,
                            chunk.get('id', 0),
                            chunk.get('text', ''),
                            chunk.get('tokens', 0),
                            chunk.get('start_sentence', 0),
                            chunk.get('end_sentence', 0),
                            Json(chunk.get('metadata', {}))
                        ))
                    
                    # Update document status
                    cur.execute("""
                        UPDATE documents 
                        SET processing_status = 'completed', updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (document_id,))
                    
                    conn.commit()
                    logging.info(f"Stored {len(chunks)} chunks for document ID: {document_id}")
                    return True
                    
        except Exception as e:
            logging.error(f"Failed to store document chunks: {str(e)}")
            return False
    
    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Retrieve document chunks from the database."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT chunk_index, chunk_text, chunk_tokens, 
                               start_position, end_position, metadata
                        FROM document_chunks 
                        WHERE document_id = %s 
                        ORDER BY chunk_index
                    """, (document_id,))
                    
                    chunks = []
                    for row in cur.fetchall():
                        chunks.append({
                            'id': row['chunk_index'],
                            'text': row['chunk_text'],
                            'tokens': row['chunk_tokens'],
                            'start_sentence': row['start_position'],
                            'end_sentence': row['end_position'],
                            'metadata': row['metadata'] or {}
                        })
                    
                    logging.info(f"Retrieved {len(chunks)} chunks for document ID: {document_id}")
                    return chunks
                    
        except Exception as e:
            logging.error(f"Failed to retrieve document chunks: {str(e)}")
            return []
    
    def log_query(self, query_text: str, parsed_query: Dict[str, Any], 
                  document_id: int = None, response_text: str = None,
                  processing_time: float = None, confidence_score: float = None,
                  metadata: Dict[str, Any] = None) -> int:
        """Log query and response for analytics."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO query_logs 
                        (query_text, parsed_query, document_id, response_text, 
                         processing_time, confidence_score, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """, (
                        query_text,
                        Json(parsed_query),
                        document_id,
                        response_text,
                        processing_time,
                        confidence_score,
                        Json(metadata or {})
                    ))
                    
                    query_id = cur.fetchone()['id']
                    conn.commit()
                    
                    logging.info(f"Logged query with ID: {query_id}")
                    return query_id
                    
        except Exception as e:
            logging.error(f"Failed to log query: {str(e)}")
            return None
    
    def store_performance_metric(self, metric_type: str, metric_value: float,
                               document_id: int = None, query_id: int = None,
                               metadata: Dict[str, Any] = None) -> bool:
        """Store performance metrics."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO performance_metrics 
                        (metric_type, metric_value, document_id, query_id, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (metric_type, metric_value, document_id, query_id, Json(metadata or {})))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logging.error(f"Failed to store performance metric: {str(e)}")
            return False
    
    def get_document_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get document information by URL."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, url, name, file_type, processing_status, 
                               created_at, updated_at, metadata
                        FROM documents 
                        WHERE url = %s
                    """, (url,))
                    
                    result = cur.fetchone()
                    if result:
                        return dict(result)
                    return None
                    
        except Exception as e:
            logging.error(f"Failed to get document by URL: {str(e)}")
            return None
    
    def get_query_analytics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get query analytics data."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT query_text, processing_time, confidence_score, 
                               created_at, metadata
                        FROM query_logs 
                        ORDER BY created_at DESC 
                        LIMIT %s
                    """, (limit,))
                    
                    return [dict(row) for row in cur.fetchall()]
                    
        except Exception as e:
            logging.error(f"Failed to get query analytics: {str(e)}")
            return []
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            metric_type,
                            COUNT(*) as count,
                            AVG(metric_value) as avg_value,
                            MIN(metric_value) as min_value,
                            MAX(metric_value) as max_value
                        FROM performance_metrics 
                        WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
                        GROUP BY metric_type
                    """, (days,))
                    
                    summary = {}
                    for row in cur.fetchall():
                        summary[row['metric_type']] = {
                            'count': row['count'],
                            'average': float(row['avg_value']),
                            'minimum': float(row['min_value']),
                            'maximum': float(row['max_value'])
                        }
                    
                    return summary
                    
        except Exception as e:
            logging.error(f"Failed to get performance summary: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days: int = 30) -> bool:
        """Clean up old data to maintain performance."""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Clean old query logs
                    cur.execute("""
                        DELETE FROM query_logs 
                        WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
                    """, (days,))
                    
                    # Clean old performance metrics
                    cur.execute("""
                        DELETE FROM performance_metrics 
                        WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '%s days'
                    """, (days,))
                    
                    conn.commit()
                    logging.info(f"Cleaned up data older than {days} days")
                    return True
                    
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {str(e)}")
            return False

# Initialize global database manager
try:
    db_manager = DatabaseManager()
    logging.info("Database manager initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize database manager: {e}")
    db_manager = None

# Backward compatibility functions
def get_db_connection():
    """Backward compatibility function."""
    if db_manager:
        return db_manager.get_db_connection()
    else:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=RealDictCursor)
        return conn

def store_document_metadata(url: str, name: str) -> int:
    """Backward compatibility function."""
    if db_manager:
        return db_manager.store_document_metadata(url, name)
    else:
        # Fallback implementation
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO documents (url, name) VALUES (%s, %s) RETURNING id", (url, name))
                    doc_id = cur.fetchone()["id"]
                    conn.commit()
                    return doc_id
        except Exception as e:
            logging.error(f"Failed to store document metadata: {str(e)}")
            raise

def store_clause(document_id: int, clause_text: str, embedding: list):
    """Backward compatibility function."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO clauses (document_id, clause_text, embedding_vector) VALUES (%s, %s, %s)",
                            (document_id, clause_text, embedding.tolist() if hasattr(embedding, 'tolist') else embedding))
                conn.commit()
                logging.info(f"Stored clause for document ID: {document_id}")
    except Exception as e:
        logging.error(f"Failed to store clause: {str(e)}")
        raise