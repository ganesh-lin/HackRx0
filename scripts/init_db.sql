-- Create documents table with all required columns
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    content_hash VARCHAR(64),
    processing_status VARCHAR(50) DEFAULT 'pending',
    is_known BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create document_chunks table
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
);

-- Create clauses table (legacy compatibility)
CREATE TABLE IF NOT EXISTS clauses (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    clause_text TEXT NOT NULL,
    embedding_vector JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create query logs table
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
);

-- Create performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_type VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    document_id INTEGER REFERENCES documents(id),
    query_id INTEGER REFERENCES query_logs(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_clauses_document_id ON clauses(document_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_document_id ON query_logs(document_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);

-- Add content_hash column if it doesn't exist (for existing databases)
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='documents' AND column_name='content_hash') THEN
        ALTER TABLE documents ADD COLUMN content_hash VARCHAR(64);
    END IF;
END $$;