CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    url VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE clauses (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    clause_text TEXT NOT NULL,
    embedding_vector FLOAT[] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);