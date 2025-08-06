LLM-Powered Query-Retrieval System
This project implements an intelligent query-retrieval system for processing natural language queries against large documents (e.g., insurance policies) using Mistral-7B-Instruct-v0.2 via Hugging Face’s Inference API, FastAPI, Pinecone, FAISS, and PostgreSQL. The system is containerized with Docker for consistent deployment.
Features

Document Processing: Extracts text from PDFs using PyPDF2.
Semantic Search: Uses Sentence-Transformers for embeddings, Pinecone for cloud-based vector search, and FAISS as a fallback.
Query Parsing: Structures queries with Mistral-7B via Hugging Face Inference API.
Clause Matching: Matches clauses using semantic similarity.
Decision Engine: Provides decisions with explainable rationale.
Database: Stores document metadata and clause embeddings in PostgreSQL.
API: FastAPI endpoint /hackrx/run with Bearer token authentication.
Caching: Uses Redis to cache embeddings and query results for low latency.

Tech Stack

Backend: FastAPI
LLM: Mistral-7B-Instruct-v0.2 (Hugging Face Inference API)
Vector Search: Pinecone, FAISS
Database: PostgreSQL (Neon-hosted)
Document Processing: PyPDF2
Environment: Docker, python-dotenv, Redis
Dependencies: See requirements.txt

Setup

Clone the Repository:
git clone <repository-url>
cd llm-query-retrieval


Create .env File:
echo "HF_TOKEN=<your-hf-token>" > .env
echo "API_TOKEN=<your-api-token>" >> .env
echo "PINECONE_API_KEY=<your-pinecone-key>" >> .env
echo "DATABASE_URL=<your-postgres-url>" >> .env


Initialize Database:
psql "$DATABASE_URL" -f scripts/init_db.sql


Build and Run with Docker:
docker-compose up --build


Test the API:
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
-H "Authorization: Bearer $API_TOKEN" \
-H "Content-Type: application/json" \
-d '{"documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...", "questions": ["Does this policy cover knee surgery, and what are the conditions?"]}'



Deployment

Push to GitHub.
Deploy on Render:
Create a web service, link to the repository.
Set environment variables in Render’s dashboard.
Deploy and obtain the public URL (e.g., https://your-app.onrender.com).


Submit the webhook URL: https://your-app.onrender.com/api/v1/hackrx/run.

Notes

Latency: Optimized with Redis caching and FAISS fallback (<30s response time).
Accuracy: Uses Mistral-7B for precise query parsing and Pinecone for semantic search.
Explainability: Responses include rationale referencing matched clauses.
Reusability: Modular code structure for easy extension.
Token Efficiency: Caches embeddings and query results to minimize API calls.

Troubleshooting

Hugging Face API Limits: Check https://huggingface.co/settings/tokens for usage limits.
Pinecone Issues: Ensure the index developer-quickstart-py is created.
Database Errors: Verify the PostgreSQL URL and schema.
Logs: Check app.log for debugging.
