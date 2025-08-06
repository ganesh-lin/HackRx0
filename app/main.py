from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from document_processor import extract_text_from_pdf
from embedding_manager import generate_embeddings, search_pinecone
from query_parser import parse_query
from clause_matcher import match_clauses
from decision_engine import evaluate_decision
from database import store_document_metadata, store_clause
from utils import download_pdf
import os

app = FastAPI()
security = HTTPBearer()

# Pydantic models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Step 1: Download and process document
        pdf_path = download_pdf(request.documents)
        document_text = extract_text_from_pdf(pdf_path)
        
        # Step 2: Store document metadata
        document_id = store_document_metadata(request.documents, "policy.pdf")
        
        # Step 3: Generate embeddings and store clauses
        clauses = document_text.split("\n\n")  # Simple clause splitting
        embeddings = generate_embeddings(clauses)
        for clause, embedding in zip(clauses, embeddings):
            store_clause(document_id, clause, embedding)
        
        # Step 4: Process queries and generate responses
        answers = []
        for question in request.questions:
            # Parse query
            structured_query = parse_query(question)
            
            # Search relevant clauses
            relevant_clauses = search_pinecone(structured_query, embeddings)
            
            # Match clauses
            matched_clauses = match_clauses(structured_query, relevant_clauses)
            
            # Evaluate decision
            answer, rationale = evaluate_decision(structured_query, matched_clauses)
            answers.append(f"{answer} (Rationale: {rationale})")
        
        return QueryResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))