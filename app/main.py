from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from document_processor import extract_text_from_pdf
from embedding_manager import generate_embeddings, search_pinecone
from query_parser import parse_query
from clause_matcher import match_clauses
from decision_engine import evaluate_decision
from database import store_document_metadata, store_clause
from utils import download_pdf
import logging

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

@app.get("/")
async def root():
    return {"message": "HackRX Policy Analysis API", "status": "running", "version": "1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2025-08-06"}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_query(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Log request
        logging.info(f"Processing query for document: {request.documents}")
        
        # Step 1: Download and process document
        pdf_path = download_pdf(request.documents)
        document_text = extract_text_from_pdf(pdf_path)
        if not document_text:
            logging.error("Failed to extract text from PDF")
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
        logging.info("Document text extracted successfully")
        
        # Step 2: Store document metadata (with error handling)
        try:
            document_id = store_document_metadata(request.documents, "policy.pdf")
        except Exception as e:
            logging.warning(f"Failed to store document metadata: {e}")
            document_id = 1  # Use a default ID if database fails
        
        # Step 3: Generate embeddings and store clauses
        clauses = document_text.split("\n\n")  # Simple clause splitting
        # Filter out empty clauses
        clauses = [clause.strip() for clause in clauses if clause.strip()]
        
        if not clauses:
            raise HTTPException(status_code=400, detail="No clauses found in document")
            
        embeddings = generate_embeddings(clauses)
        
        # Try to store clauses, but continue if it fails
        try:
            for clause, embedding in zip(clauses, embeddings):
                store_clause(document_id, clause, embedding)
        except Exception as e:
            logging.warning(f"Failed to store clauses: {e}")
        
        # Step 4: Process queries and generate responses
        answers = []
        for question in request.questions:
            logging.info(f"Processing question: {question}")
            
            try:
                # Parse query
                structured_query = parse_query(question)
                if not structured_query.get("procedure") and not structured_query.get("raw_query"):
                    answers.append("Query is too vague to process. (Rationale: No specific procedure identified.)")
                    continue
                
                # Search relevant clauses
                relevant_indices = search_pinecone(structured_query, embeddings)
                
                # Get the actual clauses from indices (for FAISS) or use directly (for Pinecone)
                if isinstance(relevant_indices, list) and len(relevant_indices) > 0:
                    if isinstance(relevant_indices[0], int):
                        # FAISS returns indices
                        relevant_clauses = [clauses[i] for i in relevant_indices if 0 <= i < len(clauses)]
                    else:
                        # Pinecone returns actual text
                        relevant_clauses = relevant_indices
                else:
                    relevant_clauses = []
                
                # Match clauses
                matched_clauses = match_clauses(structured_query, relevant_clauses)
                
                # Evaluate decision
                answer, rationale = evaluate_decision(structured_query, matched_clauses)
                answers.append(f"{answer} (Rationale: {rationale})")
                
            except Exception as e:
                logging.error(f"Error processing question '{question}': {e}")
                answers.append(f"Error processing question: {str(e)}")
        
        return QueryResponse(answers=answers)
    except Exception as e:
        logging.error(f"General error in process_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))