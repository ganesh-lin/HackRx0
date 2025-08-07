from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os
import logging
import time
import asyncio
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

# Import our enhanced modules
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from query_parser import QueryParser
from clause_matcher import ClauseMatcher
from decision_engine import DecisionEngine
from database import DatabaseManager
from llm_manager import LLMManager
import utils

# Configure enhanced logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components
class ApplicationState:
    def __init__(self):
        self.document_processor = None
        self.embedding_manager = None
        self.query_parser = None
        self.clause_matcher = None
        self.decision_engine = None
        self.database_manager = None
        self.llm_manager = None
        self.initialized = False

app_state = ApplicationState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Initializing HackRX Policy Analysis API...")
    
    try:
        # Initialize components with error handling
        app_state.document_processor = DocumentProcessor()
        app_state.embedding_manager = EmbeddingManager(use_pinecone=True)
        app_state.query_parser = QueryParser()
        app_state.clause_matcher = ClauseMatcher()
        app_state.decision_engine = DecisionEngine()
        
        # Try to initialize database manager
        try:
            app_state.database_manager = DatabaseManager()
            logger.info("Database manager initialized successfully")
        except Exception as e:
            logger.warning(f"Database manager failed to initialize: {e}")
            app_state.database_manager = None
        
        # Try to initialize LLM Manager
        try:
            logger.info("Loading LLM model... This may take a few minutes.")
            app_state.llm_manager = LLMManager()
            logger.info("LLM manager initialized successfully")
        except Exception as e:
            logger.warning(f"LLM manager failed to initialize: {e}")
            app_state.llm_manager = None
        
        # Mark as initialized even if some components failed
        app_state.initialized = True
        logger.info("Core components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize core components: {e}")
        app_state.initialized = False
    
    yield
    
    # Shutdown
    logger.info("Shutting down HackRX Policy Analysis API...")

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="HackRX Policy Analysis API",
    description="Advanced LLM-powered document analysis and query system for insurance policies and legal documents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Enhanced Pydantic models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the document to analyze")
    questions: List[str] = Field(..., description="List of questions to answer")
    
    class Config:
        schema_extra = {
            "example": {
                "documents": "https://example.com/policy.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "Does this policy cover maternity expenses?"
                ]
            }
        }

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the processing")
    
    class Config:
        schema_extra = {
            "example": {
                "answers": [
                    "The grace period for premium payment is 30 days.",
                    "Yes, maternity expenses are covered after 24 months of continuous coverage."
                ],
                "metadata": {
                    "processing_time": 15.2,
                    "document_chunks": 45,
                    "confidence_scores": [0.95, 0.89]
                }
            }
        }

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]
    version: str

class AnalyticsResponse(BaseModel):
    total_queries: int
    avg_processing_time: float
    avg_confidence: float
    recent_queries: List[Dict[str, Any]]

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    expected_token = os.getenv("API_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API token not configured"
        )
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    return credentials.credentials

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request details
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}s"
    )
    
    return response

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "HackRX Policy Analysis API v2.0",
        "status": "running",
        "version": "2.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    components_status = {}
    
    # Check component initialization
    components_status["document_processor"] = "ok" if app_state.document_processor else "error"
    components_status["embedding_manager"] = "ok" if app_state.embedding_manager else "error"
    components_status["query_parser"] = "ok" if app_state.query_parser else "error"
    components_status["clause_matcher"] = "ok" if app_state.clause_matcher else "error"
    components_status["decision_engine"] = "ok" if app_state.decision_engine else "error"
    components_status["database_manager"] = "ok" if app_state.database_manager else "error"
    components_status["llm_manager"] = "ok" if app_state.llm_manager else "error"
    
    # Check database connectivity
    try:
        if app_state.database_manager:
            with app_state.database_manager.get_db_connection() as conn:
                components_status["database_connection"] = "ok"
        else:
            components_status["database_connection"] = "disabled"
    except Exception:
        components_status["database_connection"] = "error"
    
    # Check Pinecone connectivity
    try:
        if app_state.embedding_manager and app_state.embedding_manager.use_pinecone:
            # Try to get index stats
            index_stats = app_state.embedding_manager.index.describe_index_stats()
            components_status["pinecone_connection"] = "ok"
        else:
            components_status["pinecone_connection"] = "disabled"
    except Exception:
        components_status["pinecone_connection"] = "error"
    
    # Consider system healthy if core components work, even if optional ones fail
    critical_components = ["document_processor", "embedding_manager", "query_parser", "clause_matcher", "decision_engine"]
    critical_status = all(components_status.get(comp, "error") == "ok" for comp in critical_components)
    overall_status = "healthy" if critical_status else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        components=components_status,
        version="2.0.0"
    )

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_query(request: QueryRequest, token: str = Depends(verify_token)):
    """
    Main endpoint for processing document queries using advanced RAG pipeline.
    
    This endpoint:
    1. Downloads and processes the document
    2. Chunks the content intelligently
    3. Generates embeddings and stores in vector database
    4. For each question:
       - Parses and understands the query
       - Retrieves relevant document chunks
       - Uses LLM to generate accurate answers
    5. Returns structured responses with metadata
    """
    start_time = time.time()
    
    # Check if components are initialized
    if not app_state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is still initializing. Please try again in a few moments."
        )
    
    try:
        logger.info(f"Processing query for document: {request.documents}")
        logger.info(f"Questions count: {len(request.questions)}")
        
        # Step 1: Process Document
        logger.info("Step 1: Processing document...")
        document_text = app_state.document_processor.process_document(request.documents)
        
        if not document_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract text from document"
            )
        
        logger.info(f"Extracted {len(document_text)} characters from document")
        
        # Step 2: Chunk Document
        logger.info("Step 2: Chunking document...")
        chunks = app_state.document_processor.chunk_text(document_text)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create document chunks"
            )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Store Document and Generate Embeddings
        logger.info("Step 3: Storing document and generating embeddings...")
        document_id = f"temp_{int(time.time())}"
        
        # Try to store in database if available
        if app_state.database_manager:
            try:
                document_id = app_state.database_manager.store_document_metadata(
                    url=request.documents,
                    name=utils.extract_filename_from_url(request.documents),
                    file_type="pdf",
                    content=document_text[:1000]  # Store first 1000 chars as sample
                )
                
                # Store chunks in database
                app_state.database_manager.store_document_chunks(document_id, chunks)
                logger.info(f"Stored document in database with ID: {document_id}")
                
            except Exception as e:
                logger.warning(f"Database storage failed: {e}. Continuing with processing...")
                document_id = f"temp_{int(time.time())}"
        else:
            logger.info("Database not available, using temporary document ID")
        
        # Generate and store embeddings in vector database
        namespace = f"doc_{document_id}"
        success = app_state.embedding_manager.store_embeddings(chunks, str(document_id), namespace)
        
        if not success:
            logger.warning("Vector storage failed, falling back to in-memory processing")
        
        # Step 4: Process Each Question
        logger.info("Step 4: Processing questions...")
        answers = []
        metadata = {
            "processing_time": 0,
            "document_chunks": len(chunks),
            "confidence_scores": [],
            "question_details": []
        }
        
        for i, question in enumerate(request.questions):
            question_start_time = time.time()
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
            
            try:
                # Parse the question
                parsed_query = app_state.query_parser.parse_query(question)
                
                # Search for relevant chunks
                if success:  # Use vector database if available
                    relevant_results = app_state.embedding_manager.search_similar(
                        question, namespace, str(document_id)
                    )
                    relevant_chunks = [result["text"] for result in relevant_results]
                else:  # Fallback to clause matching
                    chunk_texts = [chunk["text"] for chunk in chunks]
                    matched_results = app_state.clause_matcher.match_clauses(parsed_query, chunk_texts)
                    relevant_chunks = [result["text"] if isinstance(result, dict) else result 
                                     for result in matched_results]
                
                if not relevant_chunks:
                    # If no relevant chunks found, use top chunks
                    relevant_chunks = [chunk["text"] for chunk in chunks[:3]]
                
                logger.info(f"Found {len(relevant_chunks)} relevant chunks")
                
                # Generate answer using LLM
                if app_state.llm_manager:
                    answer = app_state.llm_manager.answer_question(
                        question, relevant_chunks, "insurance"
                    )
                else:
                    # Fallback to simpler decision engine
                    matched_clause_dicts = [{"text": chunk, "score": 1.0} for chunk in relevant_chunks]
                    answer, rationale = app_state.decision_engine.evaluate_decision(
                        parsed_query, matched_clause_dicts
                    )
                    answer = f"{answer} (Rationale: {rationale})"
                
                answers.append(answer)
                
                # Calculate processing time and confidence
                question_time = time.time() - question_start_time
                confidence = min(len(relevant_chunks) * 0.2, 1.0)  # Simple confidence metric
                
                metadata["confidence_scores"].append(confidence)
                metadata["question_details"].append({
                    "question_index": i,
                    "processing_time": question_time,
                    "relevant_chunks_count": len(relevant_chunks),
                    "confidence": confidence,
                    "query_type": parsed_query.get("query_type", "general")
                })
                
                # Log query for analytics
                try:
                    if app_state.database_manager:
                        query_id = app_state.database_manager.log_query(
                            query_text=question,
                            parsed_query=parsed_query,
                            document_id=document_id if isinstance(document_id, int) else None,
                            response_text=answer,
                            processing_time=question_time,
                            confidence_score=confidence
                        )
                except Exception as e:
                    logger.warning(f"Failed to log query: {e}")
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                answers.append(f"Error processing question: {str(e)}")
                metadata["confidence_scores"].append(0.0)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        metadata["processing_time"] = total_time
        
        logger.info(f"Completed processing in {total_time:.2f} seconds")
        
        # Store performance metrics
        try:
            if app_state.database_manager:
                app_state.database_manager.store_performance_metric(
                    "total_processing_time", total_time,
                    document_id if isinstance(document_id, int) else None
                )
                app_state.database_manager.store_performance_metric(
                    "avg_confidence", 
                    sum(metadata["confidence_scores"]) / len(metadata["confidence_scores"]) if metadata["confidence_scores"] else 0
                )
        except Exception as e:
            logger.warning(f"Failed to store performance metrics: {e}")
        
        return QueryResponse(answers=answers, metadata=metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/v1/analytics", response_model=AnalyticsResponse)
async def get_analytics(token: str = Depends(verify_token), limit: int = 50):
    """Get analytics data about queries and performance."""
    try:
        if not app_state.database_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Analytics not available"
            )
        
        # Get query analytics
        recent_queries = app_state.database_manager.get_query_analytics(limit)
        
        # Calculate aggregated metrics
        total_queries = len(recent_queries)
        avg_processing_time = 0
        avg_confidence = 0
        
        if recent_queries:
            processing_times = [q.get("processing_time", 0) for q in recent_queries if q.get("processing_time")]
            confidence_scores = [q.get("confidence_score", 0) for q in recent_queries if q.get("confidence_score")]
            
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return AnalyticsResponse(
            total_queries=total_queries,
            avg_processing_time=avg_processing_time,
            avg_confidence=avg_confidence,
            recent_queries=recent_queries[:10]  # Return only 10 most recent
        )
        
    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics"
        )

@app.get("/api/v1/performance")
async def get_performance_summary(token: str = Depends(verify_token), days: int = 7):
    """Get performance summary for the specified number of days."""
    try:
        if not app_state.database_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Performance data not available"
            )
        
        summary = app_state.database_manager.get_performance_summary(days)
        return {"performance_summary": summary, "period_days": days}
        
    except Exception as e:
        logger.error(f"Error retrieving performance summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance summary"
        )

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "type": "server_error"}
    )

# Development server
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "False").lower() == "true",
        log_level=log_level.lower()
    )