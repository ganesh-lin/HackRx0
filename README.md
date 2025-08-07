# HackRX Policy Analysis API

## 🚀 Advanced LLM-Powered Document Analysis System

A sophisticated RAG (Retrieval Augmented Generation) system that processes insurance policies, legal documents, and contracts using state-of-the-art language models and vector search technology.

### 🏗️ Architecture

```
          ┌─────────────┐
          │  User Query │
          └────┬────────┘
               ↓
        ┌──────────────┐
        │ PDF Extractor│ ◄───── PyPDF2, docx, email
        └────┬─────────┘
             ↓
    ┌────────────────────┐
    │ Chunk + Embed Docs │ ◄──── SentenceTransformers
    └────┬───────────────┘
         ↓
  ┌────────────────────┐
  │ Pinecone Vector DB │ ◄──── Pinecone Cloud
  └────┬───────────────┘
       ↓
┌────────────────────────┐
│ Prompt + Context Query │ ◄──── Mistral 7B Instruct
└────────┬───────────────┘
         ↓
  ┌────────────────────┐
  │ Structured JSON Out│ ◄──── FastAPI Endpoint
  └────────────────────┘
```

## 🎯 Features

- **Multi-Document Support**: PDF, DOCX, email processing
- **Advanced RAG Pipeline**: Intelligent chunking and semantic search
- **Mistral 7B Integration**: State-of-the-art language model for accurate responses
- **Vector Database**: Pinecone for fast semantic similarity search
- **PostgreSQL Backend**: Robust data persistence and analytics
- **Smart Query Processing**: Advanced NLP for query understanding
- **Comprehensive API**: RESTful endpoints with OpenAPI documentation
- **Real-time Analytics**: Performance monitoring and query tracking
- **Docker Support**: Easy deployment and scaling

## 📋 Requirements

### System Requirements
- Python 3.9+
- 8GB+ RAM (for LLM model)
- PostgreSQL database
- Internet connection for Pinecone and Hugging Face

### API Keys Required
- **Hugging Face Token**: For Mistral 7B model access
- **Pinecone API Key**: For vector database
- **PostgreSQL Connection**: For data persistence

## 🛠️ Installation

### Option 1: Quick Start (Docker)

1. **Clone and configure**
   ```bash
   git clone <repository-url>
   cd hackrx
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Test the API**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
     -H "Authorization: Bearer your_api_token" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://example.com/policy.pdf",
       "questions": ["What is covered under this policy?"]
     }'
   ```

### Option 2: Local Development

1. **Setup environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the application**
   ```bash
   # Linux/Mac
   ./start.sh
   
   # Windows
   start.bat
   ```

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# API Configuration
API_TOKEN=81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1

# Hugging Face Configuration
HF_TOKEN=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

# Pinecone Configuration
PINECONE_API_KEY=pcsk_313Agm_Rn7Xp4Pu7EGgT2y6WvZ9aPUdBFPuWZBzeoX4AViugzFts73j9DeX3oD8KfBFzQZ
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=hackrx-policy-index

# PostgreSQL Configuration
DATABASE_URL=postgresql://neondb_owner:npg_CVsbTxlS9c5Z@ep-orange-night-a1gigshf-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_TOKENS=2048
TEMPERATURE=0.1
TOP_K=5
```

## 📖 API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
All API endpoints require Bearer token authentication:
```
Authorization: Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1
```

### Main Endpoint

#### POST `/api/v1/hackrx/run`

Process documents and answer questions using advanced RAG pipeline.

**Example Request:**
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
```

**Example Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
    ],
    "metadata": {
        "processing_time": 15.2,
        "document_chunks": 45,
        "confidence_scores": [0.95, 0.89, 0.92]
    }
}
```

### Additional Endpoints

- **GET** `/health` - Comprehensive health check
- **GET** `/` - API information
- **GET** `/api/v1/analytics` - Query analytics
- **GET** `/docs` - Interactive API documentation (Swagger UI)

## 🧪 Testing

### Automated Testing
```bash
cd test
python comprehensive_test_new.py
```

### Manual Testing with curl
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1" \
  -H "Content-Type: application/json" \
  -d @test/test_request.json
```

### PowerShell Testing
```powershell
.\test\test_api.ps1
```

## 🔧 System Components

### 1. Enhanced Document Processing
- **Multi-format support**: PDF, DOCX, email files
- **Intelligent chunking**: Semantic-aware text segmentation
- **Content validation**: File type detection and validation
- **Preprocessing**: Text cleaning and normalization

### 2. Advanced Embedding Management
- **SentenceTransformers**: High-quality embeddings with all-MiniLM-L6-v2
- **Pinecone Integration**: Cloud-based vector database with namespacing
- **FAISS Fallback**: Local vector search for development
- **Batch Processing**: Efficient embedding generation and storage

### 3. Sophisticated Query Processing
- **NLP Analysis**: Entity extraction and intent classification
- **Context Understanding**: Temporal and medical context analysis
- **Multi-criteria Matching**: Semantic, keyword, and contextual matching
- **Confidence Scoring**: Reliability assessment for each match

### 4. LLM Integration (Mistral 7B)
- **Local Model Deployment**: Full model deployment for maximum control
- **Optimized Prompting**: Domain-specific prompt engineering
- **Token Management**: Efficient context window utilization
- **Response Generation**: Structured, explainable answers

### 5. Robust Database Layer
- **PostgreSQL Schema**: Comprehensive data modeling
- **Analytics Storage**: Query logs and performance metrics
- **Connection Management**: Pooling and error handling
- **Data Persistence**: Document metadata and chunk storage

## 📊 Performance Features

### Optimization Strategies
- **Model Quantization**: Reduced memory usage for CPU deployment
- **Smart Caching**: Embedding and response caching
- **Parallel Processing**: Concurrent question processing
- **Batch Operations**: Efficient database operations

### Monitoring & Analytics
- **Real-time Metrics**: Processing times and accuracy scores
- **Query Analytics**: Pattern analysis and trend tracking
- **Performance Dashboard**: System health monitoring
- **Error Tracking**: Comprehensive logging and alerting

## 🚀 Deployment Options

### 1. Render Deployment (Recommended)
```bash
# 1. Push to GitHub
git push origin main

# 2. Connect to Render
# 3. Set environment variables
# 4. Deploy automatically
```

### 2. Railway Deployment
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway project new
railway up
```

### 3. Local Docker
```bash
docker-compose up -d
```

## 💯 Evaluation Criteria Compliance

### ✅ Accuracy
- **Advanced Semantic Search**: Vector similarity with Pinecone
- **Context-Aware Matching**: Multi-level relevance scoring
- **LLM-Powered Responses**: Mistral 7B for accurate answer generation
- **Explainable Results**: Detailed rationale for each decision

### ✅ Token Efficiency
- **Optimized Prompting**: Minimal token usage with maximum context
- **Smart Chunking**: Efficient document segmentation
- **Context Management**: Optimal context window utilization
- **Caching Strategy**: Reduced redundant API calls

### ✅ Latency
- **Parallel Processing**: Concurrent question handling
- **Efficient Indexing**: Fast vector search with Pinecone
- **Local Model**: No external API dependencies for LLM
- **Optimized Pipeline**: Streamlined processing workflow

### ✅ Reusability
- **Modular Architecture**: Independent, configurable components
- **Standard Interfaces**: RESTful API with OpenAPI documentation
- **Flexible Configuration**: Environment-based settings
- **Extensible Design**: Easy to add new document types and features

### ✅ Explainability
- **Clause Traceability**: Direct mapping to source document sections
- **Confidence Scoring**: Reliability indicators for each answer
- **Decision Rationale**: Clear explanation of reasoning
- **Metadata Provision**: Processing details and performance metrics

## 🆘 Troubleshooting

### Common Issues

1. **Model Loading Fails**
   ```bash
   # Ensure sufficient RAM and valid HF token
   export HF_TOKEN=your_token_here
   ```

2. **Pinecone Connection Issues**
   ```bash
   # Verify API key and region
   curl -H "Api-Key: your_key" https://api.pinecone.io/indexes
   ```

3. **Database Connection**
   ```bash
   # Test PostgreSQL connection
   psql "your_database_url"
   ```

### Health Check
```bash
curl http://localhost:8000/health
```

### Logs
```bash
# Check application logs
tail -f app/app.log

# Check Docker logs
docker-compose logs -f
```

## 📞 Support

For issues and questions:
1. Check the `/docs` endpoint for API documentation
2. Review the health check endpoint: `/health`
3. Examine logs in the `logs/` directory
4. Open an issue in the repository

---

**🏆 Built for HackRX Competition - Advanced RAG System with Mistral 7B**
