# HackRX Policy Analysis API - Usage Guide

## 🚀 Quick Start

### 1. Start the Server
```bash
cd d:\webcompetation\hackrx\app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Test the API

#### Option A: Using Python (Recommended)
```bash
python test_api.py
```

#### Option B: Using PowerShell
```powershell
.\test_api.ps1
```

#### Option C: Using Windows Batch + curl
```batch
test_curl.bat
```

#### Option D: Manual PowerShell Command
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/hackrx/run" `
  -Method POST `
  -Headers @{
    "Authorization"="Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1"
    "Content-Type"="application/json"
  } `
  -Body (Get-Content "test_request.json" -Raw)
```

## 📋 API Endpoints

### Health Check
- `GET /` - Root endpoint
- `GET /health` - Health check

### Main API
- `POST /api/v1/hackrx/run` - Process policy documents and answer questions

## 📝 Request Format

```json
{
  "documents": "https://your-document-url.pdf",
  "questions": [
    "Does this policy cover knee surgery?",
    "What are the age restrictions?"
  ]
}
```

## 🔧 Troubleshooting

### Common Issues:

1. **422 Unprocessable Entity**: JSON formatting issue
   - Solution: Use the provided test scripts instead of manual curl

2. **404 Not Found**: Server not running or wrong endpoint
   - Solution: Make sure server is running and endpoint is `/api/v1/hackrx/run`

3. **401 Unauthorized**: Wrong API token
   - Solution: Check the Bearer token in your request

4. **Windows curl issues**: Escaping problems
   - Solution: Use `test_curl.bat` or PowerShell methods

## ✅ Success Response Example

```json
{
  "answers": [
    "Yes, surgery appears to be covered. (Rationale: Found relevant coverage clause: ...)"
  ]
}
```

## 🔑 Features

- ✅ PDF document processing
- ✅ Natural language query parsing  
- ✅ Semantic search using FAISS
- ✅ Intelligent clause matching
- ✅ Decision evaluation with rationale
- ✅ Robust error handling
- ✅ Windows compatibility

## 🛠️ Architecture

1. **Document Processing**: Downloads and extracts text from PDF
2. **Embedding Generation**: Creates vector embeddings using sentence-transformers
3. **Query Parsing**: Extracts structured information from natural language
4. **Similarity Search**: Uses FAISS for finding relevant clauses
5. **Decision Engine**: Evaluates coverage and provides rationale
