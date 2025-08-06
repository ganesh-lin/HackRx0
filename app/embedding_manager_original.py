from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone with error handling
try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "developer-quickstart-py"
    
    # Check if index exists before trying to use it
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        raise ValueError(f"Index '{index_name}' not found. Available indexes: {existing_indexes}")
    
    index = pc.Index(index_name)
    pinecone_available = True
    print(f"Successfully connected to Pinecone index: {index_name}")
except Exception as e:
    print(f"Warning: Pinecone initialization failed: {e}")
    print("Falling back to FAISS for similarity search")
    pc = None
    index = None
    pinecone_available = False

def generate_embeddings(texts: list) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, convert_to_numpy=True)

def search_pinecone(query: str, embeddings: np.ndarray) -> list:
    """Search Pinecone for relevant clauses, fallback to FAISS if unavailable."""
    if pinecone_available and index is not None:
        try:
            query_embedding = model.encode([query])[0]
            results = index.query(vector=query_embedding.tolist(), top_k=5)
            return [match["metadata"]["text"] for match in results["matches"]]
        except Exception as e:
            print(f"Pinecone search failed: {e}, falling back to FAISS")
            return search_faiss(query, embeddings)
    else:
        print("Pinecone not available, using FAISS")
        return search_faiss(query, embeddings)

def search_faiss(query: str, embeddings: np.ndarray) -> list:
    """Fallback to FAISS if Pinecone is unavailable."""
    if embeddings.size == 0:
        return []
    
    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(embeddings)
    query_embedding = model.encode([query])[0]
    D, I = faiss_index.search(query_embedding.reshape(1, -1), k=min(5, embeddings.shape[0]))
    
    # Return the indices of similar embeddings
    return I[0].tolist()