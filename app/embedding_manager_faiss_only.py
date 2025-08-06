from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts: list) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, convert_to_numpy=True)

def search_similar_clauses(query: str, embeddings: np.ndarray) -> list:
    """Search for relevant clauses using FAISS."""
    if embeddings.size == 0:
        return []
    
    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(embeddings)
    query_embedding = model.encode([query])[0]
    D, I = faiss_index.search(query_embedding.reshape(1, -1), k=min(5, embeddings.shape[0]))
    
    # Return the indices of similar embeddings
    return I[0].tolist()

def search_pinecone(query: str, embeddings: np.ndarray) -> list:
    """Compatibility function that uses FAISS instead of Pinecone."""
    print("Using FAISS for similarity search")
    return search_similar_clauses(query, embeddings)
