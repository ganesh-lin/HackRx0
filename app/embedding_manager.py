from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import faiss
import numpy as np
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "developer-quickstart-py"
index = pc.Index(index_name)

def generate_embeddings(texts: list) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, convert_to_numpy=True)

def search_pinecone(query: str, embeddings: np.ndarray) -> list:
    """Search Pinecone for relevant clauses."""
    query_embedding = model.encode([query])[0]
    results = index.query(vector=query_embedding.tolist(), top_k=5)
    return [match["metadata"]["text"] for match in results["matches"]]

def search_faiss(query: str, embeddings: np.ndarray) -> list:
    """Fallback to FAISS if Pinecone is unavailable."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    query_embedding = model.encode([query])[0]
    D, I = index.search(query_embedding.reshape(1, -1), k=5)
    return [I[0].tolist()]