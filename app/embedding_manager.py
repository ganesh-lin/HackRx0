from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import logging
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import json
import time

load_dotenv()

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EmbeddingManager:
    def __init__(self, use_pinecone: bool = True):
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.use_pinecone = use_pinecone
        self.top_k = int(os.getenv("TOP_K", 8))  # Increased from 5 to 8 for better coverage
        
        # Initialize Pinecone if enabled
        if self.use_pinecone:
            try:
                self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                self.index_name = os.getenv("PINECONE_INDEX_NAME", "hackrx-policy-index")
                self._setup_pinecone_index()
                self.index = self.pinecone_client.Index(self.index_name)
                logging.info("Pinecone initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize Pinecone: {e}. Falling back to FAISS.")
                self.use_pinecone = False
                self._setup_faiss()
        else:
            self._setup_faiss()
    
    def _setup_pinecone_index(self):
        """Setup Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
                logging.info(f"Created Pinecone index: {self.index_name}")
            else:
                logging.info(f"Pinecone index {self.index_name} already exists")
                
        except Exception as e:
            logging.error(f"Failed to setup Pinecone index: {e}")
            raise
    
    def _setup_faiss(self):
        """Setup FAISS index as fallback."""
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        self.faiss_metadata = []
        logging.info("FAISS index initialized")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            if not texts:
                return np.array([])
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]
            if not valid_texts:
                return np.array([])
            
            embeddings = self.model.encode(
                valid_texts, 
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=len(valid_texts) > 10
            )
            
            logging.info(f"Generated embeddings for {len(valid_texts)} texts")
            return embeddings
            
        except Exception as e:
            logging.error(f"Failed to generate embeddings: {e}")
            raise
    
    def store_embeddings(self, chunks: List[Dict[str, Any]], document_id: str, namespace: str = "default") -> bool:
        """Store embeddings in vector database."""
        try:
            if not chunks:
                logging.warning("No chunks to store")
                return False
            
            # Generate embeddings for all chunks
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            if embeddings.size == 0:
                logging.warning("No valid embeddings generated")
                return False
            
            if self.use_pinecone:
                return self._store_in_pinecone(chunks, embeddings, document_id, namespace)
            else:
                return self._store_in_faiss(chunks, embeddings, document_id)
                
        except Exception as e:
            logging.error(f"Failed to store embeddings: {e}")
            return False
    
    def _store_in_pinecone(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, 
                          document_id: str, namespace: str) -> bool:
        """Store embeddings in Pinecone."""
        try:
            vectors_to_upsert = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}_chunk_{chunk['id']}"
                metadata = {
                    "document_id": document_id,
                    "chunk_id": chunk["id"],
                    "text": chunk["text"][:1000],  # Limit metadata size
                    "tokens": chunk.get("tokens", 0),
                    "start_sentence": chunk.get("start_sentence", 0),
                    "end_sentence": chunk.get("end_sentence", 0)
                }
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
            
            logging.info(f"Stored {len(vectors_to_upsert)} vectors in Pinecone")
            return True
            
        except Exception as e:
            logging.error(f"Failed to store in Pinecone: {e}")
            return False
    
    def _store_in_faiss(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, document_id: str) -> bool:
        """Store embeddings in FAISS."""
        try:
            # Add embeddings to FAISS index
            self.faiss_index.add(embeddings)
            
            # Store metadata
            for i, chunk in enumerate(chunks):
                metadata = {
                    "document_id": document_id,
                    "chunk_id": chunk["id"],
                    "text": chunk["text"],
                    "tokens": chunk.get("tokens", 0),
                    "faiss_id": len(self.faiss_metadata)
                }
                self.faiss_metadata.append(metadata)
            
            logging.info(f"Stored {len(chunks)} vectors in FAISS")
            return True
            
        except Exception as e:
            logging.error(f"Failed to store in FAISS: {e}")
            return False
    
    def search_similar(self, query: str, namespace: str = "default", 
                      document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks with enhanced strategies for insurance queries."""
        try:
            if not query.strip():
                return []
            
            # Preprocess query to improve retrieval
            query = self._preprocess_insurance_query(query)
            
            # Generate query embedding
            query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
            
            # Enhanced retrieval logic
            if self.use_pinecone:
                return self._search_pinecone_enhanced(query, query_embedding, namespace, document_id)
            else:
                return self._search_faiss_enhanced(query, query_embedding, document_id)
                
        except Exception as e:
            logging.error(f"Failed to search similar chunks: {e}")
            return []
    
    def _preprocess_insurance_query(self, query: str) -> str:
        """Preprocess insurance query to improve retrieval."""
        # Add relevant insurance terms based on query content to enhance retrieval
        query_lower = query.lower()
        enhanced_query = query
        
        # Check for common insurance terms and enhance query
        if "grace period" in query_lower:
            enhanced_query = f"{query} premium payment due date renewal"
        elif "waiting period" in query_lower and "pre-existing" in query_lower:
            enhanced_query = f"{query} pre-existing disease PED continuous coverage inception"
        elif "waiting period" in query_lower and "cataract" in query_lower:
            enhanced_query = f"{query} cataract surgery specific disease waiting"
        elif "maternity" in query_lower:
            enhanced_query = f"{query} maternity childbirth delivery termination of pregnancy"
        elif "donor" in query_lower or "organ" in query_lower:
            enhanced_query = f"{query} organ donor transplantation harvesting medical expenses"
        elif "claim" in query_lower and ("discount" in query_lower or "bonus" in query_lower):
            enhanced_query = f"{query} no claim discount bonus NCD NCB renewal premium"
        elif "health check" in query_lower:
            enhanced_query = f"{query} preventive health check-up medical tests reimbursement"
        elif "hospital" in query_lower and "define" in query_lower:
            enhanced_query = f"{query} definition of hospital inpatient beds qualified nursing medical practitioner"
        elif "ayush" in query_lower:
            enhanced_query = f"{query} AYUSH treatment Ayurveda Yoga Naturopathy Unani Siddha Homeopathy"
        elif "room rent" in query_lower or "sub-limit" in query_lower:
            enhanced_query = f"{query} room rent ICU charges sub-limits capping sum insured percentage"
        elif "physiotherapy" in query_lower:
            enhanced_query = f"{query} physiotherapy prescribed treatment therapy out-patient physical"
        
        logging.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")
        return enhanced_query
    
    def _search_pinecone_enhanced(self, original_query: str, query_embedding: np.ndarray, 
                                 namespace: str, document_id: Optional[str]) -> List[Dict[str, Any]]:
        """Enhanced search in Pinecone with more sophisticated retrieval strategies."""
        try:
            # Build filter
            filter_dict = {}
            if document_id:
                filter_dict["document_id"] = {"$eq": document_id}
            
            # First pass: search with vector similarity
            search_results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=self.top_k,
                namespace=namespace,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            results = []
            for match in search_results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "document_id": match.metadata.get("document_id", ""),
                    "chunk_id": match.metadata.get("chunk_id", 0),
                    "tokens": match.metadata.get("tokens", 0)
                }
                results.append(result)
            
            # If we have enough results, return them
            if len(results) >= self.top_k * 0.8:  # At least 80% of requested results
                logging.info(f"Found {len(results)} similar chunks in Pinecone (standard)")
                return results
            
            # Second pass: if we have insufficient results, use hybrid search
            # Check for critical insurance terms
            query_lower = original_query.lower()
            critical_terms = []
            
            if "grace period" in query_lower:
                critical_terms = ["grace period", "premium payment", "renewal"]
            elif "waiting period" in query_lower:
                critical_terms = ["waiting period", "pre-existing", "specific disease"]
            elif "maternity" in query_lower:
                critical_terms = ["maternity", "childbirth", "pregnancy", "delivery"]
            elif "organ donor" in query_lower:
                critical_terms = ["organ donor", "transplantation", "harvest"]
            elif "no claim" in query_lower:
                critical_terms = ["no claim", "discount", "bonus"]
            elif "health check" in query_lower:
                critical_terms = ["health check-up", "preventive", "medical test"]
            elif "hospital" in query_lower:
                critical_terms = ["hospital", "definition", "beds", "nursing"]
            elif "ayush" in query_lower:
                critical_terms = ["AYUSH", "ayurveda", "yoga", "naturopathy"]
            elif "room rent" in query_lower:
                critical_terms = ["room rent", "ICU", "sub-limit", "capping"]
            elif "physiotherapy" in query_lower:
                critical_terms = ["physiotherapy", "therapy", "treatment"]
            
            # If we identified critical terms, use them to enhance search
            if critical_terms:
                logging.info(f"Using critical terms for enhanced retrieval: {critical_terms}")
                
                # Get all chunks from document
                all_results = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=100,  # Get a larger set of chunks
                    namespace=namespace,
                    filter=filter_dict if filter_dict else None,
                    include_metadata=True
                ).matches
                
                # Score chunks based on critical term presence
                scored_results = []
                for match in all_results:
                    text = match.metadata.get("text", "").lower()
                    base_score = match.score
                    term_score = sum(term.lower() in text for term in critical_terms) * 0.1
                    scored_results.append((match, base_score + term_score))
                
                # Sort by combined score and take top results
                scored_results.sort(key=lambda x: x[1], reverse=True)
                
                # Build final results
                enhanced_results = []
                for match, score in scored_results[:self.top_k]:
                    result = {
                        "id": match.id,
                        "score": score,
                        "text": match.metadata.get("text", ""),
                        "document_id": match.metadata.get("document_id", ""),
                        "chunk_id": match.metadata.get("chunk_id", 0),
                        "tokens": match.metadata.get("tokens", 0)
                    }
                    enhanced_results.append(result)
                
                logging.info(f"Found {len(enhanced_results)} similar chunks in Pinecone (enhanced)")
                return enhanced_results
            
            return results
            
        except Exception as e:
            logging.error(f"Failed to search Pinecone: {e}")
            return []
    
    def _search_faiss_enhanced(self, original_query: str, query_embedding: np.ndarray, 
                              document_id: Optional[str]) -> List[Dict[str, Any]]:
        """Enhanced search in FAISS."""
        try:
            if self.faiss_index.ntotal == 0:
                return []
            
            # Search FAISS with more results initially
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), 
                min(self.top_k * 2, self.faiss_index.ntotal)  # Get more results for filtering
            )
            
            # Check for critical insurance terms
            query_lower = original_query.lower()
            critical_terms = []
            
            if "grace period" in query_lower:
                critical_terms = ["grace period", "premium payment", "renewal"]
            elif "waiting period" in query_lower:
                critical_terms = ["waiting period", "pre-existing", "specific disease"]
            elif "maternity" in query_lower:
                critical_terms = ["maternity", "childbirth", "pregnancy"]
            elif "donor" in query_lower or "organ" in query_lower:
                critical_terms = ["organ donor", "transplantation", "harvest"]
            elif "no claim" in query_lower:
                critical_terms = ["no claim", "discount", "bonus"]
            elif "health check" in query_lower:
                critical_terms = ["health check-up", "preventive", "medical test"]
            elif "hospital" in query_lower and "define" in query_lower:
                critical_terms = ["hospital", "definition", "beds", "nursing"]
            elif "ayush" in query_lower:
                critical_terms = ["AYUSH", "ayurveda", "yoga", "naturopathy"]
            elif "room rent" in query_lower or "sub-limit" in query_lower:
                critical_terms = ["room rent", "ICU", "sub-limit", "capping"]
            elif "physiotherapy" in query_lower:
                critical_terms = ["physiotherapy", "therapy", "treatment"]
            
            results = []
            scored_results = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.faiss_metadata):
                    metadata = self.faiss_metadata[idx]
                    
                    # Filter by document_id if specified
                    if document_id and metadata.get("document_id") != document_id:
                        continue
                    
                    # Enhance scoring with critical terms if available
                    if critical_terms:
                        text = metadata.get("text", "").lower()
                        term_score = sum(term.lower() in text for term in critical_terms) * 0.1
                        final_score = float(score) + term_score
                    else:
                        final_score = float(score)
                    
                    scored_results.append((metadata, final_score))
            
            # Sort by combined score
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k results
            for metadata, final_score in scored_results[:self.top_k]:
                result = {
                    "id": f"faiss_{metadata.get('faiss_id', 0)}",
                    "score": final_score,
                    "text": metadata.get("text", ""),
                    "document_id": metadata.get("document_id", ""),
                    "chunk_id": metadata.get("chunk_id", 0),
                    "tokens": metadata.get("tokens", 0)
                }
                results.append(result)
            
            logging.info(f"Found {len(results)} similar chunks in FAISS")
            return results
            
        except Exception as e:
            logging.error(f"Failed to search FAISS: {e}")
            return []
    
    def delete_document_vectors(self, document_id: str, namespace: str = "default") -> bool:
        """Delete all vectors for a specific document."""
        try:
            if self.use_pinecone:
                # Delete from Pinecone by filter
                self.index.delete(filter={"document_id": {"$eq": document_id}}, namespace=namespace)
                logging.info(f"Deleted vectors for document {document_id} from Pinecone")
            else:
                # For FAISS, we'd need to rebuild the index (limitation of FAISS)
                # Here we just mark them for future filtering
                logging.warning("FAISS doesn't support deletion. Consider rebuilding index.")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete document vectors: {e}")
            return False

# Initialize global embedding manager
embedding_manager = EmbeddingManager()

# Backward compatibility functions
def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Backward compatibility function."""
    return embedding_manager.generate_embeddings(texts)

def search_pinecone(query_dict: dict, embeddings: np.ndarray = None) -> List[str]:
    """Backward compatibility function."""
    query_string = query_dict.get("raw_query", "") if isinstance(query_dict, dict) else str(query_dict)
    results = embedding_manager.search_similar(query_string)
    return [result["text"] for result in results]

def search_similar_clauses(query: str, embeddings: np.ndarray) -> List[int]:
    """Backward compatibility function using FAISS."""
    if embeddings.size == 0:
        return []
    
    # Use the old FAISS approach for backward compatibility
    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(embeddings)
    
    query_embedding = embedding_manager.model.encode([query])[0]
    D, I = faiss_index.search(query_embedding.reshape(1, -1), k=min(5, embeddings.shape[0]))
    
    return I[0].tolist()
