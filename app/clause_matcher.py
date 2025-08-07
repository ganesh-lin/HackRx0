from transformers import pipeline
import os
from dotenv import load_dotenv
import logging
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
load_dotenv()

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ClauseMatcher:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define domain-specific keyword categories
        self.keyword_categories = {
            "coverage": {
                "keywords": ["cover", "covered", "coverage", "include", "included", "eligible", "entitle", "benefit"],
                "weight": 3.0
            },
            "exclusion": {
                "keywords": ["exclude", "excluded", "exclusion", "not covered", "not eligible", "except", "limitation"],
                "weight": 3.0
            },
            "procedure": {
                "keywords": ["surgery", "procedure", "treatment", "operation", "therapy", "intervention"],
                "weight": 2.5
            },
            "medical_condition": {
                "keywords": ["disease", "condition", "illness", "disorder", "syndrome", "diagnosis"],
                "weight": 2.0
            },
            "time_related": {
                "keywords": ["waiting period", "grace period", "term", "duration", "annual", "monthly", "daily"],
                "weight": 2.5
            },
            "financial": {
                "keywords": ["premium", "deductible", "copay", "amount", "sum insured", "benefit amount", "limit"],
                "weight": 2.0
            },
            "body_parts": {
                "keywords": ["knee", "heart", "brain", "eye", "liver", "kidney", "lung", "spine", "dental"],
                "weight": 2.5
            }
        }
        
        # Importance weights for different types of matches
        self.match_weights = {
            "exact_phrase": 4.0,
            "semantic_similarity": 3.0,
            "keyword_match": 2.0,
            "partial_match": 1.0,
            "contextual_match": 1.5
        }
    
    def match_clauses(self, query: Dict[str, Any], clauses: List[str], 
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """Match relevant clauses based on the parsed query with scoring."""
        try:
            if not clauses:
                return []
            
            raw_query = query.get("raw_query", "").lower()
            entities = query.get("entities", {})
            query_type = query.get("query_type", "general")
            
            scored_clauses = []
            
            for i, clause in enumerate(clauses):
                if not clause or not clause.strip():
                    continue
                
                score = self._calculate_clause_score(raw_query, clause, entities, query_type)
                
                if score > 0:
                    scored_clauses.append({
                        "text": clause,
                        "score": score,
                        "index": i,
                        "match_reasons": self._get_match_reasons(raw_query, clause, entities)
                    })
            
            # Sort by score (descending) and return top matches
            scored_clauses.sort(key=lambda x: x["score"], reverse=True)
            
            logging.info(f"Matched {len(scored_clauses)} clauses, returning top {min(top_k, len(scored_clauses))}")
            return scored_clauses[:top_k]
            
        except Exception as e:
            logging.error(f"Error in clause matching: {e}")
            return []
    
    def _calculate_clause_score(self, query: str, clause: str, entities: Dict[str, Any], 
                              query_type: str) -> float:
        """Calculate relevance score for a clause."""
        try:
            clause_lower = clause.lower()
            total_score = 0.0
            
            # 1. Exact phrase matching
            exact_score = self._calculate_exact_match_score(query, clause_lower)
            total_score += exact_score * self.match_weights["exact_phrase"]
            
            # 2. Semantic similarity
            semantic_score = self._calculate_semantic_similarity(query, clause)
            total_score += semantic_score * self.match_weights["semantic_similarity"]
            
            # 3. Keyword category matching
            keyword_score = self._calculate_keyword_score(query, clause_lower, entities)
            total_score += keyword_score * self.match_weights["keyword_match"]
            
            # 4. Entity matching
            entity_score = self._calculate_entity_score(entities, clause_lower)
            total_score += entity_score * self.match_weights["partial_match"]
            
            # 5. Query type specific bonus
            type_bonus = self._calculate_type_bonus(query_type, clause_lower)
            total_score += type_bonus * self.match_weights["contextual_match"]
            
            # 6. Length penalty (very short or very long clauses are less relevant)
            length_penalty = self._calculate_length_penalty(clause)
            total_score *= length_penalty
            
            return max(0, total_score)
            
        except Exception as e:
            logging.warning(f"Error calculating clause score: {e}")
            return 0.0
    
    def _calculate_exact_match_score(self, query: str, clause: str) -> float:
        """Calculate score for exact phrase matches."""
        score = 0.0
        
        # Look for exact phrase matches
        query_phrases = self._extract_phrases(query)
        
        for phrase in query_phrases:
            if len(phrase) > 3 and phrase in clause:
                score += len(phrase.split()) * 0.5  # Longer phrases get higher scores
        
        return min(score, 3.0)  # Cap the score
    
    def _calculate_semantic_similarity(self, query: str, clause: str) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            query_embedding = self.embedding_model.encode([query])
            clause_embedding = self.embedding_model.encode([clause])
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding[0], clause_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(clause_embedding[0])
            )
            
            # Convert to score (similarity ranges from -1 to 1, we want 0 to 1)
            return max(0, (similarity + 1) / 2)
            
        except Exception as e:
            logging.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_keyword_score(self, query: str, clause: str, entities: Dict[str, Any]) -> float:
        """Calculate score based on keyword category matching."""
        score = 0.0
        
        for category, info in self.keyword_categories.items():
            keywords = info["keywords"]
            weight = info["weight"]
            
            # Count matches in query and clause
            query_matches = sum(1 for keyword in keywords if keyword in query)
            clause_matches = sum(1 for keyword in keywords if keyword in clause)
            
            if query_matches > 0 and clause_matches > 0:
                category_score = min(query_matches, clause_matches) * weight * 0.1
                score += category_score
        
        return score
    
    def _calculate_entity_score(self, entities: Dict[str, Any], clause: str) -> float:
        """Calculate score based on entity matching."""
        score = 0.0
        
        # Match procedures
        procedures = entities.get("procedures", [])
        for procedure in procedures:
            if procedure.lower() in clause:
                score += 1.0
        
        # Match body parts
        body_parts = entities.get("body_parts", [])
        for part in body_parts:
            if part.lower() in clause:
                score += 0.8
        
        # Match insurance terms
        insurance_terms = entities.get("insurance_terms", [])
        for term in insurance_terms:
            if term.lower() in clause:
                score += 0.6
        
        # Match amounts (if any numbers are mentioned)
        amounts = entities.get("amounts", [])
        for amount in amounts:
            if amount in clause:
                score += 0.5
        
        return score
    
    def _calculate_type_bonus(self, query_type: str, clause: str) -> float:
        """Give bonus based on query type alignment."""
        type_keywords = {
            "coverage": ["cover", "benefit", "include", "eligible"],
            "exclusion": ["exclude", "not cover", "limitation", "except"],
            "waiting_period": ["waiting", "period", "wait"],
            "premium": ["premium", "cost", "payment"],
            "claim": ["claim", "reimburse", "settlement"],
            "definition": ["mean", "define", "refer", "definition"]
        }
        
        keywords = type_keywords.get(query_type, [])
        matches = sum(1 for keyword in keywords if keyword in clause)
        
        return matches * 0.5
    
    def _calculate_length_penalty(self, clause: str) -> float:
        """Apply penalty for very short or very long clauses."""
        words = len(clause.split())
        
        if words < 5:
            return 0.5  # Too short
        elif words > 200:
            return 0.7  # Too long
        elif 10 <= words <= 100:
            return 1.0  # Optimal length
        else:
            return 0.8  # Moderate length
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text."""
        # Simple phrase extraction (can be enhanced)
        phrases = []
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', text)
        phrases.extend(quoted_phrases)
        
        # Extract multi-word terms
        multi_word_patterns = [
            r'\b\w+\s+\w+\s+\w+\b',  # 3-word phrases
            r'\b\w+\s+\w+\b'         # 2-word phrases
        ]
        
        for pattern in multi_word_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        
        return [phrase.strip() for phrase in phrases if len(phrase.strip()) > 3]
    
    def _get_match_reasons(self, query: str, clause: str, entities: Dict[str, Any]) -> List[str]:
        """Get reasons why a clause matched."""
        reasons = []
        
        # Check for direct keyword matches
        query_words = set(query.split())
        clause_words = set(clause.lower().split())
        common_words = query_words.intersection(clause_words)
        
        if common_words:
            reasons.append(f"Common keywords: {', '.join(list(common_words)[:3])}")
        
        # Check for entity matches
        procedures = entities.get("procedures", [])
        for procedure in procedures:
            if procedure.lower() in clause.lower():
                reasons.append(f"Procedure match: {procedure}")
        
        body_parts = entities.get("body_parts", [])
        for part in body_parts:
            if part.lower() in clause.lower():
                reasons.append(f"Body part match: {part}")
        
        return reasons

# Initialize global clause matcher
clause_matcher = ClauseMatcher()

# Backward compatibility function
def match_clauses(query: dict, clauses: list) -> list:
    """Backward compatibility function."""
    matched_results = clause_matcher.match_clauses(query, clauses)
    return [result["text"] for result in matched_results]