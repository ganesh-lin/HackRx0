from transformers import pipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def match_clauses(query: dict, clauses: list) -> list:
    """Match relevant clauses based on the parsed query."""
    if not clauses:
        return []
    
    # Simple keyword-based matching (can be enhanced with more sophisticated NLP)
    raw_query = query.get("raw_query", "").lower()
    matched_clauses = []
    
    # Define keywords for different types of queries
    coverage_keywords = ["cover", "covered", "coverage", "include", "included"]
    procedure_keywords = ["surgery", "procedure", "treatment", "operation"]
    
    for clause in clauses:
        clause_lower = clause.lower()
        
        # Check if clause is relevant to the query
        relevance_score = 0
        
        # Check for coverage-related terms
        for keyword in coverage_keywords:
            if keyword in raw_query and keyword in clause_lower:
                relevance_score += 2
        
        # Check for procedure-related terms
        for keyword in procedure_keywords:
            if keyword in raw_query and keyword in clause_lower:
                relevance_score += 1
        
        # Check for direct word matches
        query_words = raw_query.split()
        for word in query_words:
            if len(word) > 3 and word in clause_lower:  # Skip short words
                relevance_score += 1
        
        # Add clause if it has some relevance
        if relevance_score > 0:
            matched_clauses.append(clause)
    
    # Sort by relevance (approximate) and return top matches
    return matched_clauses[:5]  # Return top 5 matches