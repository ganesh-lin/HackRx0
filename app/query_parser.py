from transformers import pipeline
import os
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Load environment variables
load_dotenv()

def parse_query(query: str) -> dict:
    """Parse natural language query into structured components."""
    try:
        # Simple keyword-based parsing as fallback
        query_lower = query.lower()
        
        # Extract procedure/treatment keywords
        procedure_keywords = [
            "surgery", "operation", "procedure", "treatment", "therapy",
            "knee surgery", "heart surgery", "brain surgery", "eye surgery",
            "dental", "orthopedic", "cardiac", "neurological", "transplant",
            "chemotherapy", "radiotherapy", "dialysis", "bypass"
        ]
        
        procedure = ""
        for keyword in procedure_keywords:
            if keyword in query_lower:
                procedure = keyword
                break
        
        # Extract age if present
        age_match = None
        import re
        age_pattern = r'\b(\d{1,3})\s*(?:years?|yrs?|y\.o\.?|year-old)\b'
        age_matches = re.findall(age_pattern, query_lower)
        if age_matches:
            age_match = age_matches[0]
        
        # Extract gender if present
        gender = ""
        if any(word in query_lower for word in ["male", "m", "man"]):
            gender = "M"
        elif any(word in query_lower for word in ["female", "f", "woman"]):
            gender = "F"
        
        parsed = {
            "raw_query": query,
            "procedure": procedure,
            "age": age_match,
            "gender": gender,
            "query_type": "coverage" if any(word in query_lower for word in ["cover", "covered", "coverage"]) else "general"
        }
        
        logging.info(f"Parsed query: {parsed}")
        return parsed
        
    except Exception as e:
        logging.error(f"Failed to parse query: {str(e)}")
        return {"raw_query": query, "procedure": "", "query_type": "general"}