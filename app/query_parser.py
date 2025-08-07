from transformers import pipeline
import os
from dotenv import load_dotenv
import logging
import json
import re
from typing import Dict, List, Any, Optional

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

class QueryParser:
    def __init__(self):
        # Medical and insurance related keywords
        self.procedure_keywords = {
            "surgical": [
                "surgery", "operation", "procedure", "surgical", "operative",
                "knee surgery", "heart surgery", "brain surgery", "eye surgery",
                "cataract surgery", "bypass surgery", "transplant", "implant",
                "arthroscopy", "laparoscopy", "endoscopy", "biopsy"
            ],
            "medical": [
                "treatment", "therapy", "consultation", "diagnosis", "examination",
                "chemotherapy", "radiotherapy", "dialysis", "physiotherapy",
                "rehabilitation", "medication", "drugs", "prescription"
            ],
            "dental": [
                "dental", "tooth", "teeth", "crown", "filling", "root canal",
                "extraction", "orthodontic", "braces", "dentures"
            ],
            "maternity": [
                "maternity", "pregnancy", "childbirth", "delivery", "prenatal",
                "postnatal", "caesarean", "c-section", "abortion", "miscarriage"
            ],
            "emergency": [
                "emergency", "accident", "trauma", "urgent", "critical",
                "ambulance", "ICU", "intensive care", "hospitalization"
            ]
        }
        
        self.body_parts = [
            "knee", "heart", "brain", "eye", "liver", "kidney", "lung",
            "stomach", "spine", "back", "neck", "shoulder", "hip", "ankle",
            "wrist", "elbow", "hand", "foot", "head", "chest", "abdomen"
        ]
        
        self.insurance_terms = [
            "premium", "deductible", "copay", "coverage", "benefit", "claim",
            "policy", "waiting period", "grace period", "exclusion", "inclusion",
            "sum insured", "no claim bonus", "discount", "renewal", "lapse"
        ]
        
        self.time_periods = [
            "days?", "weeks?", "months?", "years?", "daily", "weekly", 
            "monthly", "yearly", "annual", "quarterly"
        ]
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into structured components."""
        try:
            query_lower = query.lower().strip()
            
            parsed = {
                "raw_query": query,
                "cleaned_query": query_lower,
                "entities": self._extract_entities(query_lower),
                "query_type": self._determine_query_type(query_lower),
                "intent": self._extract_intent(query_lower),
                "keywords": self._extract_keywords(query_lower),
                "medical_context": self._extract_medical_context(query_lower),
                "temporal_context": self._extract_temporal_context(query_lower)
            }
            
            # Legacy fields for backward compatibility
            parsed.update({
                "procedure": parsed["entities"].get("procedures", [""])[0] if parsed["entities"].get("procedures") else "",
                "age": parsed["entities"].get("age"),
                "gender": parsed["entities"].get("gender")
            })
            
            logging.info(f"Parsed query: {json.dumps(parsed, indent=2)}")
            return parsed
            
        except Exception as e:
            logging.error(f"Failed to parse query: {str(e)}")
            return {
                "raw_query": query,
                "procedure": "",
                "query_type": "general",
                "entities": {},
                "intent": "unknown"
            }
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract named entities from the query."""
        entities = {}
        
        # Extract age
        age_patterns = [
            r'\b(\d{1,3})\s*(?:years?|yrs?|y\.o\.?|year[-\s]old)\b',
            r'\b(\d{1,3})(?:M|F)\b',  # Pattern like "46M"
            r'\b(\d{1,3})\s*(?:male|female)\b'
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities["age"] = matches[0]
                break
        
        # Extract gender
        gender_patterns = [
            (r'\b(?:male|man|M)\b', "M"),
            (r'\b(?:female|woman|F)\b', "F")
        ]
        
        for pattern, gender in gender_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                entities["gender"] = gender
                break
        
        # Extract procedures
        procedures = []
        for category, keywords in self.procedure_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query:
                    procedures.append(keyword)
        entities["procedures"] = list(set(procedures))
        
        # Extract body parts
        body_parts = []
        for part in self.body_parts:
            if part.lower() in query:
                body_parts.append(part)
        entities["body_parts"] = body_parts
        
        # Extract insurance terms
        insurance_terms = []
        for term in self.insurance_terms:
            if term.lower() in query:
                insurance_terms.append(term)
        entities["insurance_terms"] = insurance_terms
        
        # Extract amounts/numbers
        amount_patterns = [
            r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'\b(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rupees?|dollars?|rs\.?|usd)\b'
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            amounts.extend(matches)
        entities["amounts"] = amounts
        
        return entities
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query."""
        query_types = {
            "coverage": ["cover", "covered", "coverage", "included", "eligible"],
            "exclusion": ["exclude", "excluded", "exclusion", "not covered", "not eligible"],
            "waiting_period": ["waiting period", "wait", "waiting time"],
            "premium": ["premium", "cost", "price", "payment", "fee"],
            "claim": ["claim", "reimburse", "reimbursement", "settle", "settlement"],
            "benefit": ["benefit", "advantage", "bonus", "discount"],
            "policy_terms": ["terms", "conditions", "policy", "document"],
            "definition": ["what is", "define", "definition", "meaning"],
            "procedure": ["how to", "process", "steps", "procedure"],
            "comparison": ["compare", "difference", "vs", "versus", "better"]
        }
        
        for query_type, keywords in query_types.items():
            if any(keyword in query for keyword in keywords):
                return query_type
        
        return "general"
    
    def _extract_intent(self, query: str) -> str:
        """Extract the intent of the query."""
        intent_patterns = {
            "information_seeking": [
                r"\b(?:what|how|when|where|why|which)\b",
                r"\b(?:tell me|explain|describe|define)\b"
            ],
            "coverage_check": [
                r"\b(?:does.*cover|is.*covered|covered.*by)\b",
                r"\b(?:eligible|qualify|qualifies)\b"
            ],
            "comparison": [
                r"\b(?:compare|difference|better|vs|versus)\b"
            ],
            "calculation": [
                r"\b(?:calculate|cost|price|amount|how much)\b"
            ],
            "procedure": [
                r"\b(?:how to|steps|process|procedure)\b"
            ]
        }
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        return "information_seeking"  # Default intent
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _extract_medical_context(self, query: str) -> Dict[str, Any]:
        """Extract medical context from the query."""
        medical_context = {
            "urgency": "normal",
            "complexity": "standard",
            "category": "general"
        }
        
        # Determine urgency
        urgent_keywords = ["emergency", "urgent", "critical", "immediate", "acute"]
        if any(keyword in query for keyword in urgent_keywords):
            medical_context["urgency"] = "high"
        
        # Determine complexity
        complex_keywords = ["surgery", "operation", "transplant", "cancer", "chronic"]
        if any(keyword in query for keyword in complex_keywords):
            medical_context["complexity"] = "high"
        
        # Determine category
        for category, keywords in self.procedure_keywords.items():
            if any(keyword in query for keyword in keywords):
                medical_context["category"] = category
                break
        
        return medical_context
    
    def _extract_temporal_context(self, query: str) -> Dict[str, Any]:
        """Extract time-related information from the query."""
        temporal_info = {}
        
        # Extract time periods
        time_patterns = [
            r'(\d+)\s*(?:days?|weeks?|months?|years?)',
            r'(?:within|after|before)\s+(\d+)\s*(?:days?|weeks?|months?|years?)',
            r'(\d+)[-\s](?:day|week|month|year)'
        ]
        
        periods = []
        for pattern in time_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            periods.extend(matches)
        
        temporal_info["time_periods"] = periods
        
        # Extract relative time references
        relative_patterns = [
            "recently", "currently", "now", "today", "yesterday", "tomorrow",
            "last week", "next week", "last month", "next month", "last year"
        ]
        
        relative_refs = []
        for pattern in relative_patterns:
            if pattern in query:
                relative_refs.append(pattern)
        
        temporal_info["relative_time"] = relative_refs
        
        return temporal_info
    
    def get_search_terms(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Generate search terms for vector similarity search."""
        search_terms = []
        
        # Add raw query
        search_terms.append(parsed_query["raw_query"])
        
        # Add procedures
        if parsed_query["entities"].get("procedures"):
            search_terms.extend(parsed_query["entities"]["procedures"])
        
        # Add body parts
        if parsed_query["entities"].get("body_parts"):
            search_terms.extend(parsed_query["entities"]["body_parts"])
        
        # Add insurance terms
        if parsed_query["entities"].get("insurance_terms"):
            search_terms.extend(parsed_query["entities"]["insurance_terms"])
        
        # Add keywords
        if parsed_query.get("keywords"):
            search_terms.extend(parsed_query["keywords"][:5])  # Limit to top 5
        
        # Create combined search query
        combined_query = " ".join([
            parsed_query["raw_query"],
            " ".join(parsed_query["entities"].get("procedures", [])),
            " ".join(parsed_query["entities"].get("body_parts", [])),
            " ".join(parsed_query["entities"].get("insurance_terms", []))
        ]).strip()
        
        if combined_query:
            search_terms.append(combined_query)
        
        return list(set(search_terms))  # Remove duplicates

# Initialize global query parser
query_parser = QueryParser()

# Backward compatibility function
def parse_query(query: str) -> dict:
    """Backward compatibility function."""
    return query_parser.parse_query(query)