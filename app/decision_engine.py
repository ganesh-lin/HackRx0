import logging
from typing import Dict, List, Any, Tuple, Optional
import re
from enum import Enum

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DecisionType(Enum):
    COVERED = "covered"
    NOT_COVERED = "not_covered"
    PARTIALLY_COVERED = "partially_covered"
    CONDITIONAL = "conditional"
    INSUFFICIENT_INFO = "insufficient_info"

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class DecisionEngine:
    def __init__(self):
        # Coverage indicators
        self.coverage_indicators = [
            "covered", "includes", "included", "eligible", "entitled", "benefit",
            "shall be reimbursed", "indemnify", "payable", "reimburse", "compensate"
        ]
        
        # Exclusion indicators
        self.exclusion_indicators = [
            "excluded", "not covered", "not eligible", "except", "excluding",
            "shall not", "does not cover", "not include", "limitation", "restrict"
        ]
        
        # Conditional indicators
        self.conditional_indicators = [
            "subject to", "provided that", "only if", "conditional", "depending",
            "waiting period", "after", "minimum", "maximum", "limit"
        ]
        
        # Time-related patterns
        self.time_patterns = [
            r'(\d+)\s*(?:days?|months?|years?)',
            r'waiting period of (\d+)',
            r'after (\d+)',
            r'minimum (\d+)',
            r'maximum (\d+)'
        ]
    
    def evaluate_decision(self, query: Dict[str, Any], matched_clauses: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Evaluate decision and provide detailed rationale."""
        try:
            if not matched_clauses:
                return self._handle_no_clauses(query)
            
            # Extract query information
            raw_query = query.get("raw_query", "")
            entities = query.get("entities", {})
            query_type = query.get("query_type", "general")
            
            # Analyze each clause
            clause_analyses = []
            for clause_info in matched_clauses:
                analysis = self._analyze_clause(clause_info, query, entities)
                clause_analyses.append(analysis)
            
            # Make final decision
            decision = self._make_final_decision(clause_analyses, query)
            
            # Generate response
            answer = self._generate_answer(decision, query)
            rationale = self._generate_rationale(decision, clause_analyses)
            
            logging.info(f"Decision: {decision['type']} with {decision['confidence']} confidence")
            return answer, rationale
            
        except Exception as e:
            logging.error(f"Failed to evaluate decision: {str(e)}")
            return "Unable to process query.", f"Error: {str(e)}"
    
    def _analyze_clause(self, clause_info: Dict[str, Any], query: Dict[str, Any], 
                       entities: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single clause for decision indicators."""
        clause_text = clause_info.get("text", "").lower()
        score = clause_info.get("score", 0)
        
        analysis = {
            "text": clause_info.get("text", ""),
            "score": score,
            "indicators": {
                "coverage": [],
                "exclusion": [],
                "conditional": []
            },
            "decision_type": None,
            "confidence": ConfidenceLevel.LOW,
            "time_constraints": [],
            "amount_constraints": [],
            "conditions": []
        }
        
        # Check for coverage indicators
        for indicator in self.coverage_indicators:
            if indicator in clause_text:
                analysis["indicators"]["coverage"].append(indicator)
        
        # Check for exclusion indicators
        for indicator in self.exclusion_indicators:
            if indicator in clause_text:
                analysis["indicators"]["exclusion"].append(indicator)
        
        # Check for conditional indicators
        for indicator in self.conditional_indicators:
            if indicator in clause_text:
                analysis["indicators"]["conditional"].append(indicator)
        
        # Extract time constraints
        analysis["time_constraints"] = self._extract_time_constraints(clause_text)
        
        # Extract amount constraints
        analysis["amount_constraints"] = self._extract_amount_constraints(clause_text)
        
        # Extract conditions
        analysis["conditions"] = self._extract_conditions(clause_text)
        
        # Determine clause decision type
        analysis["decision_type"] = self._determine_clause_decision_type(analysis)
        
        # Calculate confidence based on indicators and score
        analysis["confidence"] = self._calculate_confidence(analysis, score)
        
        return analysis
    
    def _determine_clause_decision_type(self, analysis: Dict[str, Any]) -> DecisionType:
        """Determine the decision type for a clause."""
        coverage_count = len(analysis["indicators"]["coverage"])
        exclusion_count = len(analysis["indicators"]["exclusion"])
        conditional_count = len(analysis["indicators"]["conditional"])
        
        if exclusion_count > coverage_count:
            return DecisionType.NOT_COVERED
        elif coverage_count > 0:
            if conditional_count > 0 or analysis["conditions"]:
                return DecisionType.CONDITIONAL
            else:
                return DecisionType.COVERED
        elif conditional_count > 0:
            return DecisionType.CONDITIONAL
        else:
            return DecisionType.INSUFFICIENT_INFO
    
    def _calculate_confidence(self, analysis: Dict[str, Any], score: float) -> ConfidenceLevel:
        """Calculate confidence level for a clause analysis."""
        indicator_count = sum(len(indicators) for indicators in analysis["indicators"].values())
        
        if score > 2.0 and indicator_count > 2:
            return ConfidenceLevel.HIGH
        elif score > 1.0 and indicator_count > 1:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _make_final_decision(self, clause_analyses: List[Dict[str, Any]], 
                           query: Dict[str, Any]) -> Dict[str, Any]:
        """Make final decision based on all clause analyses."""
        if not clause_analyses:
            return {
                "type": DecisionType.INSUFFICIENT_INFO,
                "confidence": ConfidenceLevel.LOW,
                "supporting_clauses": [],
                "conditions": [],
                "rationale": "No relevant clauses found"
            }
        
        # Score each decision type
        decision_scores = {
            DecisionType.COVERED: 0,
            DecisionType.NOT_COVERED: 0,
            DecisionType.CONDITIONAL: 0,
            DecisionType.PARTIALLY_COVERED: 0,
            DecisionType.INSUFFICIENT_INFO: 0
        }
        
        supporting_clauses = []
        all_conditions = []
        
        for analysis in clause_analyses:
            decision_type = analysis["decision_type"]
            confidence = analysis["confidence"]
            score = analysis["score"]
            
            # Weight the decision based on confidence and score
            weight = 1.0
            if confidence == ConfidenceLevel.HIGH:
                weight = 3.0
            elif confidence == ConfidenceLevel.MEDIUM:
                weight = 2.0
            
            decision_scores[decision_type] += weight * score
            
            # Collect supporting information
            if decision_type in [DecisionType.COVERED, DecisionType.NOT_COVERED, DecisionType.CONDITIONAL]:
                supporting_clauses.append(analysis)
            
            all_conditions.extend(analysis["conditions"])
        
        # Determine final decision
        final_decision_type = max(decision_scores, key=decision_scores.get)
        
        # Calculate overall confidence
        high_confidence_count = sum(1 for analysis in clause_analyses 
                                  if analysis["confidence"] == ConfidenceLevel.HIGH)
        
        if high_confidence_count > 0 and decision_scores[final_decision_type] > 2.0:
            overall_confidence = ConfidenceLevel.HIGH
        elif decision_scores[final_decision_type] > 1.0:
            overall_confidence = ConfidenceLevel.MEDIUM
        else:
            overall_confidence = ConfidenceLevel.LOW
        
        return {
            "type": final_decision_type,
            "confidence": overall_confidence,
            "supporting_clauses": supporting_clauses[:3],  # Top 3 supporting clauses
            "conditions": list(set(all_conditions))[:5],   # Top 5 unique conditions
            "decision_scores": decision_scores,
            "rationale": self._build_decision_rationale(final_decision_type, supporting_clauses)
        }
    
    def _generate_answer(self, decision: Dict[str, Any], query: Dict[str, Any]) -> str:
        """Generate human-readable answer."""
        decision_type = decision["type"]
        confidence = decision["confidence"]
        conditions = decision["conditions"]
        
        # Extract query subject
        subject = self._extract_subject(query)
        
        # Base responses
        if decision_type == DecisionType.COVERED:
            base_answer = f"Yes, {subject} is covered"
        elif decision_type == DecisionType.NOT_COVERED:
            base_answer = f"No, {subject} is not covered"
        elif decision_type == DecisionType.CONDITIONAL:
            base_answer = f"Yes, {subject} is covered with conditions"
        elif decision_type == DecisionType.PARTIALLY_COVERED:
            base_answer = f"{subject} is partially covered"
        else:
            base_answer = f"Coverage for {subject} cannot be determined from the available information"
        
        # Add conditions if present
        if conditions and decision_type in [DecisionType.COVERED, DecisionType.CONDITIONAL]:
            conditions_text = ". Conditions include: " + "; ".join(conditions[:3])
            if len(conditions) > 3:
                conditions_text += " (and others)"
            base_answer += conditions_text
        
        # Add confidence qualifier
        if confidence == ConfidenceLevel.LOW:
            base_answer += ". (Note: This assessment has low confidence due to limited information)"
        
        return base_answer + "."
    
    def _generate_rationale(self, decision: Dict[str, Any], 
                          clause_analyses: List[Dict[str, Any]]) -> str:
        """Generate detailed rationale for the decision."""
        rationale_parts = []
        
        # Add main reasoning
        rationale_parts.append(decision["rationale"])
        
        # Add supporting clause information
        supporting_clauses = decision["supporting_clauses"]
        if supporting_clauses:
            rationale_parts.append("Based on the following policy provisions:")
            
            for i, clause in enumerate(supporting_clauses[:2], 1):
                clause_text = clause["text"][:150] + "..." if len(clause["text"]) > 150 else clause["text"]
                rationale_parts.append(f"{i}. {clause_text}")
        
        # Add confidence information
        confidence = decision["confidence"]
        if confidence == ConfidenceLevel.LOW:
            rationale_parts.append("Note: This assessment has low confidence due to limited or ambiguous information in the policy document.")
        
        return " ".join(rationale_parts)
    
    def _extract_subject(self, query: Dict[str, Any]) -> str:
        """Extract the main subject of the query."""
        entities = query.get("entities", {})
        
        # Try procedures first
        procedures = entities.get("procedures", [])
        if procedures:
            return procedures[0]
        
        # Try body parts
        body_parts = entities.get("body_parts", [])
        if body_parts:
            return f"{body_parts[0]} treatment"
        
        # Try insurance terms
        insurance_terms = entities.get("insurance_terms", [])
        if insurance_terms:
            return insurance_terms[0]
        
        # Fallback to generic
        return "the requested service"
    
    def _extract_time_constraints(self, text: str) -> List[str]:
        """Extract time-related constraints from text."""
        constraints = []
        
        for pattern in self.time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                constraints.append(match.group())
        
        return constraints
    
    def _extract_amount_constraints(self, text: str) -> List[str]:
        """Extract amount/financial constraints from text."""
        amount_patterns = [
            r'₹\s*\d+(?:,\d+)*(?:\.\d+)?',
            r'\$\s*\d+(?:,\d+)*(?:\.\d+)?',
            r'\d+(?:\.\d+)?%',
            r'limit of \d+',
            r'maximum \d+',
            r'minimum \d+'
        ]
        
        constraints = []
        for pattern in amount_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                constraints.append(match.group())
        
        return constraints
    
    def _extract_conditions(self, text: str) -> List[str]:
        """Extract conditions from text."""
        condition_patterns = [
            r'provided that [^.]+',
            r'subject to [^.]+',
            r'only if [^.]+',
            r'conditional on [^.]+',
            r'waiting period of [^.]+',
            r'after [^.]+ period'
        ]
        
        conditions = []
        for pattern in condition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                conditions.append(match.group().strip())
        
        return conditions
    
    def _build_decision_rationale(self, decision_type: DecisionType, 
                                supporting_clauses: List[Dict[str, Any]]) -> str:
        """Build rationale based on decision type."""
        if decision_type == DecisionType.COVERED:
            return "Found clear coverage provisions in the policy"
        elif decision_type == DecisionType.NOT_COVERED:
            return "Found explicit exclusions in the policy"
        elif decision_type == DecisionType.CONDITIONAL:
            return "Found coverage provisions with specific conditions"
        elif decision_type == DecisionType.PARTIALLY_COVERED:
            return "Found limited coverage provisions"
        else:
            return "Insufficient information in the policy to determine coverage"
    
    def _handle_no_clauses(self, query: Dict[str, Any]) -> Tuple[str, str]:
        """Handle case when no relevant clauses are found."""
        subject = self._extract_subject(query)
        answer = f"Coverage information for {subject} was not found in the available policy document."
        rationale = "No relevant clauses were identified that address this query."
        return answer, rationale

# Initialize global decision engine
decision_engine = DecisionEngine()

# Backward compatibility function
def evaluate_decision(query: dict, clauses: list) -> tuple:
    """Backward compatibility function."""
    # Convert old format to new format
    if clauses and isinstance(clauses[0], str):
        matched_clauses = [{"text": clause, "score": 1.0} for clause in clauses]
    else:
        matched_clauses = clauses
    
    return decision_engine.evaluate_decision(query, matched_clauses)