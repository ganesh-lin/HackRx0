import logging

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_decision(query: dict, clauses: list) -> tuple:
    """Evaluate decision and provide rationale."""
    try:
        procedure = query.get("procedure", "").lower()
        raw_query = query.get("raw_query", "").lower()
        
        # If no specific procedure found, try to extract from raw query
        if not procedure:
            # Look for surgery-related terms in the raw query
            surgery_terms = ["surgery", "operation", "procedure", "treatment"]
            for term in surgery_terms:
                if term in raw_query:
                    # Extract the word before the surgery term
                    words = raw_query.split()
                    for i, word in enumerate(words):
                        if term in word and i > 0:
                            procedure = f"{words[i-1]} {term}"
                            break
                    if not procedure and term in raw_query:
                        procedure = term
                    break
        
        # Analyze clauses for coverage information
        coverage_found = False
        exclusion_found = False
        relevant_clause = ""
        
        for clause in clauses:
            clause_lower = clause.lower()
            
            # Check for coverage
            if procedure and procedure in clause_lower:
                if any(word in clause_lower for word in ["covered", "includes", "eligible", "benefit"]):
                    coverage_found = True
                    relevant_clause = clause[:200] + "..." if len(clause) > 200 else clause
                    break
                elif any(word in clause_lower for word in ["excluded", "not covered", "except", "excluding"]):
                    exclusion_found = True
                    relevant_clause = clause[:200] + "..." if len(clause) > 200 else clause
                    break
            
            # Also check for general surgery terms if specific procedure not found
            if not procedure and any(term in clause_lower for term in ["surgery", "surgical", "operation"]):
                if any(word in clause_lower for word in ["covered", "includes", "eligible"]):
                    coverage_found = True
                    relevant_clause = clause[:200] + "..." if len(clause) > 200 else clause
                    procedure = "surgical procedures"
                    break
        
        # Generate response based on findings
        if coverage_found:
            answer = f"Yes, {procedure or 'the requested procedure'} appears to be covered."
            rationale = f"Found relevant coverage clause: {relevant_clause}"
        elif exclusion_found:
            answer = f"No, {procedure or 'the requested procedure'} appears to be excluded."
            rationale = f"Found exclusion clause: {relevant_clause}"
        else:
            answer = f"Coverage for {procedure or 'the requested procedure'} is unclear from the available information."
            rationale = "No specific coverage or exclusion clauses found for this procedure."
        
        logging.info(f"Decision: {answer}")
        return answer, rationale
        
    except Exception as e:
        logging.error(f"Failed to evaluate decision: {str(e)}")
        return "Unable to process query.", f"Error: {str(e)}"