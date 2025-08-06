import logging

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_decision(query: dict, clauses: list) -> tuple:
    """Evaluate decision and provide rationale."""
    try:
        procedure = query.get("procedure", "").lower()
        for clause in clauses:
            clause_lower = clause.lower()
            if procedure and procedure in clause_lower and "covered" in clause_lower:
                logging.info(f"Decision: Covered for {procedure}")
                return f"Yes, {procedure} is covered.", f"Based on clause: {clause}"
        logging.info(f"Decision: Not covered for {procedure}")
        return f"No, {procedure} is not covered.", "No matching clause found."
    except Exception as e:
        logging.error(f"Failed to evaluate decision: {str(e)}")
        return "Unable to process query.", f"Error: {str(e)}"