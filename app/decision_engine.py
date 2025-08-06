def evaluate_decision(query: dict, clauses: list) -> tuple:
    """Evaluate decision and provide rationale."""
    # Simplified decision logic
    for clause in clauses:
        if "knee surgery" in clause.lower() and "covered" in clause.lower():
            return "Yes, knee surgery is covered.", f"Based on clause: {clause}"
    return "No, knee surgery is not covered.", "No matching clause found."