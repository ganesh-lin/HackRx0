from transformers import pipeline
import os
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Load environment variables
load_dotenv()

def parse_query(query: str) -> dict:
    """Parse natural language query using Mistral-7B via HF Inference API."""
    try:
        nlp = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", token=os.getenv("HF_TOKEN"))
        prompt = f"""Parse the query into structured components (age, procedure, location, policy_duration). Example:
        Query: "50F, heart surgery, Mumbai, 1-year policy"
        Output: {{"age": "50", "gender": "F", "procedure": "heart surgery", "location": "Mumbai", "policy_duration": "1 year"}}
        Query: {query}
        Output: """
        result = nlp(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
        # Extract JSON-like output
        output = result.split("Output: ")[-1].strip()
        parsed = json.loads(output) if output.startswith("{") else {"raw_query": query}
        parsed["raw_query"] = query
        logging.info(f"Parsed query: {parsed}")
        return parsed
    except Exception as e:
        logging.error(f"Failed to parse query: {str(e)}")
        return {"raw_query": query}