import os
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv
import re
import json

load_dotenv()

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LLMManager:
    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        self.hf_token = os.getenv("HF_TOKEN")
        self.max_tokens = int(os.getenv("MAX_TOKENS", 2048))
        self.temperature = float(os.getenv("TEMPERATURE", 0.1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Mistral model and tokenizer."""
        try:
            logging.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimization for CPU/GPU
            model_kwargs = {
                "token": self.hf_token,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # Add memory optimization for CPU
            if self.device == "cpu":
                model_kwargs.update({
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float32
                })
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                model_kwargs={"low_cpu_mem_usage": True} if self.device == "cpu" else {}
            )
            
            logging.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}")
            # Fallback to a smaller model or error handling
            raise
    
    def create_prompt(self, question: str, retrieved_chunks: List[str], 
                     context_type: str = "insurance") -> str:
        """Create a structured prompt for the LLM."""
        
        # Limit context to prevent token overflow
        max_context_length = 3000  # Reserve tokens for question and response
        combined_context = "\n\n".join(retrieved_chunks)
        
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "..."
        
        prompt_template = f"""<s>[INST] You are a highly skilled legal and insurance assistant specializing in {context_type} policy analysis.

Your task is to answer questions about insurance policies, legal documents, and related matters using ONLY the provided context. You must be precise, accurate, and cite specific information from the context.

INSTRUCTIONS:
1. Answer using ONLY the information provided in the context below
2. If the answer is not in the context, respond with "Not found in document"
3. Be specific and include relevant details like waiting periods, coverage amounts, conditions, etc.
4. Quote directly from the document when possible
5. Provide clear, actionable information

CONTEXT:
{combined_context}

QUESTION: {question}

Provide a comprehensive answer based solely on the context provided. If specific terms, conditions, or limitations apply, include them in your response. [/INST]"""

        return prompt_template
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using the LLM."""
        try:
            if not self.pipeline:
                raise Exception("Model not initialized")
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text'].strip()
            
            # Clean up the response
            generated_text = self._clean_response(generated_text)
            
            logging.info(f"Generated response: {len(generated_text)} characters")
            return generated_text
            
        except Exception as e:
            logging.error(f"Failed to generate response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _clean_response(self, text: str) -> str:
        """Clean and format the generated response."""
        # Remove potential instruction artifacts
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        text = re.sub(r'<s>|</s>', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def answer_question(self, question: str, retrieved_chunks: List[str], 
                       context_type: str = "insurance") -> str:
        """Main method to answer a question using retrieved context."""
        try:
            if not retrieved_chunks:
                return "Not found in document."
            
            # Create prompt
            prompt = self.create_prompt(question, retrieved_chunks, context_type)
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Validate response
            if not response or response.lower().strip() in ["", "not found in document", "not found in document."]:
                return "Not found in document."
            
            return response
            
        except Exception as e:
            logging.error(f"Failed to answer question: {e}")
            return f"Error processing question: {str(e)}"
    
    def batch_answer_questions(self, questions: List[str], 
                             retrieved_chunks_per_question: List[List[str]],
                             context_type: str = "insurance") -> List[str]:
        """Answer multiple questions efficiently."""
        answers = []
        
        for i, (question, chunks) in enumerate(zip(questions, retrieved_chunks_per_question)):
            try:
                logging.info(f"Processing question {i+1}/{len(questions)}")
                answer = self.answer_question(question, chunks, context_type)
                answers.append(answer)
            except Exception as e:
                logging.error(f"Error processing question {i+1}: {e}")
                answers.append(f"Error processing question: {str(e)}")
        
        return answers
    
    def extract_key_information(self, text: str, query_type: str = "policy") -> Dict[str, Any]:
        """Extract structured information from text."""
        try:
            extraction_prompt = f"""<s>[INST] Extract key information from the following {query_type} text and return it as a structured summary.

Focus on:
- Key terms and conditions
- Coverage details
- Waiting periods
- Exclusions
- Important numbers (amounts, percentages, time periods)

TEXT:
{text[:2000]}  # Limit text length

Return the information in a clear, structured format. [/INST]"""

            response = self.generate_response(extraction_prompt)
            
            # Try to parse as structured data
            structured_info = {
                "summary": response,
                "extracted_terms": self._extract_terms(response),
                "numbers": self._extract_numbers(response),
                "time_periods": self._extract_time_periods(response)
            }
            
            return structured_info
            
        except Exception as e:
            logging.error(f"Failed to extract information: {e}")
            return {"error": str(e)}
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract important terms from text."""
        # Simple keyword extraction
        important_patterns = [
            r'coverage\s+(?:for|of|includes?)\s+([^.]+)',
            r'waiting\s+period\s+(?:of|for)\s+([^.]+)',
            r'exclusion[s]?\s*:?\s*([^.]+)',
            r'condition[s]?\s*:?\s*([^.]+)',
        ]
        
        terms = []
        for pattern in important_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                terms.append(match.group(1).strip())
        
        return terms[:10]  # Limit results
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numerical information."""
        number_patterns = [
            r'\b\d+\s*(?:days?|months?|years?)\b',
            r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b',
            r'₹\s*\d+(?:,\d+)*(?:\.\d+)?',
            r'\$\s*\d+(?:,\d+)*(?:\.\d+)?',
            r'\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:rupees?|dollars?)\b'
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                numbers.append(match.group().strip())
        
        return list(set(numbers))[:10]  # Unique and limited
    
    def _extract_time_periods(self, text: str) -> List[str]:
        """Extract time-related information."""
        time_patterns = [
            r'\b(?:waiting\s+period|grace\s+period|term)\s+(?:of\s+)?(\d+\s*(?:days?|months?|years?))\b',
            r'\b(\d+\s*(?:days?|months?|years?))\s+(?:waiting|grace|term)\b',
            r'\b(?:within|after|before)\s+(\d+\s*(?:days?|months?|years?))\b'
        ]
        
        periods = []
        for pattern in time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                periods.append(match.group(1).strip())
        
        return list(set(periods))[:10]

# Initialize global LLM manager
try:
    llm_manager = LLMManager()
    logging.info("LLM Manager initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize LLM Manager: {e}")
    llm_manager = None
