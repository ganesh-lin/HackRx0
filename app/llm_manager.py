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
        # Use a more accessible model that doesn't require authentication
        self.model_name = os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium")
        self.fallback_models = [
            "microsoft/DialoGPT-medium",
            "distilbert-base-uncased",
            "gpt2"
        ]
        self.hf_token = os.getenv("HF_TOKEN")
        self.max_tokens = int(os.getenv("MAX_TOKENS", 512))  # Reduced for smaller models
        self.temperature = float(os.getenv("TEMPERATURE", 0.1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.model_type = "simple"  # Track model type for different handling
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize a lightweight model that doesn't require authentication."""
        try:
            # Try multiple models in order of preference
            for model_name in [self.model_name] + self.fallback_models:
                try:
                    logging.info(f"Attempting to load model: {model_name}")
                    
                    if "mistral" in model_name.lower() or "llama" in model_name.lower():
                        # Skip gated models
                        continue
                    
                    # Load tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )
                    
                    # Add pad token if not present
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Load model with basic configuration
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,  # Use float32 for compatibility
                        low_cpu_mem_usage=True
                    )
                    
                    # Create text generation pipeline
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=-1,  # Force CPU for stability
                        torch_dtype=torch.float32
                    )
                    
                    self.model_name = model_name
                    logging.info(f"Model loaded successfully: {model_name}")
                    return
                    
                except Exception as e:
                    logging.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            # If all models fail, use simple text processing
            logging.warning("All models failed to load, using simple text processing")
            self.model_type = "fallback"
            
        except Exception as e:
            logging.error(f"Failed to initialize any model: {e}")
            self.model_type = "fallback"
    
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
        """Generate response using the LLM or fallback method."""
        try:
            print(f"\n=== GENERATE_RESPONSE START ===")
            print(f"Model type: {self.model_type}")
            print(f"Pipeline available: {self.pipeline is not None}")
            
            if self.model_type == "fallback" or not self.pipeline:
                # Use simple rule-based processing
                print("🔄 Using simple text processing (fallback mode)")
                return self._simple_text_processing(prompt)
            
            print("🤖 Attempting to use HuggingFace model...")
            # Generate response using the model
            response = self.pipeline(
                prompt,
                max_new_tokens=min(self.max_tokens, 256),  # Limit tokens for smaller models
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
            
            print(f"✅ HuggingFace model generated response: {len(generated_text)} characters")
            
            # Check if response is empty and fallback if needed
            if not generated_text or generated_text.lower() in ['none', 'null', '']:
                print(f"❌ HuggingFace model returned empty response, falling back...")
                return self._simple_text_processing(prompt)
            
            return generated_text
            
        except Exception as e:
            print(f"❌ ERROR: HuggingFace model failed: {e}")
            print(f"🔄 Falling back to simple text processing...")
            # Fallback to simple processing
            return self._simple_text_processing(prompt)
    
    def _simple_text_processing(self, prompt: str) -> str:
        """Simple rule-based text processing as fallback."""
        try:
            print(f"\n=== SIMPLE TEXT PROCESSING START ===")
            print(f"Prompt length: {len(prompt)} characters")
            print(f"Prompt preview (first 500 chars): {prompt[:500]}")
            
            # Extract question and context from prompt
            if "QUESTION:" in prompt:
                question_part = prompt.split("QUESTION:")[-1].strip()
                question = question_part.split("\n")[0].strip()
            else:
                question = "coverage information"
            
            if "CONTEXT:" in prompt:
                context_part = prompt.split("CONTEXT:")[1]
                if "QUESTION:" in context_part:
                    context = context_part.split("QUESTION:")[0].strip()
                else:
                    context = context_part.strip()
            else:
                print(f"❌ ERROR: No CONTEXT found in prompt")
                print(f"Full prompt: {prompt}")
                return "Not found in document."
            
            # Debug logging
            print(f"✅ Question extracted: '{question}'")
            print(f"✅ Context length: {len(context)} characters")
            print(f"Context preview (first 500 chars): {context[:500]}")
            
            # Simple keyword matching for insurance questions
            question_lower = question.lower()
            context_lower = context.lower()
            
            # Split context into sentences for better matching
            sentences = context.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            print(f"Found {len(sentences)} sentences in context")
            
            # Coverage questions
            if any(word in question_lower for word in ["cover", "coverage", "include", "benefit"]):
                print("🔍 Processing coverage-related question")
                relevant_sentences = []
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(word in sentence_lower for word in ["cover", "coverage", "include", "benefit", "eligible", "insured", "policy"]):
                        relevant_sentences.append(sentence.strip())
                        print(f"  📄 Found coverage sentence: {sentence.strip()[:100]}...")
                
                print(f"Found {len(relevant_sentences)} relevant sentences for coverage")
                if relevant_sentences:
                    result = ". ".join(relevant_sentences[:3]) + "."
                    print(f"✅ Returning coverage answer: {result[:200]}...")
                    return result
            
            # Policy name questions
            if any(word in question_lower for word in ["name", "title", "policy name"]):
                print("Processing policy name question")
                relevant_sentences = []
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(word in sentence_lower for word in ["arogya", "sanjeevani", "policy", "national"]):
                        relevant_sentences.append(sentence.strip())
                        print(f"  Found policy name sentence: {sentence.strip()[:100]}...")
                
                print(f"Found {len(relevant_sentences)} relevant sentences for policy name")
                if relevant_sentences:
                    result = ". ".join(relevant_sentences[:2]) + "."
                    print(f"✅ Returning policy name answer: {result[:200]}...")
                    return result
            
            # General search - extract keywords from question and search in context
            question_words = [word for word in question_lower.split() if len(word) > 3 and word not in ["what", "when", "where", "does", "this", "policy", "and", "are", "the"]]
            print(f"Question keywords: {question_words}")
            
            if question_words:
                relevant_sentences = []
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    # Check if sentence contains any of the question keywords
                    matches = [kw for kw in question_words if kw in sentence_lower]
                    if matches:
                        relevant_sentences.append(sentence.strip())
                        print(f"  Found general sentence with keywords {matches}: {sentence.strip()[:100]}...")
                
                print(f"Found {len(relevant_sentences)} relevant sentences for general keywords")
                if relevant_sentences:
                    result = ". ".join(relevant_sentences[:3]) + "."
                    print(f"✅ Returning general answer: {result[:200]}...")
                    return result
            
            # If no specific matching, return first few sentences of context
            if sentences and len(sentences) > 0:
                print("⚠️ No specific matches, returning first sentences")
                result = ". ".join(sentences[:2]) + "."
                print(f"🔄 Returning fallback answer: {result[:200]}...")
                return result
            
            print(f"❌ ERROR: No sentences found in context")
            print(f"Context was: '{context}'")
            return "Information not found in the provided document context."
        except Exception as e:
            print(f"❌ ERROR: Simple text processing failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return "Error processing question."
    
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
            print(f"\n=== LLM ANSWER_QUESTION DEBUG ===")
            print(f"Question: {question}")
            print(f"Retrieved chunks count: {len(retrieved_chunks)}")
            print(f"Context type: {context_type}")
            
            for i, chunk in enumerate(retrieved_chunks[:3]):
                print(f"Chunk {i+1} type: {type(chunk)}")
                print(f"Chunk {i+1} preview: {str(chunk)[:100]}...")
            
            if not retrieved_chunks:
                print("❌ No chunks received!")
                return "Not found in document."
            
            # Create prompt
            prompt = self.create_prompt(question, retrieved_chunks, context_type)
            print(f"Prompt created, length: {len(prompt)}")
            
            # Generate response
            response = self.generate_response(prompt)
            print(f"Response generated (length {len(response)}): '{response}'")
            print(f"Response type: {type(response)}")
            
            # Validate response
            if not response or response.lower().strip() in ["", "not found in document", "not found in document."]:
                print(f"❌ Empty or invalid response! Response: '{response}', lower stripped: '{response.lower().strip() if response else 'None'}'")
                return "Not found in document."
            
            return response
            
        except Exception as e:
            print(f"❌ ERROR in answer_question: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
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
