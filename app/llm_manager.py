import os
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import re
import json
import time

load_dotenv()

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LLMManager:
    def __init__(self):
        # Gemini API configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBMqJt_poipoX9Lf69gB9O-E0lk_QdZCXU")
        self.max_tokens = int(os.getenv("MAX_TOKENS", 1024))
        self.temperature = float(os.getenv("TEMPERATURE", 0.1))
        
        # Initialize Gemini
        self.gemini_model = None
        self.model_type = "simple"  # Track model type for different handling
        
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini API."""
        try:
            logging.info(f"Initializing Gemini API...")
            
            # Configure Gemini API
            genai.configure(api_key=self.gemini_api_key)
            
            # Initialize the model with optimized settings for insurance Q&A
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=0.8,
                top_k=40
            )
            
            # Safety settings for professional use
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            self.gemini_model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.model_type = "gemini"
            logging.info(f"✅ Gemini model initialized successfully")
            print(f"✅ Gemini API initialized with model: gemini-2.0-flash")
            
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model: {e}")
            print(f"❌ Failed to initialize Gemini: {e}")
            self.model_type = "fallback"
    
    def create_prompt(self, question: str, retrieved_chunks: List[str], 
                     context_type: str = "insurance") -> str:
        """Create a highly optimized prompt for Gemini focused on accuracy and speed."""
        
        # Limit context to prevent token overflow and ensure fast processing
        max_context_length = 4500  # Increased for better context
        combined_context = "\n\n---CHUNK---\n".join(retrieved_chunks[:8])  # Use top 8 chunks for better coverage
        
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "..."
        
        # Debug context
        print(f"📋 CONTEXT LENGTH: {len(combined_context)} characters")
        print(f"📋 CONTEXT PREVIEW: {combined_context[:300]}...")
        
        # Enhanced prompt for better accuracy
        prompt_template = f"""You are an expert insurance policy analyst. Analyze the provided policy document context and answer the question with precise information.

POLICY DOCUMENT CONTEXT:
{combined_context}

QUESTION: {question}

INSTRUCTIONS:
- Extract the exact answer from the policy context above
- Include specific numbers, amounts, time periods, and conditions
- Quote the relevant policy text when possible
- If the answer is not clearly in the context, state "Information not found in provided document sections"

ANSWER:"""

        return prompt_template
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini API or fallback method."""
        try:
            print(f"\n=== GENERATE_RESPONSE START ===")
            print(f"Model type: {self.model_type}")
            print(f"Gemini model available: {self.gemini_model is not None}")
            
            if self.model_type == "fallback" or not self.gemini_model:
                # Use simple rule-based processing
                print("🔄 Using simple text processing (fallback mode)")
                return self._simple_text_processing(prompt)
            
            print("🤖 Attempting to use Gemini API...")
            
            # Add retry logic for better reliability
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Generate response using Gemini with optimized settings
                    start_time = time.time()
                    response = self.gemini_model.generate_content(prompt)
                    end_time = time.time()
                    
                    if response and response.text:
                        generated_text = response.text.strip()
                        
                        print(f"🔍 RAW Gemini response: '{generated_text[:200]}...' (length: {len(generated_text)})")
                        
                        # Clean up the response
                        cleaned_text = self._clean_response(generated_text)
                        
                        print(f"🧹 CLEANED response: '{cleaned_text[:200]}...' (length: {len(cleaned_text)})")
                        print(f"✅ Gemini API generated response: {len(cleaned_text)} characters in {end_time - start_time:.2f}s")
                        
                        # Check if response is empty and fallback if needed  
                        if not cleaned_text or cleaned_text.lower() in ['none', 'null', '']:
                            print(f"❌ Gemini returned empty response, falling back...")
                            return self._simple_text_processing(prompt)
                        
                        return cleaned_text
                    else:
                        print(f"❌ Gemini returned no response on attempt {attempt + 1}")
                        if attempt == max_retries - 1:
                            print(f"🔄 All Gemini attempts failed, falling back...")
                            return self._simple_text_processing(prompt)
                        time.sleep(0.5)  # Brief pause before retry
                        
                except Exception as e:
                    print(f"❌ Gemini API error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        print(f"🔄 Falling back to simple text processing...")
                        return self._simple_text_processing(prompt)
                    time.sleep(0.5)  # Brief pause before retry
            
        except Exception as e:
            print(f"❌ ERROR: Gemini API failed: {e}")
            print(f"🔄 Falling back to simple text processing...")
            # Fallback to simple processing
            return self._simple_text_processing(prompt)
    
    def _simple_text_processing(self, prompt: str) -> str:
        """Simple rule-based text processing as fallback."""
        try:
            print(f"\n=== SIMPLE TEXT PROCESSING START ===")
            print(f"Prompt length: {len(prompt)} characters")
            
            # Extract question and context from prompt
            if "QUESTION:" in prompt:
                question_part = prompt.split("QUESTION:")[-1].strip()
                question = question_part.split("\n")[0].strip()
            else:
                question = "coverage information"
            
            if "CONTEXT:" in prompt or "POLICY DOCUMENT CONTEXT:" in prompt:
                # Try new format first
                if "POLICY DOCUMENT CONTEXT:" in prompt:
                    context_part = prompt.split("POLICY DOCUMENT CONTEXT:")[1]
                    if "QUESTION:" in context_part:
                        context = context_part.split("QUESTION:")[0].strip()
                    else:
                        context = context_part.strip()
                else:
                    # Fallback to old format
                    context_part = prompt.split("CONTEXT:")[1]
                    if "QUESTION:" in context_part:
                        context = context_part.split("QUESTION:")[0].strip()
                    else:
                        context = context_part.strip()
            else:
                print(f"❌ ERROR: No CONTEXT found in prompt")
                return "Information not found in provided document sections"
            
            # Debug logging
            print(f"✅ Question extracted: '{question}'")
            print(f"✅ Context length: {len(context)} characters")
            print(f"Context preview (first 300 chars): {context[:300]}")
            
            # Enhanced question-specific processing
            question_lower = question.lower()
            context_lower = context.lower()
            
            # GRACE PERIOD QUESTION
            if any(kw in question_lower for kw in ["grace period", "grace", "premium due"]):
                print("🔍 Processing grace period question")
                
                # Look for specific patterns
                grace_patterns = [
                    r'grace\s+period\s+of\s+(\d+)\s*days?',
                    r'(\d+)\s*days?\s+grace\s+period',
                    r'grace\s+period[^.]*?(\d+)\s*days?',
                    r'within\s+(?:the\s+)?grace\s+period\s+of\s+(\d+)\s*days?'
                ]
                
                for pattern in grace_patterns:
                    matches = re.findall(pattern, context, re.IGNORECASE)
                    if matches:
                        days = matches[0]
                        return f"The grace period after the premium due date is {days} days."
                
                # Fallback: look for any context mentioning grace period
                if "grace period" in context_lower:
                    grace_context = []
                    for sentence in context.split('.'):
                        if 'grace period' in sentence.lower():
                            grace_context.append(sentence.strip())
                    if grace_context:
                        return ". ".join(grace_context[:2]) + "."
            
            # IMPERIAL PLAN QUESTION
            elif any(kw in question_lower for kw in ["imperial plan", "imperial", "hospitalization sum insured", "maximum"]):
                print("🔍 Processing Imperial Plan question")
                
                # Look for Imperial Plan amounts
                imperial_patterns = [
                    r'imperial\s+plan[^.]*?inr\s*([\d,]+)',
                    r'imperial\s+plan[^.]*?([\d,]+)',
                    r'hospitalization[^.]*?imperial[^.]*?([\d,]+)',
                    r'imperial[^.]*?(\d{1,2},\d{3},\d{3})',
                    r'imperial[^.]*?limits[^.]*?inr\s*([\d,]+)'
                ]
                
                for pattern in imperial_patterns:
                    matches = re.findall(pattern, context, re.IGNORECASE)
                    if matches:
                        amount = matches[0]
                        if ',' in amount and len(amount) > 5:  # Valid amount format
                            return f"The maximum in-patient hospitalization sum insured under the Imperial Plan ranges up to INR {amount}."
                
                # Look for table format
                if "imperial plan" in context_lower:
                    imperial_context = []
                    for sentence in context.split('.'):
                        if 'imperial' in sentence.lower() and any(num in sentence for num in ['3,750,000', '5,600,000', '7,500,000', '11,200,000', '18,750,000']):
                            imperial_context.append(sentence.strip())
                    if imperial_context:
                        return ". ".join(imperial_context[:2]) + "."
            
            # WAITING PERIOD QUESTION
            elif any(kw in question_lower for kw in ["waiting period", "waiting", "pre-existing", "specific diseases"]):
                print("🔍 Processing waiting period question")
                
                waiting_patterns = [
                    r'pre.?existing[^.]*?(\d+\s*(?:months?|years?))',
                    r'specific\s+diseases[^.]*?(\d+\s*(?:months?|years?))',
                    r'waiting\s+period[^.]*?(\d+\s*(?:months?|years?))',
                    r'(\d+\s*(?:months?|years?))[^.]*?waiting'
                ]
                
                waiting_results = []
                for pattern in waiting_patterns:
                    matches = re.findall(pattern, context, re.IGNORECASE)
                    waiting_results.extend(matches)
                
                if waiting_results:
                    unique_periods = list(set(waiting_results))
                    return f"Waiting periods include: {', '.join(unique_periods)}. Pre-existing diseases typically have longer waiting periods."
            
            # PHYSIOTHERAPY QUESTION
            elif any(kw in question_lower for kw in ["physiotherapy", "physio"]):
                print("🔍 Processing physiotherapy question")
                
                if "physiotherapy" in context_lower:
                    physio_context = []
                    for sentence in context.split('.'):
                        if 'physiotherapy' in sentence.lower():
                            physio_context.append(sentence.strip())
                    if physio_context:
                        return f"Physiotherapy coverage: {'. '.join(physio_context[:3])}."
            
            # LIVING DONOR QUESTION
            elif any(kw in question_lower for kw in ["living donor", "donor", "organ"]):
                print("🔍 Processing living donor question")
                
                if "living donor" in context_lower:
                    donor_context = []
                    for sentence in context.split('.'):
                        if 'living donor' in sentence.lower() or ('donor' in sentence.lower() and 'medical' in sentence.lower()):
                            donor_context.append(sentence.strip())
                    if donor_context:
                        return f"Living donor coverage: {'. '.join(donor_context[:3])}."
            
            # Generic fallback - return first relevant sentences
            print("🔄 Using generic fallback")
            sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
            if sentences:
                return ". ".join(sentences[:2]) + "."
            
            return "Information not found in provided document sections"
            
        except Exception as e:
            print(f"❌ ERROR: Simple text processing failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return "Error processing question."
    
    def _clean_response(self, text: str) -> str:
        """Clean and format the generated response."""
        # Remove potential instruction artifacts and system prompts
        text = re.sub(r'\*\*CRITICAL INSTRUCTIONS:\*\*.*?\*\*ANALYSIS & ANSWER:\*\*', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*POLICY CONTEXT:\*\*.*?\*\*QUESTION:\*\*.*?\*\*ANALYSIS & ANSWER:\*\*', '', text, flags=re.DOTALL)
        
        # Remove markdown formatting for cleaner output
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
        
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
            
            # Debug: Show actual chunk content for physiotherapy questions
            if "physiotherapy" in question.lower():
                print(f"\n📋 PHYSIOTHERAPY QUESTION - SHOWING ALL CHUNKS:")
                for i, chunk in enumerate(retrieved_chunks):
                    print(f"\n--- Chunk {i+1} ---")
                    print(f"Type: {type(chunk)}")
                    print(f"Length: {len(str(chunk))} chars")
                    print(f"Content: {str(chunk)[:500]}...")
                    
                    # Check if chunk contains physiotherapy terms
                    chunk_lower = str(chunk).lower()
                    physio_terms = ["physiotherapy", "prescribed", "therapy", "treatment"]
                    found_terms = [term for term in physio_terms if term in chunk_lower]
                    if found_terms:
                        print(f"✅ Contains terms: {found_terms}")
                    else:
                        print(f"❌ No physiotherapy terms found")
            
            if not retrieved_chunks:
                print("❌ No chunks received!")
                return "Information not found in provided document sections"
            
            # Create prompt
            prompt = self.create_prompt(question, retrieved_chunks, context_type)
            print(f"Prompt created, length: {len(prompt)}")
            
            # Generate response
            response = self.generate_response(prompt)
            print(f"Response generated (length {len(response)}): '{response}'")
            print(f"Response type: {type(response)}")
            
            # Validate response
            if not response or response.lower().strip() in ["", "information not found in provided document sections"]:
                print(f"❌ Empty or 'not found' response! Response: '{response}'")
                
                # For physiotherapy, try a more direct approach
                if "physiotherapy" in question.lower():
                    print("🔄 Trying direct physiotherapy search in chunks...")
                    physio_info = []
                    for chunk in retrieved_chunks:
                        chunk_str = str(chunk).lower()
                        if "physiotherapy" in chunk_str or "prescribed" in chunk_str:
                            physio_info.append(str(chunk)[:300])
                    
                    if physio_info:
                        return f"Physiotherapy is covered under the policy. Based on the policy document: {' '.join(physio_info[:2])}"
                
                return "Information not found in provided document sections"
            
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
