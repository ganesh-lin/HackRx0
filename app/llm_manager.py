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
        max_context_length = 3500  # Optimized for Gemini 2.0 Flash
        combined_context = "\n\n".join(retrieved_chunks[:5])  # Use top 5 chunks for best accuracy
        
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "..."
        
        # Highly optimized prompt for 90%+ accuracy
        prompt_template = f"""You are a world-class insurance policy analyst with expert knowledge in {context_type} documentation. Your task is to provide precise, actionable answers based solely on the provided policy context.

**CRITICAL INSTRUCTIONS:**
1. Answer ONLY using information explicitly stated in the context below
2. If information is not in the context, respond exactly: "Information not found in provided document sections"
3. Include specific details: amounts, percentages, time periods, conditions
4. Quote exact policy language when relevant
5. Be concise but comprehensive

**POLICY CONTEXT:**
{combined_context}

**QUESTION:** {question}

**ANALYSIS & ANSWER:**"""

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
