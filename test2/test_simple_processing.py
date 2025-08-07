#!/usr/bin/env python3
"""
Direct test of simple text processing method
"""

# Create a mock prompt like the LLM manager would create
def test_simple_text_processing():
    print("Testing simple text processing directly...")
    
    question = "What is coverage?"
    context_chunks = [
        "This policy provides comprehensive medical coverage for the insured person including hospitalization, surgery, and outpatient treatment.",
        "Coverage includes pre and post hospitalization expenses up to the sum insured amount.",
        "The policy covers medical expenses incurred due to illness or accident during the policy period.",
        "Emergency ambulance charges and day care procedures are also covered under this policy."
    ]
    
    # Create the prompt exactly as the LLM manager would
    max_context_length = 3000
    combined_context = "\n\n".join(context_chunks)
    
    if len(combined_context) > max_context_length:
        combined_context = combined_context[:max_context_length] + "..."
    
    prompt_template = f"""<s>[INST] You are a highly skilled legal and insurance assistant specializing in insurance policy analysis.

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

    print(f"Question: {question}")
    print(f"Context chunks: {len(context_chunks)}")
    print(f"Combined context length: {len(combined_context)}")
    print(f"Prompt length: {len(prompt_template)}")
    
    # Now simulate the simple text processing
    print("\n" + "="*50)
    print("STARTING SIMPLE TEXT PROCESSING")
    print("="*50)
    
    try:
        prompt = prompt_template
        print(f"Prompt length: {len(prompt)} characters")
        
        # Extract question and context from prompt
        if "QUESTION:" in prompt:
            question_part = prompt.split("QUESTION:")[-1].strip()
            question_extracted = question_part.split("\n")[0].strip()
        else:
            question_extracted = "coverage information"
        
        if "CONTEXT:" in prompt:
            context_part = prompt.split("CONTEXT:")[1]
            if "QUESTION:" in context_part:
                context_extracted = context_part.split("QUESTION:")[0].strip()
            else:
                context_extracted = context_part.strip()
        else:
            print(f"❌ ERROR: No CONTEXT found in prompt")
            print(f"Prompt preview: {prompt[:500]}...")
            return "Not found in document."
        
        # Debug logging
        print(f"✅ Question extracted: {question_extracted}")
        print(f"✅ Context length: {len(context_extracted)} characters")
        print(f"Context preview: {context_extracted[:300]}...")
        
        # Simple keyword matching for insurance questions
        question_lower = question_extracted.lower()
        context_lower = context_extracted.lower()
        
        # Split context into sentences for better matching
        sentences = context_extracted.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        print(f"Found {len(sentences)} sentences in context")
        
        # Coverage questions
        if any(word in question_lower for word in ["cover", "coverage", "include", "benefit"]):
            print("Processing coverage-related question")
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in ["cover", "coverage", "include", "benefit", "eligible", "insured", "policy"]):
                    relevant_sentences.append(sentence.strip())
                    print(f"  Found coverage sentence: {sentence.strip()[:100]}...")
            
            print(f"Found {len(relevant_sentences)} relevant sentences for coverage")
            if relevant_sentences:
                result = ". ".join(relevant_sentences[:3]) + "."
                print(f"✅ Returning coverage answer: {result[:200]}...")
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
            print("❌ No specific matches, returning first sentences")
            result = ". ".join(sentences[:2]) + "."
            print(f"⚠️ Returning fallback answer: {result[:200]}...")
            return result
        
        print(f"❌ ERROR: No sentences found in context")
        return "Information not found in the provided document context."
        
    except Exception as e:
        print(f"❌ ERROR: Simple text processing failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return "Error processing question."

if __name__ == "__main__":
    result = test_simple_text_processing()
    print(f"\n\nFINAL RESULT: {result}")
