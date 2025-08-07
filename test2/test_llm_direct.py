#!/usr/bin/env python3
"""
Direct test of LLM manager to debug the text processing issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from llm_manager import LLMManager

def test_llm_directly():
    print("Testing LLM Manager directly...")
    
    # Initialize LLM manager
    llm_manager = LLMManager()
    
    # Sample question and context
    question = "What is coverage?"
    
    # Sample context chunks (from a typical insurance document)
    context_chunks = [
        "This policy provides comprehensive medical coverage for the insured person including hospitalization, surgery, and outpatient treatment.",
        "Coverage includes pre and post hospitalization expenses up to the sum insured amount.",
        "The policy covers medical expenses incurred due to illness or accident during the policy period.",
        "Emergency ambulance charges and day care procedures are also covered under this policy."
    ]
    
    print(f"\nQuestion: {question}")
    print(f"Context chunks: {len(context_chunks)}")
    for i, chunk in enumerate(context_chunks):
        print(f"  Chunk {i+1}: {chunk[:100]}...")
    
    # Test the answer_question method
    print("\n" + "="*50)
    print("Testing answer_question method...")
    
    try:
        answer = llm_manager.answer_question(question, context_chunks)
        print(f"\nAnswer: {answer}")
        
        if answer == "Not found in document.":
            print("\n❌ ERROR: Getting 'Not found in document' response!")
            
            # Let's test the internal methods directly
            print("\n" + "="*50)
            print("Testing create_prompt method...")
            prompt = llm_manager.create_prompt(question, context_chunks)
            print(f"Generated prompt length: {len(prompt)} characters")
            print(f"Prompt preview:\n{prompt[:500]}...")
            
            print("\n" + "="*50)
            print("Testing generate_response method...")
            response = llm_manager.generate_response(prompt)
            print(f"Generated response: {response}")
            
        else:
            print("✅ SUCCESS: Got a meaningful response!")
            
    except Exception as e:
        print(f"❌ ERROR: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_directly()
