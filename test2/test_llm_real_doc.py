#!/usr/bin/env python3
"""
Test the LLM processing with actual document content
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_llm_with_real_document():
    print("Testing LLM with real document content...")
    
    # Test document URL from the test_simple.json
    document_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    question = "What is coverage?"
    
    try:
        # Get the document content
        print("1. Getting document content...")
        from document_processor import DocumentProcessor
        
        doc_processor = DocumentProcessor()
        text_content = doc_processor.process_document(document_url)
        
        print(f"   ✅ Document processed: {len(text_content)} characters")
        print(f"   First 500 chars: {text_content[:500]}...")
        
        # Manually create chunks by splitting the text
        print("\n2. Creating manual chunks...")
        
        # Split into chunks of reasonable size (similar to what chunking would do)
        chunk_size = 1000
        overlap = 100
        manual_chunks = []
        
        for i in range(0, len(text_content), chunk_size - overlap):
            chunk = text_content[i:i + chunk_size]
            if len(chunk.strip()) > 50:  # Only add meaningful chunks
                manual_chunks.append(chunk.strip())
        
        print(f"   ✅ Created {len(manual_chunks)} manual chunks")
        
        # Look for chunks that contain coverage information
        coverage_chunks = []
        for chunk in manual_chunks:
            chunk_lower = chunk.lower()
            if any(word in chunk_lower for word in ["coverage", "cover", "benefit", "policy", "insured"]):
                coverage_chunks.append(chunk)
        
        print(f"   ✅ Found {len(coverage_chunks)} chunks with coverage keywords")
        
        # Test with relevant chunks
        print("\n3. Testing LLM with relevant chunks...")
        
        # Use the first few relevant chunks
        test_chunks = coverage_chunks[:5] if coverage_chunks else manual_chunks[:5]
        
        for i, chunk in enumerate(test_chunks):
            print(f"   Chunk {i+1}: {chunk[:100]}...")
        
        # Now test the LLM
        from llm_manager import LLMManager
        
        llm_manager = LLMManager()
        answer = llm_manager.answer_question(question, test_chunks)
        
        print(f"\n4. LLM Answer: {answer}")
        
        if answer == "Not found in document.":
            print("\n❌ STILL GETTING 'Not found in document'")
            print("Let's debug the prompt:")
            
            prompt = llm_manager.create_prompt(question, test_chunks)
            print(f"Prompt length: {len(prompt)}")
            print("Prompt content:")
            print("="*50)
            print(prompt)
            print("="*50)
            
            print("\nTesting simple text processing directly:")
            response = llm_manager._simple_text_processing(prompt)
            print(f"Direct simple processing result: {response}")
            
        else:
            print("\n✅ SUCCESS: Got a meaningful answer!")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_with_real_document()
