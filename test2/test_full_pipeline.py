#!/usr/bin/env python3
"""
Test document processing and chunk retrieval like the API does
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_document_processing():
    print("Testing document processing pipeline...")
    
    # Test document URL from the test_simple.json
    document_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    question = "What is coverage?"
    
    try:
        # Test document processor (using the API method)
        print("1. Testing document processor...")
        from document_processor import DocumentProcessor
        
        doc_processor = DocumentProcessor()
        text_content = doc_processor.process_document(document_url)
        
        print(f"   ✅ Document processed: {len(text_content)} characters")
        print(f"   Preview: {text_content[:300]}...")
        
        # Test document chunking (using the API method)
        print("\n2. Testing document chunking...")
        chunks = doc_processor.chunk_text(text_content)
        print(f"   ✅ Document chunked: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            if isinstance(chunk, dict):
                chunk_text = chunk.get('content', str(chunk))
            else:
                chunk_text = str(chunk)
            print(f"   Chunk {i+1}: {chunk_text[:100]}...")
        
        # Test embedding manager
        print("\n3. Testing embedding manager...")
        from embedding_manager import EmbeddingManager
        
        embedding_manager = EmbeddingManager()
        
        # Store embeddings
        print("   Storing embeddings...")
        doc_id = embedding_manager.store_document_embeddings(chunks, {"url": document_url})
        print(f"   ✅ Document stored with ID: {doc_id}")
        
        # Test retrieval
        print("\n4. Testing chunk retrieval...")
        retrieved_chunks = embedding_manager.retrieve_relevant_chunks(question, top_k=5)
        print(f"   ✅ Retrieved chunks: {len(retrieved_chunks)}")
        
        for i, chunk in enumerate(retrieved_chunks):
            print(f"   Retrieved chunk {i+1}: {chunk[:100]}...")
        
        # Test LLM processing with actual chunks
        print("\n5. Testing LLM with retrieved chunks...")
        from llm_manager import LLMManager
        
        llm_manager = LLMManager()
        
        # Extract just the text content from retrieved chunks
        chunk_texts = []
        for chunk in retrieved_chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                chunk_texts.append(chunk['content'])
            elif isinstance(chunk, str):
                chunk_texts.append(chunk)
            else:
                chunk_texts.append(str(chunk))
        
        print(f"   Chunk texts for LLM: {len(chunk_texts)}")
        for i, text in enumerate(chunk_texts[:3]):
            print(f"   Text {i+1}: {text[:100]}...")
        
        answer = llm_manager.answer_question(question, chunk_texts)
        print(f"   ✅ LLM Answer: {answer}")
        
        if answer == "Not found in document.":
            print("\n❌ ISSUE FOUND: LLM returning 'Not found in document'")
            print("Let's debug the prompt creation...")
            
            prompt = llm_manager.create_prompt(question, chunk_texts)
            print(f"Prompt length: {len(prompt)}")
            print(f"Prompt preview:\n{prompt[:1000]}...")
            
        else:
            print("\n✅ SUCCESS: LLM processing working correctly!")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_document_processing()
