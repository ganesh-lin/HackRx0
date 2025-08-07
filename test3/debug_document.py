import requests
import json
from app.document_processor import DocumentProcessor
from app.embedding_manager import EmbeddingManager
from app.llm_manager import LLMManager

# Test the document processing pipeline step by step
def debug_document_processing():
    document_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    question = "What is the grace period after the premium due date?"
    
    print("🔧 DEBUGGING DOCUMENT PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Document Processing
    print("\n1. DOCUMENT PROCESSING")
    print("-" * 30)
    
    try:
        processor = DocumentProcessor()
        document_text = processor.process_document(document_url)
        
        print(f"✅ Document downloaded and processed")
        print(f"📄 Document length: {len(document_text)} characters")
        print(f"📝 First 500 characters: {document_text[:500]}")
        print(f"📝 Sample text from middle: {document_text[len(document_text)//2:len(document_text)//2+500]}")
        
        # Check for specific keywords that should be in the document
        keywords = ["grace period", "premium", "due date", "Imperial Plan", "waiting period", "physiotherapy", "living donor"]
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in document_text.lower():
                found_keywords.append(keyword)
        
        print(f"🔍 Keywords found: {found_keywords}")
        print(f"🔍 Keywords missing: {[k for k in keywords if k not in found_keywords]}")
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        return
    
    # Step 2: Chunking
    print("\n2. CHUNKING")
    print("-" * 30)
    
    try:
        chunks = processor.chunk_text(document_text)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show first few chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n📦 Chunk {i+1}:")
            print(f"   Length: {len(chunk['text'])} chars")
            print(f"   Tokens: {chunk.get('tokens', 'N/A')}")
            print(f"   Text preview: {chunk['text'][:200]}...")
            
            # Check if this chunk might be relevant
            chunk_lower = chunk['text'].lower()
            relevant_keywords = [kw for kw in keywords if kw.lower() in chunk_lower]
            if relevant_keywords:
                print(f"   🎯 Contains keywords: {relevant_keywords}")
    
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        return
    
    # Step 3: Embedding and Search
    print("\n3. EMBEDDING AND SEARCH")
    print("-" * 30)
    
    try:
        embedding_manager = EmbeddingManager(use_pinecone=True)
        
        # Store embeddings
        document_id = "debug_test"
        namespace = f"doc_{document_id}"
        success = embedding_manager.store_embeddings(chunks, document_id, namespace)
        print(f"✅ Embeddings stored: {success}")
        
        # Search for relevant chunks
        relevant_results = embedding_manager.search_similar(question, namespace, document_id)
        print(f"🔍 Found {len(relevant_results)} relevant chunks for question: '{question}'")
        
        for i, result in enumerate(relevant_results[:3]):
            print(f"\n🎯 Relevant chunk {i+1} (score: {result['score']:.3f}):")
            print(f"   Text: {result['text'][:300]}...")
            
            # Check for answer-related keywords
            text_lower = result['text'].lower()
            if any(kw in text_lower for kw in ["grace", "period", "premium", "due"]):
                print(f"   ✅ Contains answer-related keywords!")
    
    except Exception as e:
        print(f"❌ Embedding and search failed: {e}")
        return
    
    # Step 4: LLM Processing
    print("\n4. LLM PROCESSING")
    print("-" * 30)
    
    try:
        llm_manager = LLMManager()
        
        if relevant_results:
            relevant_chunks = [result["text"] for result in relevant_results]
            print(f"📤 Sending {len(relevant_chunks)} chunks to LLM")
            
            # Show what we're sending to LLM
            print("\n📋 LLM Input Preview:")
            for i, chunk in enumerate(relevant_chunks[:2]):
                print(f"   Chunk {i+1}: {chunk[:200]}...")
            
            answer = llm_manager.answer_question(question, relevant_chunks, "insurance")
            print(f"\n🤖 LLM Response: {answer}")
            
            if "Information not found" in answer:
                print("❌ LLM couldn't find information in the provided chunks!")
                
                # Let's test with a more direct approach
                print("\n🔄 Testing with manual search...")
                manual_relevant = []
                for chunk in chunks:
                    chunk_lower = chunk['text'].lower()
                    if any(kw in chunk_lower for kw in ["grace", "period", "premium", "due"]):
                        manual_relevant.append(chunk['text'])
                
                if manual_relevant:
                    print(f"📋 Found {len(manual_relevant)} manually relevant chunks")
                    manual_answer = llm_manager.answer_question(question, manual_relevant[:3], "insurance")
                    print(f"🤖 Manual search result: {manual_answer}")
                else:
                    print("❌ No manually relevant chunks found either!")
        else:
            print("❌ No relevant chunks found by embedding search")
    
    except Exception as e:
        print(f"❌ LLM processing failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🏁 DEBUG COMPLETE")

if __name__ == "__main__":
    debug_document_processing()
