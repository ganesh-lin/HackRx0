import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from llm_manager import LLMManager
import time

def test_local_physiotherapy():
    document_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    question = "Is physiotherapy covered, and if so, what is the waiting period for claims under it?"
    
    print("🔧 LOCAL PHYSIOTHERAPY TEST")
    print("=" * 60)
    
    try:
        # Process document
        print("1. Processing document...")
        processor = DocumentProcessor()
        document_text = processor.process_document(document_url)
        print(f"   ✅ Document processed: {len(document_text)} characters")
        
        # Chunk document
        print("2. Chunking document...")
        chunks = processor.chunk_text(document_text)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Find chunks containing physiotherapy
        print("3. Searching for physiotherapy in chunks...")
        physio_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"].lower()
            if "physiotherapy" in chunk_text or "prescribed" in chunk_text:
                physio_chunks.append((i, chunk))
                print(f"   ✅ Chunk {i} contains physiotherapy:")
                print(f"      {chunk['text'][:300]}...")
        
        print(f"   Found {len(physio_chunks)} chunks with physiotherapy content")
        
        # Test embedding search
        print("4. Testing embedding search...")
        embedding_manager = EmbeddingManager(use_pinecone=True)
        
        # Store embeddings
        document_id = "physio_test"
        namespace = f"doc_{document_id}"
        success = embedding_manager.store_embeddings(chunks, document_id, namespace)
        print(f"   ✅ Embeddings stored: {success}")
        
        # Search for relevant chunks
        relevant_results = embedding_manager.search_similar(question, namespace, document_id)
        print(f"   ✅ Found {len(relevant_results)} relevant chunks by embedding search")
        
        for i, result in enumerate(relevant_results):
            print(f"      Chunk {i+1} (score: {result['score']:.3f}):")
            result_text = result['text'].lower()
            if "physiotherapy" in result_text or "prescribed" in result_text:
                print(f"         ✅ Contains physiotherapy content!")
            else:
                print(f"         ❌ No physiotherapy content")
            print(f"         {result['text'][:200]}...")
        
        # Test LLM
        print("5. Testing LLM...")
        llm_manager = LLMManager()
        relevant_chunks = [result["text"] for result in relevant_results]
        
        if not relevant_chunks and physio_chunks:
            print("   🔄 No relevant chunks from search, using manually found chunks...")
            relevant_chunks = [chunk[1]["text"] for chunk in physio_chunks[:3]]
        
        answer = llm_manager.answer_question(question, relevant_chunks, "insurance")
        print(f"   ✅ LLM Answer: {answer}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_local_physiotherapy()
