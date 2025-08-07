import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from document_processor import DocumentProcessor

def find_detailed_physiotherapy():
    document_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    print("🔍 FINDING DETAILED PHYSIOTHERAPY CHUNKS")
    print("=" * 60)
    
    try:
        # Process document
        processor = DocumentProcessor()
        document_text = processor.process_document(document_url)
        chunks = processor.chunk_text(document_text)
        
        # Look for specific physiotherapy coverage details
        search_terms = [
            "prescribed physiotherapy",
            "physiotherapy refers to treatment",
            "physiotherapy coverage",
            "out-patient basis for illness",
            "dialysis, chemotherapy, radiotherapy, physiotherapy"
        ]
        
        for term in search_terms:
            print(f"\\n🔍 Searching for: '{term}'")
            found_chunks = []
            
            for i, chunk in enumerate(chunks):
                if term.lower() in chunk["text"].lower():
                    found_chunks.append((i, chunk))
            
            if found_chunks:
                print(f"✅ Found in {len(found_chunks)} chunks:")
                for chunk_id, chunk in found_chunks:
                    print(f"\\n--- Chunk {chunk_id} ---")
                    print(f"{chunk['text'][:800]}...")
            else:
                print("❌ Not found")
        
        # Look for the definition section specifically
        print(f"\\n🔍 Looking for definition/benefits section...")
        definition_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"].lower()
            if ("39. prescribed physiotherapy" in chunk_text or 
                "prescribed physiotherapy refers to treatment" in chunk_text):
                definition_chunks.append((i, chunk))
        
        if definition_chunks:
            print(f"✅ Found {len(definition_chunks)} definition chunks:")
            for chunk_id, chunk in definition_chunks:
                print(f"\\n--- Definition Chunk {chunk_id} ---")
                print(f"{chunk['text']}")
        else:
            print("❌ No definition chunks found")
            
        # Look for coverage/benefits section
        print(f"\\n🔍 Looking for coverage/benefits mentioning physiotherapy...")
        coverage_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"].lower()
            if ("expenses incurred towards prescribed physiotherapy" in chunk_text or
                "prescribed physiotherapy taken on out" in chunk_text):
                coverage_chunks.append((i, chunk))
        
        if coverage_chunks:
            print(f"✅ Found {len(coverage_chunks)} coverage chunks:")
            for chunk_id, chunk in coverage_chunks:
                print(f"\\n--- Coverage Chunk {chunk_id} ---")
                print(f"{chunk['text']}")
        else:
            print("❌ No coverage chunks found")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    find_detailed_physiotherapy()
