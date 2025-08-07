import requests
import json

# Test just the physiotherapy question
def test_physiotherapy_only():
    url = "http://localhost:8000/api/v1/hackrx/run"
    headers = {
        "Authorization": "Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1",
        "Content-Type": "application/json"
    }
    
    document_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    question = "Is physiotherapy covered, and if so, what is the waiting period for claims under it?"
    
    print("🔧 TESTING PHYSIOTHERAPY QUESTION")
    print("=" * 60)
    print(f"Question: {question}")
    print("-" * 60)
    
    payload = {
        "documents": document_url,
        "questions": [question]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answers", ["No answer"])[0]
            processing_time = result.get("metadata", {}).get("processing_time", 0)
            confidence = result.get("metadata", {}).get("confidence_scores", [0])[0]
            
            print(f"✅ Status: Success")
            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            print(f"🎯 Confidence: {confidence:.2f}")
            print(f"💬 Answer: {answer}")
            
            # Show metadata details
            metadata = result.get("metadata", {})
            question_details = metadata.get("question_details", [])
            if question_details:
                details = question_details[0]
                print(f"🔍 Relevant chunks found: {details.get('relevant_chunks_count', 0)}")
                print(f"📊 Query type: {details.get('query_type', 'unknown')}")
                
        else:
            print(f"❌ Status: Error {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_physiotherapy_only()
