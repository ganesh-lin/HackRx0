import requests
import json

# API configuration
url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1",
    "Content-Type": "application/json"
}

# Test different types of questions
test_cases = [
    {
        "name": "Easy – Grace Period",
        "question": "What is the grace period after the premium due date?"
    },
    {
        "name": "Easy – Max In-Patient Sum Insured (Imperial Plan)",
        "question": "What is the maximum in‑patient hospitalization sum insured under the Imperial Plan?"
    },
    {
        "name": "Medium – Waiting Periods",
        "question": "What are the applicable waiting periods for pre‑existing diseases and specific diseases?"
    },
    {
        "name": "Medium – Physiotherapy Coverage",
        "question": "Is physiotherapy covered, and if so, what is the waiting period for claims under it?"
    },
    {
        "name": "Hard – Living Donor Medical Expenses",
        "question": "Does the policy cover medical expenses for a living organ donor, and under what conditions or limits?"
    }
]


document_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"

print("🧪 Testing Gemini API Integration\n")
print("=" * 60)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{i}. {test_case['name']}")
    print(f"Question: {test_case['question']}")
    print("-" * 40)
    
    payload = {
        "documents": document_url,
        "questions": [test_case['question']]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answers", ["No answer"])[0]
            processing_time = result.get("metadata", {}).get("processing_time", 0)
            confidence = result.get("metadata", {}).get("confidence_scores", [0])[0]
            
            print(f"✅ Status: Success")
            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            print(f"🎯 Confidence: {confidence:.2f}")
            print(f"💬 Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            
            # Check if it's using Gemini or fallback
            if answer.startswith("Information not found"):
                print("🤖 Source: Gemini API (Information not found)")
            elif len(answer) < 100 and ("CBD" in answer or "Policy" in answer):
                print("🔄 Source: Fallback System")
            else:
                print("🤖 Source: Gemini API")
                
        else:
            print(f"❌ Status: Error {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

print("=" * 60)
print("🏁 Test completed!")
