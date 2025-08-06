import requests
import json

# API configuration
url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1",
    "Content-Type": "application/json"
}

# Test cases
test_cases = [
    {
        "name": "Knee Surgery Coverage",
        "questions": ["Does this policy cover knee surgery, and what are the conditions?"]
    },
    {
        "name": "Heart Surgery Coverage", 
        "questions": ["Is heart surgery covered under this policy?"]
    },
    {
        "name": "General Surgery Question",
        "questions": ["What surgical procedures are covered?"]
    },
    {
        "name": "Multiple Questions",
        "questions": [
            "Does this policy cover knee surgery?",
            "What about dental treatment?",
            "Are there any age restrictions?"
        ]
    }
]

document_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

print("🚀 HackRX API Comprehensive Test Suite")
print("=" * 50)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n📋 Test Case {i}: {test_case['name']}")
    print("-" * 30)
    
    payload = {
        "documents": document_url,
        "questions": test_case["questions"]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Status: SUCCESS")
            print("📝 Questions and Answers:")
            for j, (question, answer) in enumerate(zip(test_case["questions"], result.get("answers", [])), 1):
                print(f"   Q{j}: {question}")
                print(f"   A{j}: {answer}")
                print()
        else:
            print(f"❌ Status: ERROR ({response.status_code})")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Status: TIMEOUT (Request took too long)")
    except Exception as e:
        print(f"❌ Status: EXCEPTION")
        print(f"   Error: {e}")

print("\n🏁 Test Suite Complete!")
print("\n💡 For Windows curl users, use this PowerShell command:")
print('Invoke-RestMethod -Uri "http://localhost:8000/api/v1/hackrx/run" -Method POST -Headers @{"Authorization"="Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1"; "Content-Type"="application/json"} -Body (Get-Content "test_request.json" -Raw)')
