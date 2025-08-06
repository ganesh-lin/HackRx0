import requests
import json

# API configuration
url = "http://localhost:8000/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1",
    "Content-Type": "application/json"
}

# Request payload
payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": ["Does this policy cover knee surgery, and what are the conditions?"]
}

try:
    print("Testing API endpoint...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(url, headers=headers, json=payload)
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ Success! API is working correctly.")
        result = response.json()
        print("Answers:")
        for i, answer in enumerate(result.get("answers", []), 1):
            print(f"{i}. {answer}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"❌ Exception occurred: {e}")
