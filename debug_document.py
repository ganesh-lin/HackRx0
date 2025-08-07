import requests
import PyPDF2
import tempfile
import os

# Test direct document download and processing
url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

try:
    # Download the file
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(response.content)
        temp_path = temp_file.name
    
    # Extract text
    with open(temp_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        
        for page_num, page in enumerate(reader.pages):
            if page_num < 3:  # Only first 3 pages
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
    
    # Clean up
    os.remove(temp_path)
    
    print(f"Extracted {len(text)} characters")
    print("First 1000 characters:")
    print(text[:1000])
    
    # Test simple keyword matching
    if "cover" in text.lower():
        print("\n✓ Found 'cover' in text")
    else:
        print("\n✗ Did not find 'cover' in text")
        
    if "policy" in text.lower():
        print("✓ Found 'policy' in text")
    else:
        print("✗ Did not find 'policy' in text")

except Exception as e:
    print(f"Error: {e}")
