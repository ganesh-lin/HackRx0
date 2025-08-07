import requests
import PyPDF2
import tempfile
import os
import re

# Test the document directly
def debug_simple():
    document_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    print("🔧 SIMPLE DEBUG: Document Processing")
    print("=" * 60)
    
    try:
        # Download the file
        print("📥 Downloading document...")
        response = requests.get(document_url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        print(f"✅ Document downloaded to: {temp_path}")
        print(f"📊 File size: {len(response.content)} bytes")
        
        # Extract text
        print("\n📄 Extracting text from PDF...")
        with open(temp_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            text = ""
            
            print(f"📖 Total pages: {total_pages}")
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\\n--- Page {page_num + 1} ---\\n"
                        text += page_text
                        print(f"   ✅ Page {page_num + 1}: {len(page_text)} chars")
                    else:
                        print(f"   ❌ Page {page_num + 1}: No text extracted")
                except Exception as e:
                    print(f"   ❌ Page {page_num + 1}: Error - {e}")
        
        # Clean up
        os.remove(temp_path)
        
        print(f"\\n✅ Total text extracted: {len(text)} characters")
        print(f"📝 First 800 characters:")
        print("-" * 40)
        print(text[:800])
        print("-" * 40)
        
        # Test keyword search for the specific questions
        test_keywords = {
            "grace period": ["grace period", "grace", "period", "premium due"],
            "imperial plan": ["imperial plan", "imperial", "hospitalization", "sum insured"],
            "waiting period": ["waiting period", "waiting", "pre-existing", "specific diseases"],
            "physiotherapy": ["physiotherapy", "physio", "therapy"],
            "living donor": ["living donor", "donor", "organ", "transplant"]
        }
        
        print("\\n🔍 KEYWORD SEARCH RESULTS:")
        print("-" * 40)
        
        text_lower = text.lower()
        for category, keywords in test_keywords.items():
            print(f"\\n{category.upper()}:")
            found_keywords = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                print(f"   ✅ Found: {found_keywords}")
                
                # Show context around the keyword
                for keyword in found_keywords[:2]:  # Show context for first 2 matches
                    pattern = re.compile(f'(.{{0,100}}{re.escape(keyword.lower())}.{{0,100}})', re.IGNORECASE)
                    matches = pattern.findall(text_lower)
                    if matches:
                        print(f"   📝 Context for '{keyword}': {matches[0][:200]}...")
            else:
                print(f"   ❌ Not found: {keywords}")
        
        # Try to find specific answers
        print("\\n🎯 SEARCHING FOR SPECIFIC ANSWERS:")
        print("-" * 40)
        
        # Search for grace period
        grace_patterns = [
            r'grace\\s+period[^.]*\\d+\\s*days?',
            r'\\d+\\s*days?[^.]*grace\\s+period',
            r'premium[^.]*due[^.]*\\d+\\s*days?',
            r'\\d+\\s*days?[^.]*premium[^.]*due'
        ]
        
        print("Grace period search:")
        for pattern in grace_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                print(f"   ✅ Pattern '{pattern}' found: {matches[:2]}")
            else:
                print(f"   ❌ Pattern '{pattern}' not found")
        
        # Search for Imperial Plan amounts
        imperial_patterns = [
            r'imperial\\s+plan[^.]*₹[\\d,]+',
            r'₹[\\d,]+[^.]*imperial\\s+plan',
            r'imperial[^.]*hospitalization[^.]*₹[\\d,]+',
            r'sum\\s+insured[^.]*₹[\\d,]+'
        ]
        
        print("\\nImperial Plan search:")
        for pattern in imperial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                print(f"   ✅ Pattern '{pattern}' found: {matches[:2]}")
            else:
                print(f"   ❌ Pattern '{pattern}' not found")
        
        # Check if the document might be encoded differently or corrupted
        print("\\n🔬 DOCUMENT ANALYSIS:")
        print("-" * 40)
        
        # Check for common insurance document markers
        insurance_markers = ["policy", "insurance", "coverage", "premium", "benefit", "claim", "insured"]
        marker_count = sum(1 for marker in insurance_markers if marker.lower() in text_lower)
        print(f"Insurance markers found: {marker_count}/{len(insurance_markers)}")
        
        # Check text quality
        printable_chars = sum(1 for c in text if c.isprintable())
        text_quality = printable_chars / len(text) if text else 0
        print(f"Text quality: {text_quality:.2%} printable characters")
        
        # Check for garbled text
        weird_chars = sum(1 for c in text if ord(c) > 127)
        print(f"Non-ASCII characters: {weird_chars}")
        
        if text_quality < 0.8:
            print("⚠️  WARNING: Low text quality detected - PDF might be scanned or have encoding issues")
        
        print("\\n" + "=" * 60)
        print("🏁 SIMPLE DEBUG COMPLETE")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_simple()
