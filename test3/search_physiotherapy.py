import requests
import PyPDF2
import tempfile
import os
import re

def search_physiotherapy():
    document_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    print("🔍 SEARCHING FOR PHYSIOTHERAPY INFORMATION")
    print("=" * 60)
    
    try:
        # Download and extract
        response = requests.get(document_url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        with open(temp_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        
        os.remove(temp_path)
        
        # Search for physiotherapy mentions
        physio_patterns = [
            r'.{0,200}physiotherapy.{0,200}',
            r'.{0,200}physio.{0,200}',
            r'.{0,200}therapy.{0,200}'
        ]
        
        print("PHYSIOTHERAPY SEARCH RESULTS:")
        print("-" * 40)
        
        for i, pattern in enumerate(physio_patterns):
            print(f"\\nPattern {i+1}: {pattern}")
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                print(f"✅ Found {len(matches)} matches:")
                for j, match in enumerate(matches[:5]):  # Show first 5 matches
                    print(f"   Match {j+1}: {match[:300]}...")
            else:
                print("❌ No matches found")
        
        # Check if "prescribed physiotherapy" is mentioned
        print("\\n" + "="*60)
        print("SEARCHING FOR 'PRESCRIBED PHYSIOTHERAPY':")
        prescribed_matches = re.findall(r'.{0,300}prescribed.{0,20}physiotherapy.{0,300}', text, re.IGNORECASE)
        if prescribed_matches:
            print(f"✅ Found {len(prescribed_matches)} 'prescribed physiotherapy' mentions:")
            for i, match in enumerate(prescribed_matches):
                print(f"\\nMatch {i+1}:")
                print(f"{match}")
        else:
            print("❌ No 'prescribed physiotherapy' mentions found")
        
        # Check specific coverage section
        print("\\n" + "="*60)
        print("CHECKING IN-PATIENT BENEFITS SECTION:")
        inpatient_section = re.search(r'in.patient.*?benefits.*?domestic.*?cover.*?(?=part\s+b|section\s+d)', text, re.IGNORECASE | re.DOTALL)
        if inpatient_section:
            section_text = inpatient_section.group()
            if 'physiotherapy' in section_text.lower():
                print("✅ Physiotherapy found in In-patient Benefits section:")
                physio_context = re.findall(r'.{0,200}physiotherapy.{0,200}', section_text, re.IGNORECASE)
                for context in physio_context:
                    print(f"   {context}")
            else:
                print("❌ Physiotherapy not found in In-patient Benefits section")
        else:
            print("❌ Could not locate In-patient Benefits section")
        
        print("\\n" + "=" * 60)
        print("🏁 PHYSIOTHERAPY SEARCH COMPLETE")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    search_physiotherapy()
