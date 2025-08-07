import requests
import PyPDF2
import tempfile
import os
import re

def find_specific_answers():
    document_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    print("🔍 FINDING SPECIFIC ANSWERS IN DOCUMENT")
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
        
        # Search for specific answer patterns
        questions_and_patterns = [
            {
                "question": "What is the grace period after the premium due date?",
                "patterns": [
                    r'grace\s+period[^.]*?(\d+\s*(?:days?|months?))',
                    r'(\d+\s*(?:days?|months?))[^.]*?grace\s+period',
                    r'premium.*?due.*?(\d+\s*(?:days?|months?))',
                    r'(\d+\s*(?:days?|months?)).*?premium.*?due',
                    r'grace\s+period.*?is.*?(\d+)',
                    r'(\d+).*?days?.*?grace'
                ]
            },
            {
                "question": "What is the maximum in‑patient hospitalization sum insured under the Imperial Plan?",
                "patterns": [
                    r'imperial\s+plan[^.]*?(₹[\d,]+(?:\.\d+)?)',
                    r'(₹[\d,]+(?:\.\d+)?)[^.]*?imperial\s+plan',
                    r'imperial.*?hospitalization.*?(₹[\d,]+)',
                    r'imperial.*?sum\s+insured.*?(₹[\d,]+)',
                    r'imperial.*?(?:upto|up\s+to|maximum).*?(₹[\d,]+)',
                    r'imperial.*?(\d{1,2},?\d{1,3},?\d{3})',
                    r'imperial.*?plan.*?(\d+,\d+,\d+)'
                ]
            },
            {
                "question": "What are the applicable waiting periods for pre‑existing diseases and specific diseases?",
                "patterns": [
                    r'pre.?existing.*?waiting.*?(\d+\s*(?:days?|months?|years?))',
                    r'waiting.*?pre.?existing.*?(\d+\s*(?:days?|months?|years?))',
                    r'specific\s+diseases.*?waiting.*?(\d+\s*(?:days?|months?|years?))',
                    r'waiting.*?specific\s+diseases.*?(\d+\s*(?:days?|months?|years?))',
                    r'(\d+\s*(?:months?|years?))\s*waiting.*?pre.?existing',
                    r'(\d+\s*(?:months?|years?))\s*waiting.*?specific'
                ]
            },
            {
                "question": "Is physiotherapy covered, and if so, what is the waiting period for claims under it?",
                "patterns": [
                    r'physiotherapy[^.]*?(covered|eligible|included|benefit)',
                    r'physiotherapy[^.]*?waiting.*?(\d+\s*(?:days?|months?))',
                    r'waiting.*?physiotherapy.*?(\d+\s*(?:days?|months?))',
                    r'physiotherapy.*?treatment.*?(covered|eligible)',
                    r'prescribed\s+physiotherapy[^.]*'
                ]
            },
            {
                "question": "Does the policy cover medical expenses for a living organ donor, and under what conditions or limits?",
                "patterns": [
                    r'living\s+donor[^.]*?(covered|eligible|₹[\d,]+)',
                    r'donor.*?medical.*?(covered|costs|expenses)',
                    r'organ.*?donor[^.]*?(covered|eligible)',
                    r'living\s+donor.*?medical.*?costs?[^.]*',
                    r'transplant.*?donor[^.]*'
                ]
            }
        ]
        
        for q_data in questions_and_patterns:
            print(f"\\n{'='*50}")
            print(f"Q: {q_data['question']}")
            print('='*50)
            
            found_any = False
            for i, pattern in enumerate(q_data['patterns']):
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    print(f"\\n✅ Pattern {i+1} FOUND:")
                    print(f"   Pattern: {pattern}")
                    print(f"   Matches: {matches[:3]}")
                    found_any = True
                    
                    # Get context around matches
                    for match in matches[:2]:
                        if isinstance(match, tuple):
                            search_term = match[0] if match[0] else 'grace period'
                        else:
                            search_term = str(match)
                        
                        # Find context
                        context_pattern = f'.{{0,200}}{re.escape(search_term)}.{{0,200}}'
                        context_matches = re.findall(context_pattern, text, re.IGNORECASE)
                        if context_matches:
                            print(f"   Context: {context_matches[0][:400]}...")
                
                else:
                    print(f"❌ Pattern {i+1}: No matches")
            
            if not found_any:
                print("\\n🔍 Manual search for keywords...")
                # Manual keyword search
                question_lower = q_data['question'].lower()
                if 'grace period' in question_lower:
                    grace_contexts = []
                    for match in re.finditer(r'.{0,100}grace.{0,100}', text, re.IGNORECASE):
                        grace_contexts.append(match.group())
                    if grace_contexts:
                        print(f"Found 'grace' contexts: {grace_contexts[:3]}")
                
                elif 'imperial' in question_lower:
                    imperial_contexts = []
                    for match in re.finditer(r'.{0,150}imperial.{0,150}', text, re.IGNORECASE):
                        imperial_contexts.append(match.group())
                    if imperial_contexts:
                        print(f"Found 'imperial' contexts: {imperial_contexts[:3]}")
                
                elif 'waiting period' in question_lower:
                    waiting_contexts = []
                    for match in re.finditer(r'.{0,100}waiting.{0,100}', text, re.IGNORECASE):
                        waiting_contexts.append(match.group())
                    if waiting_contexts:
                        print(f"Found 'waiting' contexts: {waiting_contexts[:3]}")
                
                elif 'physiotherapy' in question_lower:
                    physio_contexts = []
                    for match in re.finditer(r'.{0,100}physiotherapy.{0,100}', text, re.IGNORECASE):
                        physio_contexts.append(match.group())
                    if physio_contexts:
                        print(f"Found 'physiotherapy' contexts: {physio_contexts[:3]}")
                
                elif 'living donor' in question_lower:
                    donor_contexts = []
                    for match in re.finditer(r'.{0,100}(?:living.{0,20}donor|donor.{0,20}medical).{0,100}', text, re.IGNORECASE):
                        donor_contexts.append(match.group())
                    if donor_contexts:
                        print(f"Found 'donor' contexts: {donor_contexts[:3]}")
        
        print("\\n" + "=" * 60)
        print("🏁 ANSWER SEARCH COMPLETE")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    find_specific_answers()
