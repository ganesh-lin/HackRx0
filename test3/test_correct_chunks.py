import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from llm_manager import LLMManager

def test_with_correct_chunks():
    question = "Is physiotherapy covered, and if so, what is the waiting period for claims under it?"
    
    # The correct chunks that contain physiotherapy information
    correct_chunks = [
        # Definition chunk (Chunk 15)
        """39. Prescribed physiotherapy: - Prescribed physiotherapy refers to treatment provided by a registered physiotherapist following referral by a Doctor. Physiotherapy is initially restricted to 12 sessions per condition, after which treatment must be reviewed by the Doctor who referred You. If You need further sessions, You must send Us a new progress report after every set of 12 sessions, indicating the medical necessity for more treatment. Physiotherapy does not include therapies such as Rolfing, massage, Pilates, Fango and Milta.""",
        
        # Coverage chunk (Chunk 33)
        """Physiotherapy Benefit We will pay the expenses incurred towards Prescribed Physiotherapy taken on Out-patient basis for Illness/Injury contracted during the Policy Period, maximum up to the limit specified in the Policy Schedule, provided that, a. The treatment is referred by a Doctor or prescribed by a Specialist consultant for Muskulo-skeletal/Neurological diseases/Injuries or other Systemic diseases b. The treatment should be carried out by a registered physiotherapist in a Hospital or a clinic as defined under the Policy c. Physiotherapy is initially restricted to 12 sessions per condition, after which treatment must be reviewed by the Doctor who referred You. If You need further sessions, You must send Us a new progress report after every set of 12 sessions, indicating the medical necessity for more treatment. Exclusion: a. During the first year of Global Health Care Policy with Us, 90 days waiting period would be applicable for all claims under Physiotherapy Benefit except those arising out of Accidental Injury, however the waiting period would not be applied during subsequent renewals b. Physiotherapy does not include therapies such as Rolfing, massage, Pilates, Fango and Milta.""",
        
        # In-patient coverage (Chunk 24)
        """v. Anesthesia, Blood, Oxygen, Operation Theatre Charges, surgical appliances, vi. Dialysis, Chemotherapy, Radiotherapy, Physiotherapy vii. Prescription drugs and materials viii. Cost of Artificial Limbs, cost of prosthetic devices implanted during surgical procedure like Pacemaker, orthopedic implants, cardiac valve replacements, vascular stents."""
    ]
    
    print("🧪 TESTING WITH CORRECT PHYSIOTHERAPY CHUNKS")
    print("=" * 60)
    print(f"Question: {question}")
    print("-" * 60)
    
    try:
        llm_manager = LLMManager()
        answer = llm_manager.answer_question(question, correct_chunks, "insurance")
        
        print(f"✅ LLM Answer: {answer}")
        
        if "Information not found" in answer:
            print("❌ LLM still couldn't find the information!")
            print("📋 Testing with a more direct prompt...")
            
            # Try with a more direct prompt
            direct_prompt = f"""Based on the following insurance policy text, answer this question: {question}

POLICY TEXT:
{' '.join(correct_chunks)}

Please provide a clear answer about physiotherapy coverage and waiting periods."""
            
            response = llm_manager.generate_response(direct_prompt)
            print(f"🔄 Direct prompt result: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_with_correct_chunks()
