@echo off
curl -X POST "http://localhost:8000/api/v1/hackrx/run" ^
-H "Authorization: Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1" ^
-H "Content-Type: application/json" ^
-d "{\"documents\": \"https://hackrx.blob.core.windows.net/assets/Arogya%%20Sanjeevani%%20Policy%%20-%%20CIN%%20-%%20U10200WB1906GOI001713%%201.pdf?sv=2023-01-03&st=2025-07-21T08%%3A29%%3A02Z&se=2025-09-22T08%%3A29%%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%%2BBXom%%2FB%%2BMPTFMFP3PRnIvEsipAX10Ig4%%3D\", \"questions\": [\"Does this policy cover knee surgery, and what are the conditions?\", \"What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?\", \"What is the waiting period for pre-existing diseases (PED) to be covered?\", \"Does this policy cover maternity expenses, and what are the conditions?\"]}"
