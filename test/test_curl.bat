@echo off
echo Testing HackRX API with Windows curl...
echo.

REM Use curl with the JSON file to avoid escaping issues
curl -X POST "http://localhost:8000/api/v1/hackrx/run" ^
     -H "Authorization: Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1" ^
     -H "Content-Type: application/json" ^
     -d @test_request.json

echo.
echo Test complete!
