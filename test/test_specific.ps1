# Test specific insurance question
$headers = @{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1"
}

$body = @{
    user_input = "What is the waiting period for pre-existing diseases?"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/hackrx/run" -Method POST -Headers $headers -Body $body
    
    Write-Host "Success!" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Status Code: $($_.Exception.Response.StatusCode)" -ForegroundColor Red
}
