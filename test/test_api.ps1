# PowerShell script to test the API
$headers = @{
    "Authorization" = "Bearer 81c4c164366a640333a4e6786fad3ab382a3eaf34d53d200cdf35bed368bedd1"
    "Content-Type" = "application/json"
}

$body = Get-Content "test_request.json" -Raw

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/hackrx/run" -Method POST -Headers $headers -Body $body
    Write-Host "Success!" -ForegroundColor Green
    Write-Host ($response | ConvertTo-Json -Depth 10)
} catch {
    Write-Host "Error:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response Body: $responseBody"
    }
}
