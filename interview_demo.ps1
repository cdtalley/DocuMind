param(
    [string]$ApiBase = "http://127.0.0.1:8001"
)

$ErrorActionPreference = "Stop"

Write-Host "DocuMind Interview Demo Sequence"
Write-Host "1) Boot stack"
Write-Host "2) Verify health"
Write-Host "3) Show library"
Write-Host "4) Run dataset query"
Write-Host ""

if (-not (Test-Path ".\start_documind.ps1")) {
    throw "start_documind.ps1 not found in project root."
}

powershell -ExecutionPolicy Bypass -File .\start_documind.ps1 -SkipModelPull
Start-Sleep -Seconds 2

$health = Invoke-RestMethod -Uri "$ApiBase/health" -Method GET -TimeoutSec 15
Write-Host "[health] status=$($health.status), ollama=$($health.ollama_available), papers=$($health.collection_stats.paper_count)"

$papers = Invoke-RestMethod -Uri "$ApiBase/api/v1/papers" -Method GET -TimeoutSec 20
Write-Host "[library] indexed papers=$(@($papers).Count)"

$queryBody = @{
    query = "List every dataset mentioned across my papers and summarize usage."
    top_k = 10
    query_mode = "datasets"
    section_filter = $null
}
$answer = Invoke-RestMethod -Uri "$ApiBase/api/v1/query" -Method POST -Body ($queryBody | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 120

Write-Host ""
Write-Host "[query] has_answer=$($answer.has_answer), confidence=$($answer.confidence), sources=$(@($answer.sources).Count)"
Write-Host "[query preview]"
Write-Host ($answer.answer.Substring(0, [Math]::Min(350, $answer.answer.Length)))
Write-Host ""
Write-Host "Open dashboard: http://localhost:3002"
