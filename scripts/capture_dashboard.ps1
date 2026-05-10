param(
    [string]$WebBase = "http://127.0.0.1:3002",
    [string]$ApiBase = "http://127.0.0.1:8001",
    [int]$MinDocs = 400,
    [int]$WaitIndexMinutes = 90,
    [int]$MaxLivenessWaitMinutes = 180
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$apiRoot = $ApiBase.TrimEnd("/")
$liveUrl = "$apiRoot/health/live"
Write-Host "Waiting for API liveness (GET $liveUrl), max $MaxLivenessWaitMinutes min..."
$deadline = (Get-Date).AddMinutes($MaxLivenessWaitMinutes)
while ((Get-Date) -lt $deadline) {
    try {
        $null = Invoke-RestMethod -Uri $liveUrl -TimeoutSec 10
        break
    } catch {
        Start-Sleep -Seconds 10
    }
}
if ((Get-Date) -ge $deadline) {
    throw "API did not become live at $liveUrl within $MaxLivenessWaitMinutes minutes. Start the API (e.g. start_documind.ps1) or raise -MaxLivenessWaitMinutes."
}

Write-Host "API is live. Running dashboard capture (min docs: $MinDocs, index wait: ${WaitIndexMinutes}m)..."
$waitMs = $WaitIndexMinutes * 60 * 1000
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $py ".\scripts\capture_dashboard_playwright.py" --url $WebBase/ --min-docs $MinDocs --wait-index-ms $waitMs
