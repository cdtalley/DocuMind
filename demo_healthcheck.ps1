param(
    [string]$ApiBase = "http://127.0.0.1:8001",
    [string]$WebBase = "http://localhost:3002"
)

$ErrorActionPreference = "Stop"

function Write-CheckResult([string]$Name, [bool]$Passed, [string]$Detail) {
    $status = if ($Passed) { "[PASS]" } else { "[FAIL]" }
    Write-Host "$status $Name - $Detail"
}

function Try-Request([string]$Method, [string]$Url, $Body = $null, $Files = $null, [int]$TimeoutSec = 20) {
    try {
        if ($Files) {
            return Invoke-RestMethod -Uri $Url -Method $Method -Form $Files -TimeoutSec $TimeoutSec
        }
        if ($Body) {
            return Invoke-RestMethod -Uri $Url -Method $Method -Body ($Body | ConvertTo-Json -Depth 8) -ContentType "application/json" -TimeoutSec $TimeoutSec
        }
        return Invoke-RestMethod -Uri $Url -Method $Method -TimeoutSec $TimeoutSec
    } catch {
        throw $_
    }
}

$allPassed = $true

Write-Host "Running DocuMind demo healthcheck..."
Write-Host "  API: $ApiBase"
Write-Host "  WEB: $WebBase"
Write-Host ""

try {
    $null = Invoke-WebRequest -Uri $WebBase -UseBasicParsing -TimeoutSec 15
    Write-CheckResult "Frontend Reachability" $true $WebBase
} catch {
    $allPassed = $false
    Write-CheckResult "Frontend Reachability" $false $_.Exception.Message
}

try {
    $health = Try-Request -Method "GET" -Url "$ApiBase/health"
    Write-CheckResult "API /health" $true ("status={0}, ollama_available={1}" -f $health.status, $health.ollama_available)
    if (-not $health.ollama_available) {
        $allPassed = $false
        Write-CheckResult "Ollama Availability" $false "Ollama reported unavailable by API"
    } else {
        Write-CheckResult "Ollama Availability" $true "reachable from API"
    }
} catch {
    $allPassed = $false
    Write-CheckResult "API /health" $false $_.Exception.Message
}

try {
    $papers = Try-Request -Method "GET" -Url "$ApiBase/api/v1/papers"
    $paperCount = @($papers).Count
    Write-CheckResult "Paper Library Endpoint" $true ("papers={0}" -f $paperCount)
    if ($paperCount -eq 0) {
        $allPassed = $false
        Write-CheckResult "Paper Library Content" $false "No papers indexed yet. Ingest papers before demo."
    } else {
        Write-CheckResult "Paper Library Content" $true ("indexed papers={0}" -f $paperCount)
    }
} catch {
    $allPassed = $false
    Write-CheckResult "Paper Library Endpoint" $false $_.Exception.Message
}

try {
    $queryBody = @{
        query = "List every dataset mentioned across my papers and summarize how each dataset is used."
        top_k = 10
        query_mode = "datasets"
        section_filter = $null
    }
    $query = Try-Request -Method "POST" -Url "$ApiBase/api/v1/query" -Body $queryBody -TimeoutSec 90
    if ($query.has_answer -eq $true) {
        Write-CheckResult "RAG Query (datasets mode)" $true ("sources={0}, confidence={1}" -f @($query.sources).Count, $query.confidence)
    } else {
        $allPassed = $false
        Write-CheckResult "RAG Query (datasets mode)" $false "No answer returned. Upload more papers or review retrieval settings."
    }
} catch {
    $allPassed = $false
    Write-CheckResult "RAG Query (datasets mode)" $false $_.Exception.Message
}

Write-Host ""
if ($allPassed) {
    Write-Host "Demo healthcheck complete: ALL SYSTEMS GO"
    exit 0
}

Write-Host "Demo healthcheck complete: ACTION REQUIRED"
exit 1
