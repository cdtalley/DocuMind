param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8001,
    [int]$WebPort = 3002,
    [int]$MaxApiWaitMinutes = 180,
    [switch]$SkipModelPull
)

$ErrorActionPreference = "Stop"

function Test-OllamaReady {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

function Ensure-OllamaRunning {
    if (Test-OllamaReady) {
        Write-Host "[ok] Ollama already running on 127.0.0.1:11434"
        return
    }

    Write-Host "[start] Launching Ollama server..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized | Out-Null

    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-OllamaReady) {
            $ready = $true
            break
        }
    }

    if (-not $ready) {
        throw "Ollama did not become ready. Run 'ollama serve' manually and retry."
    }
    Write-Host "[ok] Ollama is ready."
}

function Ensure-Model([string]$ModelName) {
    $modelsRaw = & ollama list
    if ($modelsRaw -match ("(^|\s)" + [regex]::Escape($ModelName) + "(\s|$)")) {
        Write-Host "[ok] Model present: $ModelName"
        return
    }
    Write-Host "[pull] Downloading model: $ModelName"
    & ollama pull $ModelName
}

Push-Location $PSScriptRoot
try {
    $pythonExe = if (Test-Path (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")) {
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
    } else {
        "python"
    }
    if ($pythonExe -ne "python") {
        Write-Host "[start] Using virtualenv Python: $pythonExe"
    }

    Ensure-OllamaRunning

    if (-not $SkipModelPull) {
        Ensure-Model "llama3"
        Ensure-Model "nomic-embed-text"
    }

    Write-Host "[start] Launching API on http://$ApiHost`:$ApiPort ..."
    Write-Host "[info] First boot can block on sample corpus ingest for a long time; waiting up to $MaxApiWaitMinutes min for /health ..."
    Start-Process -FilePath $pythonExe -ArgumentList @(
        "-m", "uvicorn", "app.main:app", "--host", $ApiHost, "--port", $ApiPort.ToString(), "--reload"
    ) -WorkingDirectory $PSScriptRoot -WindowStyle Minimized | Out-Null

    $apiReady = $false
    $deadline = (Get-Date).AddMinutes($MaxApiWaitMinutes)
    $lastProgress = [datetime]::MinValue
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        try {
            $health = Invoke-RestMethod -Uri "http://$ApiHost`:$ApiPort/health" -Method Get -TimeoutSec 10
            if ($health.status) {
                $apiReady = $true
                break
            }
        } catch {
        }
        if (((Get-Date) - $lastProgress).TotalSeconds -ge 60) {
            $remaining = [math]::Max(0, [int]($deadline - (Get-Date)).TotalMinutes)
            Write-Host "[wait] API not ready yet (ingest may still be running). ~$remaining min left before timeout."
            $lastProgress = Get-Date
        }
    }
    if (-not $apiReady) {
        throw "API failed readiness check at http://$ApiHost`:$ApiPort/health within $MaxApiWaitMinutes minutes. Extend with -MaxApiWaitMinutes or check the minimized uvicorn window for errors."
    }
    Write-Host "[ok] API is ready."

    Write-Host "[start] Installing frontend dependencies if needed..."
    if (-not (Test-Path (Join-Path $PSScriptRoot "web\node_modules"))) {
        Push-Location (Join-Path $PSScriptRoot "web")
        try {
            npm install
        } finally {
            Pop-Location
        }
    }

    Write-Host "[start] Launching frontend on http://localhost:$WebPort ..."
    $cmd = '$env:NEXT_PUBLIC_API_BASE_URL="http://' + $ApiHost + ':' + $ApiPort + '"; npx next dev -p ' + $WebPort
    Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-Command", $cmd) -WorkingDirectory (Join-Path $PSScriptRoot "web") -WindowStyle Minimized | Out-Null

    Write-Host ""
    Write-Host "DocuMind is booting:"
    Write-Host "  - Frontend: http://localhost:$WebPort"
    Write-Host "  - Backend : http://$ApiHost`:$ApiPort"
    Write-Host "  - API Docs: http://$ApiHost`:$ApiPort/docs"
    Write-Host ""
    Write-Host "Tip: run with -SkipModelPull for faster repeated boots; use -MaxApiWaitMinutes to allow longer first-boot ingest."
} finally {
    Pop-Location
}
