param(
    [int[]]$Ports = @(3002, 8001, 11434)
)

$ErrorActionPreference = "SilentlyContinue"

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        # Do not use $PID — it is a built-in automatic variable in PowerShell.
        $owningPid = $conn.OwningProcess
        if ($owningPid -and $owningPid -ne 0) {
            try {
                Stop-Process -Id $owningPid -Force
                Write-Host "[stopped] PID $owningPid on port $port"
            } catch {
            }
        }
    }
}

Write-Host "DocuMind stop sweep complete."
