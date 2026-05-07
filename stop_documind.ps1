param(
    [int[]]$Ports = @(3002, 8001, 11434)
)

$ErrorActionPreference = "SilentlyContinue"

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen
    foreach ($conn in $connections) {
        $pid = $conn.OwningProcess
        if ($pid -and $pid -ne 0) {
            try {
                Stop-Process -Id $pid -Force
                Write-Host "[stopped] PID $pid on port $port"
            } catch {
            }
        }
    }
}

Write-Host "DocuMind stop sweep complete."
