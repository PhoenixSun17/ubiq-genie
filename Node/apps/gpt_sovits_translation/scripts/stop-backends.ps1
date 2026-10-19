[CmdletBinding()]
param([string]$RuntimeRoot = '')
. (Join-Path $PSScriptRoot 'common.ps1')
Initialize-Runtime
Stop-OwnedProcess 'gpt-sovits'
$recordPath = Join-Path $RuntimeRoot 'pids\ollama.json'
if (Test-Path -LiteralPath $recordPath) {
    $record = Get-Content -Raw -LiteralPath $recordPath | ConvertFrom-Json
    $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    try {
        if ($process -and $process.Path -eq $record.path -and $process.StartTime.ToUniversalTime().ToString('o') -eq $record.startTime) {
            $listeners = @(Get-NetTCPConnection -State Listen -LocalPort 11435 -ErrorAction SilentlyContinue)
            $ownedListeners = @($listeners | Where-Object { $_.OwningProcess -eq $record.pid -and $_.LocalAddress -eq '127.0.0.1' })
            $otherListeners = @($listeners | Where-Object { $_.OwningProcess -ne $record.pid })
            if ($ownedListeners.Count -gt 0 -and $otherListeners.Count -eq 0) {
                $appConfig = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot '..\config.json') | ConvertFrom-Json
                $model = $appConfig.translation.ollama.model
                if ([string]::IsNullOrWhiteSpace($model)) { throw 'The app config does not specify an Ollama model to unload.' }
                $unloadBody = @{ model = $model; keep_alive = 0 } | ConvertTo-Json -Compress
                # An empty unload request cannot generate speech or text.
                Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/generate' -Method Post -ContentType 'application/json' -Body $unloadBody -TimeoutSec 10 | Out-Null
            }
        }
    } catch {
        Write-Warning "Ollama model unload did not complete: $_"
    } finally {
        Stop-OwnedProcess 'ollama'
    }
}
Write-Host 'Stopped the recorded project backends. Stop the Node app separately to stop its ASR worker.'
