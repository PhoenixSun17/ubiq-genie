[CmdletBinding()]
param([string]$RuntimeRoot = '')
. (Join-Path $PSScriptRoot 'common.ps1')
Initialize-Runtime
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OLLAMA_TMPDIR = Join-Path $RuntimeRoot 'ollama\tmp'
New-Item -ItemType Directory -Force -Path $env:OLLAMA_TMPDIR | Out-Null

function Assert-PortAvailableOrOwned {
    param([int]$Port, [string]$Name)
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) { return }
    $pidFile = Join-Path $RuntimeRoot "pids\$Name.json"
    if (-not (Test-Path -LiteralPath $pidFile)) { throw "Port $Port is already occupied by an unowned process." }
    $record = Get-Content -Raw -LiteralPath $pidFile | ConvertFrom-Json
    $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if (-not $process -or $process.Path -ne $record.path -or $process.StartTime.ToUniversalTime().ToString('o') -ne $record.startTime) {
        throw "Port $Port is occupied; the recorded $Name process identity does not match."
    }
    foreach ($listener in $listeners) {
        if ($listener.OwningProcess -ne $record.pid) { throw "Port $Port has an unowned listener." }
    }
}

function Wait-BackendReady {
    param([string]$Name, [System.Diagnostics.Process]$Process, [string]$Uri, [int]$TimeoutSeconds = 600)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $logPath = Join-Path $RuntimeRoot "logs\$Name.err.log"
    Write-Host "Waiting for $Name at $Uri..."
    while ([DateTime]::UtcNow -lt $deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "$Name exited before HTTP readiness. Inspect $logPath"
        }
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Host "$Name HTTP endpoint is ready."
                return
            }
        } catch {
            # Connection refusal is expected while models are still loading.
        }
        Start-Sleep -Seconds 1
    }
    throw "$Name did not become HTTP-ready within $TimeoutSeconds seconds. Inspect $logPath"
}

$ollama = Join-Path $RuntimeRoot 'ollama\ollama.exe'
$python = Join-Path $RuntimeRoot 'tts-env\python.exe'
$tts = Join-Path $RuntimeRoot 'GPT-SoVITS'
$config = Join-Path $RuntimeRoot 'tts_infer.yaml'
foreach ($required in @($ollama, $python, (Join-Path $tts 'api_v2.py'), $config,
                         (Join-Path $RuntimeRoot 'receipts\tts-install.json'),
                         (Join-Path $RuntimeRoot 'receipts\ollama-ffmpeg.json'))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing installation artifact: $required. Run setup.ps1 first." }
}
Assert-PortAvailableOrOwned 11435 'ollama'
Assert-PortAvailableOrOwned 9880 'gpt-sovits'
$previousUserProfile = $env:USERPROFILE
try {
    $env:USERPROFILE = Join-Path $RuntimeRoot 'ollama\home'
    New-Item -ItemType Directory -Force -Path $env:USERPROFILE | Out-Null
    $ollamaProcess = Start-OwnedProcess 'ollama' $ollama @('serve') (Split-Path $ollama)
} finally {
    $env:USERPROFILE = $previousUserProfile
}
$ttsProcess = Start-OwnedProcess 'gpt-sovits' $python @('api_v2.py', '-a', '127.0.0.1', '-p', '9880', '-c', $config) $tts
Write-Host "Started Ollama PID $($ollamaProcess.Id), GPT-SoVITS PID $($ttsProcess.Id)."
Wait-BackendReady 'ollama' $ollamaProcess 'http://127.0.0.1:11435/api/tags'
Wait-BackendReady 'gpt-sovits' $ttsProcess 'http://127.0.0.1:9880/openapi.json'
Write-Host 'Both backend HTTP endpoints are ready. You can now start the Node app; inference has not been tested.'
Write-Host 'ASR is started by the Node app when the app is run.'
