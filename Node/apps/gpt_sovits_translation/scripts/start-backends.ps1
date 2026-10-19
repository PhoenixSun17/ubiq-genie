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
Write-Host 'Models may still be loading. Inspect runtime/logs; starting processes is not an inference check.'
Write-Host 'ASR is started by the Node app when the app is run.'
