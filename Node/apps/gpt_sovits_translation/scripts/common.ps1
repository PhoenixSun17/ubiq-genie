$ErrorActionPreference = 'Stop'
if (-not $RuntimeRoot) { $RuntimeRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\.runtime\gpt_sovits_translation')) }
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$NodeRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
function Initialize-Runtime {
    foreach ($relative in @('', 'logs', 'receipts', 'pids', 'downloads', 'tmp', 'cache\pip', 'cache\conda', 'cache\huggingface', 'models', 'tools')) {
        New-Item -ItemType Directory -Force -Path (Join-Path $RuntimeRoot $relative) | Out-Null
    }
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:TEMP = Join-Path $RuntimeRoot 'tmp'
    $env:TMP = $env:TEMP
    $env:PIP_CACHE_DIR = Join-Path $RuntimeRoot 'cache\pip'
    $env:CONDA_PKGS_DIRS = Join-Path $RuntimeRoot 'cache\conda'
    $env:CONDA_REGISTER_ENVS = 'false'
    $env:HF_HOME = Join-Path $RuntimeRoot 'cache\huggingface'
    $env:NLTK_DATA = Join-Path $RuntimeRoot 'tools\nltk_data'
    $env:OLLAMA_MODELS = Join-Path $RuntimeRoot 'models\ollama'
    $env:OLLAMA_HOST = '127.0.0.1:11435'
    $env:OLLAMA_NO_CLOUD = '1'
    $env:OLLAMA_NUM_PARALLEL = '1'
    $env:OLLAMA_MAX_LOADED_MODELS = '1'
    $env:OLLAMA_CONTEXT_LENGTH = '4096'
    $torchDlls = Join-Path $RuntimeRoot 'tts-env\Lib\site-packages\torch\lib'
    $ffmpeg = Join-Path $RuntimeRoot 'tools\ffmpeg\bin'
    $env:PATH = "$ffmpeg;$torchDlls;$env:PATH"
}
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}
function Get-Download {
    param([string]$Url, [string]$Destination)
    if (Test-Path -LiteralPath $Destination) { return }
    New-Item -ItemType Directory -Force -Path (Split-Path $Destination) | Out-Null
    Invoke-Checked 'curl.exe' @('-L', '--fail', '--show-error', '--retry', '4', '--retry-delay', '3', '-C', '-', '-o', "$Destination.partial", $Url)
    $moveSource = Assert-ManagedPath "$Destination.partial"
    $moveTarget = Assert-ManagedPath $Destination
    Move-Item -LiteralPath $moveSource -Destination $moveTarget
}
function Write-Receipt {
    param([string]$Name, [object]$Data)
    $Data | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $RuntimeRoot "receipts\$Name.json") -Encoding UTF8
}
function Assert-ManagedPath {
    param([string]$Path)
    $resolved = [IO.Path]::GetFullPath($Path)
    $root = [IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the selected project runtime: $resolved"
    }
    return $resolved
}
function Start-OwnedProcess {
    param([string]$Name, [string]$Executable, [string[]]$Arguments, [string]$WorkingDirectory)
    $pidFile = Join-Path $RuntimeRoot "pids\$Name.json"
    if (Test-Path -LiteralPath $pidFile) {
        $record = Get-Content -Raw -LiteralPath $pidFile | ConvertFrom-Json
        $existing = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
        if ($existing -and $existing.StartTime.ToUniversalTime().ToString('o') -eq $record.startTime -and $existing.Path -eq $record.path) {
            Write-Host "$Name already running (owned PID $($record.pid))."
            return $existing
        }
        Remove-Item -LiteralPath $pidFile
    }
    $quoted = @($Arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' })
    $executablePath = (Get-Item -LiteralPath $Executable -ErrorAction Stop).FullName
    $process = Start-Process -FilePath $executablePath -ArgumentList $quoted -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RuntimeRoot "logs\$Name.out.log") -RedirectStandardError (Join-Path $RuntimeRoot "logs\$Name.err.log")
    # Process.Path can still be null immediately after Start-Process on Windows.
    @{pid=$process.Id; path=$executablePath; startTime=$process.StartTime.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8
    return $process
}
function Stop-OwnedProcess {
    param([string]$Name)
    $pidFile = Join-Path $RuntimeRoot "pids\$Name.json"
    if (-not (Test-Path -LiteralPath $pidFile)) { return }
    $record = Get-Content -Raw -LiteralPath $pidFile | ConvertFrom-Json
    $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if ($process) {
        if ($process.Path -ne $record.path -or $process.StartTime.ToUniversalTime().ToString('o') -ne $record.startTime) {
            throw "Refusing to stop ${Name}: PID identity no longer matches the recorded process."
        }
        Stop-Process -Id $process.Id
    }
    Remove-Item -LiteralPath $pidFile
}
