[CmdletBinding()]
param(
    [string]$RuntimeRoot = 'C:\ubiq-genie\Node\.runtime\gpt_sovits_translation',
    [string]$CondaExe = 'C:\ProgramData\anaconda3\Scripts\conda.exe'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$AsrPrefix = Join-Path $RuntimeRoot 'asr-env'
$AsrPython = Join-Path $AsrPrefix 'python.exe'
$Requirements = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\services\speech_translation\providers\faster_whisper\requirements.txt'))
if (-not (Test-Path -LiteralPath $CondaExe -PathType Leaf)) { throw "Conda executable not found: $CondaExe" }
if (-not (Test-Path -LiteralPath $Requirements -PathType Leaf)) { throw "ASR requirements not found: $Requirements" }

function Invoke-AsrInstallCommand {
    param([string]$Executable, [string[]]$ArgumentList)
    & $Executable @ArgumentList
    if ($LASTEXITCODE -ne 0) { throw "ASR installation command failed with exit code ${LASTEXITCODE}: $Executable" }
}

# Changes apply only to this installer process and are restored for a caller
# that invokes this script while installing the other project-local backends.
$LocalEnvironment = @{
    PYTHONNOUSERSITE = '1'
    PYTHONUTF8 = '1'
    PYTHONIOENCODING = 'utf-8'
    PIP_CACHE_DIR = (Join-Path $RuntimeRoot 'cache\pip-asr')
    PIP_DISABLE_PIP_VERSION_CHECK = '1'
    CONDA_PKGS_DIRS = (Join-Path $RuntimeRoot 'cache\conda-asr')
    CONDA_ENVS_PATH = (Join-Path $RuntimeRoot 'cache\conda-envs-asr')
    CONDA_REGISTER_ENVS = 'false'
    CONDA_AUTO_UPDATE_CONDA = 'false'
    CONDA_REPORT_ERRORS = 'false'
    CONDA_NO_PLUGINS = 'true'
    HF_HOME = (Join-Path $RuntimeRoot 'cache\huggingface-asr')
}
$PreviousEnvironment = @{}
foreach ($Name in $LocalEnvironment.Keys) {
    $PreviousEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
    [Environment]::SetEnvironmentVariable($Name, $LocalEnvironment[$Name], 'Process')
}

try {
    foreach ($Directory in @($RuntimeRoot, 'cache\pip-asr', 'cache\conda-asr', 'cache\conda-envs-asr', 'cache\huggingface-asr', 'models')) {
        $Target = if ([IO.Path]::IsPathRooted($Directory)) { $Directory } else { Join-Path $RuntimeRoot $Directory }
        New-Item -ItemType Directory -Force -Path $Target | Out-Null
    }
    if (-not (Test-Path -LiteralPath $AsrPython -PathType Leaf)) {
        if (Test-Path -LiteralPath $AsrPrefix) {
            throw "ASR prefix exists without python.exe; inspect this incomplete install before retrying: $AsrPrefix"
        }
        Write-Host 'Creating the independent Python 3.11 ASR environment (no global environment registration)...'
        Invoke-AsrInstallCommand $CondaExe @(
            '--no-plugins', 'create', '--yes', '--copy', '--prefix', $AsrPrefix,
            '--solver', 'classic', '--override-channels', '--channel', 'conda-forge',
            'python=3.11', 'pip'
        )
    }

    Write-Host 'Installing pinned Faster-Whisper / CTranslate2 / CPU ONNX / resampler dependencies...'
    Invoke-AsrInstallCommand $AsrPython @(
        '-m', 'pip', '--isolated', '--disable-pip-version-check', '--cache-dir', $LocalEnvironment.PIP_CACHE_DIR,
        'install', '--no-user', '--no-input', '--no-compile', '--only-binary=:all:', '-r', $Requirements
    )
    # pip check examines installed package metadata; it does not import models.
    Invoke-AsrInstallCommand $AsrPython @('-m', 'pip', '--isolated', '--disable-pip-version-check', 'check')
    Write-Host 'Downloading the pinned complete ASR model; model weights are about 1.6 GB...'
    Invoke-AsrInstallCommand $AsrPython @(
        (Join-Path $PSScriptRoot 'download-asr.py'), '--runtime-root', $RuntimeRoot,
        '--requirements', $Requirements
    )
    Write-Host "ASR installation recorded at $(Join-Path $RuntimeRoot 'asr-manifest.json'). Runtime checks were not performed."
} finally {
    foreach ($Name in $PreviousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $PreviousEnvironment[$Name], 'Process')
    }
}
