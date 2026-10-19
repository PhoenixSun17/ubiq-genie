[CmdletBinding()]
param([string]$RuntimeRoot = '', [string]$CondaExe = 'C:\ProgramData\anaconda3\Scripts\conda.exe', [switch]$BuildJiebaFast)
. (Join-Path $PSScriptRoot 'common.ps1')
Initialize-Runtime
$installLock = [IO.File]::Open((Join-Path $RuntimeRoot 'tts-install.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
$env:CONDA_PKGS_DIRS = Join-Path $RuntimeRoot 'cache\conda-tts'
$env:PIP_NO_INPUT = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$sourceCommit = 'd7c2210da8c013e81a94bfc7b811a477c99fd506'
$sourceRoot = Join-Path $RuntimeRoot 'GPT-SoVITS'
$sourceZip = Join-Path $RuntimeRoot 'downloads\GPT-SoVITS-d7c2210.zip'
$python = Join-Path $RuntimeRoot 'tts-env\python.exe'
$constraints = Join-Path $PSScriptRoot 'tts-constraints.txt'

function Install-Command {
    param([string]$Name, [string]$Executable, [string[]]$Arguments)
    Write-Host "Installing $Name (logs/$Name.log)..."
    $quoted = @($Arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' })
    $process = Start-Process -FilePath $Executable -ArgumentList $quoted -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $RuntimeRoot "logs\$Name.log") -RedirectStandardError (Join-Path $RuntimeRoot "logs\$Name.err.log")
    if ($process.ExitCode -ne 0) { throw "$Name failed; see logs/$Name.err.log (exit $($process.ExitCode))." }
}

if (-not (Test-Path -LiteralPath $python)) {
    Install-Command 'conda-tts' $CondaExe @('create', '--yes', '--prefix', (Join-Path $RuntimeRoot 'tts-env'), 'python=3.11', 'pip', '--override-channels', '-c', 'defaults', '--no-default-packages')
}
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    Get-Download "https://codeload.github.com/RVC-Boss/GPT-SoVITS/zip/$sourceCommit" $sourceZip
    $expanded = Join-Path $RuntimeRoot 'downloads\source-expanded'
    if (-not (Test-Path -LiteralPath (Join-Path $expanded "GPT-SoVITS-$sourceCommit"))) {
        Expand-Archive -LiteralPath $sourceZip -DestinationPath $expanded
    }
    $moveSource = Assert-ManagedPath (Join-Path $expanded "GPT-SoVITS-$sourceCommit")
    $moveTarget = Assert-ManagedPath $sourceRoot
    Move-Item -LiteralPath $moveSource -Destination $moveTarget
}
$sourceReceipt = Join-Path $RuntimeRoot 'receipts\tts-source.json'
if (Test-Path -LiteralPath $sourceReceipt) {
    if ((Get-Content -Raw -LiteralPath $sourceReceipt | ConvertFrom-Json).commit -ne $sourceCommit) {
        throw 'Existing source revision differs; refusing to overwrite it.'
    }
} elseif (-not (Test-Path -LiteralPath $sourceZip)) {
    throw 'Unrecognized existing GPT-SoVITS source. Refusing to reuse it without its pinned archive receipt.'
}
Write-Receipt 'tts-source' @{ repository='https://github.com/RVC-Boss/GPT-SoVITS'; tag='20250606v2pro'; commit=$sourceCommit; archiveSha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $sourceZip).Hash }
$torchWheel = Join-Path $RuntimeRoot 'downloads\torch-2.7.1+cu128-cp311-cp311-win_amd64.whl'
$torchSha256 = '138c66dcd0ed2f07aafba3ed8b7958e2bed893694990e0b4b55b6b2b4a336aa6'
Get-Download 'https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp311-cp311-win_amd64.whl' $torchWheel
if ((Get-FileHash -LiteralPath $torchWheel -Algorithm SHA256).Hash -ne $torchSha256) { throw 'PyTorch wheel checksum mismatch or incomplete download.' }
Install-Command 'pip-tts-torch' $python @('-m', 'pip', 'install', '--no-user', '--only-binary=:all:', $torchWheel, 'torchaudio==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu128')
Install-Command 'pip-tts-dependencies' $python @('-m', 'pip', 'install', '--no-user', '--only-binary=:all:', '--no-binary=jieba,wordsegment,distance', '-c', $constraints, '-r', (Join-Path $PSScriptRoot 'tts-inference-requirements.txt'))
# Never trigger a native compiler automatically. The wheel must be supplied by
# an explicitly approved build or a verified compatible upstream distribution.
$jiebaWheels = @(Get-ChildItem -LiteralPath (Join-Path $RuntimeRoot 'downloads') -Filter 'jieba_fast-0.53-cp311-cp311-win_amd64.whl' -ErrorAction SilentlyContinue)
if ($jiebaWheels.Count -ne 1 -and $BuildJiebaFast) {
    & (Join-Path $PSScriptRoot 'build-jieba-fast.ps1') -RuntimeRoot $RuntimeRoot
    $jiebaWheels = @(Get-ChildItem -LiteralPath (Join-Path $RuntimeRoot 'downloads') -Filter 'jieba_fast-0.53-cp311-cp311-win_amd64.whl')
}
if ($jiebaWheels.Count -ne 1) { throw 'Missing approved jieba_fast 0.53 Windows Python 3.11 wheel. Run setup.ps1 -BuildJiebaFast only after granting permission for this native build.' }
Install-Command 'pip-tts-jieba-fast' $python @('-m', 'pip', 'install', '--no-user', '--only-binary=:all:', $jiebaWheels[0].FullName)
Install-Command 'download-tts' $python @((Join-Path $PSScriptRoot 'download-tts.py'), '--runtime-root', $RuntimeRoot)
$configFile = Join-Path $RuntimeRoot 'tts_infer.yaml'
if (-not (Test-Path -LiteralPath $configFile)) {
    @'
custom:
  device: cuda
  is_half: true
  version: v2Pro
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1v3.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
'@ | Set-Content -LiteralPath $configFile -Encoding ASCII
}
Install-Command 'pip-tts-freeze' $python @('-m', 'pip', 'freeze', '--all')
Copy-Item -LiteralPath (Join-Path $RuntimeRoot 'logs\pip-tts-freeze.log') -Destination (Join-Path $RuntimeRoot 'receipts\tts-packages.txt')
Install-Command 'pip-tts-check' $python @('-m', 'pip', 'check')
Write-Receipt 'tts-install' @{ completedAt=[DateTime]::UtcNow.ToString('o'); python='3.11'; torch='2.7.1+cu128'; torchaudio='2.7.1+cu128'; sourceCommit=$sourceCommit; languages=@('zh', 'en'); omittedOptionalPackages=@('pyopenjtalk', 'g2pk2', 'ko_pron', 'ToJyutping', 'python_mecab_ko', 'funasr', 'modelscope'); runtimeValidated=$false }
Write-Host 'TTS source, environment, and assets installed. Runtime validation is pending.'
} finally {
    $installLock.Dispose()
}
