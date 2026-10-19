# This is the ONLY native compilation performed by the installer.
# Run only with explicit approval; Visual Studio C++ Build Tools are required.
[CmdletBinding()]
param([string]$RuntimeRoot = '')
. (Join-Path $PSScriptRoot 'common.ps1')
Initialize-Runtime
$python = Join-Path $RuntimeRoot 'tts-env\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create the TTS Python 3.11 environment first.' }
$source = Join-Path $RuntimeRoot 'downloads\jieba_fast-0.53.tar.gz'
$url = 'https://files.pythonhosted.org/packages/87/6f/9c22f7b0ecc043f8e7d324e30767fdc9ce8d3cf5fd66e60823dd2b84432e/jieba_fast-0.53.tar.gz'
$expectedHash = 'e92089d52faa91d51b6a7c1e6e4c4c85064a0e36f6a29257af2254b9e558ddd0'
Get-Download $url $source
if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $expectedHash) { throw 'jieba_fast source checksum mismatch.' }
$arguments = @('-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--wheel-dir', (Join-Path $RuntimeRoot 'downloads'), $source)
$quoted = @($arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' })
$wheel = Join-Path $RuntimeRoot 'downloads\jieba_fast-0.53-cp311-cp311-win_amd64.whl'
if (-not (Test-Path -LiteralPath $wheel)) {
    $process = Start-Process -FilePath $python -ArgumentList $quoted -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RuntimeRoot 'logs\build-jieba-fast.log') -RedirectStandardError (Join-Path $RuntimeRoot 'logs\build-jieba-fast.err.log')
    # Wait for pip itself; the MSVC compiler's reusable PDB server can outlive it.
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw 'jieba_fast build failed. See runtime/logs/build-jieba-fast.err.log; no other native builds were requested.' }
}
if (-not (Test-Path -LiteralPath $wheel)) { throw 'Build did not produce the expected Python 3.11 Windows wheel.' }
Write-Receipt 'jieba-fast-build' @{ package='jieba_fast'; version='0.53'; sourceUrl=$url; sourceSha256=$expectedHash; wheel=$wheel; wheelSha256=(Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash; approvedNativeBuild=$true; inferenceValidated=$false }
