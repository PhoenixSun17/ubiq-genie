[CmdletBinding()]
param(
    [string]$RuntimeRoot = 'C:\ubiq-genie\Node\.runtime\gpt_sovits_translation'
)

$ErrorActionPreference = 'Stop'
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$OllamaVersion = 'v0.40.3'
$OllamaUrl = "https://github.com/ollama/ollama/releases/download/$OllamaVersion/ollama-windows-amd64.zip"
$OllamaSha256 = 'e5ecff02dfc7cae8ddf6d6b7a7ac5f727ffeedf4491f44ae75ff5b0820dbbbd5'
$FfmpegVersion = '9.0.2'
$FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-$FfmpegVersion-essentials_build.zip"
$FfmpegSha256 = '60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba'
$Model = 'qwen3:4b-q4_K_M'
$ModelDigest = '2bfd38a7daaf4b1037efe517ccb73d1a3bbd4822cf89f1a82be1569050a114e0'
$ServerOrigin = 'http://127.0.0.1:11435'

function Assert-RuntimePath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($RuntimeRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the selected runtime directory: $full"
    }
    return $full
}

function Invoke-CheckedCurl([string[]]$Arguments) {
    & curl.exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "curl failed with exit code $LASTEXITCODE" }
}

function Get-VerifiedArchive([string]$Url, [string]$Destination, [string]$ExpectedHash) {
    $Destination = Assert-RuntimePath $Destination
    if (-not (Test-Path -LiteralPath $Destination)) {
        Write-Host "Downloading $Url"
        $partial = Assert-RuntimePath "$Destination.partial"
        Invoke-CheckedCurl @('-L', '--fail', '--show-error', '--retry', '4', '--retry-delay', '3', '--connect-timeout', '30', '-C', '-', '-o', $partial, $Url)
        if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $ExpectedHash) {
            throw "SHA256 mismatch: $partial. Preserve the file for inspection; do not install it."
        }
        Move-Item -LiteralPath $partial -Destination $Destination
    }
    if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -ne $ExpectedHash) {
        throw "SHA256 mismatch: $Destination"
    }
}

foreach ($relative in @('downloads', 'receipts', 'logs', 'pids', 'ollama', 'ollama\home', 'ollama\tmp', 'models\ollama', 'tools')) {
    New-Item -ItemType Directory -Force -Path (Assert-RuntimePath (Join-Path $RuntimeRoot $relative)) | Out-Null
}

$ollamaArchive = Join-Path $RuntimeRoot "downloads\ollama-$OllamaVersion-windows-amd64.zip"
$ffmpegArchive = Join-Path $RuntimeRoot "downloads\ffmpeg-$FfmpegVersion-essentials_build.zip"
Get-VerifiedArchive $OllamaUrl $ollamaArchive $OllamaSha256
Get-VerifiedArchive $FfmpegUrl $ffmpegArchive $FfmpegSha256

$ollamaDirectory = Assert-RuntimePath (Join-Path $RuntimeRoot 'ollama')
$ollamaExecutable = Join-Path $ollamaDirectory 'ollama.exe'
$ollamaMarker = Join-Path $ollamaDirectory 'archive-sha256.txt'
if (-not (Test-Path -LiteralPath $ollamaExecutable) -or
    -not (Test-Path -LiteralPath $ollamaMarker) -or (Get-Content -Raw -LiteralPath $ollamaMarker).Trim() -ne $OllamaSha256) {
    $usingOllama = Get-Process -Name ollama -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $ollamaExecutable }
    if ($usingOllama) { throw 'Project-local Ollama is running. Stop its owned service before installing.' }
    Write-Host "Extracting Ollama $OllamaVersion"
    Expand-Archive -LiteralPath $ollamaArchive -DestinationPath $ollamaDirectory -Force
    if (-not (Test-Path -LiteralPath $ollamaExecutable)) { throw 'Ollama archive did not contain ollama.exe' }
    Set-Content -LiteralPath $ollamaMarker -Value $OllamaSha256 -Encoding ASCII
}

$ffmpegDirectory = Assert-RuntimePath (Join-Path $RuntimeRoot 'tools\ffmpeg')
$ffmpegExecutable = Join-Path $ffmpegDirectory 'bin\ffmpeg.exe'
$ffmpegMarker = Join-Path $ffmpegDirectory 'archive-sha256.txt'
if (-not (Test-Path -LiteralPath $ffmpegExecutable) -or
    -not (Test-Path -LiteralPath $ffmpegMarker) -or (Get-Content -Raw -LiteralPath $ffmpegMarker).Trim() -ne $FfmpegSha256) {
    Write-Host "Extracting FFmpeg $FfmpegVersion"
    $extraction = Assert-RuntimePath (Join-Path $RuntimeRoot "downloads\ffmpeg-$FfmpegVersion-extracted")
    Expand-Archive -LiteralPath $ffmpegArchive -DestinationPath $extraction -Force
    $packageDirectory = Assert-RuntimePath (Join-Path $extraction "ffmpeg-$FfmpegVersion-essentials_build")
    if (-not (Test-Path -LiteralPath (Join-Path $packageDirectory 'bin\ffmpeg.exe'))) { throw 'FFmpeg archive layout was unexpected' }
    New-Item -ItemType Directory -Force -Path $ffmpegDirectory | Out-Null
    Get-ChildItem -LiteralPath $packageDirectory | Copy-Item -Destination $ffmpegDirectory -Recurse -Force
    Set-Content -LiteralPath $ffmpegMarker -Value $FfmpegSha256 -Encoding ASCII
}

# No global installation, registry changes, startup entry, or persistent PATH changes.
# Only these child processes see the project-local model and home directories.
$environment = @{
    OLLAMA_HOST = '127.0.0.1:11435'
    OLLAMA_MODELS = (Join-Path $RuntimeRoot 'models\ollama')
    OLLAMA_NO_CLOUD = '1'
    OLLAMA_NUM_PARALLEL = '1'
    OLLAMA_MAX_LOADED_MODELS = '1'
    OLLAMA_CONTEXT_LENGTH = '4096'
    OLLAMA_TMPDIR = (Join-Path $RuntimeRoot 'ollama\tmp')
    USERPROFILE = (Join-Path $RuntimeRoot 'ollama\home')
}
$savedEnvironment = @{}
$ownedProcess = $null
$record = $null
$pidFile = Assert-RuntimePath (Join-Path $RuntimeRoot 'pids\ollama-install.json')
try {
    if (Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue) {
        throw 'Port 11435 is already listening. Stop the owned translation Ollama service before installing; an existing process will not be reused or stopped.'
    }
    foreach ($entry in $environment.GetEnumerator()) {
        $savedEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, 'Process')
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    Write-Host 'Starting a temporary owned Ollama server for model download only.'
    $ownedProcess = Start-Process -FilePath $ollamaExecutable -ArgumentList @('serve') -WorkingDirectory $ollamaDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RuntimeRoot 'logs\ollama-install.out.log') -RedirectStandardError (Join-Path $RuntimeRoot 'logs\ollama-install.err.log')
    $record = @{ pid = $ownedProcess.Id; path = $ownedProcess.Path; startTime = $ownedProcess.StartTime.ToUniversalTime().ToString('o') }
    $record | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding UTF8
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        $ownedProcess.Refresh()
        if ($ownedProcess.HasExited) { throw 'Temporary Ollama exited; see logs/ollama-install.err.log' }
        try {
            $versionResponse = Invoke-RestMethod -Uri "$ServerOrigin/api/version" -Method Get -TimeoutSec 2
            $ready = $true
            break
        } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $ready) { throw 'Temporary Ollama startup timed out' }
    $listener = @(Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction Stop)
    if ($listener.Count -eq 0 -or @($listener | Where-Object { $_.OwningProcess -ne $ownedProcess.Id }).Count -gt 0) {
        throw 'Port 11435 listener does not match the process started by this installer'
    }

    $pullRequest = Assert-RuntimePath (Join-Path $RuntimeRoot 'downloads\ollama-pull-request.json')
    $pullBody = @{ model = $Model; stream = $false } | ConvertTo-Json -Compress
    [IO.File]::WriteAllText($pullRequest, $pullBody, [Text.UTF8Encoding]::new($false))
    $pullResult = Assert-RuntimePath (Join-Path $RuntimeRoot 'downloads\ollama-pull-result.json')
    Write-Host "Pulling $Model into $($environment.OLLAMA_MODELS); this does not run inference."
    Invoke-CheckedCurl @('-L', '--fail', '--show-error', '--connect-timeout', '10', '--max-time', '7200', '-H', 'Content-Type: application/json', '--data-binary', "@$pullRequest", '-o', $pullResult, "$ServerOrigin/api/pull")
    $pull = Get-Content -Raw -LiteralPath $pullResult | ConvertFrom-Json
    if ($pull.status -ne 'success') { throw "Ollama pull did not succeed: $($pull.error)" }
    $tags = Invoke-RestMethod -Uri "$ServerOrigin/api/tags" -Method Get -TimeoutSec 15
    $modelInfo = @($tags.models | Where-Object { $_.name -eq $Model -or $_.model -eq $Model })
    if ($modelInfo.Count -ne 1 -or -not $modelInfo[0].digest) { throw "Downloaded $Model was not present in the local model manifest" }
    if (($modelInfo[0].digest -replace '^sha256:', '') -ne $ModelDigest) {
        throw "Unexpected model digest for ${Model}. Expected $ModelDigest; found $($modelInfo[0].digest). The upstream tag may have changed; verify the model before using it."
    }
    $receipt = @{
        installedAtUtc = [DateTime]::UtcNow.ToString('o')
        ollama = @{ version = $OllamaVersion; reportedVersion = $versionResponse.version; source = $OllamaUrl; archiveSha256 = $OllamaSha256; executableSha256 = (Get-FileHash -LiteralPath $ollamaExecutable -Algorithm SHA256).Hash.ToLowerInvariant() }
        ffmpeg = @{ version = $FfmpegVersion; source = $FfmpegUrl; archiveSha256 = $FfmpegSha256; executableSha256 = (Get-FileHash -LiteralPath $ffmpegExecutable -Algorithm SHA256).Hash.ToLowerInvariant() }
        model = @{ name = $Model; digest = $modelInfo[0].digest; size = $modelInfo[0].size; details = $modelInfo[0].details; source = "https://ollama.com/library/$Model"; directory = $environment.OLLAMA_MODELS }
        validation = 'Installation, archive hashes, server ownership, and model download manifest only. No generation, audio conversion, model inference, unit tests, or Ubiq tests executed.'
    }
    $receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $RuntimeRoot 'receipts\ollama-ffmpeg.json') -Encoding UTF8
    Write-Host "Installed portable Ollama, $Model, and FFmpeg. Model inference remains unverified."
} finally {
    if ($ownedProcess -and $record) {
        $running = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
        if ($running) {
            if ($running.Path -ne $record.path -or $running.StartTime.ToUniversalTime().ToString('o') -ne $record.startTime) {
                Write-Warning 'Refusing to stop a process whose PID/path/start time no longer match the installer record.'
            } else {
                Stop-Process -Id $running.Id -ErrorAction Stop
                $running.WaitForExit(10000) | Out-Null
                Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
            }
        } else { Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue }
    }
    foreach ($entry in $savedEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
}
