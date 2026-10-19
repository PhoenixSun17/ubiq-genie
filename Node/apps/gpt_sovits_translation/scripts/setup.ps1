[CmdletBinding()]
param(
    [ValidateSet('all', 'tts', 'asr', 'ollama')][string]$Component = 'all',
    [string]$RuntimeRoot = '',
    [string]$CondaExe = 'C:\ProgramData\anaconda3\Scripts\conda.exe',
    [switch]$BuildJiebaFast
)
. (Join-Path $PSScriptRoot 'common.ps1')
Initialize-Runtime
if ($Component -eq 'tts' -or $Component -eq 'all') {
    & (Join-Path $PSScriptRoot 'install-tts.ps1') -RuntimeRoot $RuntimeRoot -CondaExe $CondaExe -BuildJiebaFast:$BuildJiebaFast
}
if ($Component -eq 'asr' -or $Component -eq 'all') {
    & (Join-Path $PSScriptRoot 'install-asr.ps1') -RuntimeRoot $RuntimeRoot -CondaExe $CondaExe
}
if ($Component -eq 'ollama' -or $Component -eq 'all') {
    & (Join-Path $PSScriptRoot 'install-ollama.ps1') -RuntimeRoot $RuntimeRoot
}
Write-Host 'Installation finished. No ASR/TTS inference, Ubiq connection, or GPU test was run.'
