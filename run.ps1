# Toolset-Training Unified CLI - PowerShell wrapper
# Usage: .\run.ps1 [train|upload|eval|pipeline]
# Or: python tuner.py

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Find Python from conda environment
$CondaEnvs = @(
    "$env:USERPROFILE\miniconda3\envs\unsloth_env\python.exe",
    "$env:USERPROFILE\anaconda3\envs\unsloth_env\python.exe",
    "C:\ProgramData\miniconda3\envs\unsloth_env\python.exe",
    "C:\ProgramData\anaconda3\envs\unsloth_env\python.exe"
)

$Python = $null
foreach ($path in $CondaEnvs) {
    if (Test-Path $path) {
        $Python = $path
        break
    }
}

if (-not $Python) {
    # Fallback to system python
    $Python = "python"
}

# Run CLI
& $Python tuner.py @args
