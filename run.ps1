# Toolset-Training Unified CLI - PowerShell wrapper
# Usage: .\run.ps1 [train|upload|eval|pipeline]
#
# NOTE: For GPU operations (training, upload), use WSL instead:
#       wsl -d Ubuntu-22.04 bash -c 'cd /mnt/f/Code/Toolset-Training && ./run.sh'

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Standard environment
$UnslothEnv = "unsloth_latest"

# For GPU operations, recommend WSL
Write-Host ""
Write-Host "NOTE: For GPU training/upload, use WSL:" -ForegroundColor Yellow
Write-Host "  wsl -d Ubuntu-22.04 bash -c 'cd /mnt/f/Code/Toolset-Training && ./run.sh $($Arguments -join ' ')'" -ForegroundColor Cyan
Write-Host ""

# Find Python - check WSL first for GPU support
$UseWsl = $false
$WslDistro = "Ubuntu-22.04"

# Check if this is a GPU operation
$GpuOps = @("train", "upload", "pipeline")
$NeedsGpu = $Arguments | Where-Object { $GpuOps -contains $_ }

if ($NeedsGpu) {
    Write-Host "This operation requires GPU. Running via WSL..." -ForegroundColor Cyan
    $WslCmd = "cd /mnt/f/Code/Toolset-Training && ./run.sh $($Arguments -join ' ')"
    wsl -d $WslDistro bash -c $WslCmd
    exit $LASTEXITCODE
}

# For non-GPU operations (eval), try to find local Python
$CondaPaths = @(
    "$env:USERPROFILE\miniconda3\envs\$UnslothEnv\python.exe",
    "$env:USERPROFILE\anaconda3\envs\$UnslothEnv\python.exe",
    "C:\ProgramData\miniconda3\envs\$UnslothEnv\python.exe",
    "C:\ProgramData\anaconda3\envs\$UnslothEnv\python.exe"
)

$Python = $null
foreach ($path in $CondaPaths) {
    if (Test-Path $path) {
        $Python = $path
        Write-Host "Using Python: $path" -ForegroundColor Green
        break
    }
}

if (-not $Python) {
    Write-Host "Environment '$UnslothEnv' not found." -ForegroundColor Yellow
    $RunSetup = Read-Host "Would you like to run setup now? (Y/n)"
    if ($RunSetup -match "^[Yy]$" -or $RunSetup -eq "") {
        .\setup_env.ps1
        # Try finding python again
        foreach ($path in $CondaPaths) {
            if (Test-Path $path) {
                $Python = $path
                break
            }
        }
    }
}

# Fallback to other python paths if setup failed or was skipped (legacy behavior)
if (-not $Python) {
    $FallbackPaths = @(
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\anaconda3\python.exe"
    )
    foreach ($path in $FallbackPaths) {
        if (Test-Path $path) {
            $Python = $path
            Write-Host "Using fallback Python: $path" -ForegroundColor Yellow
            break
        }
    }
}

# Try system Python if no conda
if (-not $Python) {
    try {
        $SysPython = (Get-Command python -ErrorAction SilentlyContinue).Source
        if ($SysPython) {
            $Python = $SysPython
            Write-Host "Using system Python: $Python" -ForegroundColor Green
        }
    } catch { }
}

if (-not $Python) {
    # Fallback to WSL only if no local Python at all
    Write-Host "No local Python found, using WSL..." -ForegroundColor Yellow
    $WslCmd = "cd /mnt/f/Code/Toolset-Training && ./run.sh $($Arguments -join ' ')"
    wsl -d $WslDistro bash -c $WslCmd
    exit $LASTEXITCODE
}

# Run CLI
& $Python tuner.py @Arguments
