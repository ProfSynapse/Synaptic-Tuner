# Toolset-Training Unified CLI - PowerShell wrapper
# Usage: .\run.ps1 [train|upload|eval|pipeline]
#
# NOTE: For best results, run directly in WSL:
#   ./run.sh [train|upload|eval|pipeline]

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Load environment variables from .env if it exists
$EnvFile = Join-Path $ScriptDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        # Skip comments and empty lines
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line -split "=", 2
            if ($parts.Length -eq 2) {
                $name = $parts[0].Trim()
                $value = $parts[1].Trim()
                # Remove quotes if present
                $value = $value -replace '^["'']|["'']$', ''
                [Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
    }
}

# Standard environment
$UnslothEnv = "unsloth_latest"

# Auto-detect WSL distro
Write-Host "Detecting WSL distribution..." -ForegroundColor Gray
$WslDistros = (wsl -l -q 2>$null) -replace "`0", "" | Where-Object { $_ -ne "" }
if ($WslDistros) {
    # Find Ubuntu distro (prefer Ubuntu-22.04, then any Ubuntu, then first available)
    $WslDistro = $WslDistros | Where-Object { $_ -eq "Ubuntu-22.04" } | Select-Object -First 1
    if (-not $WslDistro) {
        $WslDistro = $WslDistros | Where-Object { $_ -like "Ubuntu*" } | Select-Object -First 1
    }
    if (-not $WslDistro) {
        $WslDistro = $WslDistros | Select-Object -First 1
    }
    Write-Host "  Using WSL distro: $WslDistro" -ForegroundColor Green
} else {
    Write-Host "  ERROR: No WSL distributions found!" -ForegroundColor Red
    Write-Host "  Install WSL with: wsl --install" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

$UseWsl = $false

# Check if this is a GPU operation
$GpuOps = @("train", "upload", "pipeline", "gguf")
$NeedsGpu = $Arguments | Where-Object { $GpuOps -contains $_ }

if ($NeedsGpu) {
    Write-Host "This operation requires GPU. Running via WSL..." -ForegroundColor Cyan
    Write-Host ""

    # Delegate to run.sh which handles dependency checks and animations
    Write-Host "Delegating to WSL..." -ForegroundColor Cyan
    Write-Host ""

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
