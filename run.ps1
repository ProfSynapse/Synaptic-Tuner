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
$GpuOps = @("train", "upload", "pipeline")
$NeedsGpu = $Arguments | Where-Object { $GpuOps -contains $_ }

if ($NeedsGpu) {
    Write-Host "This operation requires GPU. Running via WSL..." -ForegroundColor Cyan
    Write-Host ""

    # Check dependencies in WSL before running
    Write-Host "Checking dependencies..." -ForegroundColor Cyan

    # Check unsloth
    $unslothCheck = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c 'import unsloth' 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] unsloth" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] unsloth" -ForegroundColor Red
    }

    # Check FastVisionModel
    $visionCheck = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c 'from unsloth import FastVisionModel' 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] FastVisionModel (Vision Language support)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] FastVisionModel (Vision Language support)" -ForegroundColor Red
        Write-Host ""
        Write-Host "Vision Model support is required for Qwen-VL, LLaVA, Pixtral models." -ForegroundColor Yellow
        $install = Read-Host "Install missing dependencies? (Y/n)"
        if ($install -match "^[Yy]$" -or $install -eq "") {
            Write-Host ""
            Write-Host "Installing unsloth + unsloth_zoo (this may take 1-2 minutes)..." -ForegroundColor Cyan
            wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && pip install --upgrade unsloth unsloth_zoo xformers"
            Write-Host ""

            # Verify
            $verifyCheck = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c 'from unsloth import FastVisionModel' 2>&1"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "FastVisionModel installed successfully!" -ForegroundColor Green
            } else {
                Write-Host "FastVisionModel still not available. Check errors above." -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "Skipping installation. VL model uploads will fail." -ForegroundColor Yellow
        }
    }

    # Check xformers
    $xformersCheck = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c 'import xformers' 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] xformers" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] xformers" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Running operation..." -ForegroundColor Cyan
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
