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
$GpuOps = @("train", "upload", "pipeline")
$NeedsGpu = $Arguments | Where-Object { $GpuOps -contains $_ }

if ($NeedsGpu) {
    Write-Host "This operation requires GPU. Running via WSL..." -ForegroundColor Cyan
    Write-Host ""

    # Check dependencies in WSL before running
    Write-Host "Checking dependencies..." -ForegroundColor Cyan
    $MissingDeps = @()
    $NeedInstall = $false

    # Check unsloth
    $unslothCheck = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c 'import unsloth' 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] unsloth" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] unsloth" -ForegroundColor Red
        $MissingDeps += "unsloth"
        $NeedInstall = $true
    }

    # Check FastVisionModel (required for Ministral 3 and VL models)
    $visionCheck = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c 'from unsloth import FastVisionModel' 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] FastVisionModel (Ministral 3 / VL support)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] FastVisionModel (required for Ministral 3)" -ForegroundColor Red
        $MissingDeps += "unsloth_zoo (Vision Model support)"
        $NeedInstall = $true
    }

    # Check xformers
    $xformersCheck = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c 'import xformers' 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] xformers" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] xformers" -ForegroundColor Yellow
        $MissingDeps += "xformers"
        $NeedInstall = $true
    }

    # Check Transformers version (5.0.0.dev0 required for Ministral 3)
    $TransformersVersion = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c `"import transformers; print(transformers.__version__)`" 2>/dev/null"
    if ($TransformersVersion -like "5.0.0*") {
        Write-Host "  [OK] transformers $TransformersVersion" -ForegroundColor Green
    } else {
        Write-Host "  [OUTDATED] transformers $TransformersVersion (need 5.0.0.dev0)" -ForegroundColor Red
        $MissingDeps += "transformers 5.0.0.dev0 (current: $TransformersVersion)"
        $NeedInstall = $true
    }

    # Check TRL version (0.22.2 required for Transformers 5 + Unsloth)
    $TrlVersion = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c `"import trl; print(trl.__version__)`" 2>/dev/null"
    if ($TrlVersion -eq "0.22.2") {
        Write-Host "  [OK] trl $TrlVersion" -ForegroundColor Green
    } else {
        Write-Host "  [OUTDATED] trl $TrlVersion (need 0.22.2)" -ForegroundColor Red
        $MissingDeps += "trl 0.22.2 (current: $TrlVersion)"
        $NeedInstall = $true
    }

    # Install missing dependencies if needed
    if ($NeedInstall) {
        Write-Host ""
        Write-Host "Missing or outdated dependencies detected:" -ForegroundColor Yellow
        foreach ($dep in $MissingDeps) {
            Write-Host "  - $dep" -ForegroundColor Yellow
        }
        Write-Host ""
        Write-Host "Ministral 3 and VL models require these specific versions." -ForegroundColor Yellow
        $install = Read-Host "Install/update dependencies? (Y/n)"
        if ($install -match "^[Yy]$" -or $install -eq "") {
            Write-Host ""
            Write-Host "Installing dependencies for Ministral 3 / Transformers 5 (this may take 2-3 minutes)..." -ForegroundColor Cyan
            Write-Host ""

            # Install Transformers 5 from special branch (required for Ministral 3)
            Write-Host "[1/4] Installing Transformers 5 (Ministral 3 branch)..." -ForegroundColor Cyan
            wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && pip install git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce -q"

            # Install TRL 0.22.2 (compatible with Transformers 5 + Unsloth)
            Write-Host "[2/4] Installing TRL 0.22.2..." -ForegroundColor Cyan
            wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && pip install --no-deps trl==0.22.2 -q"

            # Install Unsloth (with --no-deps to avoid version conflicts)
            Write-Host "[3/4] Installing Unsloth (latest)..." -ForegroundColor Cyan
            wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo -q"

            # Install xformers
            Write-Host "[4/4] Installing xformers..." -ForegroundColor Cyan
            wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && pip install --upgrade xformers -q"

            Write-Host ""

            # Verify installation
            $verifyCheck = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c 'from unsloth import FastVisionModel' 2>&1"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Dependencies installed successfully!" -ForegroundColor Green
                Write-Host "FastVisionModel available (Ministral 3 ready)" -ForegroundColor Green
                $newTransformers = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c `"import transformers; print(transformers.__version__)`""
                $newTrl = wsl -d $WslDistro bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest && python -c `"import trl; print(trl.__version__)`""
                Write-Host "Transformers: $newTransformers" -ForegroundColor Green
                Write-Host "TRL: $newTrl" -ForegroundColor Green
            } else {
                Write-Host "FastVisionModel still not available after install." -ForegroundColor Red
                Write-Host "Try running: ./setup_env.sh in WSL" -ForegroundColor Yellow
                exit 1
            }
        } else {
            Write-Host "Skipping dependency installation." -ForegroundColor Yellow
            Write-Host "Ministral 3 and VL model operations may fail." -ForegroundColor Yellow
        }
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
