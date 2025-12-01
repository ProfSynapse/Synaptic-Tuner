# Setup script for Toolset-Training Environment (unsloth_latest)
# Usage: .\setup_env.ps1

$ErrorActionPreference = "Stop"
$EnvName = "unsloth_latest"
$PythonVersion = "3.10"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Toolset-Training Environment Setup" -ForegroundColor Cyan
Write-Host "Environment: $EnvName" -ForegroundColor Cyan
Write-Host "Python: $PythonVersion" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Detect Conda
$CondaPath = $null
$PossiblePaths = @(
    "$env:USERPROFILE\miniconda3",
    "$env:USERPROFILE\anaconda3",
    "C:\ProgramData\miniconda3",
    "C:\ProgramData\anaconda3"
)

foreach ($path in $PossiblePaths) {
    if (Test-Path "$path\Scripts\conda.exe") {
        $CondaPath = "$path\Scripts\conda.exe"
        break
    }
}

if (-not $CondaPath) {
    Write-Error "Conda not found. Please install Miniconda or Anaconda."
}

# Check if environment exists
$EnvList = & $CondaPath env list
if ($EnvList -match $EnvName) {
    Write-Host "Environment $EnvName already exists." -ForegroundColor Yellow
    $Recreate = Read-Host "Recreate it? (y/N)"
    if ($Recreate -match "^[Yy]$") {
        & $CondaPath env remove -n $EnvName -y
    } else {
        Write-Host "Skipping creation."
        exit 0
    }
}

# Create environment
Write-Host "Creating conda environment $EnvName..." -ForegroundColor Green
& $CondaPath create -y -n $EnvName python=$PythonVersion

# We can't easily "activate" conda in a script and have it persist or use it for subsequent commands easily without initializing shell.
# Instead, we will use 'conda run' or find the python executable directly.

$PythonPath = $null
# Try to find the python executable in the new environment
# It's usually in <CondaRoot>\envs\$EnvName\python.exe
$CondaRoot = Split-Path (Split-Path $CondaPath)
$PythonPath = "$CondaRoot\envs\$EnvName\python.exe"

if (-not (Test-Path $PythonPath)) {
    Write-Error "Could not find python.exe in the new environment at $PythonPath"
}

Write-Host "Using Python at: $PythonPath" -ForegroundColor Gray

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Green
& $PythonPath -m pip install --upgrade pip setuptools wheel -q

if (Test-Path "Trainers\rtx3090_sft\requirements.txt") {
    & $PythonPath -m pip install -r Trainers\rtx3090_sft\requirements.txt
} else {
    Write-Error "Trainers\rtx3090_sft\requirements.txt not found"
}

# Install Unsloth and Xformers
Write-Host "Installing Unsloth and Xformers..." -ForegroundColor Green
& $PythonPath -m pip install --no-deps unsloth==2024.9
& $PythonPath -m pip install --no-deps xformers==0.0.27.post2

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Cyan
Write-Host "You can now run the CLI using: .\run.ps1" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
