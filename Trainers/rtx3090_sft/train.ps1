# ============================================================================
# SFT Training Script for Windows PowerShell
# ============================================================================
# This script will:
# 1. Check prerequisites (conda, GPU, disk space)
# 2. Activate unsloth_env
# 3. Verify dataset exists
# 4. Run SFT training with Mistral 7B
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "SFT Training - Mistral 7B v0.3" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 1: Navigate to training directory and load config
# ============================================================================
Write-Host "[1/6] Loading configuration..." -ForegroundColor Yellow

# Get script directory (should be Trainers/rtx3090_sft)
$TrainingDir = $PSScriptRoot
Set-Location $TrainingDir

# Find project root (two levels up from training dir)
$ProjectRoot = (Get-Item $TrainingDir).Parent.Parent.FullName

Write-Host "  [OK] Training directory: $TrainingDir" -ForegroundColor Green
Write-Host "  [OK] Project root: $ProjectRoot" -ForegroundColor Green

# Read dataset path from YAML config using Python
$ConfigScript = @"
import sys
sys.path.insert(0, r'$TrainingDir')
from configs.config_loader import get_7b_config
config = get_7b_config()
# Resolve relative path to absolute
import os
if config.dataset.local_file:
    dataset_path = os.path.normpath(os.path.join(r'$TrainingDir', config.dataset.local_file))
    print(dataset_path)
else:
    print(f'{config.dataset.dataset_name}/{config.dataset.dataset_file}')
"@

# Find Python first (will do this in next step but need it now for config)
$UnslothEnvPaths = @(
    "$env:USERPROFILE\miniconda3\envs\unsloth_env\python.exe",
    "$env:USERPROFILE\anaconda3\envs\unsloth_env\python.exe",
    "C:\ProgramData\miniconda3\envs\unsloth_env\python.exe",
    "C:\ProgramData\anaconda3\envs\unsloth_env\python.exe"
)

$PythonExe = $null
foreach ($path in $UnslothEnvPaths) {
    if (Test-Path $path) {
        $PythonExe = $path
        break
    }
}

if (-not $PythonExe) {
    Write-Host "  [ERROR] unsloth_env Python not found (needed to read config)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$DatasetPath = & $PythonExe -c $ConfigScript
if (-not $DatasetPath) {
    Write-Host "  [ERROR] Could not read dataset path from config" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  [OK] Dataset from config: $DatasetPath" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 2: Check dataset exists
# ============================================================================
Write-Host "[2/6] Checking dataset..." -ForegroundColor Yellow

if (-not (Test-Path $DatasetPath)) {
    Write-Host "  [ERROR] Dataset not found at $DatasetPath" -ForegroundColor Red
    Write-Host "  [INFO] This path was read from configs/training_config.py" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

$DatasetSizeMB = [math]::Round((Get-Item $DatasetPath).Length / 1MB, 1)
Write-Host "  [OK] Dataset found: $DatasetSizeMB MB" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 3: Check disk space
# ============================================================================
Write-Host "[3/6] Checking disk space..." -ForegroundColor Yellow

$Drive = Get-PSDrive F
$FreeSpaceGB = [math]::Round($Drive.Free / 1GB, 2)
$RequiredGB = 30

if ($FreeSpaceGB -lt $RequiredGB) {
    Write-Host "  [ERROR] Insufficient disk space" -ForegroundColor Red
    Write-Host "  [ERROR] Available: $FreeSpaceGB GB" -ForegroundColor Red
    Write-Host "  [ERROR] Required: $RequiredGB GB" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  [OK] Disk space available: $FreeSpaceGB GB" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 4: Verify Python (already found in step 1)
# ============================================================================
Write-Host "[4/6] Verifying Python environment..." -ForegroundColor Yellow
Write-Host "  [OK] Python: $PythonExe" -ForegroundColor Green

$PythonVersion = & $PythonExe --version 2>&1
Write-Host "  [OK] Version: $PythonVersion" -ForegroundColor Green

# Test CUDA availability
Write-Host "  -> Testing CUDA availability..." -ForegroundColor Cyan
$CudaTest = & $PythonExe -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1
if ($CudaTest -match "CUDA") {
    $GpuName = & $PythonExe -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "  [OK] CUDA available: $GpuName" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] CUDA not available!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# ============================================================================
# STEP 5: Display training configuration
# ============================================================================
Write-Host "[5/6] Training configuration..." -ForegroundColor Yellow
Write-Host ""

# Read config values from YAML
$ConfigReadScript = @"
import sys
sys.path.insert(0, r'$TrainingDir')
from configs.config_loader import get_7b_config
config = get_7b_config()

# Extract values
model_name = config.model.model_name.split('/')[-1]  # Get just the model name part
batch_size = config.training.per_device_train_batch_size
grad_accum = config.training.gradient_accumulation_steps
effective_batch = batch_size * grad_accum
learning_rate = config.training.learning_rate
epochs = config.training.num_train_epochs
max_seq = config.training.max_seq_length

# Print in PowerShell-parseable format
print(f'MODEL={model_name}')
print(f'BATCH_SIZE={batch_size}')
print(f'GRAD_ACCUM={grad_accum}')
print(f'EFFECTIVE_BATCH={effective_batch}')
print(f'LEARNING_RATE={learning_rate}')
print(f'EPOCHS={epochs}')
print(f'MAX_SEQ={max_seq}')
"@

$ConfigValues = & $PythonExe -c $ConfigReadScript
$ConfigDict = @{}
foreach ($line in $ConfigValues) {
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2) {
        $ConfigDict[$parts[0]] = $parts[1]
    }
}

# Get dataset filename
$DatasetName = Split-Path $DatasetPath -Leaf

Write-Host "  Model:           $($ConfigDict['MODEL'])" -ForegroundColor White
Write-Host "  Method:          SFT (Supervised Fine-Tuning)" -ForegroundColor White
Write-Host "  Dataset:         $DatasetName" -ForegroundColor White
Write-Host "  Batch size:      $($ConfigDict['BATCH_SIZE']) (effective: $($ConfigDict['EFFECTIVE_BATCH']) with grad accum)" -ForegroundColor White
Write-Host "  Learning rate:   $($ConfigDict['LEARNING_RATE'])" -ForegroundColor White
Write-Host "  Epochs:          $($ConfigDict['EPOCHS'])" -ForegroundColor White
Write-Host "  Max seq length:  $($ConfigDict['MAX_SEQ']) tokens" -ForegroundColor White
Write-Host ""

# ============================================================================
# STEP 6: Confirm and run training
# ============================================================================
Write-Host "[6/6] Ready to start training" -ForegroundColor Yellow
Write-Host ""
$Confirmation = Read-Host "Start training? (y/n)"

if ($Confirmation -ne "y" -and $Confirmation -ne "Y") {
    Write-Host "Training cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "STARTING TRAINING" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Run training with local dataset
& $PythonExe train_sft.py `
    --model-size 7b `
    --local-file $DatasetPath

$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
if ($ExitCode -eq 0) {
    Write-Host "TRAINING COMPLETED SUCCESSFULLY" -ForegroundColor Green
} else {
    Write-Host "TRAINING FAILED (Exit code: $ExitCode)" -ForegroundColor Red
}
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
exit $ExitCode
