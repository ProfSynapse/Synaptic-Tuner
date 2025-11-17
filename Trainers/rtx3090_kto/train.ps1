# ============================================================================
# KTO Training Script for Windows PowerShell
# ============================================================================
# This script will:
# 1. Check prerequisites (conda, GPU, disk space)
# 2. Activate unsloth_env
# 3. Verify dataset exists
# 4. Run KTO training with Qwen 2.5 3B
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "KTO Training - Qwen 2.5 3B Instruct" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 1: Navigate to training directory
# ============================================================================
Write-Host "[1/6] Navigating to training directory..." -ForegroundColor Yellow

$ProjectRoot = "C:\Users\Joseph\Documents\Code\Toolset-Training"
$TrainingDir = Join-Path $ProjectRoot "Trainers\rtx3090_kto"
$DatasetPath = Join-Path $ProjectRoot "Datasets\syngen_tools_11.14.25.jsonl"

Set-Location $TrainingDir
Write-Host "  [OK] Current directory: $TrainingDir" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 2: Check dataset exists
# ============================================================================
Write-Host "[2/6] Checking dataset..." -ForegroundColor Yellow

if (-not (Test-Path $DatasetPath)) {
    Write-Host "  [ERROR] Dataset not found at $DatasetPath" -ForegroundColor Red
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

$Drive = Get-PSDrive C
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
# STEP 4: Find Python from unsloth_env
# ============================================================================
Write-Host "[4/6] Finding Python from unsloth_env..." -ForegroundColor Yellow

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
    Write-Host "  [ERROR] unsloth_env Python not found" -ForegroundColor Red
    Write-Host "  [INFO] Searched in:" -ForegroundColor Yellow
    foreach ($path in $UnslothEnvPaths) {
        Write-Host "    - $path" -ForegroundColor Yellow
    }
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  [OK] Python found: $PythonExe" -ForegroundColor Green

$PythonVersion = & $PythonExe --version 2>&1
Write-Host "  [OK] Version: $PythonVersion" -ForegroundColor Green

# Test CUDA availability
Write-Host "  -> Testing CUDA availability..." -ForegroundColor Cyan
$CudaTest = & $PythonExe -c "import torch; print(f'CUDA:{torch.cuda.is_available()}'); print(f'GPU:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')" 2>&1

if ($CudaTest -match "CUDA:True") {
    Write-Host "  [OK] CUDA is available" -ForegroundColor Green
    $GpuLine = $CudaTest | Select-String "GPU:" | Out-String
    Write-Host "  [OK] $($GpuLine.Trim())" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] CUDA not available in this environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# ============================================================================
# STEP 5: Show configuration summary
# ============================================================================
Write-Host "[5/6] Training Configuration:" -ForegroundColor Yellow
Write-Host "  Model: Qwen 2.5 3B Instruct" -ForegroundColor Cyan
Write-Host "  Dataset: syngen_tools_11.14.25.jsonl (4652 examples)" -ForegroundColor Cyan
Write-Host "  Batch size: 4 x 4 accum = 16 effective" -ForegroundColor Cyan
Write-Host "  Learning rate: 5e-7" -ForegroundColor Cyan
Write-Host "  Epochs: 2" -ForegroundColor Cyan
Write-Host "  Estimated steps: ~582" -ForegroundColor Cyan
Write-Host "  Estimated time: ~1.5-2 hours" -ForegroundColor Cyan
Write-Host ""

$Confirmation = Read-Host "Ready to start training? [Y/n]"
if ($Confirmation -eq "n" -or $Confirmation -eq "N") {
    Write-Host "  Cancelled by user." -ForegroundColor Yellow
    exit 0
}

Write-Host ""

# ============================================================================
# STEP 6: Run training
# ============================================================================
Write-Host "[6/6] Starting training..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Training output below:" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Run training with local dataset file
& $PythonExe train_kto.py --local-file "..\..\Datasets\syngen_tools_11.14.25.jsonl"

$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan

if ($ExitCode -eq 0) {
    Write-Host "[SUCCESS] Training completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Check output in: kto_output_rtx3090\" -ForegroundColor Cyan
    Write-Host "  2. Review training logs for metrics" -ForegroundColor Cyan
    Write-Host "  3. Upload to HuggingFace and test with evaluator" -ForegroundColor Cyan
} else {
    Write-Host "[ERROR] Training failed with exit code: $ExitCode" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
