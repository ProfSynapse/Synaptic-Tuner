# ============================================================================
# GGUF Creation Script for nexus-tools-v0.0.1
# ============================================================================
# This script will:
# 1. Check prerequisites (conda, disk space, files)
# 2. Read HuggingFace token from .env
# 3. Create GGUF quantized versions (Q4_K_M, Q5_K_M, Q8_0)
# 4. Upload to HuggingFace
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "GGUF Creation Script for nexus-tools-v0.0.1" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 1: Navigate to correct directory
# ============================================================================
Write-Host "[1/7] Navigating to project directory..." -ForegroundColor Yellow

$ProjectRoot = "C:\Users\Joseph\Documents\Code\Toolset-Training"
$TrainingDir = Join-Path $ProjectRoot "code\rtx3090_kto"
$TrainingOutput = Join-Path $TrainingDir "kto_output_rtx3090\20251116_121211"
$EnvFile = Join-Path $ProjectRoot ".env"

Set-Location $TrainingDir
Write-Host "  [OK] Current directory: $TrainingDir" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 2: Check if training output exists
# ============================================================================
Write-Host "[2/7] Checking training output..." -ForegroundColor Yellow

if (-not (Test-Path $TrainingOutput)) {
    Write-Host "  [ERROR] Training output not found at $TrainingOutput" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$AdapterFile = Join-Path $TrainingOutput "adapter_model.safetensors"
if (-not (Test-Path $AdapterFile)) {
    Write-Host "  [ERROR] adapter_model.safetensors not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$AdapterSize = (Get-Item $AdapterFile).Length / 1MB
Write-Host "  [OK] Training output found" -ForegroundColor Green
Write-Host "  [OK] Adapter model: $([math]::Round($AdapterSize, 1)) MB" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 3: Check disk space
# ============================================================================
Write-Host "[3/7] Checking disk space..." -ForegroundColor Yellow

$Drive = Get-PSDrive C
$FreeSpaceGB = [math]::Round($Drive.Free / 1GB, 2)
$RequiredGB = 20

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
Write-Host "[4/7] Finding Python from unsloth_env..." -ForegroundColor Yellow

# Look for unsloth_env Python specifically
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
    Write-Host ""
    Write-Host "  [INFO] Please ensure you have created the unsloth_env conda environment" -ForegroundColor Yellow
    Write-Host "  [INFO] Or modify this script to use a different environment name" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Test Python and CUDA
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
    Write-Host "  [INFO] Make sure unsloth_env has GPU-enabled PyTorch installed" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# ============================================================================
# STEP 5: Read HuggingFace token
# ============================================================================
Write-Host "[5/7] Reading HuggingFace token..." -ForegroundColor Yellow

if (-not (Test-Path $EnvFile)) {
    Write-Host "  [ERROR] .env file not found at $EnvFile" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$TokenLine = Get-Content $EnvFile | Where-Object { $_ -match "HF_API_KEY=" }
if (-not $TokenLine) {
    Write-Host "  [ERROR] HF_API_KEY not found in .env file" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$HFToken = ($TokenLine -split "=", 2)[1].Trim()
if (-not $HFToken -or $HFToken.Length -lt 10) {
    Write-Host "  [ERROR] Invalid HuggingFace token" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$MaskedToken = $HFToken.Substring(0, 7) + "..." + $HFToken.Substring($HFToken.Length - 4)
Write-Host "  [OK] Token found: $MaskedToken" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 6: Confirm and show summary
# ============================================================================
Write-Host "[6/7] Configuration Summary:" -ForegroundColor Yellow
Write-Host "  Model: professorsynapse/nexus-tools-v0.0.1" -ForegroundColor Cyan
Write-Host "  Training output: .\kto_output_rtx3090\20251116_121211" -ForegroundColor Cyan
Write-Host "  Quantizations: Q4_K_M (~3.5GB), Q5_K_M (~4.5GB), Q8_0 (~7GB)" -ForegroundColor Cyan
Write-Host "  Skip standard upload: Yes (already on HuggingFace)" -ForegroundColor Cyan
Write-Host "  Python: $PythonExe" -ForegroundColor Cyan
Write-Host "  CUDA: Available with RTX 3090" -ForegroundColor Cyan
Write-Host "  Estimated time: 30-40 minutes" -ForegroundColor Cyan
Write-Host ""

$Confirmation = Read-Host "Ready to start GGUF creation? [Y/n]"
if ($Confirmation -eq "n" -or $Confirmation -eq "N") {
    Write-Host "  Cancelled by user." -ForegroundColor Yellow
    exit 0
}

Write-Host ""

# ============================================================================
# STEP 7: Run GGUF creation
# ============================================================================
Write-Host "[7/7] Starting GGUF creation..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "This will take 30-40 minutes. Output from Python script:" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Build the command
$PythonScript = Join-Path $TrainingDir "src\upload_to_hf.py"
$ModelPath = ".\kto_output_rtx3090\20251116_121211"
$RepoID = "professorsynapse/nexus-tools-v0.0.1"

# Run the upload script
& $PythonExe $PythonScript `
    $ModelPath `
    $RepoID `
    --token $HFToken `
    --create-gguf `
    --skip-standard

$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan

if ($ExitCode -eq 0) {
    Write-Host "[SUCCESS] GGUF creation completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. View your model: https://huggingface.co/professorsynapse/nexus-tools-v0.0.1" -ForegroundColor Cyan
    Write-Host "  2. Pull into Ollama: ollama pull hf.co/professorsynapse/nexus-tools-v0.0.1" -ForegroundColor Cyan
    Write-Host "  3. Test the model: ollama run nexus-tools-v0.0.1 'Hello!'" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "[ERROR] GGUF creation failed with exit code: $ExitCode" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
    Write-Host ""
}

Read-Host "Press Enter to exit"
