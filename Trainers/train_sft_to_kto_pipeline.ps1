# ============================================================================
# SFT → KTO Training Pipeline for Windows PowerShell
# ============================================================================
# This script chains supervised fine-tuning with preference learning:
# 1. Runs SFT training (teaches tool-calling syntax)
# 2. Captures SFT output directory
# 3. Updates KTO config with SFT model path
# 4. Runs KTO training (refines quality)
#
# Usage:
#   .\train_sft_to_kto_pipeline.ps1
# ============================================================================

Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         SFT → KTO Training Pipeline                           ║" -ForegroundColor Cyan
Write-Host "║              Using YAML Configurations                        ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 1: Find Python from unsloth_env
# ============================================================================
Write-Host "[1/5] Finding Python environment..." -ForegroundColor Yellow

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
    Write-Host "  [INFO] Expected locations:" -ForegroundColor Yellow
    foreach ($path in $UnslothEnvPaths) {
        Write-Host "    - $path" -ForegroundColor Gray
    }
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  [OK] Python: $PythonExe" -ForegroundColor Green

# Verify CUDA
$CudaTest = & $PythonExe -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1
if ($CudaTest -match "CUDA") {
    $GpuName = & $PythonExe -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "  [OK] GPU: $GpuName" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] CUDA not available!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# ============================================================================
# STEP 2: Get script directory and navigate
# ============================================================================
Write-Host "[2/5] Preparing directories..." -ForegroundColor Yellow

$ScriptDir = $PSScriptRoot
$TrainersDir = $ScriptDir
Set-Location $TrainersDir

Write-Host "  [OK] Working directory: $TrainersDir" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 3: Display configuration
# ============================================================================
Write-Host "[3/5] Configuration summary..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Pipeline Steps:" -ForegroundColor White
Write-Host "    1. SFT Training (Supervised Fine-Tuning)" -ForegroundColor White
Write-Host "       - Config: rtx3090_sft/configs/config.yaml" -ForegroundColor Gray
Write-Host "       - Model: mistral-7b-instruct-v0.3-bnb-4bit" -ForegroundColor Gray
Write-Host "       - Dataset: syngen_tools_sft_11.22.25.jsonl" -ForegroundColor Gray
Write-Host "       - Learning rate: 2e-4, Epochs: 3" -ForegroundColor Gray
Write-Host ""
Write-Host "    2. KTO Refinement (Preference Learning)" -ForegroundColor White
Write-Host "       - Config: rtx3090_kto/configs/config.yaml" -ForegroundColor Gray
Write-Host "       - Model: SFT output (auto-detected)" -ForegroundColor Gray
Write-Host "       - Dataset: syngen_tools_11.18.25.jsonl" -ForegroundColor Gray
Write-Host "       - Learning rate: 2e-7, Epochs: 1" -ForegroundColor Gray
Write-Host ""

# ============================================================================
# STEP 4: Confirm start
# ============================================================================
Write-Host "[4/5] Ready to start pipeline" -ForegroundColor Yellow
Write-Host ""
$Confirmation = Read-Host "Start SFT → KTO training? (y/n)"

if ($Confirmation -ne "y" -and $Confirmation -ne "Y") {
    Write-Host "Pipeline cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""

# ============================================================================
# PHASE 1: SFT Training
# ============================================================================
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Phase 1: SFT Training (Teaching Tool-Calling Syntax)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Set-Location "rtx3090_sft"

Write-Host "  Starting SFT training..." -ForegroundColor Yellow
Write-Host "  (This may take 45-60 minutes for 3 epochs)" -ForegroundColor Gray
Write-Host ""

# Run SFT training
& $PythonExe train_sft.py

$SftExitCode = $LASTEXITCODE

if ($SftExitCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] SFT training failed (Exit code: $SftExitCode)" -ForegroundColor Red
    Write-Host "Pipeline aborted." -ForegroundColor Red
    Set-Location $TrainersDir
    Read-Host "Press Enter to exit"
    exit $SftExitCode
}

Write-Host ""
Write-Host "✓ Phase 1 complete!" -ForegroundColor Green

# Find the most recent SFT output directory
$SftOutputDirs = Get-ChildItem -Path "sft_output_rtx3090" -Directory | Sort-Object LastWriteTime -Descending
if ($SftOutputDirs.Count -eq 0) {
    Write-Host "[ERROR] No SFT output directory found" -ForegroundColor Red
    Set-Location $TrainersDir
    Read-Host "Press Enter to exit"
    exit 1
}

$SftOutputDir = $SftOutputDirs[0].FullName
$SftFinalModel = Join-Path $SftOutputDir "final_model"

if (-not (Test-Path $SftFinalModel)) {
    Write-Host "[ERROR] SFT final_model directory not found: $SftFinalModel" -ForegroundColor Red
    Set-Location $TrainersDir
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  Output: $SftOutputDir" -ForegroundColor White
Write-Host "  Model: $SftFinalModel" -ForegroundColor White
Write-Host ""

# ============================================================================
# PHASE 2: KTO Training
# ============================================================================
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Phase 2: KTO Training (Preference Learning Refinement)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Set-Location "../rtx3090_kto"

Write-Host "  Base model: $SftFinalModel" -ForegroundColor White
Write-Host ""
Write-Host "  Updating KTO config to use SFT output model..." -ForegroundColor Yellow

# Update KTO config with SFT model path using Python
$UpdateConfigScript = @"
import yaml
from pathlib import Path

config_path = Path('configs/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update model_name to point to SFT output
config['model']['model_name'] = r'$SftFinalModel'

# Write back with preserved formatting
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

print(f'✓ Updated KTO config: model.model_name = {config[\"model\"][\"model_name\"]}')
"@

& $PythonExe -c $UpdateConfigScript

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Failed to update KTO config" -ForegroundColor Red
    Set-Location $TrainersDir
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "  Starting KTO training..." -ForegroundColor Yellow
Write-Host "  (This may take 15-25 minutes for 1 epoch)" -ForegroundColor Gray
Write-Host ""

# Run KTO training
& $PythonExe train_kto.py

$KtoExitCode = $LASTEXITCODE

if ($KtoExitCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] KTO training failed (Exit code: $KtoExitCode)" -ForegroundColor Red
    Set-Location $TrainersDir
    Read-Host "Press Enter to exit"
    exit $KtoExitCode
}

Write-Host ""
Write-Host "✓ Phase 2 complete!" -ForegroundColor Green

# Find the most recent KTO output directory
$KtoOutputDirs = Get-ChildItem -Path "kto_output_rtx3090" -Directory | Sort-Object LastWriteTime -Descending
if ($KtoOutputDirs.Count -gt 0) {
    $KtoOutputDir = $KtoOutputDirs[0].FullName
    $KtoFinalModel = Join-Path $KtoOutputDir "final_model"
    Write-Host "  Output: $KtoOutputDir" -ForegroundColor White
    Write-Host "  Model: $KtoFinalModel" -ForegroundColor White
}

Write-Host ""

# ============================================================================
# STEP 5: Summary
# ============================================================================
Write-Host "[5/5] Pipeline Summary" -ForegroundColor Yellow
Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✓ Complete SFT→KTO Pipeline Finished Successfully!          ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Training Outputs:" -ForegroundColor White
Write-Host "  SFT Output:  $SftOutputDir" -ForegroundColor Gray
Write-Host "  KTO Output:  $KtoOutputDir" -ForegroundColor Gray
Write-Host ""
Write-Host "Configuration Files:" -ForegroundColor White
Write-Host "  SFT used: Trainers/rtx3090_sft/configs/config.yaml" -ForegroundColor Gray
Write-Host "  KTO used: Trainers/rtx3090_kto/configs/config.yaml (updated with SFT model path)" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor White
Write-Host "  1. Test the model:" -ForegroundColor White
Write-Host "     cd Evaluator" -ForegroundColor Gray
Write-Host "     python cli.py --model `"$KtoFinalModel`" --prompt-set prompts/baseline.json" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Upload to HuggingFace:" -ForegroundColor White
Write-Host "     cd Trainers\rtx3090_kto" -ForegroundColor Gray
Write-Host "     .\upload_model.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Create GGUF quantizations:" -ForegroundColor White
Write-Host "     Select 'Create GGUF' option during upload" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. To revert KTO config to original model:" -ForegroundColor White
Write-Host "     git checkout Trainers/rtx3090_kto/configs/config.yaml" -ForegroundColor Gray
Write-Host ""

Set-Location $TrainersDir

Read-Host "Press Enter to exit"
exit 0
