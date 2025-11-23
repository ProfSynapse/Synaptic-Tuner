# ============================================================================
# Interactive Training CLI for Windows PowerShell
# Unified script for SFT, KTO, or SFT->KTO pipeline training
# ============================================================================

# Get script directory
$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir

# ============================================================================
# Display Header
# ============================================================================
Clear-Host
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "            Toolset-Training Interactive CLI                              " -ForegroundColor Cyan
Write-Host "         SFT & KTO Training for RTX 3090 / 4090                           " -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Find Python from unsloth_env
# ============================================================================
Write-Host "Locating Python environment..." -ForegroundColor Yellow

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
    Write-Host "[ERROR] unsloth_env Python not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  [OK] Python: $PythonExe" -ForegroundColor Green

# Verify CUDA
$CudaTest = & $PythonExe -c 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")' 2>&1
if ($CudaTest -match "CUDA") {
    $GpuName = & $PythonExe -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1
    Write-Host "  [OK] GPU: $GpuName" -ForegroundColor Green
} else {
    Write-Host "[ERROR] CUDA not available!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# ============================================================================
# Main Menu
# ============================================================================
Write-Host "Select Training Mode:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1) SFT Only           - Supervised Fine-Tuning (teaches tool-calling)"
Write-Host "  2) KTO Only           - Preference Learning (refines existing model)"
Write-Host "  3) SFT -> KTO Pipeline - Full training pipeline (recommended)"
Write-Host "  4) Exit"
Write-Host ""

$Choice = Read-Host "Enter choice [1-4]"

switch ($Choice) {
    "1" { $Mode = "sft" }
    "2" { $Mode = "kto" }
    "3" { $Mode = "pipeline" }
    "4" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# ============================================================================
# W&B Configuration
# ============================================================================
Write-Host "Weights & Biases Logging:" -ForegroundColor Yellow
Write-Host ""

$WandbEnable = Read-Host "Enable W&B logging? (y/n) [n]"
if ([string]::IsNullOrEmpty($WandbEnable)) { $WandbEnable = "n" }

$WandbFlag = ""
$WandbProject = ""

if ($WandbEnable -eq "y" -or $WandbEnable -eq "Y") {
    $WandbProjectName = Read-Host "W&B project name [toolset-training]"
    if ([string]::IsNullOrEmpty($WandbProjectName)) { $WandbProjectName = "toolset-training" }

    $WandbFlag = "--wandb"
    $WandbProject = "--wandb-project $WandbProjectName"
}

Write-Host ""

# ============================================================================
# Configuration Summary
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Configuration Summary" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

$ModeText = switch ($Mode) {
    "sft" { "SFT Only" }
    "kto" { "KTO Only" }
    "pipeline" { "SFT -> KTO Pipeline" }
}

Write-Host "  Training Mode:    $ModeText"
if ($WandbFlag) {
    Write-Host "  W&B Logging:      Enabled ($WandbProjectName)"
} else {
    Write-Host "  W&B Logging:      Disabled"
}
Write-Host ""

if ($Mode -eq "sft" -or $Mode -eq "pipeline") {
    Write-Host "  SFT Configuration:" -ForegroundColor White
    Write-Host "    - Config: rtx3090_sft/configs/config.yaml" -ForegroundColor Gray

    # Load SFT config dynamically
    $SftConfigScript = @"
import sys
sys.path.insert(0, r'$ScriptDir/rtx3090_sft')
from configs.config_loader import get_7b_config
config = get_7b_config()
model_name = config.model.model_name.split('/')[-1]
dataset_file = config.dataset.dataset_file
learning_rate = config.training.learning_rate
epochs = config.training.num_train_epochs
print(f'MODEL={model_name}')
print(f'DATASET={dataset_file}')
print(f'LR={learning_rate}')
print(f'EPOCHS={epochs}')
"@

    $SftConfigValues = & $PythonExe -c $SftConfigScript
    $SftConfig = @{}
    foreach ($line in $SftConfigValues) {
        $parts = $line -split '=', 2
        if ($parts.Count -eq 2) {
            $SftConfig[$parts[0]] = $parts[1]
        }
    }

    Write-Host "    - Model: $($SftConfig['MODEL'])" -ForegroundColor Gray
    Write-Host "    - Dataset: $($SftConfig['DATASET'])" -ForegroundColor Gray
    Write-Host "    - Learning rate: $($SftConfig['LR']), Epochs: $($SftConfig['EPOCHS'])" -ForegroundColor Gray
    Write-Host ""
}

if ($Mode -eq "kto" -or $Mode -eq "pipeline") {
    Write-Host "  KTO Configuration:" -ForegroundColor White
    Write-Host "    - Config: rtx3090_kto/configs/config.yaml" -ForegroundColor Gray

    # Load KTO config dynamically
    $KtoConfigScript = @"
import sys
sys.path.insert(0, r'$ScriptDir/rtx3090_kto')
from configs.config_loader import get_7b_config
config = get_7b_config()
model_name = config.model.model_name.split('/')[-1] if '/' in config.model.model_name else config.model.model_name
dataset_file = config.dataset.dataset_file
learning_rate = config.training.learning_rate
epochs = config.training.num_train_epochs
beta = config.training.beta
desirable_weight = config.training.desirable_weight
undesirable_weight = config.training.undesirable_weight
print(f'MODEL={model_name}')
print(f'DATASET={dataset_file}')
print(f'LR={learning_rate}')
print(f'EPOCHS={epochs}')
print(f'BETA={beta}')
print(f'DESIRABLE_WEIGHT={desirable_weight}')
print(f'UNDESIRABLE_WEIGHT={undesirable_weight}')
"@

    $KtoConfigValues = & $PythonExe -c $KtoConfigScript
    $KtoConfig = @{}
    foreach ($line in $KtoConfigValues) {
        $parts = $line -split '=', 2
        if ($parts.Count -eq 2) {
            $KtoConfig[$parts[0]] = $parts[1]
        }
    }

    if ($Mode -eq "kto") {
        Write-Host "    - Model: $($KtoConfig['MODEL'])" -ForegroundColor Gray
    } else {
        Write-Host "    - Model: SFT output (auto-detected)" -ForegroundColor Gray
    }
    Write-Host "    - Dataset: $($KtoConfig['DATASET'])" -ForegroundColor Gray
    Write-Host "    - Learning rate: $($KtoConfig['LR']), Epochs: $($KtoConfig['EPOCHS'])" -ForegroundColor Gray
    Write-Host "    - Beta: $($KtoConfig['BETA']), Weights: $($KtoConfig['DESIRABLE_WEIGHT'])/$($KtoConfig['UNDESIRABLE_WEIGHT']) (desirable/undesirable)" -ForegroundColor Gray
    Write-Host ""
}

$Confirm = Read-Host "Continue with this configuration? (y/n)"
if ($Confirm -ne "y" -and $Confirm -ne "Y") {
    Write-Host "Training cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""

# ============================================================================
# Execute Training
# ============================================================================

if ($Mode -eq "sft") {
    # ========================================================================
    # SFT Only
    # ========================================================================
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "SFT Training (Supervised Fine-Tuning)" -ForegroundColor Cyan
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host ""

    Set-Location "rtx3090_sft"
    & $PythonExe train_sft.py $WandbFlag $WandbProject

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] SFT training failed (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        Set-Location $ScriptDir
        Read-Host "Press Enter to exit"
        exit $LASTEXITCODE
    }

    # Find output
    $SftOutputDirs = Get-ChildItem -Path "sft_output_rtx3090" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($SftOutputDirs) {
        $SftOutputDir = $SftOutputDirs[0].FullName
        Write-Host ""
        Write-Host "[SUCCESS] SFT Training Complete!" -ForegroundColor Green
        Write-Host "  Output: $SftOutputDir" -ForegroundColor White
    }

} elseif ($Mode -eq "kto") {
    # ========================================================================
    # KTO Only
    # ========================================================================
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "KTO Training (Preference Learning)" -ForegroundColor Cyan
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Note: KTO is designed for refinement. Ensure you're using an SFT-trained model." -ForegroundColor Yellow
    Write-Host ""

    Set-Location "rtx3090_kto"
    & $PythonExe train_kto.py $WandbFlag $WandbProject

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] KTO training failed (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        Set-Location $ScriptDir
        Read-Host "Press Enter to exit"
        exit $LASTEXITCODE
    }

    # Find output
    $KtoOutputDirs = Get-ChildItem -Path "kto_output_rtx3090" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($KtoOutputDirs) {
        $KtoOutputDir = $KtoOutputDirs[0].FullName
        Write-Host ""
        Write-Host "[SUCCESS] KTO Training Complete!" -ForegroundColor Green
        Write-Host "  Output: $KtoOutputDir" -ForegroundColor White
    }

} elseif ($Mode -eq "pipeline") {
    # ========================================================================
    # SFT -> KTO Pipeline
    # ========================================================================
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "Phase 1: SFT Training (Teaching Tool-Calling Syntax)" -ForegroundColor Cyan
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host ""

    Set-Location "rtx3090_sft"
    & $PythonExe train_sft.py $WandbFlag $WandbProject

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] SFT training failed (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        Set-Location $ScriptDir
        Read-Host "Press Enter to exit"
        exit $LASTEXITCODE
    }

    # Find SFT output
    $SftOutputDirs = Get-ChildItem -Path "sft_output_rtx3090" -Directory | Sort-Object LastWriteTime -Descending
    if ($SftOutputDirs.Count -eq 0) {
        Write-Host "[ERROR] No SFT output directory found" -ForegroundColor Red
        Set-Location $ScriptDir
        Read-Host "Press Enter to exit"
        exit 1
    }

    $SftOutputDir = $SftOutputDirs[0].FullName
    $SftFinalModel = Join-Path $SftOutputDir "final_model"

    Write-Host ""
    Write-Host "[SUCCESS] Phase 1 Complete!" -ForegroundColor Green
    Write-Host "  Output: $SftOutputDir" -ForegroundColor White
    Write-Host "  Model: $SftFinalModel" -ForegroundColor White
    Write-Host ""

    # Ask to continue to KTO
    $ContinueKto = Read-Host "Continue to KTO refinement? (y/n)"
    if ($ContinueKto -ne "y" -and $ContinueKto -ne "Y") {
        Write-Host "Pipeline stopped after SFT." -ForegroundColor Yellow
        Set-Location $ScriptDir
        exit 0
    }

    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "Phase 2: KTO Training (Preference Learning Refinement)" -ForegroundColor Cyan
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host ""

    Set-Location "../rtx3090_kto"

    Write-Host "  Updating KTO config to use SFT output model..." -ForegroundColor Yellow
    Write-Host "  New model_name: $SftFinalModel" -ForegroundColor White

    # Update KTO config with SFT model path
    $UpdateConfigScript = @"
import yaml
from pathlib import Path

config_path = Path('configs/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['model']['model_name'] = r'$SftFinalModel'

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

print('[OK] Updated KTO config')
"@

    & $PythonExe -c $UpdateConfigScript

    Write-Host ""
    & $PythonExe train_kto.py $WandbFlag $WandbProject

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[ERROR] KTO training failed (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        Set-Location $ScriptDir
        Read-Host "Press Enter to exit"
        exit $LASTEXITCODE
    }

    # Find KTO output
    $KtoOutputDirs = Get-ChildItem -Path "kto_output_rtx3090" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($KtoOutputDirs) {
        $KtoOutputDir = $KtoOutputDirs[0].FullName
    }

    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host "  [SUCCESS] Complete SFT->KTO Pipeline Finished Successfully!" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Training Outputs:" -ForegroundColor White
    Write-Host "  SFT Output:  $SftOutputDir" -ForegroundColor Gray
    Write-Host "  KTO Output:  $KtoOutputDir" -ForegroundColor Gray
}

# ============================================================================
# Next Steps
# ============================================================================
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Next Steps" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Test the model:" -ForegroundColor White
Write-Host "   cd Evaluator" -ForegroundColor Gray
if ($Mode -eq "kto" -or $Mode -eq "pipeline") {
    if ($KtoOutputDir) {
        Write-Host "   python cli.py --model `"$KtoOutputDir\final_model`" --prompt-set prompts/baseline.json" -ForegroundColor Gray
    }
} else {
    if ($SftOutputDir) {
        Write-Host "   python cli.py --model `"$SftOutputDir\final_model`" --prompt-set prompts/baseline.json" -ForegroundColor Gray
    }
}
Write-Host ""
Write-Host "2. Upload to HuggingFace:" -ForegroundColor White
if ($Mode -eq "kto" -or $Mode -eq "pipeline") {
    Write-Host "   cd Trainers\rtx3090_kto" -ForegroundColor Gray
} else {
    Write-Host "   cd Trainers\rtx3090_sft" -ForegroundColor Gray
}
Write-Host "   .\upload_model.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Create GGUF quantizations:" -ForegroundColor White
Write-Host "   Select 'Create GGUF' option during upload" -ForegroundColor Gray
Write-Host ""

Set-Location $ScriptDir
Read-Host "Press Enter to exit"
exit 0
