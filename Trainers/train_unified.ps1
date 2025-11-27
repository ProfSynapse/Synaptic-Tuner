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
Write-Host "Select Mode:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1) SFT Only            - Supervised Fine-Tuning (teaches tool-calling)"
Write-Host "  2) KTO Only            - Preference Learning (refines existing model)"
Write-Host "  3) SFT -> KTO Pipeline - Full training pipeline (recommended)"
Write-Host "  4) Evaluate Model      - Run evaluation on a trained model"
Write-Host "  5) Exit"
Write-Host ""

$Choice = Read-Host "Enter choice [1-5]"

switch ($Choice) {
    "1" { $Mode = "sft" }
    "2" { $Mode = "kto" }
    "3" { $Mode = "pipeline" }
    "4" { $Mode = "evaluate" }
    "5" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

# ============================================================================
# Standalone Evaluation Mode
# ============================================================================
if ($Mode -eq "evaluate") {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "Model Evaluation" -ForegroundColor Cyan
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This will run the evaluation CLI against a model running in LM Studio or Ollama." -ForegroundColor White
    Write-Host ""

    # Backend selection
    Write-Host "Select Backend:" -ForegroundColor Yellow
    Write-Host "  1) LM Studio (recommended)"
    Write-Host "  2) Ollama"
    Write-Host ""
    $BackendChoice = Read-Host "Enter choice [1]"
    if ([string]::IsNullOrEmpty($BackendChoice)) { $BackendChoice = "1" }

    $Backend = switch ($BackendChoice) {
        "1" { "lmstudio" }
        "2" { "ollama" }
        default { "lmstudio" }
    }

    # Set API endpoint based on backend
    $ApiBaseUrl = if ($Backend -eq "lmstudio") { "http://localhost:1234" } else { "http://localhost:11434" }

    Write-Host ""
    Write-Host "Prerequisites:" -ForegroundColor Yellow
    if ($Backend -eq "lmstudio") {
        Write-Host "  1. Open LM Studio" -ForegroundColor White
        Write-Host "  2. Load your model (GGUF format)" -ForegroundColor White
        Write-Host "  3. Start the local server (Developer tab -> Start Server)" -ForegroundColor White
    } else {
        Write-Host "  1. Ensure Ollama is running" -ForegroundColor White
        Write-Host "  2. Pull/create your model: ollama pull <model-name>" -ForegroundColor White
    }
    Write-Host ""

    $Ready = Read-Host "Is your model loaded and server running? (y/n)"
    if ($Ready -ne "y" -and $Ready -ne "Y") {
        Write-Host "Please set up your model first, then run this script again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 0
    }

    # Fetch available models from API
    Write-Host ""
    Write-Host "Fetching available models..." -ForegroundColor Yellow

    $ModelName = $null
    try {
        $ModelsResponse = Invoke-RestMethod -Uri "$ApiBaseUrl/v1/models" -Method Get -TimeoutSec 5
        $Models = $ModelsResponse.data | ForEach-Object { $_.id }

        if ($Models.Count -eq 0) {
            Write-Host "[WARN] No models found. Please load a model first." -ForegroundColor Yellow
            Read-Host "Press Enter to exit"
            exit 0
        } elseif ($Models.Count -eq 1) {
            $ModelName = $Models[0]
            Write-Host "  Using only available model: $ModelName" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "Available Models:" -ForegroundColor Yellow
            for ($i = 0; $i -lt $Models.Count; $i++) {
                Write-Host "  $($i + 1)) $($Models[$i])"
            }
            Write-Host ""
            $ModelChoice = Read-Host "Select model [1]"
            if ([string]::IsNullOrEmpty($ModelChoice)) { $ModelChoice = "1" }

            $ModelIndex = [int]$ModelChoice - 1
            if ($ModelIndex -ge 0 -and $ModelIndex -lt $Models.Count) {
                $ModelName = $Models[$ModelIndex]
            } else {
                $ModelName = $Models[0]
            }
        }
        Write-Host "  Selected: $ModelName" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Could not connect to $Backend API at $ApiBaseUrl" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Make sure the server is running and try again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host ""
    Write-Host "Evaluation Type:" -ForegroundColor Yellow
    Write-Host "  1) Tool Evaluation      - Tests tool-calling accuracy (47 prompts)"
    Write-Host "  2) Behavior Evaluation  - Tests behavioral patterns"
    Write-Host "  3) Full Evaluation      - Both tool and behavior tests"
    Write-Host ""
    $EvalTypeChoice = Read-Host "Select evaluation type [3]"
    if ([string]::IsNullOrEmpty($EvalTypeChoice)) { $EvalTypeChoice = "3" }

    Write-Host ""
    Write-Host "Starting evaluation..." -ForegroundColor Yellow
    Write-Host ""

    # Run evaluator based on choice
    Set-Location (Join-Path $ScriptDir "../Evaluator")

    switch ($EvalTypeChoice) {
        "1" {
            # Tool evaluation only
            Write-Host "Running Tool Evaluation..." -ForegroundColor Cyan
            $EvalArgs = @(
                "cli.py",
                "--backend", $Backend,
                "--model", $ModelName,
                "--prompt-set", "prompts/full_coverage.json"
            )
            & $PythonExe @EvalArgs
        }
        "2" {
            # Behavior evaluation only
            Write-Host "Running Behavior Evaluation..." -ForegroundColor Cyan
            $EvalArgs = @(
                "cli.py",
                "--backend", $Backend,
                "--model", $ModelName,
                "--prompt-set", "prompts/behavioral_patterns.json"
            )
            & $PythonExe @EvalArgs
        }
        default {
            # Full evaluation (both)
            Write-Host "Running Full Evaluation (Tools + Behavior)..." -ForegroundColor Cyan
            Write-Host ""

            Write-Host "--- Tool Evaluation ---" -ForegroundColor Yellow
            $EvalArgs = @(
                "cli.py",
                "--backend", $Backend,
                "--model", $ModelName,
                "--prompt-set", "prompts/full_coverage.json"
            )
            & $PythonExe @EvalArgs

            Write-Host ""
            Write-Host "--- Behavior Evaluation ---" -ForegroundColor Yellow
            $EvalArgs = @(
                "cli.py",
                "--backend", $Backend,
                "--model", $ModelName,
                "--prompt-set", "prompts/behavioral_patterns.json"
            )
            & $PythonExe @EvalArgs

            Write-Host ""
            Write-Host "============================================================================" -ForegroundColor Green
            Write-Host "  Full Evaluation Complete!" -ForegroundColor Green
            Write-Host "============================================================================" -ForegroundColor Green
        }
    }

    Set-Location $ScriptDir
    Read-Host "Press Enter to exit"
    exit 0
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
# HuggingFace Upload Configuration (Collect Upfront)
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "HuggingFace Upload Configuration" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

$DoUpload = Read-Host "Upload to HuggingFace when training completes? (y/n) [n]"
if ([string]::IsNullOrEmpty($DoUpload)) { $DoUpload = "n" }

$HfToken = $null
$RepoName = $null
$SaveMethod = $null
$GgufFlag = ""

if ($DoUpload -eq "y" -or $DoUpload -eq "Y") {
    # Get HF token and username from .env file
    $EnvFile = Join-Path $ScriptDir "../.env"
    $HfUsernameFromEnv = $null
    if (Test-Path $EnvFile) {
        $EnvContent = Get-Content $EnvFile
        foreach ($line in $EnvContent) {
            if ($line -match "^HF_TOKEN=(.+)$") {
                $HfToken = $Matches[1].Trim('"').Trim("'")
            } elseif ($line -match "^HF_API_KEY=(.+)$" -and -not $HfToken) {
                $HfToken = $Matches[1].Trim('"').Trim("'")
            } elseif ($line -match "^HF_USERNAME=(.+)$") {
                $HfUsernameFromEnv = $Matches[1].Trim('"').Trim("'")
            }
        }
    }

    if (-not $HfToken) {
        Write-Host "[WARN] HF_TOKEN not found in .env file" -ForegroundColor Yellow
        $HfToken = Read-Host "Enter HuggingFace Token (with write access)"
        if ([string]::IsNullOrEmpty($HfToken)) {
            Write-Host "No token provided - upload will be skipped" -ForegroundColor Yellow
            $DoUpload = "n"
        }
    } else {
        Write-Host "  [OK] HF_TOKEN loaded from .env" -ForegroundColor Green
    }

    if ($HfUsernameFromEnv) {
        Write-Host "  [OK] HF_USERNAME loaded from .env: $HfUsernameFromEnv" -ForegroundColor Green
    }
}

if ($DoUpload -eq "y" -or $DoUpload -eq "Y") {
    Write-Host ""

    # Get HuggingFace username (use from .env if available)
    if ($HfUsernameFromEnv) {
        $HfUsername = Read-Host "HuggingFace username [$HfUsernameFromEnv]"
        if ([string]::IsNullOrEmpty($HfUsername)) { $HfUsername = $HfUsernameFromEnv }
    } else {
        $HfUsername = Read-Host "HuggingFace username"
    }

    # Generate default repo name based on mode
    $DefaultModelName = switch ($Mode) {
        "sft" { "toolset-sft-" + (Get-Date -Format "yyyyMMdd") }
        "kto" { "toolset-kto-" + (Get-Date -Format "yyyyMMdd") }
        "pipeline" { "toolset-sft-kto-" + (Get-Date -Format "yyyyMMdd") }
    }

    $ModelName = Read-Host "Model name [$DefaultModelName]"
    if ([string]::IsNullOrEmpty($ModelName)) { $ModelName = $DefaultModelName }

    $RepoName = "$HfUsername/$ModelName"

    Write-Host ""
    Write-Host "Save Method Options:" -ForegroundColor Yellow
    Write-Host "  1) merged_16bit - Full quality (~14GB) [Recommended]"
    Write-Host "  2) merged_4bit  - Smaller size (~3.5GB)"
    Write-Host "  3) lora         - LoRA adapters only (~320MB)"
    Write-Host ""
    $SaveMethodChoice = Read-Host "Select save method [1]"
    if ([string]::IsNullOrEmpty($SaveMethodChoice)) { $SaveMethodChoice = "1" }

    $SaveMethod = switch ($SaveMethodChoice) {
        "1" { "merged_16bit" }
        "2" { "merged_4bit" }
        "3" { "lora" }
        default { "merged_16bit" }
    }

    Write-Host ""
    $CreateGguf = Read-Host "Create GGUF quantizations (Q4_K_M, Q5_K_M, Q8_0)? (y/n) [n]"
    if ([string]::IsNullOrEmpty($CreateGguf)) { $CreateGguf = "n" }

    if ($CreateGguf -eq "y" -or $CreateGguf -eq "Y") {
        $GgufFlag = "--create-gguf"
    }
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

# Upload settings
if ($DoUpload -eq "y" -or $DoUpload -eq "Y") {
    Write-Host "  HuggingFace Upload:" -ForegroundColor Green
    Write-Host "    Repository:     $RepoName" -ForegroundColor White
    Write-Host "    Save Method:    $SaveMethod" -ForegroundColor White
    Write-Host "    Create GGUF:    $(if ($GgufFlag) { 'Yes (Q4_K_M, Q5_K_M, Q8_0)' } else { 'No' })" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "  HuggingFace Upload: Disabled" -ForegroundColor Gray
    Write-Host ""
}

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
# Determine Final Model Path
# ============================================================================
$FinalModelPath = $null
$TrainerDir = $null

if ($Mode -eq "kto" -or $Mode -eq "pipeline") {
    if ($KtoOutputDir) {
        $FinalModelPath = "$KtoOutputDir\final_model"
        $TrainerDir = "rtx3090_kto"
    }
} else {
    if ($SftOutputDir) {
        $FinalModelPath = "$SftOutputDir\final_model"
        $TrainerDir = "rtx3090_sft"
    }
}

# ============================================================================
# Automatic Upload to HuggingFace (uses settings collected upfront)
# ============================================================================
if ($DoUpload -eq "y" -or $DoUpload -eq "Y") {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host "Uploading to HuggingFace (Automatic)" -ForegroundColor Cyan
    Write-Host "============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Model Path:    $FinalModelPath" -ForegroundColor White
    Write-Host "  Repository:    $RepoName" -ForegroundColor White
    Write-Host "  Save Method:   $SaveMethod" -ForegroundColor White
    Write-Host "  Create GGUF:   $(if ($GgufFlag) { 'Yes' } else { 'No' })" -ForegroundColor White
    Write-Host ""

    # Check for training lineage
    $LineagePath = Join-Path (Split-Path $FinalModelPath -Parent) "training_lineage.json"
    if (Test-Path $LineagePath) {
        Write-Host "  [OK] Training lineage found - generating comprehensive model card" -ForegroundColor Green
    } else {
        Write-Host "  [INFO] No training lineage found - generating basic model card" -ForegroundColor Yellow
    }
    Write-Host ""

    Write-Host "Starting upload..." -ForegroundColor Yellow
    Write-Host ""

    # Run the upload script
    Set-Location (Join-Path $ScriptDir $TrainerDir)

    $UploadArgs = @(
        "src/upload_to_hf.py",
        $FinalModelPath,
        $RepoName,
        "--save-method", $SaveMethod
    )

    if ($GgufFlag) {
        $UploadArgs += $GgufFlag
    }

    # Set HF_TOKEN environment variable for the upload
    $env:HF_TOKEN = $HfToken

    & $PythonExe @UploadArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "============================================================================" -ForegroundColor Green
        Write-Host "  [SUCCESS] Upload Complete!" -ForegroundColor Green
        Write-Host "============================================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "  View your model at: https://huggingface.co/$RepoName" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "[ERROR] Upload failed (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        Write-Host ""
    }

    Set-Location $ScriptDir
}

# ============================================================================
# Optional: Run Evaluation
# ============================================================================
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Model Evaluation (Optional)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Would you like to run evaluation on the trained model?" -ForegroundColor White
Write-Host ""
Write-Host "To evaluate, you'll need to:" -ForegroundColor Yellow
Write-Host "  1. Download the model to LM Studio (from HuggingFace or local GGUF)" -ForegroundColor White
Write-Host "  2. Load the model in LM Studio" -ForegroundColor White
Write-Host "  3. Start the local server (Developer tab -> Start Server)" -ForegroundColor White
Write-Host ""

$RunEvalNow = Read-Host "Run evaluation now? (y/n) [n]"
if ([string]::IsNullOrEmpty($RunEvalNow)) { $RunEvalNow = "n" }

if ($RunEvalNow -eq "y" -or $RunEvalNow -eq "Y") {
    Write-Host ""
    Write-Host "Please set up your model in LM Studio now..." -ForegroundColor Yellow
    Write-Host ""
    if ($DoUpload -eq "y" -or $DoUpload -eq "Y") {
        Write-Host "Your model is available at: https://huggingface.co/$RepoName" -ForegroundColor White
        if ($GgufFlag) {
            Write-Host "GGUF files are in the 'gguf' folder on HuggingFace" -ForegroundColor White
        }
    } else {
        Write-Host "Your model is at: $FinalModelPath" -ForegroundColor White
    }
    Write-Host ""

    $ModelReady = Read-Host "Is your model loaded and LM Studio server running? (y/n)"

    if ($ModelReady -eq "y" -or $ModelReady -eq "Y") {
        # Fetch available models from LM Studio API
        Write-Host ""
        Write-Host "Fetching available models from LM Studio..." -ForegroundColor Yellow

        $EvalModelName = $null
        try {
            $ModelsResponse = Invoke-RestMethod -Uri "http://localhost:1234/v1/models" -Method Get -TimeoutSec 5
            $Models = $ModelsResponse.data | ForEach-Object { $_.id }

            if ($Models.Count -eq 0) {
                Write-Host "[WARN] No models found. Please load a model first." -ForegroundColor Yellow
            } elseif ($Models.Count -eq 1) {
                $EvalModelName = $Models[0]
                Write-Host "  Using only available model: $EvalModelName" -ForegroundColor Green
            } else {
                Write-Host ""
                Write-Host "Available Models:" -ForegroundColor Yellow
                for ($i = 0; $i -lt $Models.Count; $i++) {
                    Write-Host "  $($i + 1)) $($Models[$i])"
                }
                Write-Host ""
                $ModelChoice = Read-Host "Select model [1]"
                if ([string]::IsNullOrEmpty($ModelChoice)) { $ModelChoice = "1" }

                $ModelIndex = [int]$ModelChoice - 1
                if ($ModelIndex -ge 0 -and $ModelIndex -lt $Models.Count) {
                    $EvalModelName = $Models[$ModelIndex]
                } else {
                    $EvalModelName = $Models[0]
                }
            }
        } catch {
            Write-Host "[ERROR] Could not connect to LM Studio API" -ForegroundColor Red
            Write-Host "Make sure LM Studio server is running and try again." -ForegroundColor Yellow
            $EvalModelName = $null
        }

        if ($EvalModelName) {
            Write-Host "  Selected: $EvalModelName" -ForegroundColor Green

            Write-Host ""
            Write-Host "Evaluation Type:" -ForegroundColor Yellow
            Write-Host "  1) Tool Evaluation      - Tests tool-calling accuracy (47 prompts)"
            Write-Host "  2) Behavior Evaluation  - Tests behavioral patterns"
            Write-Host "  3) Full Evaluation      - Both tool and behavior tests"
            Write-Host ""
            $EvalTypeChoice = Read-Host "Select evaluation type [3]"
            if ([string]::IsNullOrEmpty($EvalTypeChoice)) { $EvalTypeChoice = "3" }

            Write-Host ""
            Write-Host "Starting evaluation..." -ForegroundColor Yellow
            Write-Host ""

            # Run evaluator based on choice
            Set-Location (Join-Path $ScriptDir "../Evaluator")

            switch ($EvalTypeChoice) {
                "1" {
                    Write-Host "Running Tool Evaluation..." -ForegroundColor Cyan
                    $EvalArgs = @("cli.py", "--backend", "lmstudio", "--model", $EvalModelName, "--prompt-set", "prompts/full_coverage.json")
                    & $PythonExe @EvalArgs
                }
                "2" {
                    Write-Host "Running Behavior Evaluation..." -ForegroundColor Cyan
                    $EvalArgs = @("cli.py", "--backend", "lmstudio", "--model", $EvalModelName, "--prompt-set", "prompts/behavioral_patterns.json")
                    & $PythonExe @EvalArgs
                }
                default {
                    Write-Host "Running Full Evaluation (Tools + Behavior)..." -ForegroundColor Cyan
                    Write-Host ""
                    Write-Host "--- Tool Evaluation ---" -ForegroundColor Yellow
                    $EvalArgs = @("cli.py", "--backend", "lmstudio", "--model", $EvalModelName, "--prompt-set", "prompts/full_coverage.json")
                    & $PythonExe @EvalArgs
                    Write-Host ""
                    Write-Host "--- Behavior Evaluation ---" -ForegroundColor Yellow
                    $EvalArgs = @("cli.py", "--backend", "lmstudio", "--model", $EvalModelName, "--prompt-set", "prompts/behavioral_patterns.json")
                    & $PythonExe @EvalArgs
                    Write-Host ""
                    Write-Host "Full Evaluation Complete!" -ForegroundColor Green
                }
            }

            Set-Location $ScriptDir
        }
    } else {
        Write-Host ""
        Write-Host "No problem! You can run evaluation later:" -ForegroundColor Yellow
        Write-Host "  .\train_unified.ps1  (then select option 4)" -ForegroundColor Gray
        Write-Host ""
    }
}

# ============================================================================
# Final Summary
# ============================================================================
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "  Pipeline Complete!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Training Output: $FinalModelPath" -ForegroundColor White

if ($DoUpload -eq "y" -or $DoUpload -eq "Y") {
    Write-Host "  HuggingFace:     https://huggingface.co/$RepoName" -ForegroundColor White
}

Write-Host ""

# Show what was accomplished
Write-Host "Completed Steps:" -ForegroundColor Cyan
Write-Host "  [OK] Training ($ModeText)" -ForegroundColor Green
Write-Host "  [OK] Training lineage saved" -ForegroundColor Green
if ($DoUpload -eq "y" -or $DoUpload -eq "Y") {
    Write-Host "  [OK] Uploaded to HuggingFace" -ForegroundColor Green
    if ($GgufFlag) {
        Write-Host "  [OK] GGUF quantizations created" -ForegroundColor Green
    }
    Write-Host "  [OK] Model card generated from lineage" -ForegroundColor Green
}

Write-Host ""

# Show next steps if upload wasn't done
if ($DoUpload -ne "y" -and $DoUpload -ne "Y") {
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "  - Upload to HuggingFace:" -ForegroundColor White
    Write-Host "    cd Trainers\$TrainerDir" -ForegroundColor Gray
    Write-Host "    python src/upload_to_hf.py `"$FinalModelPath`" username/model-name" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "To run evaluation later:" -ForegroundColor Cyan
Write-Host "  .\train_unified.ps1  (select option 4: Evaluate Model)" -ForegroundColor Gray
Write-Host ""

if ($DoUpload -eq "y" -or $DoUpload -eq "Y") {
    Write-Host "Quick Links:" -ForegroundColor Cyan
    Write-Host "  Model:  https://huggingface.co/$RepoName" -ForegroundColor White
    Write-Host ""
}

Set-Location $ScriptDir
Read-Host "Press Enter to exit"
exit 0
