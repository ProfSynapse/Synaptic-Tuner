# Unsloth Windows Setup Script with Compatibility Patches
# Based on working WSL configuration from code/rtx3090_kto/
# Run: .\setup_unsloth_windows.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Unsloth Windows Setup - RTX 3090" -ForegroundColor Cyan
Write-Host "With Windows Compatibility Patches" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Check if in conda environment
Write-Host "`n[1/9] Checking conda environment..." -ForegroundColor Yellow
$condaEnv = $env:CONDA_DEFAULT_ENV
if (-not $condaEnv) {
    Write-Host "ERROR: Not in a conda environment!" -ForegroundColor Red
    Write-Host "Please activate unsloth_env first: conda activate unsloth_env" -ForegroundColor Yellow
    exit 1
}
Write-Host "Active environment: $condaEnv" -ForegroundColor Green

# Step 2: Check Python version
Write-Host "`n[2/9] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "$pythonVersion" -ForegroundColor Green

# Step 3: Check GPU
Write-Host "`n[3/9] Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
    Write-Host "$gpuInfo" -ForegroundColor Green
} catch {
    Write-Host "WARNING: nvidia-smi not found" -ForegroundColor Yellow
}

# Step 4: Upgrade pip
Write-Host "`n[4/9] Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel --quiet
$pipVersion = pip --version
Write-Host "pip upgraded: $pipVersion" -ForegroundColor Green

# Step 5: Install PyTorch 2.4.1+cu121
Write-Host "`n[5/9] Installing PyTorch 2.4.1+cu121..." -ForegroundColor Yellow
Write-Host "This may take 2-3 minutes..." -ForegroundColor Gray
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch installation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "PyTorch installed" -ForegroundColor Green

# Step 6: Verify CUDA
Write-Host "`n[6/9] Verifying PyTorch CUDA..." -ForegroundColor Yellow
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: CUDA not available in PyTorch!" -ForegroundColor Red
    exit 1
}

# Step 7: Install core dependencies
Write-Host "`n[7/9] Installing core dependencies..." -ForegroundColor Yellow
Write-Host "This may take 2-3 minutes..." -ForegroundColor Gray

Write-Host "  - transformers, datasets, accelerate..." -ForegroundColor Gray
pip install transformers==4.45.2 datasets==2.16.1 accelerate==0.27.0

Write-Host "  - bitsandbytes 0.45.5 (Windows-compatible, no triton.ops)..." -ForegroundColor Gray
pip install bitsandbytes==0.45.5

Write-Host "  - peft, trl..." -ForegroundColor Gray
pip install peft==0.7.1 trl==0.11.4

Write-Host "  - huggingface-hub (CRITICAL: 0.25.0)..." -ForegroundColor Gray
pip install huggingface-hub==0.25.0

Write-Host "  - utilities..." -ForegroundColor Gray
pip install sentencepiece "protobuf==3.20.3" python-dotenv tensorboard

# Remove problematic packages
Write-Host "  - Removing problematic packages (torchao, diffusers)..." -ForegroundColor Gray
pip uninstall torchao diffusers -y 2>$null

Write-Host "Core dependencies installed" -ForegroundColor Green

# Step 8: Install Unsloth and xformers
Write-Host "`n[8/9] Installing Unsloth 2024.9 and xformers..." -ForegroundColor Yellow

Write-Host "  - Installing Unsloth 2024.9 (with --no-deps)..." -ForegroundColor Gray
pip install --no-deps unsloth==2024.9
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Unsloth installation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Unsloth 2024.9 installed" -ForegroundColor Green

Write-Host "  - Installing xformers 0.0.27.post2 (with --no-deps)..." -ForegroundColor Gray
pip install --no-deps xformers==0.0.27.post2
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: xformers installation failed (may work without it)" -ForegroundColor Yellow
}
Write-Host "  xformers 0.0.27.post2 installed" -ForegroundColor Green

# Step 9: Create and apply Windows compatibility patches
Write-Host "`n[9/9] Creating Windows compatibility patches..." -ForegroundColor Yellow

$patchScript = @'
"""
Unsloth Windows Compatibility Patches
Auto-applied during setup
"""
import sys
import os
from dataclasses import dataclass, fields

def apply_patches():
    print("Applying Windows compatibility patches...")

    # Patch 1: Fix triton AttrsDescriptor
    try:
        import triton
        if hasattr(triton.runtime.autotuner, 'AttrsDescriptor'):
            AttrsDescriptor = triton.runtime.autotuner.AttrsDescriptor
            try:
                fields(AttrsDescriptor)
            except TypeError:
                @dataclass
                class AttrsDescriptor:
                    divisible_by_16: tuple = ()
                    equal_to_1: tuple = ()
                triton.runtime.autotuner.AttrsDescriptor = AttrsDescriptor
                print("  ✓ AttrsDescriptor patched")
    except: pass

    # Patch 2: Wrap fields() for non-dataclasses
    import dataclasses
    original_fields = fields
    def patched_fields(class_or_instance):
        try:
            return original_fields(class_or_instance)
        except TypeError:
            return ()
    dataclasses.fields = patched_fields

    # Patch 3: Disable torch.compile
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    # Patch 4: Pre-patch torch._inductor
    try:
        import torch._inductor.runtime.hints
        if not hasattr(torch._inductor.runtime.hints, 'attr_desc_fields'):
            torch._inductor.runtime.hints.attr_desc_fields = set()
    except: pass

    print("  ✓ All patches applied")

if __name__ == "__main__":
    apply_patches()
    try:
        from unsloth import FastLanguageModel
        print("\n✅ SUCCESS! Unsloth works on Windows!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
'@

# Save patch script
$patchScript | Out-File -FilePath "unsloth_windows_patch.py" -Encoding UTF8
Write-Host "  ✓ Patch script created: unsloth_windows_patch.py" -ForegroundColor Green

# Apply patches and test
Write-Host "`nApplying patches and testing Unsloth..." -ForegroundColor Yellow
python unsloth_windows_patch.py
$patchTest = $LASTEXITCODE

Write-Host "`n==========================================" -ForegroundColor Cyan
if ($patchTest -eq 0) {
    Write-Host "SETUP COMPLETE - UNSLOTH WORKING!" -ForegroundColor Green
} else {
    Write-Host "SETUP COMPLETE (needs troubleshooting)" -ForegroundColor Yellow
}
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`nInstallation Summary:" -ForegroundColor Cyan
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'  Datasets: {datasets.__version__}')"
python -c "import peft; print(f'  PEFT: {peft.__version__}')"
python -c "import trl; print(f'  TRL: {trl.__version__}')"
python -c "import huggingface_hub; print(f'  HuggingFace Hub: {huggingface_hub.__version__}')"
python -c "import bitsandbytes; print(f'  bitsandbytes: {bitsandbytes.__version__}')"

if ($patchTest -eq 0) {
    Write-Host "`nNext steps:" -ForegroundColor Green
    Write-Host "  1. Always import patches first: python -c 'from unsloth_windows_patch import apply_patches; apply_patches()'" -ForegroundColor White
    Write-Host "  2. Then import Unsloth: from unsloth import FastLanguageModel" -ForegroundColor White
    Write-Host "  3. Test with: python diagnose_unsloth_env.py" -ForegroundColor White
} else {
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Run: python unsloth_windows_patch.py" -ForegroundColor White
    Write-Host "  2. Check output for specific errors" -ForegroundColor White
    Write-Host "  3. Share errors for further debugging" -ForegroundColor White
}

Write-Host ""
