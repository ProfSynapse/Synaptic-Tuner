# Installation Fixes Applied - November 2024

## Summary
This document describes the fixes applied to ensure reliable installation of the RTX 3090 KTO training environment.

## Key Changes Made

### 1. requirements.txt Updates

**Critical Fixes:**
- `datasets`: Changed from `2.14.0` → `2.16.1` (fixes pyarrow compatibility)
- `peft`: Changed from `0.7.0` → `0.7.1` (required by Unsloth 2024.9)
- `huggingface-hub`: Pinned to `==0.25.0` (CRITICAL - newer versions break Unsloth)

**New Dependencies Added:**
```
tokenizers>=0.20,<0.21       # Required version range for transformers 4.45.2
fsspec[http]>=2023.1.0,<=2023.10.0  # datasets 2.16.1 compatibility
pyarrow>=14.0.0               # datasets backend
pyarrow_hotfix                # security fixes for pyarrow
```

### 2. setup.sh Updates

**Python Version Check:**
- Added warning for Python 3.11+ recommending Python 3.10 for best compatibility
- Suggests using conda for Python 3.10: `conda create -p ./venv python=3.10`

**Unsloth Installation:**
Changed from:
```bash
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

To:
```bash
pip install --no-deps unsloth==2024.9
pip install --no-deps xformers==0.0.27.post2
```

**Reasoning:** Using `--no-deps` prevents Unsloth from upgrading/downgrading other packages, maintaining our stable dependency versions.

**Installation Summary:**
- Enhanced to show all critical package versions including:
  - Datasets version
  - PEFT version
  - HuggingFace Hub version (with reminder that 0.25.0 is critical)
  - Xformers version

## Why These Changes Matter

### The HuggingFace Hub 0.25.0 Requirement
**Problem:** Unsloth 2024.9 imports from `huggingface_hub.utils._token` which was removed in newer versions (0.36.0+).

**Solution:** Pin to version 0.25.0 which contains the required internal module.

**Source:** https://github.com/unslothai/unsloth/issues/1148

### The --no-deps Strategy
**Problem:** When installing Unsloth normally, it tries to manage its own dependencies and can:
- Downgrade PyTorch from 2.4.1 to 2.4.0
- Upgrade packages that break compatibility
- Install incompatible versions of xformers

**Solution:** Install with `--no-deps` after all other dependencies are in place, preventing version conflicts.

## Installation Order (Critical)

The correct installation sequence is now:

1. Install PyTorch 2.4.1 + CUDA 12.1
2. Install all other requirements from requirements.txt
3. Install Unsloth 2024.9 with `--no-deps`
4. Install Xformers 0.0.27.post2 with `--no-deps`
5. (Optional) Compile Flash Attention 2.5.9.post1

## Recommended Python Version

**Best:** Python 3.10.x
**Works:** Python 3.9+
**Not Recommended:** Python 3.12+ (limited pre-built wheels)

Use conda for easy Python 3.10 installation:
```bash
conda create -p ./venv python=3.10
```

## Tested Configuration

✓ Python: 3.10.19 (conda environment)
✓ PyTorch: 2.4.1 + CUDA 12.1
✓ Transformers: 4.45.2
✓ Datasets: 2.16.1
✓ PEFT: 0.7.1
✓ TRL: 0.11.4
✓ BitsAndBytes: 0.43.0
✓ HuggingFace Hub: 0.25.0 ⚠️ CRITICAL VERSION
✓ Unsloth: 2024.9
✓ Xformers: 0.0.27.post2

## Next Time Setup

To replicate this working setup from scratch:

```bash
# 1. Create Python 3.10 environment
conda create -p ./venv python=3.10
conda activate ./venv

# 2. Run setup script
bash setup.sh

# That's it! The setup script now handles everything correctly.
```

## Known Warnings (Safe to Ignore)

1. "Unsloth unsuccessfully patched LoraLayer.update_layer"
   - **Impact:** None - training still works

2. "Your Flash Attention 2 installation seems to be broken"
   - **Impact:** None - Unsloth uses Xformers instead (no performance loss)

3. pip dependency resolver warnings about xformers wanting torch==2.4.0
   - **Impact:** None - torch 2.4.1 is compatible

## Contact

If you encounter issues with this setup, check:
1. Python version (should be 3.10.x)
2. HuggingFace Hub version (must be 0.25.0)
3. All packages installed with correct versions from requirements.txt

Last Updated: November 14, 2024
