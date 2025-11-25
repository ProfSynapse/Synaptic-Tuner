# RTX 3090 KTO Training - Installation Guide

Complete, tested installation guide for RTX 3090 with dependency troubleshooting.

## Hardware Requirements

✅ **GPU**: NVIDIA RTX 3090 (24GB VRAM, Ampere architecture, compute capability 8.6)
✅ **Driver**: NVIDIA Driver 535+ (for CUDA 12.1)
✅ **RAM**: 32GB+ recommended
✅ **Storage**: 50GB+ free space
✅ **OS**: Linux (Ubuntu 20.04+), Windows 10/11, or WSL2

## Quick Installation (Recommended)

```bash
# Clone/navigate to project
cd rtx3090_kto

# Run setup script
bash setup.sh

# Activate environment
source venv/bin/activate

# Test installation
python test_installation.py
```

Done! Skip to [Testing](#testing-your-installation) section.

## Manual Installation (If Script Fails)

### Step 1: Verify Prerequisites

```bash
# Check Python version (3.9+ required, 3.10+ recommended)
python3 --version

# Check NVIDIA driver
nvidia-smi

# Check CUDA capability (should show 8.6 for RTX 3090)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

**Expected Output**:
- Python: 3.9.x or higher
- NVIDIA Driver: 535.x or higher
- Compute Capability: 8.6

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install PyTorch with CUDA Support

**CRITICAL**: Install PyTorch FIRST with correct CUDA version.

```bash
# For CUDA 12.1 (recommended for RTX 3090)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

**Verify PyTorch CUDA**:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

**Expected**:
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3090
```

### Step 4: Install Core Dependencies

```bash
pip install transformers==4.45.2
pip install datasets==2.14.0
pip install accelerate==0.27.0
pip install bitsandbytes==0.43.0
pip install peft==0.7.0
pip install trl==0.11.4
```

### Step 5: Install Unsloth (Special Installation)

```bash
# For PyTorch 2.4.1 + CUDA 12.1 + Ampere (RTX 3090)
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

**Verify Unsloth**:
```bash
python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"
```

### Step 6: Install Utilities

```bash
pip install numpy>=1.24.0 pandas>=2.0.0 tqdm>=4.65.0 huggingface-hub>=0.20.0
```

### Step 7: Optional - Flash Attention (Recommended)

```bash
# This compiles from source and takes 5-10 minutes
MAX_JOBS=4 pip install flash-attn==2.5.9 --no-build-isolation
```

**Note**: If you have < 32GB RAM, use `MAX_JOBS=2` instead.

## Testing Your Installation

Run the comprehensive test script:

```bash
python test_installation.py
```

**All tests should pass**. If any fail, see [Troubleshooting](#troubleshooting) below.

Quick test:

```bash
# Test dry run
python train_kto.py --model-size 3b --dry-run
```

## Version Compatibility Matrix

### ✅ Tested & Stable Configuration

| Component | Version | Notes |
|-----------|---------|-------|
| PyTorch | 2.4.1 | Stable with Unsloth |
| CUDA | 12.1 | Best compatibility |
| Transformers | 4.45.2 | Stable with TRL 0.11.4 |
| TRL | 0.11.4 | Compatible with transformers 4.45.2 |
| BitsAndBytes | 0.43.0 | Full RTX 3090 support |
| PEFT | 0.7.0 | Stable LoRA implementation |
| Accelerate | 0.27.0 | Good mixed precision support |
| Datasets | 2.14.0 | Stable |
| Unsloth | Latest | cu121-ampere-torch240 |
| Flash Attention | 2.5.9 | Optional, 2-4x speedup |

### ⚠️ Alternative Configuration (Experimental)

| Component | Version | Notes |
|-----------|---------|-------|
| PyTorch | 2.5.1 | May have Unsloth issues |
| CUDA | 12.4 | Latest, less tested |
| Transformers | 4.51.3 | Latest features |
| TRL | 0.21.0 | Requires transformers 4.51.3+ |
| BitsAndBytes | 0.45.3 | Latest |
| PEFT | 0.14.0 | Latest |
| Accelerate | 1.4.0 | Latest |
| Datasets | 3.3.2 | Latest |

**Use experimental only if you need latest features and are willing to debug compatibility issues.**

## Troubleshooting

### Issue 1: "CUDA not available" / "sm_86 not compatible"

**Symptoms**:
```python
torch.cuda.is_available()  # Returns False
# or
RuntimeError: CUDA capability sm_86 is not compatible
```

**Solutions**:

1. **Check NVIDIA drivers**:
   ```bash
   nvidia-smi
   # Should show driver 535+ and CUDA 12.1
   ```

2. **Reinstall PyTorch with correct CUDA**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
     --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify you didn't install CPU-only PyTorch**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   # Should show 12.1, not None
   ```

4. **Check CUDA_HOME environment variable** (if compiling from source):
   ```bash
   echo $CUDA_HOME  # Should point to /usr/local/cuda-12.1
   ```

### Issue 2: Flash Attention Compilation Takes 40+ Minutes

**Symptoms**: `pip install flash-attn` hangs or takes very long.

**Solutions**:

1. **Use PyTorch 2.4.1** (not 2.5+):
   ```bash
   # PyTorch 2.5 can cause flash-attn to take 40+ minutes
   # Downgrade to 2.4.1 first
   ```

2. **Limit parallel compilation jobs**:
   ```bash
   MAX_JOBS=2 pip install flash-attn==2.5.9 --no-build-isolation
   ```

3. **Ensure enough RAM** (64GB+ recommended for compilation):
   ```bash
   free -h  # Check available RAM
   ```

4. **Skip Flash Attention** (optional anyway):
   - Training will work without it, just 2-4x slower on attention

### Issue 3: Unsloth Import Error

**Symptoms**:
```python
ModuleNotFoundError: No module named 'unsloth'
# or
ImportError: Unsloth version mismatch
```

**Solutions**:

1. **Use version-specific installation**:
   ```bash
   pip uninstall unsloth unsloth-zoo -y
   pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
   ```

2. **Check PyTorch version compatibility**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   # Must be 2.4.1 for cu121-torch240
   ```

3. **Install dependencies first**:
   - Unsloth requires transformers, peft, etc. to be installed first

### Issue 4: TRL/Transformers Version Mismatch

**Symptoms**:
```
ImportError: TRL requires transformers>=4.51.3 but you have 4.45.2
```

**Solutions**:

1. **Use compatible versions** (recommended):
   ```bash
   pip install transformers==4.45.2 trl==0.11.4
   ```

2. **Or upgrade both**:
   ```bash
   pip install transformers==4.51.3 trl==0.21.0
   ```

3. **Check compatibility**:
   ```bash
   pip show transformers trl
   ```

### Issue 5: BitsAndBytes "No CUDA GPU" Warning

**Symptoms**:
```
Warning: The installed version of bitsandbytes was compiled without GPU support.
```

**Solutions**:

1. **Reinstall bitsandbytes**:
   ```bash
   pip uninstall bitsandbytes -y
   pip install bitsandbytes==0.43.0
   ```

2. **Verify CUDA is available**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **On Windows**: Use WSL2 or build from source

### Issue 6: Out of Memory During Installation

**Symptoms**: Installation crashes or system freezes.

**Solutions**:

1. **Close other applications**

2. **Install one package at a time**:
   ```bash
   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
   # Wait for completion
   pip install transformers==4.45.2
   # Continue one by one
   ```

3. **Use smaller pip cache**:
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

### Issue 7: "No matching distribution found"

**Symptoms**: `pip` can't find a package version.

**Solutions**:

1. **Check Python version**:
   ```bash
   python --version  # Must be 3.9+
   ```

2. **Update pip**:
   ```bash
   pip install --upgrade pip
   ```

3. **Check CUDA URL**:
   ```bash
   # Ensure requirements.txt has:
   --index-url https://download.pytorch.org/whl/cu121
   ```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

**Best support**. Follow standard installation.

Additional packages you might need:
```bash
sudo apt update
sudo apt install build-essential python3-dev git
```

### Windows 10/11

**Moderate support**. Recommended to use WSL2.

**Native Windows**:
1. Install Visual Studio Build Tools
2. Install CUDA Toolkit 12.1
3. Set `dataloader_num_workers=0` in training config
4. Use `--no-build-isolation` for compilations

**WSL2** (Recommended):
```bash
# Install WSL2 with Ubuntu
wsl --install

# Install NVIDIA drivers in Windows (not WSL)
# Follow standard Linux installation in WSL
```

### macOS

**Not supported** - No NVIDIA GPU support on macOS.

Use the Mac M4 implementation instead (`mistral_lora_mac/`).

## Advanced Installation Options

### Using Conda Instead of Pip

```bash
conda create -n kto python=3.10
conda activate kto
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.45.2 datasets==2.14.0 accelerate==0.27.0
pip install bitsandbytes==0.43.0 peft==0.7.0 trl==0.11.4
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

### Installing Latest Versions (Experimental)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers trl accelerate peft bitsandbytes datasets --upgrade
pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

### Offline Installation

1. Download wheels on a machine with internet:
   ```bash
   pip download -r requirements.txt -d ./wheels
   ```

2. Transfer `wheels/` directory to offline machine

3. Install from wheels:
   ```bash
   pip install --no-index --find-links ./wheels -r requirements.txt
   ```

## Verification Checklist

After installation, verify each component:

- [ ] Python 3.9+ installed
- [ ] NVIDIA drivers 535+ working (`nvidia-smi`)
- [ ] PyTorch 2.4.1 with CUDA 12.1
- [ ] CUDA available in PyTorch (`torch.cuda.is_available() == True`)
- [ ] GPU recognized (`torch.cuda.get_device_name(0)` shows RTX 3090)
- [ ] Transformers 4.45.2 installed
- [ ] TRL 0.11.4 installed
- [ ] Unsloth imports successfully
- [ ] All test_installation.py tests pass
- [ ] Dry run completes without errors

## Getting Help

1. **Run diagnostic script**:
   ```bash
   python test_installation.py
   ```

2. **Check versions**:
   ```bash
   pip list | grep -E "torch|transformers|trl|unsloth|bitsandbytes"
   ```

3. **Review this guide's troubleshooting section**

4. **Check official docs**:
   - PyTorch: https://pytorch.org/get-started/
   - Unsloth: https://docs.unsloth.ai/
   - TRL: https://huggingface.co/docs/trl/

5. **Community support**:
   - Unsloth Discord
   - PyTorch Forums
   - HuggingFace Forums

## Quick Reference

### Stable Installation Command (All-in-One)

```bash
# Create environment
python3 -m venv venv && source venv/bin/activate

# Install PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install core
pip install transformers==4.45.2 datasets==2.14.0 accelerate==0.27.0 \
  bitsandbytes==0.43.0 peft==0.7.0 trl==0.11.4

# Install Unsloth
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Install utilities
pip install numpy pandas tqdm huggingface-hub

# Test
python -c "import torch; from unsloth import FastLanguageModel; print('✓ Ready')"
```

---

**Last Updated**: November 2025
**Tested On**: RTX 3090, Ubuntu 22.04, Driver 535.x, CUDA 12.1
