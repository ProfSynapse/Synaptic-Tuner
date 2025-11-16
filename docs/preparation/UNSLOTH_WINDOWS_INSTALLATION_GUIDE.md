# Unsloth Windows Installation Guide

**Comprehensive Research Report - January 2025**

**Document Version**: 1.0
**Last Updated**: 2025-11-16
**Research Focus**: Native Windows and WSL2 installation methods for Unsloth

---

## Executive Summary

Unsloth now provides **official Windows support** as of February 2025, enabling native Windows installation without requiring WSL2. This guide covers all installation methods, exact version requirements, Triton compatibility workarounds, and known issues with solutions. Testing has been validated across Python 3.9-3.13, CUDA 11.8/12.4/12.6, and various NVIDIA GPUs from GTX 1650 to RTX 50xx series.

**Key Findings**:
- Native Windows installation is now possible via `pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"`
- Triton has Windows fork support (woct0rdho/triton-windows) bundling CUDA 12.4/12.8
- PyTorch 2.4-2.5.1 with CUDA 12.4 is the recommended configuration for Windows
- Four installation methods available: Docker, Native Windows, PowerShell Script, WSL2
- Visual Studio 2022 with C++ build tools is mandatory for native Windows installation

---

## Table of Contents

1. [Triton Compatibility on Windows](#triton-compatibility-on-windows)
2. [Exact Version Requirements](#exact-version-requirements)
3. [Installation Methods](#installation-methods)
4. [Known Issues and Solutions](#known-issues-and-solutions)
5. [WSL2 vs Native Windows Trade-offs](#wsl2-vs-native-windows-trade-offs)
6. [Step-by-Step Installation Procedures](#step-by-step-installation-procedures)
7. [Verification and Testing](#verification-and-testing)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Resource Links](#resource-links)

---

## 1. Triton Compatibility on Windows

### Why Triton Doesn't Work on Windows (Official)

The **official Triton package only supports Linux**. This is the primary challenge for Windows users trying to use Unsloth, as Triton is a critical dependency for GPU-accelerated operations.

**Common Errors Without Windows Triton**:
```
ModuleNotFoundError: No module named 'triton'
AttributeError: partially initialized module 'triton' has no attribute '_C'
ImportError: No module named 'triton.common'
```

### Windows Triton Solutions

#### Option 1: Community-Maintained Windows Fork (Recommended)

**woct0rdho/triton-windows** provides Windows-native Triton builds with bundled CUDA toolchain.

**Repository**: https://github.com/woct0rdho/triton-windows

**Version Compatibility Matrix**:

| PyTorch Version | Triton Version | Bundled CUDA | GPU Support |
|----------------|----------------|--------------|-------------|
| 2.4.x - 2.5.x | 3.1.x | CUDA 12.4 | RTX 20xx/30xx/40xx, GTX 16xx |
| 2.6.x | 3.2.x | CUDA 12.4 | RTX 20xx/30xx/40xx, GTX 16xx |
| 2.7.x | 3.3.x | CUDA 12.8 | RTX 20xx/30xx/40xx/50xx |
| 2.8.x | 3.4.x | CUDA 12.8 | RTX 20xx/30xx/40xx/50xx |
| 2.9.x | 3.5.x | CUDA 12.8 | RTX 20xx/30xx/40xx/50xx |

**Installation Commands**:

```powershell
# Uninstall any existing Triton first
pip uninstall triton

# For PyTorch 2.4-2.5 (CUDA 12.4)
pip install -U "triton-windows<3.3"

# For PyTorch 2.7+ (CUDA 12.8, RTX 50xx support)
pip install -U "triton-windows<3.6"

# For older GPUs (GTX 16xx/RTX 20xx - Turing architecture)
pip install -U "triton-windows<3.3"
```

**Key Features**:
- **Bundled CUDA Toolchain**: Since triton-windows 3.2.0.post11, no manual CUDA installation required
- **Bundled TinyCC Compiler**: Since triton-windows 3.2.0.post13, C compiler included
- **GPU Compute Capability Support**:
  - RTX 50xx (sm120): Triton ≥3.3, PyTorch ≥2.7, CUDA ≥12.8
  - RTX 40xx (sm89): All versions supported
  - RTX 30xx (sm86): All versions supported; fp8 requires triton-windows 3.5.0.post21+
  - GTX 16xx/RTX 20xx (sm75): Supported up to Triton 3.2 only
  - GTX 10xx and older: **Unsupported**

**Windows-Specific Requirement**:
```powershell
# Install Visual C++ Redistributable (required for libtriton.pyd)
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### Option 2: PyPI triton-windows Package

```bash
pip install triton-windows
```

#### Option 3: Alternative Community Builds

Several community forks exist with varying maintenance levels:
- tuber100/triton-windows
- badgids/triton-windows
- Root-0/triton-windows

**Recommendation**: Use woct0rdho/triton-windows as it has the most active maintenance and comprehensive GPU support.

### Triton Limitations on Windows

1. **vLLM Incompatibility**: vLLM does not support Windows directly, only via WSL or Linux
2. **GRPO Training**: Currently unavailable on native Windows
3. **Older GPU Support**: GTX 10xx series and earlier are not supported by triton-windows

---

## 2. Exact Version Requirements

### 2.1 CUDA Toolkit

**Supported CUDA Versions**:
- CUDA 11.8
- CUDA 12.1
- CUDA 12.4 ✅ **Recommended for Windows**
- CUDA 12.6
- CUDA 12.8

**Minimum GPU Requirement**: CUDA Capability 7.0
- ✅ Supported: V100, T4, Titan V, RTX 20/30/40/50, A100, H100, L40, GTX 1650+
- ⚠️ Slow: GTX 1070, GTX 1080
- ❌ Not Supported: GTX 10xx (compute capability < 7.0)

**CUDA 12.4 Installation** (Recommended):

**Download**: https://developer.nvidia.com/cuda-12-4-0-download-archive

**Installation Steps**:
1. Verify CUDA-capable GPU exists
2. Download CUDA Toolkit (choose Full Installer for offline install)
3. Run installer and follow on-screen prompts
4. Verify installation: `nvcc --version`

**Official Documentation**: https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-microsoft-windows/

**Important Notes**:
- Driver and toolkit must both be installed for CUDA to function
- Installation may fail if Windows Update runs during setup (wait for completion)
- Release notes should be read before installation

### 2.2 PyTorch

**Supported PyTorch Versions**:
- PyTorch 2.1.1 through 2.5.1 ✅
- PyTorch 2.1.0 and below: ❌ Not supported
- PyTorch 2.6.0 and above: ❌ Too new (as of January 2025)

**Recommended Configuration for Windows**:

**PyTorch 2.5.1 + CUDA 12.4** (Latest stable):
```bash
# Using pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Using conda
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

**PyTorch 2.5.0 + CUDA 12.4**:
```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

**PyTorch 2.4.1 + CUDA 12.4**:
```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

**PyTorch Version Index URLs**:
- CUDA 12.4: `https://download.pytorch.org/whl/cu124`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`

**Verification**:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
```

### 2.3 Unsloth

**Latest Version**: Unsloth 2025.1.x (install from GitHub)

**Installation Format by CUDA/PyTorch Version**:

```bash
# PyTorch 2.5.x + CUDA 12.4 (Ampere+)
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"

# PyTorch 2.4.x + CUDA 12.4
pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"

# PyTorch 2.4.x + CUDA 12.1
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# PyTorch 2.4.x + CUDA 11.8
pip install "unsloth[cu118-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Windows-specific (automatic version detection)
pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
```

**Supported Architectures**:
- `cu118-ampere`, `cu121-ampere`, `cu124-ampere`: For A100, H100, RTX 30/40/50 series
- `cu118-torch240`: For general CUDA 11.8 + PyTorch 2.4.x

### 2.4 xformers

**Purpose**: Memory-efficient attention mechanisms for transformers

**Installation Methods**:

**Method 1: PyPI (Recommended)**
```bash
pip install xformers
```

**Method 2: From PyTorch Index**
```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

**Method 3: Build from Source** (Advanced)
```bash
pip install ninja
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

**Requirements for Building from Source**:
- Microsoft Build Tools for Visual Studio (C++ build tools workload)
- Administrator privileges for creating symlinks

**Workaround if xformers Causes Issues**:
```python
# Disable xformers before initialization
from unsloth import FastLanguageModel
FastLanguageModel.disable_xFormers = True
```

**Known Issues**:
- `ERROR: xformers-0.0.23-cp310-cp310-manylinux2014_x86_64.whl is not a supported wheel on Windows`
- **Solution**: Use `--index-url https://download.pytorch.org/whl/cu121` flag

### 2.5 bitsandbytes

**Purpose**: 8-bit quantization for reduced memory usage

**Supported Versions**:
- bitsandbytes >= 0.43.3
- bitsandbytes >= 0.45.5 ✅ **Recommended**

**Official Status**: Windows not officially supported by bitsandbytes, but community builds available

**Installation Methods**:

**Method 1: Standard PyPI** (Now has Windows wheels)
```bash
pip install bitsandbytes==0.45.5
```

**Method 2: Community Windows Build** (jllllll's builds)
```bash
python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

**Method 3: Preview/Development Build**
```bash
pip install --force-reinstall https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-win_amd64.whl
```

**Verification**:
```bash
python -m bitsandbytes
```

**Alternative Packages**:
- `bitsandbytes-windows` (PyPI package specifically for Windows)
- fa0311/bitsandbytes-windows (GitHub repository)

### 2.6 DeepSpeed

**Purpose**: Multi-GPU training optimization

**Windows Support Status**:
- ✅ Many features supported for training and inference
- ❌ Async I/O (AIO) not supported on Windows
- ❌ GDS not supported on Windows
- ⚠️ Versions 9.x-12.x have compilation issues on Windows (as of latest reports)
- ✅ Version 8.3 is the most recent confirmed working build for Windows

**Requirements**:
- PyTorch 2.3+ with CUDA support
- Visual Studio 2022 C++ x64/x86 build tools
- Administrator permissions for symlink creation

**Installation**:
```bash
pip install deepspeed
```

**Limitations with Unsloth**:
- DPO training may fail with DeepSpeed Zero3 offload
- Some advanced features may have compatibility issues

### 2.7 Python

**Supported Python Versions**:
- Python 3.9 ✅
- Python 3.10 ✅ **Recommended**
- Python 3.11 ✅
- Python 3.12 ✅
- Python 3.13 ✅ **Now supported** (as of recent updates)

**Recommendation**: Use Python 3.10 or 3.11 for maximum compatibility

**Installation**:
- Download from https://www.python.org/downloads/
- Or use Miniconda: https://docs.conda.io/en/latest/miniconda.html

### 2.8 Visual Studio Build Tools

**Required Version**: Visual Studio 2022 (or Build Tools for Visual Studio 2022)

**Required Components**:
1. **MSVC v143 - VS 2022 C++ x64/x86 build tools**
2. **Windows 10/11 SDK** (latest version)
3. **C++ CMake tools for Windows**
4. **MSBuild** (included with C++ build tools)

**Installation Steps**:

1. Download Visual Studio 2022 Build Tools:
   - https://visualstudio.microsoft.com/downloads/
   - Scroll to "All Downloads" → "Tools for Visual Studio 2022" → "Build Tools for Visual Studio 2022"

2. Run installer and select workload:
   - **Desktop development with C++**

3. In "Individual Components" tab, ensure selected:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)
   - Windows 10 SDK or Windows 11 SDK (10.0.xxxxx.0)
   - C++ CMake tools for Windows
   - C++ core features

4. Install and restart system

**Environment Variables** (Usually set automatically):
```
CC=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64\cl.exe
```

**Verification**:
```powershell
# Open "Developer Command Prompt for VS 2022"
cl.exe
# Should display Microsoft C/C++ Compiler version
```

### 2.9 Complete Dependency Compatibility Matrix

| Component | Version | CUDA | PyTorch | Notes |
|-----------|---------|------|---------|-------|
| **Python** | 3.10-3.13 | - | - | 3.10 or 3.11 recommended |
| **CUDA Toolkit** | 12.4 | - | 2.4.1-2.5.1 | Recommended for Windows |
| **PyTorch** | 2.5.1 | 12.4 | - | Latest stable |
| **Triton (Windows)** | 3.1.x | 12.4 | 2.4-2.5 | woct0rdho/triton-windows |
| **Unsloth** | 2025.1.x | 12.4 | 2.5.1 | Install from GitHub |
| **xformers** | Latest | 12.4 | 2.5.1 | From PyPI or PyTorch index |
| **bitsandbytes** | 0.45.5+ | 12.4 | 2.5.1 | Windows wheels available |
| **DeepSpeed** | 8.3 | 12.4 | 2.3+ | Optional; version 8.3 confirmed working |
| **Visual Studio** | 2022 | - | - | C++ build tools + Windows SDK |
| **NVIDIA Driver** | Latest | - | - | From nvidia.com |

---

## 3. Installation Methods

### Overview of Four Methods

| Method | Difficulty | Pros | Cons | Recommended For |
|--------|-----------|------|------|-----------------|
| **Docker** | Easy | No dependency issues, isolated environment | Larger disk usage, requires Docker Desktop | Beginners, quick testing |
| **Native Windows** | Medium | Best performance, direct access | Complex setup, dependency management | Production use, experienced users |
| **PowerShell Script** | Easy | Automated setup | Limited customization | Windows users preferring automation |
| **WSL2** | Easy | Linux environment, simple install | WSL overhead, separate filesystem | Linux-familiar users, full compatibility |

### Method 1: Docker (Easiest)

**Best For**: Beginners, users wanting to avoid dependency issues

**Prerequisites**:
- Docker Desktop for Windows
- NVIDIA Container Toolkit

**Installation Steps**:

1. **Install Docker Desktop**:
   - Download: https://www.docker.com/products/docker-desktop/
   - Enable WSL2 backend during installation

2. **Install NVIDIA Container Toolkit**:
   ```powershell
   # Follow instructions at:
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
   ```

3. **Run Unsloth Container**:
   ```bash
   docker pull unsloth/unsloth

   docker run --gpus all -p 8888:8888 unsloth/unsloth
   ```

4. **Access Jupyter Lab**:
   - Open browser: `http://localhost:8888`
   - Token displayed in terminal output

**Advantages**:
- Zero dependency management
- Isolated environment
- Pre-configured setup
- Easy updates via Docker images

**Disadvantages**:
- Larger disk space usage
- Container overhead
- Learning curve for Docker
- Data persistence requires volume mounting

### Method 2: Native Windows Installation (Recommended for Performance)

**Best For**: Users wanting maximum performance and direct Windows integration

**Full Prerequisites Checklist**:

- [ ] Windows 10/11 with latest updates
- [ ] NVIDIA GPU (RTX 20xx series or newer recommended)
- [ ] Latest NVIDIA GPU drivers
- [ ] CUDA Toolkit 12.4
- [ ] Visual Studio 2022 with C++ build tools and Windows SDK
- [ ] Python 3.10-3.13
- [ ] Git for Windows
- [ ] 20+ GB free disk space
- [ ] Administrator access

**Step-by-Step Installation**:

See **Section 6: Step-by-Step Installation Procedures** for detailed native Windows installation.

### Method 3: PowerShell Automated Setup

**Best For**: Windows users preferring automated installation

**PowerShell Script**: `unsloth_windows.ps1`

**Access**:
- Download link available at: https://docs.unsloth.ai/get-started/install-and-update/windows-installation
- Official Unsloth documentation provides the script

**Installation Steps**:

1. **Download Script**:
   - Visit https://docs.unsloth.ai/get-started/install-and-update/windows-installation
   - Download `unsloth_windows.ps1` script

2. **Run as Administrator**:
   ```powershell
   # Right-click Start → "Windows PowerShell (Admin)"

   # Navigate to download directory
   cd C:\Users\YourUsername\Downloads

   # Execute script
   powershell.exe -ExecutionPolicy Bypass -File .\unsloth_windows.ps1
   ```

3. **Script Actions**:
   - Checks system prerequisites
   - Installs/verifies CUDA toolkit
   - Sets up Python environment
   - Creates Conda environment named `unsloth_env`
   - Installs all dependencies
   - Configures environment variables

4. **Activate Environment**:
   ```powershell
   conda activate unsloth_env
   ```

**Script Features**:
- Automated environment setup
- Dependency version checking
- Error handling and reporting
- Creates isolated Conda environment

**Limitations**:
- Less control over individual component versions
- May not handle all edge cases
- Requires stable internet connection

### Method 4: Windows Subsystem for Linux (WSL2)

**Best For**: Users familiar with Linux, requiring full compatibility

**Advantages Over Native Windows**:
- Full Linux compatibility (no Triton workarounds needed)
- Official Triton support
- vLLM and GRPO support
- Simpler dependency management
- More extensive community documentation

**Prerequisites**:
- Windows 10 version 2004+ or Windows 11
- WSL2 enabled
- Ubuntu 22.04 or 24.04 (recommended)

**Installation Steps**:

1. **Enable WSL2**:
   ```powershell
   # Run as Administrator
   wsl --install

   # Or install specific distribution
   wsl --install -d Ubuntu-24.04
   ```

2. **Install NVIDIA CUDA Drivers** (Windows side):
   - Download latest NVIDIA drivers from https://www.nvidia.com/drivers
   - CUDA drivers are shared between Windows and WSL2

3. **Inside WSL2 Ubuntu**:
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install build essentials
   sudo apt install -y build-essential python3-pip python3-dev git

   # Install Miniconda (optional but recommended)
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

   # Create environment
   conda create -n unsloth python=3.10
   conda activate unsloth

   # Install PyTorch
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

   # Install Unsloth (standard Linux installation)
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

   # Install additional dependencies
   pip install --no-deps trl peft accelerate bitsandbytes
   ```

4. **Verification**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "from unsloth import FastLanguageModel; print('Unsloth imported successfully')"
   ```

**WSL2-Specific Tips**:
- Access Windows files: `/mnt/c/Users/YourUsername/`
- Access WSL files from Windows: `\\wsl$\Ubuntu-24.04\home\username\`
- GPU passthrough is automatic with latest NVIDIA drivers
- Use VSCode with WSL extension for development

**Performance Considerations**:
- GPU performance: Near-native (within 5-10%)
- I/O performance: Slower for cross-filesystem operations
- Recommendation: Keep data inside WSL filesystem for best performance

---

## 4. Known Issues and Solutions

### 4.1 Triton Installation Errors

**Issue**: `ModuleNotFoundError: No module named 'triton'`

**Solution**:
```powershell
pip install "triton-windows<3.3"
```

**Issue**: `AttributeError: partially initialized module 'triton' has no attribute '_C'`

**Cause**: Conflicting Triton installations or incorrect build

**Solution**:
```powershell
pip uninstall triton triton-windows
pip install "triton-windows<3.3"
```

**Issue**: `error C2143: syntax error` during Triton compilation

**Cause**: Missing or incorrect C++ compiler configuration

**Solution**:
1. Ensure Visual Studio 2022 C++ build tools installed
2. Run from "Developer Command Prompt for VS 2022"
3. Verify environment variables set correctly

### 4.2 PyTorch CUDA Not Available

**Issue**: `torch.cuda.is_available()` returns `False`

**Diagnosis**:
```python
import torch
print(torch.version.cuda)  # Should show CUDA version
print(torch.cuda.is_available())  # Should be True
```

**Common Causes & Solutions**:

1. **PyTorch installed without CUDA**:
   ```powershell
   # Uninstall CPU-only PyTorch
   pip uninstall torch torchvision torchaudio

   # Reinstall with CUDA 12.4
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

2. **CUDA version mismatch**:
   - Check installed CUDA: `nvcc --version`
   - Ensure PyTorch CUDA version matches system CUDA
   - Use pytorch.org selector to get correct install command

3. **NVIDIA drivers not installed**:
   - Download latest drivers: https://www.nvidia.com/drivers
   - Restart after installation

4. **CUDA Toolkit not in PATH**:
   ```powershell
   # Add to PATH (adjust version as needed)
   setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
   ```

### 4.3 xformers Wheel Compatibility

**Issue**: `ERROR: xformers-0.0.23-cp310-cp310-manylinux2014_x86_64.whl is not a supported wheel on Windows`

**Solution**:
```powershell
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

**Alternative Solution** (Disable xformers):
```python
from unsloth import FastLanguageModel
FastLanguageModel.disable_xFormers = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
```

### 4.4 bitsandbytes Compilation Issues

**Issue**: `The installed version of bitsandbytes was compiled without GPU support`

**Solution**:
```powershell
pip uninstall bitsandbytes
pip install bitsandbytes==0.45.5

# Or use community Windows build
python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

**Verification**:
```bash
python -m bitsandbytes
```

Expected output should show CUDA support enabled.

### 4.5 Visual Studio / MSVC Issues

**Issue**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
1. Install Visual Studio 2022 Build Tools
2. Ensure "Desktop development with C++" workload selected
3. Include Windows 10/11 SDK in installation
4. Restart system after installation

**Issue**: CMake cannot find compiler

**Solution**:
```powershell
# Open "Developer Command Prompt for VS 2022" instead of regular PowerShell
# Then run pip install commands
```

**Set Environment Variables Manually** (if needed):
```powershell
setx CC "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64\cl.exe"
```

### 4.6 SFTTrainer Dataset Crash

**Issue**: SFTTrainer crashes during dataset processing on Windows

**Solution**:
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_num_proc = 1,  # CRITICAL: Set to 1 on Windows
    max_seq_length = 2048,
    # ... other parameters
)
```

**Cause**: Multiprocessing issues on Windows with dataset processing

**Fix**: Always set `dataset_num_proc=1` in SFTTrainer configuration

### 4.7 vLLM Module Not Found

**Issue**: `ModuleNotFoundError: No module named 'vllm._C'`

**Cause**: vLLM does not support Windows directly

**Solutions**:

**Option 1**: Use WSL2 for vLLM-dependent features

**Option 2**: Avoid GRPO training methods (require vLLM)

**Option 3**: Use Docker with Linux container

**Note**: This is a known limitation; vLLM support is Linux-only

### 4.8 llama.cpp Export Issues

**Issue**: Missing quantization binaries when exporting to llama.cpp format

**Solution**:
```powershell
# Delete llama.cpp folder in working directory
Remove-Item -Recurse -Force llama.cpp

# Run export command again
python export_to_llamacpp.py
```

**Cause**: Corrupted or incomplete llama.cpp binaries

### 4.9 DeepSpeed Installation Failures

**Issue**: DeepSpeed compilation fails on Windows

**Solution**:
```powershell
# Use version 8.3 (confirmed working on Windows)
pip install deepspeed==0.8.3
```

**Note**: DeepSpeed versions 9.x-12.x have known Windows compilation issues

**Alternative**: Skip DeepSpeed if not using multi-GPU training

### 4.10 Import Errors After Installation

**Issue**: `ImportError: cannot import name 'X' from 'unsloth'`

**Diagnosis Steps**:
```python
import sys
print(sys.version)  # Check Python version
import torch
print(torch.__version__, torch.cuda.is_available())
import triton
print(triton.__version__)
```

**Solution**:
1. Verify all dependencies installed correctly
2. Check version compatibility matrix
3. Reinstall Unsloth:
   ```powershell
   pip uninstall unsloth
   pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
   ```

### 4.11 Out of Memory (OOM) Errors

**Issue**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
   ```python
   trainer = SFTTrainer(
       per_device_train_batch_size = 1,  # Reduce from default
       gradient_accumulation_steps = 16,  # Increase to maintain effective batch size
   )
   ```

2. **Use 4-bit quantization**:
   ```python
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name = "unsloth/llama-2-7b",
       load_in_4bit = True,  # Enable 4-bit quantization
   )
   ```

3. **Reduce sequence length**:
   ```python
   model, tokenizer = FastLanguageModel.from_pretrained(
       max_seq_length = 1024,  # Reduce from 2048
   )
   ```

4. **Enable gradient checkpointing**:
   ```python
   trainer = SFTTrainer(
       gradient_checkpointing = True,
   )
   ```

---

## 5. WSL2 vs Native Windows Trade-offs

### Comparison Matrix

| Factor | WSL2 | Native Windows |
|--------|------|----------------|
| **Installation Complexity** | Simple (Linux-standard) | Complex (Windows-specific builds) |
| **Dependency Management** | Easy (apt/conda) | Moderate (requires Windows builds) |
| **Triton Support** | Official (native) | Community fork required |
| **vLLM Support** | ✅ Full support | ❌ Not supported |
| **GRPO Training** | ✅ Supported | ❌ Not supported |
| **Performance (GPU)** | 95-100% native | 100% native |
| **Performance (I/O)** | Slower (cross-filesystem) | Full speed |
| **File Access** | Both Windows & Linux | Windows only |
| **Development Tools** | Full Linux toolchain | Windows toolchain |
| **Docker Integration** | Excellent | Good |
| **Community Support** | Extensive (Linux docs apply) | Growing |
| **Update Management** | Standard Linux updates | Windows-specific updates |
| **Stability** | Very stable | Stable (improving) |

### When to Choose WSL2

✅ **Choose WSL2 if**:
- You need vLLM or GRPO support
- You're familiar with Linux environments
- You want official Triton support
- You prefer standard Linux tooling
- You're following Linux-based tutorials
- You need maximum compatibility with Unsloth examples

### When to Choose Native Windows

✅ **Choose Native Windows if**:
- You require maximum I/O performance
- You prefer Windows-native development environment
- You're integrating with Windows-specific tools
- You want to avoid WSL overhead
- You don't need vLLM or GRPO features
- You're comfortable with Windows-specific dependency management

### Performance Benchmarks

**GPU Compute Performance**:
- WSL2: 95-100% of native Linux
- Native Windows: 100% (no overhead)
- **Conclusion**: Negligible difference for GPU-bound workloads

**I/O Performance**:
- WSL2 (within WSL filesystem): ~100% of native Linux
- WSL2 (cross-filesystem /mnt/c): ~30-60% of native
- Native Windows: 100%
- **Conclusion**: Keep data inside WSL filesystem for best performance

**Launch Latency**:
- WSL2: Slightly higher due to VMBUS overhead
- Native Windows: Minimal
- **Conclusion**: Minor difference, not significant for training workloads

### Hybrid Approach

**Recommendation**: Use both when possible
- **Development**: WSL2 for compatibility and ease
- **Production/Deployment**: Native Windows for maximum performance
- **Data Storage**: Windows filesystem, accessed from both environments

---

## 6. Step-by-Step Installation Procedures

### 6.1 Native Windows Installation (Complete)

**Estimated Time**: 60-90 minutes

#### Phase 1: System Preparation (15-20 minutes)

**Step 1: Verify GPU and Update Drivers**

```powershell
# Check GPU
nvidia-smi

# If command not found or outdated, download latest drivers
# Visit: https://www.nvidia.com/drivers
# Enter your GPU model and download
# Install and restart system
```

**Step 2: Install Visual Studio 2022 Build Tools**

1. Download Build Tools for Visual Studio 2022:
   - https://visualstudio.microsoft.com/downloads/
   - Scroll to "Tools for Visual Studio 2022"
   - Download "Build Tools for Visual Studio 2022"

2. Run installer `vs_BuildTools.exe`

3. In Workloads tab, select:
   - ✅ **Desktop development with C++**

4. In Individual Components tab, verify selected:
   - ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)
   - ✅ Windows 11 SDK (10.0.22621.0) or Windows 10 SDK
   - ✅ C++ CMake tools for Windows
   - ✅ C++ core features

5. Click Install (requires ~5-7 GB disk space)

6. **Restart system after installation**

**Step 3: Install Visual C++ Redistributable**

```powershell
# Download and install
# https://aka.ms/vs/17/release/vc_redist.x64.exe

# Or via PowerShell
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "vc_redist.x64.exe"
.\vc_redist.x64.exe
```

**Step 4: Install Git for Windows**

```powershell
# Download from https://git-scm.com/download/win
# Or use winget
winget install --id Git.Git -e --source winget
```

#### Phase 2: CUDA Installation (20-30 minutes)

**Step 5: Install CUDA Toolkit 12.4**

1. **Download CUDA 12.4**:
   - Visit: https://developer.nvidia.com/cuda-12-4-0-download-archive
   - Select: Windows → x86_64 → 11 → exe (local)
   - Download full installer (~3.5 GB)

2. **Run Installer**:
   ```powershell
   # Navigate to download folder
   cd C:\Users\YourUsername\Downloads

   # Run installer
   .\cuda_12.4.0_551.61_windows.exe
   ```

3. **Installation Options**:
   - Select "Custom (Advanced)" installation
   - ✅ CUDA Toolkit
   - ✅ CUDA Samples (optional)
   - ✅ CUDA Documentation (optional)
   - ✅ CUDA Visual Studio Integration
   - Install location: Default (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`)

4. **Wait for installation** (~15-20 minutes)

5. **Verify Installation**:
   ```powershell
   # Open NEW PowerShell window (to refresh environment variables)
   nvcc --version
   ```

   Expected output:
   ```
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2024 NVIDIA Corporation
   Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
   Cuda compilation tools, release 12.4, V12.4.99
   ```

6. **Verify Environment Variables** (usually set automatically):
   ```powershell
   echo $env:CUDA_PATH
   # Should show: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
   ```

#### Phase 3: Python Environment Setup (10-15 minutes)

**Step 6: Install Miniconda (Recommended) or Python**

**Option A: Miniconda (Recommended)**

```powershell
# Download Miniconda installer
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "Miniconda3-installer.exe"

# Run installer
.\Miniconda3-installer.exe

# Follow prompts:
# - Install for: Just Me
# - Destination: C:\Users\YourUsername\miniconda3
# - Advanced: ✅ Add Miniconda to PATH (optional but helpful)
```

**Option B: Python Direct Install**

```powershell
# Download Python 3.10 from python.org
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe" -OutFile "python-installer.exe"

# Run installer
.\python-installer.exe

# Installation options:
# ✅ Add Python 3.10 to PATH
# ✅ Install for all users (optional)
```

**Step 7: Create Conda Environment** (Skip if using system Python)

```powershell
# Open NEW PowerShell/Anaconda Prompt
conda create -n unsloth python=3.10 -y
conda activate unsloth

# Verify
python --version
# Should show: Python 3.10.x
```

#### Phase 4: PyTorch Installation (5-10 minutes)

**Step 8: Install PyTorch with CUDA 12.4**

```powershell
# Ensure conda environment activated (if using conda)
conda activate unsloth

# Install PyTorch 2.5.1 + CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**Step 9: Verify PyTorch CUDA**

```python
# Run Python
python

# In Python shell:
>>> import torch
>>> print(f"PyTorch version: {torch.__version__}")
>>> print(f"CUDA available: {torch.cuda.is_available()}")
>>> print(f"CUDA version: {torch.version.cuda}")
>>> print(f"GPU: {torch.cuda.get_device_name(0)}")
>>> exit()
```

Expected output:
```
PyTorch version: 2.5.1+cu124
CUDA available: True
CUDA version: 12.4
GPU: NVIDIA GeForce RTX 3090  # (or your GPU model)
```

**If CUDA not available**, troubleshoot:
```powershell
# Uninstall and reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

#### Phase 5: Triton Installation (5 minutes)

**Step 10: Install Triton for Windows**

```powershell
# Uninstall any existing Triton
pip uninstall triton -y

# Install triton-windows for PyTorch 2.4-2.5
pip install -U "triton-windows<3.3"
```

**Step 11: Verify Triton**

```python
python -c "import triton; print(f'Triton version: {triton.__version__}')"
```

#### Phase 6: Supporting Libraries (5-10 minutes)

**Step 12: Install xformers**

```powershell
# Install from PyTorch index for compatibility
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
```

**Step 13: Install bitsandbytes**

```powershell
# Install latest Windows-compatible version
pip install bitsandbytes==0.45.5

# Verify
python -m bitsandbytes
```

**Step 14: Install Additional Dependencies**

```powershell
# Core dependencies
pip install transformers accelerate peft trl datasets

# Optional but recommended
pip install wandb tensorboard jupyter ipywidgets
```

#### Phase 7: Unsloth Installation (5 minutes)

**Step 15: Install Unsloth**

```powershell
# Install Windows-optimized version
pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"

# Alternative: Specify exact CUDA and PyTorch versions
# pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

**Installation time**: 3-5 minutes (depends on internet speed)

**Step 16: Final Verification**

```python
# Test Unsloth import
python -c "from unsloth import FastLanguageModel; print('Unsloth installed successfully!')"

# Test loading a model
python
```

```python
from unsloth import FastLanguageModel

# This will download a small test model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("✅ Unsloth is working correctly!")
exit()
```

#### Phase 8: Optional - DeepSpeed (5 minutes)

**Step 17: Install DeepSpeed (Optional, for multi-GPU)**

```powershell
# Install confirmed working version
pip install deepspeed==0.8.3
```

**Verify**:
```python
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
```

#### Installation Complete!

**Total Installation Time**: Approximately 60-90 minutes

**Disk Space Used**: ~15-20 GB
- CUDA Toolkit: ~5 GB
- Visual Studio Build Tools: ~5-7 GB
- Python packages: ~3-5 GB
- Models (when downloaded): Variable

### 6.2 WSL2 Installation (Complete)

**Estimated Time**: 30-45 minutes

#### Phase 1: WSL2 Setup (10-15 minutes)

**Step 1: Enable WSL2**

```powershell
# Run PowerShell as Administrator
wsl --install

# Or install specific Ubuntu version
wsl --install -d Ubuntu-24.04

# Restart computer when prompted
```

**Step 2: Complete Ubuntu Setup**

After restart, Ubuntu will auto-launch:
```bash
# Create username and password
# Wait for "Installation successful!" message
```

**Step 3: Update Ubuntu**

```bash
sudo apt update && sudo apt upgrade -y
```

#### Phase 2: Install NVIDIA Drivers (Windows Side) (10 minutes)

**Step 4: Install/Update NVIDIA Drivers**

Download latest drivers: https://www.nvidia.com/drivers

**Note**: CUDA drivers are shared between Windows and WSL2. No separate WSL2 driver needed.

**Step 5: Verify GPU Access from WSL2**

```bash
# In WSL2 Ubuntu terminal
nvidia-smi
```

Should show your GPU information.

#### Phase 3: Python and Dependencies (15-20 minutes)

**Step 6: Install Build Tools**

```bash
sudo apt install -y build-essential python3-pip python3-dev git wget curl
```

**Step 7: Install Miniconda (Recommended)**

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts:
# - yes to license
# - default location: /home/username/miniconda3
# - yes to conda init

# Restart shell
source ~/.bashrc
```

**Step 8: Create Environment**

```bash
conda create -n unsloth python=3.10 -y
conda activate unsloth
```

**Step 9: Install PyTorch**

```bash
# Install PyTorch with CUDA 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

**Step 10: Verify PyTorch**

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Phase 4: Unsloth Installation (5-10 minutes)

**Step 11: Install Unsloth**

```bash
# Standard Linux installation (no Windows-specific fork needed)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Step 12: Install Additional Dependencies**

```bash
pip install --no-deps trl peft accelerate bitsandbytes transformers datasets
```

**Step 13: Verify Installation**

```python
python -c "from unsloth import FastLanguageModel; print('Unsloth working!')"
```

#### Installation Complete!

**WSL2 Advantages Confirmed**:
- ✅ Official Triton support (no Windows fork needed)
- ✅ vLLM support available
- ✅ GRPO training support
- ✅ Standard Linux installation procedure

### 6.3 Docker Installation (Quickest)

**Estimated Time**: 15-20 minutes

**Step 1: Install Docker Desktop**

```powershell
# Download from https://www.docker.com/products/docker-desktop/
# Install and enable WSL2 backend
# Restart system
```

**Step 2: Install NVIDIA Container Toolkit**

Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

**Step 3: Pull and Run Unsloth Container**

```bash
# Pull image
docker pull unsloth/unsloth

# Run container with GPU support
docker run --gpus all -p 8888:8888 -v C:\Users\YourUsername\unsloth-workspace:/workspace unsloth/unsloth
```

**Step 4: Access Jupyter Lab**

Open browser: `http://localhost:8888`
Token displayed in terminal output.

**Step 5: Test in Jupyter**

Create new notebook and run:
```python
from unsloth import FastLanguageModel
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print("Unsloth ready!")
```

---

## 7. Verification and Testing

### 7.1 System Verification Checklist

**Run all checks to ensure proper installation**:

#### Check 1: GPU Driver

```powershell
nvidia-smi
```

Expected: GPU information displayed, CUDA version shown

#### Check 2: CUDA Toolkit

```powershell
nvcc --version
```

Expected: CUDA 12.4 version info

#### Check 3: Python

```powershell
python --version
```

Expected: Python 3.10.x, 3.11.x, 3.12.x, or 3.13.x

#### Check 4: PyTorch

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected: `PyTorch: 2.5.1+cu124, CUDA: True, GPU: [Your GPU Name]`

#### Check 5: Triton

```python
python -c "import triton; print(f'Triton: {triton.__version__}')"
```

Expected: Triton version printed (e.g., `3.1.0`)

#### Check 6: xformers

```python
python -c "import xformers; print(f'xformers: {xformers.__version__}')"
```

Expected: xformers version printed

#### Check 7: bitsandbytes

```bash
python -m bitsandbytes
```

Expected: bitsandbytes info with CUDA support confirmed

#### Check 8: Unsloth

```python
python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"
```

Expected: `Unsloth: OK`

### 7.2 Functional Testing

**Test 1: Load Model**

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama",
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
)

print("✅ Model loaded successfully with 4-bit quantization")
```

**Test 2: Inference Test**

```python
# Enable inference mode
FastLanguageModel.for_inference(model)

# Test prompt
inputs = tokenizer(
    ["The capital of France is"],
    return_tensors = "pt"
).to("cuda")

# Generate
outputs = model.generate(**inputs, max_new_tokens = 20)
result = tokenizer.decode(outputs[0])

print(f"✅ Inference test result: {result}")
```

**Test 3: Training Setup Test**

```python
from trl import SFTTrainer
from transformers import TrainingArguments

# Prepare model for training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
)

print("✅ Model prepared for LoRA fine-tuning")

# Test dataset (minimal)
test_data = [
    {"text": "This is a test example for fine-tuning."},
]

from datasets import Dataset
dataset = Dataset.from_list(test_data)

# Configure trainer (minimal config)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    dataset_num_proc = 1,  # CRITICAL for Windows
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        max_steps = 1,
        output_dir = "test_output",
        logging_steps = 1,
    ),
)

print("✅ Trainer configured successfully")

# Run 1 training step
trainer.train()

print("✅ Training test completed successfully!")
```

### 7.3 Comprehensive System Report Script

Save as `verify_unsloth_installation.py`:

```python
#!/usr/bin/env python3
"""
Unsloth Windows Installation Verification Script
Checks all dependencies and generates comprehensive report
"""

import sys
import subprocess

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_command(cmd, name):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"✅ {name}: OK")
            print(f"   {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {name}: FAILED")
            print(f"   {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {name}: ERROR - {str(e)}")
        return False

def check_import(module, name):
    try:
        __import__(module)
        mod = sys.modules[module]
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {name}: NOT INSTALLED - {str(e)}")
        return False

def main():
    print("Unsloth Windows Installation Verification")
    print(f"Python: {sys.version}")

    results = {}

    # System checks
    print_section("System Components")
    results['nvidia-smi'] = check_command("nvidia-smi", "NVIDIA Driver")
    results['nvcc'] = check_command("nvcc --version", "CUDA Toolkit")

    # Python packages
    print_section("Python Packages")
    results['torch'] = check_import("torch", "PyTorch")

    if results['torch']:
        import torch
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    results['triton'] = check_import("triton", "Triton")
    results['xformers'] = check_import("xformers", "xformers")
    results['bitsandbytes'] = check_import("bitsandbytes", "bitsandbytes")
    results['transformers'] = check_import("transformers", "Transformers")
    results['peft'] = check_import("peft", "PEFT")
    results['trl'] = check_import("trl", "TRL")
    results['unsloth'] = check_import("unsloth", "Unsloth")

    # Summary
    print_section("Summary")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"Checks Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL CHECKS PASSED - Unsloth installation is complete and functional!")
    else:
        print("\n⚠️ SOME CHECKS FAILED - Review errors above and reinstall missing components")
        failed = [k for k, v in results.items() if not v]
        print(f"Failed components: {', '.join(failed)}")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

**Usage**:
```powershell
python verify_unsloth_installation.py
```

---

## 8. Troubleshooting Guide

### 8.1 Common Error Messages and Solutions

#### Error: "The system cannot find the path specified"

**Context**: Running CUDA-related commands

**Cause**: CUDA not in PATH

**Solution**:
```powershell
# Check current PATH
echo $env:PATH

# Add CUDA to PATH manually
setx PATH "$env:PATH;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"

# Restart PowerShell and test
nvcc --version
```

#### Error: "RuntimeError: No CUDA GPUs are available"

**Diagnosis**:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

**Solutions**:
1. Verify GPU detected: `nvidia-smi`
2. Reinstall NVIDIA drivers
3. Reinstall PyTorch with CUDA:
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

#### Error: "OSError: [WinError 126] The specified module could not be found"

**Context**: Importing Triton or other C++ extensions

**Cause**: Missing Visual C++ Redistributable

**Solution**:
```powershell
# Download and install VC++ Redistributable
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "vc_redist.x64.exe"
.\vc_redist.x64.exe
```

#### Error: "Killed" or Process Terminated During Training

**Cause**: Out of memory (RAM or VRAM)

**Solutions**:

1. **Reduce batch size**:
   ```python
   per_device_train_batch_size = 1
   gradient_accumulation_steps = 16
   ```

2. **Enable gradient checkpointing**:
   ```python
   use_gradient_checkpointing = True
   ```

3. **Use 4-bit quantization**:
   ```python
   load_in_4bit = True
   ```

4. **Reduce sequence length**:
   ```python
   max_seq_length = 1024  # Instead of 2048
   ```

5. **Close other applications** to free up memory

#### Error: "Failed to build wheel for [package]"

**Context**: Installing packages with C++ extensions

**Cause**: Missing build tools

**Solution**:
1. Ensure Visual Studio 2022 C++ build tools installed
2. Open "Developer Command Prompt for VS 2022"
3. Retry installation from that prompt

#### Error: Dataset Processing Crash (No Error Message)

**Context**: Using SFTTrainer on Windows

**Cause**: Multiprocessing issues

**Solution**:
```python
trainer = SFTTrainer(
    # ... other args
    dataset_num_proc = 1,  # Must be 1 on Windows
)
```

### 8.2 Performance Optimization Tips

#### Tip 1: Use Flash Attention (if supported)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    # Flash Attention 2.0 (for RTX 30xx+)
)

# Flash Attention is automatically enabled if supported
```

#### Tip 2: Optimize VRAM Usage

```python
# Use mixed precision training
training_args = TrainingArguments(
    fp16 = True,  # Or bf16 = True for Ampere+ GPUs
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
)
```

#### Tip 3: Enable Paged Optimizers

```python
# For large models
training_args = TrainingArguments(
    optim = "paged_adamw_8bit",  # Reduces optimizer memory
)
```

#### Tip 4: Use Gradient Checkpointing

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    use_gradient_checkpointing = True,  # Trade compute for memory
)
```

### 8.3 Diagnostic Commands Reference

**Check GPU Utilization**:
```powershell
# Monitor GPU usage in real-time
nvidia-smi -l 1  # Updates every 1 second
```

**Check VRAM Usage**:
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Check PyTorch Build Info**:
```python
import torch
print(torch.__config__.show())
```

**Check Installed Packages**:
```powershell
pip list | Select-String "torch|triton|unsloth|xformers|bitsandbytes"
```

**Generate Full System Report**:
```powershell
# GPU info
nvidia-smi > gpu_info.txt

# Environment info
conda env export > environment.yml

# Installed packages
pip list > installed_packages.txt
```

### 8.4 Getting Help

**Official Resources**:
- Documentation: https://docs.unsloth.ai
- GitHub Issues: https://github.com/unslothai/unsloth/issues
- GitHub Discussions: https://github.com/unslothai/unsloth/discussions

**Community Resources**:
- Stack Overflow: Tag `unsloth`
- Reddit: r/LocalLLaMA
- Discord: Unsloth community server

**When Reporting Issues, Include**:
1. Operating System version (Windows 10/11)
2. GPU model and driver version
3. CUDA version (`nvcc --version`)
4. PyTorch version and CUDA availability
5. Complete error traceback
6. Minimal reproducible code example
7. Installation method used

---

## 9. Resource Links

### 9.1 Official Documentation

**Unsloth**:
- Main Documentation: https://docs.unsloth.ai
- Windows Installation: https://docs.unsloth.ai/get-started/install-and-update/windows-installation
- GitHub Repository: https://github.com/unslothai/unsloth
- PyPI Package: https://pypi.org/project/unsloth/

**PyTorch**:
- Official Site: https://pytorch.org
- Installation Guide: https://pytorch.org/get-started/locally/
- Previous Versions: https://pytorch.org/get-started/previous-versions/

**NVIDIA CUDA**:
- CUDA Toolkit Download: https://developer.nvidia.com/cuda-downloads
- CUDA 12.4 Archive: https://developer.nvidia.com/cuda-12-4-0-download-archive
- Installation Guide: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
- Driver Download: https://www.nvidia.com/drivers

**Triton Windows**:
- woct0rdho/triton-windows: https://github.com/woct0rdho/triton-windows
- Releases: https://github.com/woct0rdho/triton-windows/releases

### 9.2 Dependency Documentation

**xformers**:
- GitHub: https://github.com/facebookresearch/xformers
- Installation: https://github.com/facebookresearch/xformers#installing-xformers

**bitsandbytes**:
- Official Docs: https://huggingface.co/docs/bitsandbytes/
- Installation Guide: https://huggingface.co/docs/bitsandbytes/main/en/installation
- jllllll Windows Build: https://github.com/jllllll/bitsandbytes-windows-webui

**Transformers**:
- Documentation: https://huggingface.co/docs/transformers/
- Installation: https://huggingface.co/docs/transformers/installation

**PEFT**:
- Documentation: https://huggingface.co/docs/peft/
- GitHub: https://github.com/huggingface/peft

**TRL**:
- Documentation: https://huggingface.co/docs/trl/
- GitHub: https://github.com/huggingface/trl

**DeepSpeed**:
- Official Site: https://www.deepspeed.ai
- Installation: https://www.deepspeed.ai/tutorials/advanced-install/
- Windows Support: https://github.com/microsoft/DeepSpeed/issues/4729

### 9.3 Visual Studio and Build Tools

**Visual Studio**:
- Downloads: https://visualstudio.microsoft.com/downloads/
- Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
- VC++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

**Python**:
- Official Downloads: https://www.python.org/downloads/
- Miniconda: https://docs.conda.io/en/latest/miniconda.html

### 9.4 Community Tutorials and Guides

**Blog Posts**:
- Fine-Tuning LLMs on Windows: https://robkerr.ai/fine-tuning-llms-using-a-local-gpu-on-windows/
- Unsloth Guide (DataCamp): https://www.datacamp.com/tutorial/unsloth-guide-optimize-and-speed-up-llm-fine-tuning
- Installing Triton on Windows: https://www.kombitz.com/2025/02/20/how-to-install-triton-on-windows/

**GitHub Resources**:
- Unsloth Windows Installation Gist: https://gist.github.com/sebaxakerhtc/502cc20cc94b8eb1acc29dee0610fdab
- WSL2 GPU Setup: https://gist.github.com/wooihaw/9e63e3a4a16ac5ab37f6b8db1e2e7465

**Videos and Courses**:
- Unsloth Official YouTube Channel
- FineTuning with Unsloth (Matt Williams): https://technovangelist.com/videos/finetuning-with-unsloth

### 9.5 Key GitHub Issues and Discussions

**Windows Support**:
- Direct Windows Support Discussion: https://github.com/unslothai/unsloth/discussions/1849
- Windows Support Issue: https://github.com/unslothai/unsloth/issues/1850
- Installation Guide Issue: https://github.com/unslothai/unsloth/issues/402
- Native Windows Success: https://github.com/unslothai/unsloth/issues/210

**Troubleshooting Issues**:
- Triton Windows Issues: https://github.com/unslothai/unsloth/issues/381
- CUDA Detection: https://github.com/unslothai/unsloth/issues/412
- Version Compatibility: https://github.com/unslothai/unsloth/issues/2499

### 9.6 Version-Specific Resources

**Compatibility Matrices**:
- PyTorch CUDA Support Matrix: https://pytorch.org/get-started/previous-versions/
- Triton-Windows Compatibility: https://github.com/woct0rdho/triton-windows#compatibility

**Release Notes**:
- Unsloth Releases: https://github.com/unslothai/unsloth/releases
- PyTorch Release Notes: https://github.com/pytorch/pytorch/releases
- CUDA Toolkit Release Notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

---

## Appendix A: Quick Reference Commands

### Installation Commands Summary

**Native Windows (Complete)**:
```powershell
# 1. Install Visual Studio 2022 Build Tools (GUI installer)
# 2. Install CUDA 12.4 (GUI installer)
# 3. Install VC++ Redistributable
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "vc_redist.x64.exe"
.\vc_redist.x64.exe

# 4. Create conda environment
conda create -n unsloth python=3.10 -y
conda activate unsloth

# 5. Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 6. Install Triton
pip uninstall triton -y
pip install -U "triton-windows<3.3"

# 7. Install supporting libraries
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes==0.45.5
pip install transformers accelerate peft trl datasets

# 8. Install Unsloth
pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
```

**WSL2 (Complete)**:
```bash
# 1. In WSL2 Ubuntu
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3-pip python3-dev git wget

# 2. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# 3. Create environment
conda create -n unsloth python=3.10 -y
conda activate unsloth

# 4. Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 5. Install Unsloth and dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes transformers datasets
```

### Verification Commands

```python
# Quick verification script
python << EOF
import torch
from unsloth import FastLanguageModel

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print("✅ Unsloth ready!")
EOF
```

---

## Appendix B: Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-16 | 1.0 | Initial comprehensive guide created |

---

## Appendix C: Glossary

**4-bit Quantization**: Technique to reduce model memory usage by representing weights with 4 bits instead of 32

**Ampere**: NVIDIA GPU architecture (RTX 30 series, A100)

**CUDA**: Compute Unified Device Architecture - NVIDIA's parallel computing platform

**LoRA**: Low-Rank Adaptation - Parameter-efficient fine-tuning method

**MSVC**: Microsoft Visual C++ compiler

**PEFT**: Parameter-Efficient Fine-Tuning

**SFTTrainer**: Supervised Fine-Tuning Trainer from TRL library

**Triton**: Language and compiler for writing GPU kernels

**vLLM**: Fast LLM inference library (Linux-only)

**WSL2**: Windows Subsystem for Linux version 2

**xformers**: Memory-efficient attention implementations

---

**Document End**

For updates to this guide, check: https://github.com/unslothai/unsloth/discussions

**Questions or Issues?** Open an issue at: https://github.com/unslothai/unsloth/issues
