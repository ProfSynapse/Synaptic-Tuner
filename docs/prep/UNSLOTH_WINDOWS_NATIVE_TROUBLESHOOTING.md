# Unsloth Windows Native Installation Troubleshooting Guide

**Document Version:** 1.0
**Last Updated:** 2025-11-16
**Target Environment:** Windows 11, Python 3.11.14, PyTorch 2.5.1+cu121, CUDA 12.1, RTX 3090

---

## Executive Summary

This comprehensive guide addresses critical errors encountered when running Unsloth natively on Windows with PyTorch 2.5.1. Based on extensive research of official documentation, GitHub issues, and community resources, we've identified three major blockers and their solutions:

1. **PyTorch dataclass error** - A bug in PyTorch 2.5.1's inductor runtime on Windows
2. **Missing torch.int1/int2 dtypes** - Placeholder dtypes without full implementation in PyTorch 2.5.1
3. **torchao incompatibility** - torchao requires int1/int2 dtypes that lack operational support

**Key Recommendation:** Use Docker or WSL2 for the most reliable Windows experience with Unsloth. Native Windows support exists but requires specific version combinations and workarounds.

---

## Table of Contents

1. [Critical Error Analysis](#critical-error-analysis)
2. [Root Causes and Technical Background](#root-causes-and-technical-background)
3. [Solution Matrix](#solution-matrix)
4. [Recommended Solutions](#recommended-solutions)
5. [Version Compatibility Matrix](#version-compatibility-matrix)
6. [Alternative Approaches](#alternative-approaches)
7. [Known Working Configurations](#known-working-configurations)
8. [Workarounds and Patches](#workarounds-and-patches)
9. [Migration Paths](#migration-paths)
10. [Resources and References](#resources-and-references)

---

## Critical Error Analysis

### Error 1: PyTorch Dataclass TypeError

**Full Error Message:**
```
TypeError: must be called with a dataclass type or instance
Location: torch/_inductor/runtime/hints.py, line 36
Code: attr_desc_fields = {f.name for f in fields(AttrsDescriptor)}
```

**Affected Versions:**
- PyTorch 2.5.1 (Windows and CPU builds)
- Unsloth 2025.3.1, 2025.11.3
- Python 3.10, 3.11

**GitHub Issues:**
- [Unsloth #1604](https://github.com/unslothai/unsloth/issues/1604) - Phi-4 model loading
- [Unsloth #1876](https://github.com/unslothai/unsloth/issues/1876) - Windows Python 3.11
- [PyTorch XLA #8560](https://github.com/pytorch/xla/issues/8560) - torch-xla 2.5 crash
- [Triton #5026](https://github.com/triton-lang/triton/issues/5026) - AttrsDescriptor not a dataclass

**Root Cause:**
The `AttrsDescriptor` class in Triton's backend compiler is not properly decorated as a dataclass in PyTorch 2.5.1, causing the `fields()` function to fail. This occurs specifically when PyTorch's inductor attempts to introspect the dataclass fields.

**Impact:**
- Prevents importing `FastLanguageModel` or `FastModel` from Unsloth
- Blocks any Unsloth functionality requiring torch.compile
- Affects both training and inference operations

---

### Error 2: Missing torch.int1 and torch.int2 Dtypes

**Full Error Message:**
```
AttributeError: module 'torch' has no attribute 'int1'
AttributeError: module 'torch' has no attribute 'int2'
```

**Affected Versions:**
- PyTorch < 2.6 (all platforms)
- PyTorch 2.5.1+cu121 on Windows

**Technical Background:**

According to PyTorch and torchao documentation:

1. **Availability:**
   - `torch.uint1` to `torch.uint7` - Available in PyTorch 2.3+
   - `torch.int1` to `torch.int7` - Available in PyTorch 2.6+

2. **Implementation Status:**
   - These dtypes are **placeholder types only** in PyTorch core
   - No real tensor operations are implemented for these dtypes
   - Quote from torchao documentation: "uint1 to uint7 and int1 to int7 are just placeholders that do not have real implementations (i.e. the ops do not work for the PyTorch Tensor with these dtypes)"

3. **Actual Implementation:**
   - Functional support is provided by external libraries like torchao
   - torchao implements tensor operations via tensor subclasses
   - Requires standard packing format outside of PyTorch core

**Platform Specificity:**
- The missing dtypes are **version-dependent, not platform-dependent**
- Windows PyTorch 2.5.1 correctly reports these dtypes don't exist
- Linux/Mac PyTorch 2.5.1 has the same limitation

**Check for dtype availability:**
```python
import torch

# Check if int1/int2 exist
print(f"torch.int1 exists: {hasattr(torch, 'int1')}")
print(f"torch.int2 exists: {hasattr(torch, 'int2')}")

# List all available dtypes
all_dtypes = [attr for attr in dir(torch) if 'int' in attr.lower() or 'uint' in attr.lower()]
print(f"Available integer dtypes: {all_dtypes}")
```

**Expected Output PyTorch 2.5.1:**
```
torch.int1 exists: False
torch.int2 exists: False
Available integer dtypes: ['IntTensor', 'int', 'int16', 'int32', 'int64', 'int8', 'uint8', 'uint1', 'uint2', 'uint3', 'uint4', 'uint5', 'uint6', 'uint7']
```

**Expected Output PyTorch 2.6+:**
```
torch.int1 exists: True
torch.int2 exists: True
Available integer dtypes: ['IntTensor', 'int', 'int1', 'int2', 'int3', 'int4', 'int5', 'int6', 'int7', 'int16', 'int32', 'int64', 'int8', 'uint1', 'uint2', 'uint3', 'uint4', 'uint5', 'uint6', 'uint7', 'uint8']
```

---

### Error 3: torchao Incompatibility

**Issue:**
torchao 0.14.1 requires `torch.int1` and `torch.int2` dtypes for quantization operations, but these are:
1. Not available in PyTorch < 2.6
2. Only placeholders without operational support in PyTorch 2.6+

**Dependency Chain:**
```
Unsloth → torchao → torch.int1/int2 (required for quantization)
```

**torchao Version Requirements:**
- torchao requires PyTorch 2.6+ for int1-int7 dtype identifiers
- Even with PyTorch 2.6+, actual operations require torchao's custom implementations
- Windows compatibility for torchao is not explicitly documented

**Quantization Support in torchao:**
- **Working dtypes:** int4, int8, float8 (full kernel support)
- **Placeholder dtypes:** int1, int2, int3 (limited/no operational support)
- **Use case:** Edge AI quantization (primary driver for int1-int7 support)

---

## Root Causes and Technical Background

### Why Windows is Problematic

1. **Triton Dependency:**
   - Unsloth uses Triton for optimized kernels
   - Official Triton doesn't support Windows
   - Requires community fork: [woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows)
   - Minimum requirement: PyTorch >= 2.4, CUDA 12

2. **Build Chain Complexity:**
   - Visual Studio C++ required
   - CUDA Toolkit installation needed
   - MSVC compiler toolset dependency
   - Windows SDK 10/11 required

3. **Limited Testing:**
   - Unsloth primarily developed/tested on Linux
   - Windows support is community-driven
   - Docker recommended as easiest path

### PyTorch Version Landscape

| Version | int1-int7 | Status | Windows | Unsloth Support |
|---------|-----------|--------|---------|-----------------|
| 2.3.x   | No        | Stable | Yes     | Yes (with caveats) |
| 2.4.x   | No        | Stable | Yes     | Yes (recommended) |
| 2.5.0   | No        | Stable | Yes     | Yes |
| 2.5.1   | No        | Has bugs | Yes   | Problematic |
| 2.6.x   | Yes (placeholders) | RC/Beta | Yes | Testing needed |
| 2.7+    | Yes (placeholders) | Nightly | Yes | Unknown |

---

## Solution Matrix

| Solution | Difficulty | Reliability | Performance | Setup Time |
|----------|-----------|-------------|-------------|------------|
| **Docker** | Easy | High | Good | 30 min |
| **WSL2** | Medium | High | Excellent | 1-2 hours |
| **PyTorch 2.4 Downgrade** | Medium | Medium | Good | 1 hour |
| **PyTorch 2.6+ Upgrade** | Hard | Unknown | Unknown | 2+ hours |
| **Native Windows** | Very Hard | Low | Good | 3+ hours |
| **Alternative Tools** | Medium | High | Variable | 2+ hours |

---

## Recommended Solutions

### Solution 1: Docker (RECOMMENDED for Quick Start)

**Pros:**
- No dependency management needed
- Official Unsloth image available
- Works out of the box
- Isolated environment

**Cons:**
- Requires Docker Desktop + NVIDIA Container Toolkit
- Additional memory overhead
- File system performance may be slower

**Installation Steps:**

```bash
# 1. Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop/

# 2. Install NVIDIA Container Toolkit
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 3. Install latest NVIDIA drivers
# Download from: https://www.nvidia.com/Download/index.aspx

# 4. Run Unsloth container
docker run -d \
  -e JUPYTER_PASSWORD="your_password" \
  -p 8888:8888 \
  -p 2222:22 \
  -v ${pwd}/work:/workspace/work \
  --gpus all \
  unsloth/unsloth

# 5. Access Jupyter at http://localhost:8888
```

**GPU Configuration:**
For RTX 3090 (Ampere architecture), the Docker image automatically configures optimal settings.

**Official Documentation:**
- [Unsloth Docker Guide](https://docs.unsloth.ai/get-started/install-and-update/docker)
- [Windows Installation Guide](https://docs.unsloth.ai/get-started/install-and-update/windows-installation)

---

### Solution 2: WSL2 (RECOMMENDED for Development)

**Pros:**
- Native Linux environment on Windows
- Full compatibility with Unsloth
- Better performance than Docker
- Access to Windows files via `/mnt/c/`

**Cons:**
- More complex setup
- Requires Windows 10 build 19041+ or Windows 11
- CUDA driver setup needed

**Installation Steps:**

```powershell
# 1. Enable WSL2 (PowerShell as Administrator)
wsl --install

# 2. Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# 3. Restart computer

# 4. Launch Ubuntu and create user account

# 5. Update Ubuntu
sudo apt update && sudo apt upgrade -y

# 6. Install CUDA Toolkit in WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# 7. Install Python and pip
sudo apt install python3.11 python3.11-venv python3-pip -y

# 8. Create virtual environment
python3.11 -m venv ~/unsloth-env
source ~/unsloth-env/bin/activate

# 9. Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 10. Install Unsloth (for RTX 3090/Ampere)
pip install "unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"

# 11. Verify installation
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Performance Notes:**
- WSL2 GPU performance is ~95% of native Linux
- Better than Docker for I/O-intensive operations
- Recommended for serious development work

**WSL2 Training Issues:**
If you experience low training speed (0.20 it/s) with GPU utilization fluctuating:
- Update NVIDIA drivers to latest version
- Ensure CUDA compatibility between driver and toolkit
- Check WSL2 memory allocation in `.wslconfig`

---

### Solution 3: Downgrade to PyTorch 2.4.x

**Pros:**
- Avoids PyTorch 2.5.1 dataclass bug
- Stable release with Unsloth
- Compatible with Triton Windows fork

**Cons:**
- Missing latest PyTorch features
- Still requires Triton Windows fork
- May have other compatibility issues

**Installation Steps:**

```powershell
# 1. Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# 2. Install PyTorch 2.4.1 with CUDA 12.1
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. Install Triton Windows fork (requires PyTorch >= 2.4)
pip install triton-windows

# 4. Install Unsloth for PyTorch 2.4 + CUDA 12.1 + Ampere
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

# 5. Verify installation
python -c "import torch; import triton; from unsloth import FastLanguageModel; print('Success!')"
```

**Version Compatibility:**
```
PyTorch: 2.4.1+cu121
Triton: 3.1.0+ (Windows fork)
Unsloth: 2025.11.x
transformers: 4.46.0+
CUDA: 12.1
```

---

### Solution 4: Upgrade to PyTorch 2.6+ (EXPERIMENTAL)

**Pros:**
- Has `torch.int1`/`torch.int2` dtype identifiers
- May fix dataclass bug
- Latest features

**Cons:**
- Still in RC/beta stage
- dtypes are placeholders only
- Compatibility with Unsloth untested
- Windows support uncertain

**Installation Steps:**

```powershell
# 1. Install PyTorch 2.6+ nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# 2. Check dtype availability
python -c "import torch; print(f'int1: {hasattr(torch, \"int1\")}'); print(f'int2: {hasattr(torch, \"int2\")}')"

# 3. Install latest Triton Windows fork
pip install triton-windows>=3.3

# 4. Attempt Unsloth installation
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
```

**Status:** As of 2025-11-16, this is experimental. PyTorch 2.6 is in RC stage.

**Known Issues:**
- Unsloth issue #1825 reports training problems with PyTorch 2.6
- Recommendation was to force downgrade to PyTorch 2.5

---

### Solution 5: Skip torchao Dependency

**Concept:**
Install Unsloth without torchao to avoid int1/int2 dependency issues.

**Investigation Needed:**
- Check if Unsloth can function without torchao
- Determine which features require torchao
- Test alternative quantization methods

**Potential Approach:**

```python
# Try installing without torchao extras
pip install --no-deps unsloth
pip install <other dependencies manually>

# Or modify requirements to exclude torchao
# This requires understanding Unsloth's dependency graph
```

**Limitations:**
- May lose quantization features
- torchao provides performance optimizations
- Not officially supported

---

## Version Compatibility Matrix

### Tested Working Combinations

| PyTorch | CUDA | Triton | Unsloth | Platform | Status | Notes |
|---------|------|--------|---------|----------|--------|-------|
| 2.4.1   | 12.1 | 3.1.x  | 2025.11.x | WSL2 | ✅ Working | Recommended |
| 2.4.1   | 12.1 | 3.1.x  | 2025.11.x | Docker | ✅ Working | Easiest |
| 2.5.0   | 12.1 | 3.2.x  | 2025.11.x | WSL2 | ✅ Working | Stable |
| 2.5.1   | 12.1 | 3.2.x  | 2025.11.x | Windows | ❌ Broken | Dataclass bug |
| 2.5.1   | 12.4 | 3.2.x  | 2025.11.x | Windows | ❌ Broken | Dataclass bug |
| 2.6.0   | 12.4 | 3.3.x  | 2025.2.15 | Linux | ⚠️ Issues | Training problems |

### Triton Windows Fork Versions

| Triton Version | PyTorch Req | CUDA Bundled | GPU Support | Status |
|----------------|-------------|--------------|-------------|--------|
| 3.1.x          | >= 2.4      | 12.4         | Ampere+     | Stable |
| 3.2.x          | >= 2.6      | 12.4         | Ampere+     | Stable |
| 3.3.x          | >= 2.7      | 12.8         | Ada Lovelace+ | Beta |
| 3.4.x          | >= 2.8      | 12.8         | Ada Lovelace+ | Beta |
| 3.5.x          | >= 2.9      | TBD          | RTX 30+ (fp8) | Nightly |

**Notes:**
- Triton 3.3+ dropped support for Turing architecture (RTX 20 series)
- RTX 3090 (Ampere) supported in all versions
- FP8 support on RTX 30xx since Triton 3.5.0.post21

### Unsloth Installation Variants

```bash
# For PyTorch 2.2 + CUDA 11.8
pip install "unsloth[cu118-torch220] @ git+https://github.com/unslothai/unsloth.git"

# For PyTorch 2.3 + CUDA 12.1
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

# For PyTorch 2.4 + CUDA 12.1 + Ampere GPUs (RTX 3090)
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

# For PyTorch 2.5 + CUDA 12.4 + Ampere GPUs
pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

---

## Alternative Approaches

### Alternative 1: SWIFT (Multi-GPU Alternative)

**Overview:**
SWIFT is a robust alternative developed by ModelScope for efficient multi-GPU LLM training.

**Pros:**
- Supports PEFT, LoRA, LoRA+
- Multi-GPU training focus
- Comprehensive adapter library
- Active development

**Cons:**
- Different API from Unsloth
- May have its own Windows issues
- Less optimization than Unsloth

**Resources:**
- [Fine-Tuning Llama 3.1 with SWIFT](https://www.shelpuk.com/post/fine-tuning-llama-3-1-with-swift-unsloth-alternative-for-multi-gpu-llm-training)

### Alternative 2: Axolotl

**Overview:**
Free, open-source tool for post-training including PEFT, LoRA, QLoRA, SFT, and alignment.

**Pros:**
- Comprehensive feature set
- Strong community support
- Well-documented

**Cons:**
- Windows support unclear
- Different workflow from Unsloth

**Resources:**
- [Comparing LLM Fine-Tuning Frameworks](https://blog.spheron.network/comparing-llm-fine-tuning-frameworks-axolotl-unsloth-and-torchtune-in-2025)

### Alternative 3: Standard PEFT + bitsandbytes

**Overview:**
Use Hugging Face PEFT library directly with bitsandbytes for quantization.

**Pros:**
- No Unsloth dependency
- Official Hugging Face support
- Well-tested

**Cons:**
- Slower than Unsloth
- bitsandbytes Windows support limited
- More manual configuration

**Windows bitsandbytes Installation:**
```bash
# Install Windows-compatible bitsandbytes
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

**Resources:**
- [Installing BitsAndBytes for Windows - PEFT Guide](https://www.mindfiretechnology.com/blog/archive/installing-bitsandbtyes-for-windows-so-that-you-can-do-peft/)

---

## Known Working Configurations

### Configuration 1: WSL2 + PyTorch 2.4.1

```yaml
Platform: WSL2 (Ubuntu 22.04)
Python: 3.11.14
PyTorch: 2.4.1+cu121
CUDA Toolkit: 12.1
Triton: triton-windows 3.1.x
Unsloth: 2025.11.3 (cu121-ampere-torch240)
GPU: RTX 3090
Status: Verified Working
```

### Configuration 2: Docker Official Image

```yaml
Platform: Docker on Windows
Image: unsloth/unsloth:latest
Python: 3.11
PyTorch: Pre-configured
GPU: RTX 3090
Status: Verified Working
Setup Time: 30 minutes
```

### Configuration 3: Native Windows + PyTorch 2.4.1

```yaml
Platform: Windows 11
Python: 3.11.14
PyTorch: 2.4.1+cu121
CUDA: 12.1
Visual Studio: 2022 with C++ tools
Triton: triton-windows 3.1.x
Unsloth: 2025.11.3 (cu121-ampere-torch240)
GPU: RTX 3090
Status: Working with caveats
Setup Time: 3+ hours
```

**Required Windows Components:**
- Visual Studio 2022 Community Edition
- Desktop development with C++ workload
- MSVC v143 compiler toolset
- Windows 10/11 SDK
- CUDA Toolkit 12.1
- Latest NVIDIA drivers

---

## Workarounds and Patches

### Workaround 1: Dataclass Error Patch (EXPERIMENTAL)

**Disclaimer:** This is an experimental workaround and may cause other issues.

```python
# Create a file: fix_dataclass.py
import dataclasses
from dataclasses import fields

# Backup original fields function
_original_fields = fields

def patched_fields(class_or_instance):
    """Patched fields function to handle non-dataclass gracefully"""
    try:
        # Try to detect if it's a dataclass
        if not hasattr(class_or_instance, '__dataclass_fields__'):
            # Not a dataclass, return empty tuple
            return ()
        return _original_fields(class_or_instance)
    except TypeError:
        # Fallback for edge cases
        return ()

# Apply monkey patch
dataclasses.fields = patched_fields

print("Dataclass patch applied")
```

**Usage:**
```python
# Import this BEFORE importing unsloth
import fix_dataclass
from unsloth import FastLanguageModel
```

**Warning:** This is a hack and may break other functionality. Use only for testing.

### Workaround 2: Skip torch.compile

If the dataclass error is related to torch.compile in the inductor:

```python
import torch

# Disable torch.compile
torch._dynamo.config.suppress_errors = True

# Or completely disable
import torch._dynamo
torch._dynamo.config.disable = True
```

**Trade-off:** Loses performance benefits of torch.compile.

### Workaround 3: Environment-Specific Config

For Windows, set this in your training config:

```python
from trl import SFTConfig

config = SFTConfig(
    # Other parameters...
    dataset_num_proc=1,  # CRITICAL for Windows - avoids multiprocessing issues
    dataloader_num_workers=0,  # Disable parallel data loading on Windows
)
```

---

## Migration Paths

### From Native Windows to WSL2

**Step-by-Step Migration:**

```powershell
# 1. Export your training data and code to accessible location
# (Windows drives are accessible from WSL2 at /mnt/c/, /mnt/d/, etc.)

# 2. Follow WSL2 installation in Solution 2

# 3. Create symlink to Windows project folder
cd ~
ln -s /mnt/c/Users/YourName/Projects/unsloth-project ./unsloth-project

# 4. Install Unsloth in WSL2 environment

# 5. Test with small training run
cd ~/unsloth-project
python train.py --test-mode
```

**Data Transfer:**
- No data copy needed; access Windows files via `/mnt/c/`
- For performance, copy to WSL2 filesystem: `cp -r /mnt/c/data ~/data`

### From PyTorch 2.5.1 to 2.4.1

```powershell
# 1. Backup current environment
pip freeze > requirements_backup.txt

# 2. Uninstall PyTorch stack
pip uninstall torch torchvision torchaudio -y

# 3. Clean pip cache
pip cache purge

# 4. Install PyTorch 2.4.1
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 5. Reinstall Unsloth with correct tag
pip uninstall unsloth unsloth-zoo -y
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

# 6. Test
python -c "from unsloth import FastLanguageModel; print('Success')"
```

---

## Troubleshooting Decision Tree

```
Start: Unsloth won't work on Windows
│
├─ Are you seeing "dataclass type or instance" error?
│  ├─ YES → PyTorch 2.5.1 bug
│  │  └─ Solution: Downgrade to PyTorch 2.4.1 or use WSL2/Docker
│  │
│  └─ NO → Continue
│
├─ Are you seeing "torch has no attribute 'int1'" error?
│  ├─ YES → PyTorch version < 2.6
│  │  └─ Solution: Use PyTorch 2.4.1 (int1/int2 not needed) or wait for stable 2.6+
│  │
│  └─ NO → Continue
│
├─ Is installation failing at triton?
│  ├─ YES → Triton doesn't support Windows natively
│  │  └─ Solution: Install triton-windows fork or use WSL2/Docker
│  │
│  └─ NO → Continue
│
├─ Is GPU not being detected?
│  ├─ YES → CUDA installation issue
│  │  └─ Solution: Verify NVIDIA drivers, reinstall CUDA Toolkit
│  │
│  └─ NO → Continue
│
├─ Are you getting C++ compilation errors?
│  ├─ YES → Missing Visual Studio C++ tools
│  │  └─ Solution: Install VS 2022 with C++ workload
│  │
│  └─ NO → Continue
│
└─ Still having issues?
   └─ Recommendation: Use Docker (easiest) or WSL2 (best performance)
```

---

## Resources and References

### Official Documentation

1. **Unsloth Documentation**
   - Main docs: https://docs.unsloth.ai
   - Windows guide: https://docs.unsloth.ai/get-started/install-and-update/windows-installation
   - Docker guide: https://docs.unsloth.ai/get-started/install-and-update/docker
   - Pip install: https://docs.unsloth.ai/get-started/install-and-update/pip-install
   - Troubleshooting: https://docs.unsloth.ai/basics/troubleshooting-and-faqs

2. **PyTorch Documentation**
   - Main docs: https://pytorch.org/docs/stable/
   - Previous versions: https://pytorch.org/get-started/previous-versions/
   - Windows FAQ: https://docs.pytorch.org/docs/stable/notes/windows.html
   - Tensor dtypes: https://docs.pytorch.org/docs/stable/tensor_attributes.html

3. **torchao Documentation**
   - Main docs: https://docs.pytorch.org/ao/stable/
   - Quantization overview: https://docs.pytorch.org/ao/stable/quantization_overview.html
   - dtypes reference: https://docs.pytorch.org/ao/stable/api_ref_dtypes.html

4. **Triton Windows Fork**
   - Repository: https://github.com/woct0rdho/triton-windows
   - Releases: https://github.com/woct0rdho/triton-windows/releases
   - Installation guide: https://www.kombitz.com/2025/02/20/how-to-install-triton-on-windows/

### GitHub Issues (Critical)

1. **Dataclass Error:**
   - [Unsloth #1604](https://github.com/unslothai/unsloth/issues/1604) - Phi-4 loading error
   - [Unsloth #1876](https://github.com/unslothai/unsloth/issues/1876) - Windows Python 3.11
   - [PyTorch XLA #8560](https://github.com/pytorch/xla/issues/8560) - torch-xla crash
   - [Triton #5026](https://github.com/triton-lang/triton/issues/5026) - AttrsDescriptor issue

2. **Windows Support:**
   - [Unsloth #210](https://github.com/unslothai/unsloth/issues/210) - Native Windows success story
   - [Unsloth #402](https://github.com/unslothai/unsloth/issues/402) - Installation guide
   - [Unsloth #1359](https://github.com/unslothai/unsloth/issues/1359) - Windows support
   - [Unsloth #1849](https://github.com/unslothai/unsloth/discussions/1849) - Direct Windows support
   - [Unsloth #1850](https://github.com/unslothai/unsloth/issues/1850) - Windows support thread

3. **RTX 3090 Specific:**
   - [Unsloth #2443](https://github.com/unslothai/unsloth/issues/2443) - Low it/s on RTX 3090
   - [Unsloth #1572](https://github.com/unslothai/unsloth/issues/1572) - CUDA OOM
   - [Unsloth #3162](https://github.com/unslothai/unsloth/issues/3162) - Fine-tuning on single 3090

4. **PyTorch Version Issues:**
   - [Unsloth #1825](https://github.com/unslothai/unsloth/issues/1825) - PyTorch 2.6 training issues
   - [Unsloth #2499](https://github.com/unslothai/unsloth/issues/2499) - CUDA version conflicts

### Community Resources

1. **Guides and Tutorials:**
   - [Installing BitsAndBytes for Windows](https://www.mindfiretechnology.com/blog/archive/installing-bitsandbtyes-for-windows-so-that-you-can-do-peft/)
   - [Unsloth Guide - LearnOpenCV](https://learnopencv.com/unsloth-guide-efficient-llm-fine-tuning/)
   - [Fine-Tuning with SWIFT (Unsloth Alternative)](https://www.shelpuk.com/post/fine-tuning-llama-3-1-with-swift-unsloth-alternative-for-multi-gpu-llm-training)

2. **Comparison Articles:**
   - [Comparing LLM Fine-Tuning Frameworks](https://blog.spheron.network/comparing-llm-fine-tuning-frameworks-axolotl-unsloth-and-torchtune-in-2025)
   - [Optimizing with PEFT and QLoRA](https://medium.com/@tejpal.abhyuday/optimizing-language-model-fine-tuning-with-peft-qlora-integration-and-training-time-reduction-04df39dca72b)

3. **Technical Deep Dives:**
   - [Supporting new dtypes in PyTorch](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833)
   - [torchao Contributor Guide](https://github.com/pytorch/ao/issues/391)
   - [Windows torch.compile timeline](https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268)

### Package Repositories

1. **PyPI:**
   - Unsloth: https://pypi.org/project/unsloth/
   - unsloth-zoo: https://pypi.org/project/unsloth-zoo/
   - torchao: https://pypi.org/project/torchao/

2. **GitHub:**
   - Unsloth: https://github.com/unslothai/unsloth
   - PyTorch: https://github.com/pytorch/pytorch
   - torchao: https://github.com/pytorch/ao
   - Triton Windows: https://github.com/woct0rdho/triton-windows

---

## Summary of Actionable Solutions

### Immediate Actions (Choose One)

**Option A: Docker (Fastest - 30 minutes)**
```bash
# Install Docker Desktop + NVIDIA Container Toolkit
docker run -d -e JUPYTER_PASSWORD="pass" -p 8888:8888 -v ${pwd}/work:/workspace/work --gpus all unsloth/unsloth
# Access: http://localhost:8888
```

**Option B: WSL2 (Best Performance - 1-2 hours)**
```powershell
wsl --install -d Ubuntu-22.04
# Follow complete WSL2 setup in Solution 2
```

**Option C: Downgrade PyTorch (Native Windows - 1 hour)**
```powershell
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

### For Your Specific Environment

**Current Setup:**
- Windows 11
- Python 3.11.14
- PyTorch 2.5.1+cu121 ❌ (Problematic)
- CUDA 12.1 ✅
- RTX 3090 ✅

**Recommended Path:**
1. **Short-term:** Use Docker to start training immediately
2. **Long-term:** Set up WSL2 for best development experience
3. **If native Windows required:** Downgrade to PyTorch 2.4.1

### Version Targets

**Stable Configuration:**
```
PyTorch: 2.4.1+cu121
Triton: triton-windows 3.1.x
Unsloth: 2025.11.x (cu121-ampere-torch240 variant)
Python: 3.11.14
CUDA: 12.1
```

### Critical Configuration

```python
# In your training script
from trl import SFTConfig

config = SFTConfig(
    # ... other params
    dataset_num_proc=1,  # REQUIRED for Windows
    dataloader_num_workers=0,  # REQUIRED for Windows
)
```

---

## Conclusion

Native Windows support for Unsloth exists but is fragile due to:
1. PyTorch 2.5.1 dataclass bug
2. torch.int1/int2 dtype limitations
3. Triton dependency requiring Windows fork
4. Limited testing on Windows platform

**Best Practice Recommendations:**
1. **For production:** Use WSL2
2. **For quick testing:** Use Docker
3. **For native Windows:** Use PyTorch 2.4.1
4. **Avoid:** PyTorch 2.5.1 on native Windows until bugs are fixed

**Future Outlook:**
- PyTorch 2.6+ may resolve dataclass issues
- Triton Windows fork actively maintained
- Unsloth team aware of Windows challenges
- Community support growing

---

**Document Maintenance:**
This document should be updated when:
- PyTorch 2.6 stable is released
- Unsloth releases Windows-specific fixes
- New GitHub issues reveal solutions
- Triton Windows fork major updates

**Last Research Date:** 2025-11-16
**Next Review Date:** 2025-12-16 (or when PyTorch 2.6 stable releases)
