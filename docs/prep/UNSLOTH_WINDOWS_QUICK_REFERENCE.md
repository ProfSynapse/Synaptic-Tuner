# Unsloth Windows Quick Reference

**Fast lookup guide for common tasks and version requirements**

---

## TL;DR - Fastest Installation

### Option 1: PowerShell Script (Automated)
```powershell
# Download from: https://docs.unsloth.ai/get-started/install-and-update/windows-installation
powershell.exe -ExecutionPolicy Bypass -File .\unsloth_windows.ps1
```

### Option 2: Manual (Native Windows)
```powershell
# Prerequisites: Visual Studio 2022 C++ Build Tools, CUDA 12.4

conda create -n unsloth python=3.10 -y
conda activate unsloth

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -U "triton-windows<3.3"
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes==0.45.5

pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
```

### Option 3: WSL2 (Recommended for Beginners)
```bash
wsl --install -d Ubuntu-24.04  # PowerShell as Admin
# Restart, then in WSL:

conda create -n unsloth python=3.10 -y
conda activate unsloth

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## Recommended Versions (January 2025)

| Component | Version | Download/Command |
|-----------|---------|------------------|
| **Windows** | 10/11 latest | Windows Update |
| **Python** | 3.10 or 3.11 | https://www.python.org/downloads/ |
| **CUDA Toolkit** | 12.4 | https://developer.nvidia.com/cuda-12-4-0-download-archive |
| **NVIDIA Driver** | Latest | https://www.nvidia.com/drivers |
| **Visual Studio** | 2022 Build Tools | https://visualstudio.microsoft.com/downloads/ |
| **PyTorch** | 2.5.1+cu124 | `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124` |
| **Triton** | 3.1.x (Windows) | `pip install -U "triton-windows<3.3"` |
| **xformers** | Latest | `pip install -U xformers --index-url https://download.pytorch.org/whl/cu124` |
| **bitsandbytes** | 0.45.5+ | `pip install bitsandbytes==0.45.5` |
| **Unsloth** | Latest | `pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"` |

---

## GPU Compatibility

| GPU Series | Supported | CUDA Capability | Notes |
|------------|-----------|-----------------|-------|
| RTX 50xx | ✅ | 12.0 (sm120) | Requires Triton ≥3.3, PyTorch ≥2.7 |
| RTX 40xx | ✅ | 8.9 (sm89) | Fully supported |
| RTX 30xx | ✅ | 8.6 (sm86) | Fully supported |
| RTX 20xx | ✅ | 7.5 (sm75) | Triton ≤3.2 only |
| GTX 16xx | ✅ | 7.5 (sm75) | Triton ≤3.2 only |
| GTX 1650+ | ✅ | 7.0 | Minimum supported |
| GTX 1070/1080 | ⚠️ | 6.1 | Slow performance |
| GTX 10xx (older) | ❌ | <7.0 | Not supported |

---

## Common Commands

### Verification
```powershell
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"
```

### Environment Management
```powershell
# Create environment
conda create -n unsloth python=3.10 -y

# Activate
conda activate unsloth

# Deactivate
conda deactivate

# List environments
conda env list

# Delete environment
conda env remove -n unsloth
```

### Update Unsloth
```powershell
pip install --upgrade --force-reinstall "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
```

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `torch.cuda.is_available()` is False | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| "No module named triton" | `pip install -U "triton-windows<3.3"` |
| xformers wheel error | `pip install -U xformers --index-url https://download.pytorch.org/whl/cu124` |
| Training crashes without error | Set `dataset_num_proc=1` in SFTTrainer |
| Out of memory | Reduce batch size, enable 4-bit: `load_in_4bit=True` |
| Visual C++ error | Install VC++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe |

---

## Minimal Working Example

```python
from unsloth import FastLanguageModel
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama",
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
)

# Inference
FastLanguageModel.for_inference(model)
inputs = tokenizer(["Hello, my name is"], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

---

## Windows-Specific Settings

### Always Use These in Windows:
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    # ... other args
    dataset_num_proc = 1,  # CRITICAL: Must be 1 on Windows
)
```

### Optional: Disable xformers if Issues
```python
from unsloth import FastLanguageModel
FastLanguageModel.disable_xFormers = True
```

---

## Installation Time Estimates

| Method | Time | Difficulty | Best For |
|--------|------|------------|----------|
| Docker | 15-20 min | Easy | Quick testing |
| PowerShell Script | 30-40 min | Easy | Automation |
| WSL2 | 30-45 min | Easy | Linux-familiar users |
| Native Windows | 60-90 min | Medium | Production use |

---

## Disk Space Requirements

- CUDA Toolkit: ~5 GB
- Visual Studio Build Tools: ~5-7 GB
- Python + packages: ~3-5 GB
- Models (variable): 2-30 GB each
- **Total**: ~20-25 GB minimum

---

## WSL2 vs Native Windows

| Feature | WSL2 | Native Windows |
|---------|------|----------------|
| vLLM Support | ✅ Yes | ❌ No |
| GRPO Training | ✅ Yes | ❌ No |
| Triton | ✅ Official | ⚠️ Windows fork |
| Performance | 95-100% | 100% |
| Setup Time | Faster | Slower |
| Compatibility | Full | Growing |

**Recommendation**: Use WSL2 unless you specifically need native Windows integration.

---

## Critical Links

- **Official Docs**: https://docs.unsloth.ai
- **Windows Install Guide**: https://docs.unsloth.ai/get-started/install-and-update/windows-installation
- **GitHub Issues**: https://github.com/unslothai/unsloth/issues
- **GitHub Discussions**: https://github.com/unslothai/unsloth/discussions
- **Triton Windows Fork**: https://github.com/woct0rdho/triton-windows
- **CUDA Download**: https://developer.nvidia.com/cuda-downloads
- **PyTorch Index**: https://pytorch.org/get-started/previous-versions/

---

## One-Line Test

```powershell
python -c "from unsloth import FastLanguageModel; import torch; print(f'✅ Ready! CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
✅ Ready! CUDA: True, GPU: NVIDIA GeForce RTX 3090
```

---

**Last Updated**: 2025-11-16
