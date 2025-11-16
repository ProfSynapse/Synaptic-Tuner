#!/usr/bin/env python3
"""
GPU Setup Verification Script
Diagnoses GPU configuration and ensures RTX 3090 is being used correctly.
"""

import os
import torch

def check_gpu_configuration():
    """Check GPU configuration and identify issues."""
    
    print("=" * 70)
    print("GPU CONFIGURATION CHECK")
    print("=" * 70)
    
    # 1. Check environment variables
    print("\n1. ENVIRONMENT VARIABLES:")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    if cuda_visible == "Not set":
        print("   ⚠ WARNING: CUDA_VISIBLE_DEVICES not set!")
        print("   → This may cause PyTorch to use all GPUs or wrong GPU")
        print("   → Recommendation: Set CUDA_VISIBLE_DEVICES=0 before training")
    
    # 2. Check CUDA availability
    print("\n2. CUDA AVAILABILITY:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("   ✗ ERROR: CUDA not available!")
        print("   → Check CUDA installation")
        print("   → Run: nvidia-smi")
        return
    
    # 3. Check visible GPUs
    print(f"\n3. VISIBLE GPUs:")
    num_gpus = torch.cuda.device_count()
    print(f"   Number of GPUs visible to PyTorch: {num_gpus}")
    
    if num_gpus == 0:
        print("   ✗ ERROR: No GPUs visible to PyTorch!")
        return
    
    if num_gpus > 1:
        print(f"   ⚠ WARNING: Multiple GPUs visible ({num_gpus})")
        print("   → This can cause memory to split across GPUs")
        print("   → Set CUDA_VISIBLE_DEVICES=0 to use only RTX 3090")
    
    # 4. List all visible GPUs
    print("\n4. GPU DETAILS:")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        memory_gb = props.total_memory / 1e9
        
        is_rtx3090 = "3090" in name
        marker = "✓ RTX 3090 (CORRECT)" if is_rtx3090 else "⚠ Not RTX 3090"
        
        print(f"\n   GPU {i}: {name} {marker}")
        print(f"   └─ Total Memory: {memory_gb:.1f} GB")
        print(f"   └─ Compute Capability: {props.major}.{props.minor}")
        
        if is_rtx3090 and i != 0:
            print(f"   ⚠ WARNING: RTX 3090 is GPU {i}, not GPU 0!")
            print(f"   → Set CUDA_VISIBLE_DEVICES={i} to make it GPU 0")
    
    # 5. Check current device
    print("\n5. CURRENT DEFAULT DEVICE:")
    current_device = torch.cuda.current_device()
    current_name = torch.cuda.get_device_name(current_device)
    print(f"   Default device: GPU {current_device}")
    print(f"   Device name: {current_name}")
    
    if "3090" in current_name:
        print("   ✓ Currently using RTX 3090")
    else:
        print("   ✗ NOT using RTX 3090!")
        print("   → This will use shared/integrated GPU instead")
        print("   → Performance will be severely degraded")
    
    # 6. Memory test
    print("\n6. MEMORY TEST:")
    try:
        # Try to allocate 1GB on GPU 0
        torch.cuda.set_device(0)
        test_tensor = torch.randn(1024, 1024, 256, device='cuda:0')
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"   ✓ Successfully allocated test memory on GPU 0")
        print(f"   └─ Allocated: {allocated:.2f} GB")
        print(f"   └─ Reserved: {reserved:.2f} GB")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ Failed to allocate memory: {e}")
    
    # 7. Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    
    rtx3090_idx = None
    for i in range(num_gpus):
        if "3090" in torch.cuda.get_device_name(i):
            rtx3090_idx = i
            break
    
    if rtx3090_idx is None:
        print("\n✗ NO RTX 3090 FOUND!")
        print("   Check your system has RTX 3090 installed and drivers are correct")
    elif rtx3090_idx == 0 and num_gpus == 1:
        print("\n✓ CONFIGURATION IS CORRECT!")
        print("   RTX 3090 is GPU 0 and is the only visible GPU")
        print("   You can proceed with training")
    elif rtx3090_idx == 0 and num_gpus > 1:
        print(f"\n⚠ RTX 3090 is GPU 0, but {num_gpus} GPUs are visible")
        print("   Before training, run:")
        print("   → export CUDA_VISIBLE_DEVICES=0  (Linux/Mac)")
        print("   → $env:CUDA_VISIBLE_DEVICES=\"0\"  (PowerShell)")
        print("   → set CUDA_VISIBLE_DEVICES=0  (CMD)")
    else:
        print(f"\n⚠ RTX 3090 is GPU {rtx3090_idx}, not GPU 0")
        print("   Before training, run:")
        print(f"   → export CUDA_VISIBLE_DEVICES={rtx3090_idx}  (Linux/Mac)")
        print(f"   → $env:CUDA_VISIBLE_DEVICES=\"{rtx3090_idx}\"  (PowerShell)")
        print(f"   → set CUDA_VISIBLE_DEVICES={rtx3090_idx}  (CMD)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_gpu_configuration()
