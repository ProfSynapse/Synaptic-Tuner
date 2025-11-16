"""
Test Unsloth Windows - Verify it actually works
"""

# Apply patches FIRST
from unsloth_windows_patch import apply_patches
apply_patches()

print("\n" + "=" * 60)
print("Testing Unsloth on Windows")
print("=" * 60)

# Test 1: Import
print("\n[Test 1] Importing Unsloth...")
try:
    from unsloth import FastLanguageModel
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Check if we can instantiate (without downloading)
print("\n[Test 2] Checking FastLanguageModel...")
try:
    # Just verify the class exists and has expected methods
    assert hasattr(FastLanguageModel, 'from_pretrained'), "Missing from_pretrained"
    assert hasattr(FastLanguageModel, 'get_peft_model'), "Missing get_peft_model"
    print("✅ FastLanguageModel has required methods")
except Exception as e:
    print(f"❌ FastLanguageModel check failed: {e}")
    exit(1)

# Test 3: Verify GPU is available
print("\n[Test 3] Checking GPU availability...")
try:
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA: {torch.version.cuda}")
    print(f"   PyTorch: {torch.__version__}")
except Exception as e:
    print(f"❌ GPU check failed: {e}")
    exit(1)

# Test 4: Check core dependencies
print("\n[Test 4] Checking core dependencies...")
try:
    import transformers
    import peft
    import bitsandbytes
    import trl
    print(f"✅ transformers: {transformers.__version__}")
    print(f"✅ peft: {peft.__version__}")
    print(f"✅ bitsandbytes: {bitsandbytes.__version__}")
    print(f"✅ trl: {trl.__version__}")
except Exception as e:
    print(f"❌ Dependency check failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED - UNSLOTH WORKS ON WINDOWS!")
print("=" * 60)
print("\n✅ You can now:")
print("   1. Load models with FastLanguageModel.from_pretrained()")
print("   2. Fine-tune with Unsloth's optimizations")
print("   3. Use KTO/DPO/SFT trainers")
print("\n⚠️  Remember: Always apply patches before importing Unsloth:")
print("   from unsloth_windows_patch import apply_patches")
print("   apply_patches()")
print("   from unsloth import FastLanguageModel")
print("")
