#!/usr/bin/env python3
"""
Test installation and verify all dependencies are working correctly.
Run after setup to ensure everything is configured properly.
"""

import sys
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def test_python_version():
    """Test Python version."""
    print("\n[1/8] Testing Python version...")
    version = sys.version_info

    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("✗ FAIL: Python 3.10+ required")
        return False

    print("✓ PASS")
    return True


def test_pytorch():
    """Test PyTorch installation."""
    print("\n[2/8] Testing PyTorch...")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        if not torch.cuda.is_available():
            print("⚠ WARNING: CUDA not available")
            print("Make sure NVIDIA drivers are installed")
            return False

        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        print("✓ PASS")
        return True

    except ImportError as e:
        print(f"✗ FAIL: {e}")
        return False


def test_gpu():
    """Test GPU availability."""
    print("\n[3/8] Testing GPU...")

    try:
        import torch

        if not torch.cuda.is_available():
            print("✗ FAIL: No CUDA GPU available")
            return False

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_memory:.1f} GB")

        if gpu_memory < 20:
            print("⚠ WARNING: GPU has less than 20GB VRAM")
            print("Recommended: RTX 3090 (24GB) or equivalent")

        print("✓ PASS")
        return True

    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_transformers():
    """Test Transformers library."""
    print("\n[4/8] Testing Transformers...")

    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print("✓ PASS")
        return True
    except ImportError as e:
        print(f"✗ FAIL: {e}")
        return False


def test_unsloth():
    """Test Unsloth library."""
    print("\n[5/8] Testing Unsloth...")

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        print("Unsloth installed: ✓")
        print(f"BFloat16 supported: {is_bfloat16_supported()}")
        print("✓ PASS")
        return True
    except ImportError as e:
        print(f"✗ FAIL: {e}")
        print("Install with: pip install unsloth")
        return False


def test_trl():
    """Test TRL library."""
    print("\n[6/8] Testing TRL (Transformer Reinforcement Learning)...")

    try:
        from trl import KTOConfig, KTOTrainer
        import trl
        print(f"TRL version: {trl.__version__}")
        print("KTOTrainer available: ✓")
        print("✓ PASS")
        return True
    except ImportError as e:
        print(f"✗ FAIL: {e}")
        print("Install with: pip install trl")
        return False


def test_datasets():
    """Test Datasets library."""
    print("\n[7/8] Testing Datasets...")

    try:
        from datasets import load_dataset
        import datasets
        print(f"Datasets version: {datasets.__version__}")
        print("✓ PASS")
        return True
    except ImportError as e:
        print(f"✗ FAIL: {e}")
        return False


def test_project_structure():
    """Test project structure."""
    print("\n[8/8] Testing project structure...")

    required_files = [
        "train_kto.py",
        "requirements.txt",
        "configs/training_config.py",
        "src/data_loader.py",
        "src/model_loader.py",
        "src/inference.py",
        "src/upload_to_hf.py",
    ]

    all_exist = True
    for file in required_files:
        file_path = Path(__file__).parent / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_exist = False

    if all_exist:
        print("✓ PASS")
    else:
        print("✗ FAIL: Some files are missing")

    return all_exist


def main():
    """Run all tests."""
    print_header("RTX 3090 KTO Training - Installation Test")

    tests = [
        test_python_version,
        test_pytorch,
        test_gpu,
        test_transformers,
        test_unsloth,
        test_trl,
        test_datasets,
        test_project_structure,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)

    # Summary
    print_header("TEST SUMMARY")
    passed = sum(results)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Test with dry run: python train_kto.py --model-size 3b --dry-run")
        print("  2. Start training: python train_kto.py --model-size 7b")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check NVIDIA drivers: nvidia-smi")
        print("  3. Verify CUDA installation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
