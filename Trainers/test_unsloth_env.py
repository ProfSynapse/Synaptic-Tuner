#!/usr/bin/env python3
"""Test script for unsloth_latest environment."""

def main():
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: Not available")
        print("GPU: None")

    import unsloth
    print(f"Unsloth: {unsloth.__version__}")

    import trl
    print(f"TRL: {trl.__version__}")

    import transformers
    print(f"Transformers: {transformers.__version__}")

    import peft
    print(f"PEFT: {peft.__version__}")

    print("\nEnvironment is working correctly!")

if __name__ == "__main__":
    main()
