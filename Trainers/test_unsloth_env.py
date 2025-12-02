#!/usr/bin/env python3
"""Test script for unsloth environment - checks core and vision model support."""

def main():
    print("=" * 50)
    print("UNSLOTH ENVIRONMENT TEST")
    print("=" * 50)

    # Core dependencies
    print("\nCore Dependencies:")
    print("-" * 30)

    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA: Not available")
        print("  GPU: None")

    import unsloth
    print(f"  Unsloth: {getattr(unsloth, '__version__', 'unknown')}")

    import trl
    print(f"  TRL: {trl.__version__}")

    import transformers
    print(f"  Transformers: {transformers.__version__}")

    import peft
    print(f"  PEFT: {peft.__version__}")

    # Vision model support
    print("\nVision Model Support:")
    print("-" * 30)

    try:
        from unsloth import FastVisionModel
        print("  FastVisionModel: AVAILABLE")
        print("  Supported models: Qwen-VL, LLaVA, Pixtral, PaliGemma")
    except ImportError as e:
        print("  FastVisionModel: NOT AVAILABLE")
        print(f"  Error: {e}")
        print("  To enable: pip install --upgrade unsloth unsloth_zoo")

    # Language model support (should always work)
    print("\nLanguage Model Support:")
    print("-" * 30)

    try:
        from unsloth import FastLanguageModel
        print("  FastLanguageModel: AVAILABLE")
        print("  Supported models: Llama, Mistral, Qwen, Gemma, Phi, etc.")
    except ImportError as e:
        print("  FastLanguageModel: NOT AVAILABLE")
        print(f"  Error: {e}")

    print("\n" + "=" * 50)
    print("Environment test complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
