#!/usr/bin/env python3
"""
Simple script to download the Mistral 7B 4-bit quantized model from mlx-community.
This runs independently of the training pipeline.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

from mlx_lm import load

print("=" * 80)
print("Downloading MLX Mistral 7B 4-bit Quantized Model")
print("=" * 80)
print()

model_name = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

print(f"Model: {model_name}")
print(f"Cache dir: ~/.cache/huggingface/hub/")
print()
print("Starting download...")
print()

try:
    model, tokenizer = load(model_name)
    print()
    print("=" * 80)
    print("✅ Model downloaded and loaded successfully!")
    print("=" * 80)
    print(f"Model type: {type(model)}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print()
    print("You can now run: python main.py --config config_test.yaml")
    print()
except Exception as e:
    print()
    print("=" * 80)
    print(f"❌ Error downloading model: {e}")
    print("=" * 80)
    import traceback
    traceback.print_exc()
    sys.exit(1)
