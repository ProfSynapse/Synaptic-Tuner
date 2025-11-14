#!/usr/bin/env python3
"""
Quick config verification script to check batch size settings.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from configs.training_config import get_7b_config

print("=" * 60)
print("CONFIGURATION VERIFICATION")
print("=" * 60)

config = get_7b_config()

print(f"\nModel: {config.model.model_name}")
print(f"Max sequence length: {config.model.max_seq_length}")

print(f"\nBatch Configuration:")
print(f"  per_device_train_batch_size: {config.training.per_device_train_batch_size}")
print(f"  gradient_accumulation_steps: {config.training.gradient_accumulation_steps}")
print(f"  Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")

print(f"\nLoRA Configuration:")
print(f"  Rank: {config.lora.r}")
print(f"  Alpha: {config.lora.lora_alpha}")

print(f"\nMemory Settings:")
print(f"  gradient_checkpointing: {config.training.gradient_checkpointing}")
print(f"  optim: {config.training.optim}")

print(f"\nCheckpointing:")
print(f"  logging_steps: {config.training.logging_steps}")
print(f"  save_steps: {config.training.save_steps}")
print(f"  save_total_limit: {config.training.save_total_limit}")

print("\n" + "=" * 60)

# Check if batch size is optimized
if config.training.per_device_train_batch_size == 8:
    print("✓ VRAM OPTIMIZED: batch_size=8 (should use ~20GB)")
elif config.training.per_device_train_batch_size == 4:
    print("⚠ CONSERVATIVE: batch_size=4 (only ~5-6GB VRAM)")
    print("  To optimize: batch_size should be 8 for 7B models")
else:
    print(f"? CUSTOM: batch_size={config.training.per_device_train_batch_size}")

print("=" * 60)
