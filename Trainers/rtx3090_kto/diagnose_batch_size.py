#!/usr/bin/env python3
"""Quick diagnostic to show what batch size will be used"""
from configs.config_loader import load_config

config = load_config()

print("=" * 60)
print("BATCH SIZE DIAGNOSTIC")
print("=" * 60)
print(f"per_device_train_batch_size: {config.training.per_device_train_batch_size}")
print(f"gradient_accumulation_steps: {config.training.gradient_accumulation_steps}")
print(f"Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
print("=" * 60)

if config.training.per_device_train_batch_size == 8:
    print("✓ CORRECT: Batch size is 8 (optimized for 24GB)")
    print("  Expected VRAM: 18-23GB")
elif config.training.per_device_train_batch_size == 4:
    print("✗ PROBLEM: Batch size is still 4 (conservative)")
    print("  Expected VRAM: 5-6GB only")
    print("  Solution: Use explicit flag --batch-size 8")
else:
    print(f"? CUSTOM: Batch size is {config.training.per_device_train_batch_size}")
