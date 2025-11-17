#!/usr/bin/env python3
"""
KTO Training for Windows
Uses existing code/rtx3090_kto modules with Windows compatibility patches

Usage:
    python train_windows.py
"""

import os
import sys
from pathlib import Path

# CRITICAL: Apply Windows patches FIRST
print("Applying Windows compatibility patches...")
from unsloth_windows_patch import apply_patches
apply_patches()

import torch
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "code" / "rtx3090_kto"))

# Import existing modules (DRY!)
from configs.training_config import Config
from src.data_loader import load_and_prepare_dataset, validate_kto_dataset
from src.model_loader import load_model_and_tokenizer, apply_lora_adapters, check_gpu_memory
from src.training_callbacks import MetricsTableCallback
from trl import KTOConfig, KTOTrainer
from unsloth import is_bfloat16_supported
from datetime import datetime

# Load environment
load_dotenv()


def setup_environment():
    """Setup Windows-specific environment"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("=" * 60)
    print("WINDOWS KTO TRAINING")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("=" * 60)


def main():
    setup_environment()

    # Load config from existing training_config.py
    config = Config()
    model_config = config.model
    lora_config = config.lora
    training_config = config.training
    dataset_config = config.dataset

    # WINDOWS-SPECIFIC OVERRIDES (CRITICAL)
    dataset_config.num_proc = 1  # Must be 1 on Windows
    training_config.dataloader_num_workers = 0  # Must be 0 on Windows

    # Create timestamped run folder (same as WSL)
    base_output = Path(__file__).parent / "code" / "rtx3090_kto" / "kto_output_rtx3090"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = base_output / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)

    training_config.output_dir = str(run_output_dir)

    print(f"\nConfiguration:")
    print(f"  Model: {model_config.model_name}")
    print(f"  Dataset: syngen_tools_11.14.25.jsonl")
    print(f"  Output: {training_config.output_dir}")

    # Load model using existing model_loader
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_config.model_name,
        max_seq_length=model_config.max_seq_length,
        dtype=model_config.dtype,
        load_in_4bit=model_config.load_in_4bit,
    )

    model = apply_lora_adapters(
        model,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=lora_config.target_modules,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
        random_state=lora_config.random_state,
    )

    check_gpu_memory()

    # Load dataset using existing data_loader
    train_dataset, eval_dataset = load_and_prepare_dataset(
        local_file="syngen_tools_11.14.25.jsonl",
        num_proc=dataset_config.num_proc,
        split_dataset=False
    )

    # Validate dataset
    validate_kto_dataset(train_dataset)

    # KTO Training Arguments
    kto_config = KTOConfig(
        output_dir=training_config.output_dir,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_train_epochs=training_config.num_train_epochs,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        optim=training_config.optim,
        max_grad_norm=training_config.max_grad_norm,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        lr_scheduler_type=training_config.lr_scheduler_type,

        # Windows-specific (CRITICAL)
        dataloader_num_workers=training_config.dataloader_num_workers,

        # KTO-specific parameters
        beta=training_config.beta,
        desirable_weight=training_config.desirable_weight,
        undesirable_weight=training_config.undesirable_weight,
        max_length=training_config.max_length,
        max_prompt_length=training_config.max_prompt_length,

        # Reporting
        report_to=["tensorboard"],
        logging_dir=f"{training_config.output_dir}/logs",
    )

    # Setup callbacks for logging
    metrics_callback = MetricsTableCallback(
        log_every_n_steps=training_config.logging_steps,
        output_dir=training_config.output_dir
    )

    # KTO Trainer
    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[metrics_callback],
    )

    # Clean up GPU memory before training
    if torch.cuda.is_available():
        print(f"\nCleaning GPU memory cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"\nGPU Memory before training:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f}GB")

    # Train!
    print("\n" + "=" * 60)
    print("STARTING KTO TRAINING")
    print("=" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)

    # Save
    print(f"\nSaving model to {training_config.output_dir}...")
    model.save_pretrained(training_config.output_dir)
    tokenizer.save_pretrained(training_config.output_dir)
    print("✓ Model saved")

    # Show final memory
    if torch.cuda.is_available():
        print(f"\nGPU Memory after training:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

    print(f"\n✅ Done! Model saved to: {training_config.output_dir}")


if __name__ == "__main__":
    main()
