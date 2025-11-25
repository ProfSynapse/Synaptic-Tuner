#!/usr/bin/env python3
"""
SFT Training Script for RTX 3090 (24GB VRAM)
Supervised Fine-Tuning for tool-calling instruction learning

Usage:
    python train_sft.py --model-size 7b
    python train_sft.py --model-size 13b --dataset-file my_data.jsonl
    python train_sft.py --config custom_config.py
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Load .env file for API keys (HF_TOKEN, WANDB_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

# ============================================================================
# WINDOWS COMPATIBILITY PATCHES - Apply BEFORE importing unsloth
# ============================================================================
if sys.platform == 'win32':
    print("Applying Windows compatibility patches for Unsloth...")
    from dataclasses import dataclass, fields
    import dataclasses

    # Patch 1: Wrap fields() for non-dataclasses
    original_fields = fields
    def patched_fields(class_or_instance):
        try:
            return original_fields(class_or_instance)
        except TypeError:
            return ()
    dataclasses.fields = patched_fields

    # Patch 2: Disable torch.compile
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    # Patch 3: Pre-patch torch._inductor
    try:
        import torch._inductor.runtime.hints
        if not hasattr(torch._inductor.runtime.hints, 'attr_desc_fields'):
            torch._inductor.runtime.hints.attr_desc_fields = set()
    except:
        pass

    print("[OK] Windows patches applied")
# ============================================================================

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unsloth import is_bfloat16_supported
from trl import SFTConfig, SFTTrainer

from configs.config_loader import (
    get_3b_config,
    get_7b_config,
    get_13b_config,
    get_20b_config,
    load_config,
)
from src.data_loader import load_and_prepare_dataset, print_dataset_samples
from src.model_loader import (
    load_model_and_tokenizer,
    apply_lora_adapters,
    check_gpu_memory
)
from src.training_callbacks import MetricsTableCallback, CheckpointMonitorCallback


def setup_wandb():
    """Auto-setup W&B if API key is in environment."""
    wandb_key = os.environ.get("WANDB_API_KEY")

    if not wandb_key:
        return False  # No key, W&B disabled

    try:
        import wandb

        # Login with API key from .env
        wandb.login(key=wandb_key, relogin=True, force=True)

        print("[OK] W&B: Logged in automatically (using WANDB_API_KEY from .env)")
        return True
    except Exception as e:
        print(f"[WARN] W&B: Login failed ({e})")
        return False


def extract_previous_log_entries(checkpoint_path: str):
    """Extract previous training log entries when resuming from checkpoint."""
    import json

    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.is_dir():
        checkpoint_dir = checkpoint_dir.parent

    # Find training.jsonl in checkpoint dir or its parent
    log_paths = list(checkpoint_dir.glob("**/training_*.jsonl"))
    if not log_paths:
        return None

    log_file = log_paths[0]
    print(f"Loading previous log entries from: {log_file}")

    entries = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except:
                continue

    return entries if entries else None


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SFT Training for RTX 3090")

    # Model configuration
    parser.add_argument("--model-size", type=str, choices=["3b", "7b", "13b", "20b"],
                       help="Model size preset (3b, 7b, 13b, or 20b)")
    parser.add_argument("--config", type=str,
                       help="Path to custom config file")

    # Training parameters
    parser.add_argument("--batch-size", type=int,
                       help="Override per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int,
                       help="Override gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float,
                       help="Override learning rate")
    parser.add_argument("--num-epochs", type=int,
                       help="Override number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum training steps (overrides epochs)")
    parser.add_argument("--max-seq-length", type=int,
                       help="Override maximum sequence length")

    # Dataset parameters
    parser.add_argument("--dataset-name", type=str,
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset-file", type=str,
                       help="Dataset file within HuggingFace dataset")
    parser.add_argument("--local-file", type=str,
                       help="Path to local dataset file (overrides HF dataset)")
    parser.add_argument("--split-dataset", action="store_true",
                       help="Create train/validation split")

    # W&B tracking
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str,
                       help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str,
                       help="W&B run name")

    # Utility
    parser.add_argument("--dry-run", action="store_true",
                       help="Setup only, don't train")
    parser.add_argument("--resume-from-checkpoint", type=str,
                       help="Resume from checkpoint path")
    parser.add_argument("--hf-token", type=str,
                       help="HuggingFace API token (or set HF_TOKEN env var)")

    return parser.parse_args(argv)


def run(args: argparse.Namespace):
    """Execute training with the provided CLI arguments."""
    run_metadata = {
        "train_size": None,
        "eval_size": None,
        "run_dir": None,
        "final_model_dir": None,
        "logs_dir": None,
        "config_path": args.config,
    }

    # Load configuration
    if args.config:
        # Custom config file
        print(f"Loading custom config from: {args.config}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        config = custom_config.Config()
    elif args.model_size:
        # Use preset configuration
        if args.model_size == "3b":
            config = get_3b_config()
        elif args.model_size == "7b":
            config = get_7b_config()
        elif args.model_size == "13b":
            config = get_13b_config()
        elif args.model_size == "20b":
            config = get_20b_config()
        print(f"Using {args.model_size.upper()} preset configuration")
    else:
        # Default configuration (YAML)
        config = load_config()
        print("Using default YAML configuration")

    # Apply CLI overrides
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs
    if args.max_seq_length:
        config.training.max_seq_length = args.max_seq_length
        config.model.max_seq_length = args.max_seq_length

    # Dataset overrides
    if args.dataset_name:
        config.dataset.dataset_name = args.dataset_name
    if args.dataset_file:
        config.dataset.dataset_file = args.dataset_file
    if args.local_file:
        config.dataset.local_file = args.local_file

    # W&B setup
    if args.wandb:
        config.use_wandb = setup_wandb()
        if config.use_wandb and args.wandb_project:
            config.wandb_project = args.wandb_project
        if config.use_wandb and args.wandb_run_name:
            config.wandb_run_name = args.wandb_run_name

    # HuggingFace token
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")

    # Create timestamped run directory (following KTO pattern)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(config.training.output_dir)
    run_dir = base_output_dir / timestamp

    # Create subdirectories for this run
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Update config to use checkpoints directory (trainer saves here)
    config.training.output_dir = str(checkpoints_dir)

    print(f"Training run directory: {run_dir}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Logs: {logs_dir}\n")
    run_metadata.update(
        {
            "run_dir": str(run_dir),
            "final_model_dir": str(run_dir / "final_model"),
            "logs_dir": str(logs_dir),
        }
    )

    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=config.dataset.dataset_name if not config.dataset.local_file else None,
        data_files=config.dataset.dataset_file if not config.dataset.local_file else None,
        local_file=config.dataset.local_file,
        num_proc=config.dataset.num_proc,
        test_size=config.dataset.test_size,
        split_dataset=args.split_dataset or config.dataset.split_dataset,
        filter_desirable=config.dataset.filter_desirable
    )
    run_metadata["train_size"] = len(train_dataset)
    run_metadata["eval_size"] = len(eval_dataset) if eval_dataset else None

    # Print samples
    print_dataset_samples(train_dataset, num_samples=2)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        dtype=config.model.dtype,
        load_in_4bit=config.model.load_in_4bit,
        hf_token=args.hf_token
    )

    # Apply LoRA adapters
    model = apply_lora_adapters(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
        random_state=config.lora.random_state
    )

    # Check initial GPU memory
    check_gpu_memory()

    def format_chat_template(batch):
        """Apply the model chat template to each example for TRL's SFTTrainer."""
        messages_key = "messages" if "messages" in batch else "conversations"
        conversations = batch.get(messages_key)
        if conversations is None:
            raise ValueError("Dataset must contain 'messages' or 'conversations' columns")

        # TRL calls formatting_func with batched examples; handle both batched and single-example shapes
        if isinstance(conversations, dict):
            conversations = [conversations]
        elif len(conversations) > 0 and isinstance(conversations[0], dict):
            conversations = [conversations]

        formatted = []
        for msgs in conversations:
            if not isinstance(msgs, list):
                raise ValueError(f"Expected list of messages, got {type(msgs)}")
            formatted.append(
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return formatted

    # Configure SFT training arguments
    training_args = SFTConfig(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type=config.training.lr_scheduler_type,
        max_seq_length=config.training.max_seq_length,
        packing=config.training.packing,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        fp16=not is_bfloat16_supported() if config.training.fp16 is False else config.training.fp16,
        bf16=is_bfloat16_supported() if config.training.bf16 is True else config.training.bf16,
        num_train_epochs=1 if args.max_steps else config.training.num_train_epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        eval_strategy=config.training.eval_strategy if eval_dataset else "no",
        eval_steps=config.training.eval_steps if eval_dataset else None,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.wandb_run_name if config.use_wandb else None,
        seed=config.seed,
        dataset_kwargs={"add_special_tokens": False},
    )

    # Print training configuration
    print("\n" + "=" * 60)
    print("SFT TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.model.model_name}")
    print(f"Output directory: {config.training.output_dir}")
    print(f"Dataset: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Validation: {len(eval_dataset)} examples")
    print(f"\nBatch configuration:")
    print(f"  Batch size: {config.training.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    effective_batch = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    print(f"  Effective batch size: {effective_batch}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Warmup ratio: {config.training.warmup_ratio}")
    print(f"  Max sequence length: {config.training.max_seq_length}")
    print(f"  Number of epochs: {config.training.num_train_epochs}")
    print(f"\nLoRA configuration:")
    print(f"  Rank: {config.lora.r}")
    print(f"  Alpha: {config.lora.lora_alpha}")
    print(f"  Dropout: {config.lora.lora_dropout}")
    print(f"\nSFT-specific:")
    print(f"  Packing: {config.training.packing}")
    print(f"  Completion-only loss: {config.training.completion_only_loss}")
    print(f"\nOptimizations:")
    print(f"  Optimizer: {config.training.optim}")
    print(f"  FP16: {training_args.fp16}")
    print(f"  BF16: {training_args.bf16}")
    print(f"  Gradient checkpointing: {config.training.gradient_checkpointing}")
    print(f"\nCheckpointing & Logging:")
    print(f"  Log metrics every: {config.training.logging_steps} steps")
    print(f"  Save checkpoint every: {config.training.save_steps} steps")
    print(f"  Keep last: {config.training.save_total_limit} checkpoints")
    print("=" * 60 + "\n")

    if args.dry_run:
        print("[OK] Dry run completed. Exiting without training.")
        return run_metadata

    # Extract previous log entries if resuming from checkpoint
    previous_log_entries = None
    if args.resume_from_checkpoint:
        previous_log_entries = extract_previous_log_entries(args.resume_from_checkpoint)

    # Initialize callbacks
    callbacks = [
        MetricsTableCallback(
            log_every_n_steps=config.training.logging_steps,
            output_dir=str(run_dir),  # Pass run_dir, callback adds /logs
            previous_log_entries=previous_log_entries
        ),
        CheckpointMonitorCallback()
    ]

    # Initialize SFT Trainer (NO ref_model needed!)
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,  # Use 'tokenizer' or 'processing_class'
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        formatting_func=format_chat_template,
    )

    print("[OK] SFT trainer initialized with metrics tracking")
    print()

    # Check memory before training
    check_gpu_memory()

    # Train the model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    # Save final model
    final_model_path = run_dir / "final_model"
    print(f"\nSaving final model to: {final_model_path}")
    trainer.save_model(str(final_model_path))

    print(f"\n[OK] Training complete!")
    print(f"  Model saved to: {final_model_path}")
    print(f"  Logs saved to: {logs_dir}/")

    # Final GPU memory check
    print()
    check_gpu_memory()
    return run_metadata


def main(argv=None):
    """Main training function."""
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
