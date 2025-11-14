#!/usr/bin/env python3
"""
KTO Training Script for RTX 3090 (24GB VRAM)
Based on rtx3090-kto-finetuning.md specification

Usage:
    python train_kto.py --model-size 7b
    python train_kto.py --model-size 13b --dataset-file my_data.jsonl
    python train_kto.py --config custom_config.py
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unsloth import is_bfloat16_supported
from trl import KTOConfig, KTOTrainer

from configs.training_config import (
    Config,
    get_3b_config,
    get_7b_config,
    get_13b_config,
    get_20b_config
)
from src.data_loader import load_and_prepare_dataset, validate_kto_dataset, print_dataset_samples
from src.model_loader import (
    load_model_and_tokenizer,
    apply_lora_adapters,
    check_gpu_memory
)
from src.training_callbacks import MetricsTableCallback, CheckpointMonitorCallback


def setup_environment():
    """Setup environment variables and configurations."""
    # Enable CUDA error debugging (optional, can slow down training)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Disable tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("=" * 60)
    print("RTX 3090 KTO TRAINING")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    print(f"BFloat16 supported: {is_bfloat16_supported()}")
    print("=" * 60 + "\n")


def get_config_by_size(model_size: str) -> Config:
    """Get configuration based on model size."""
    configs = {
        "3b": get_3b_config,
        "7b": get_7b_config,
        "13b": get_13b_config,
        "20b": get_20b_config
    }

    if model_size not in configs:
        raise ValueError(f"Invalid model size: {model_size}. Choose from: {list(configs.keys())}")

    return configs[model_size]()


def main():
    parser = argparse.ArgumentParser(description="KTO Training on RTX 3090")

    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default="7b",
        choices=["3b", "7b", "13b", "20b"],
        help="Model size preset (default: 7b)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name (e.g., unsloth/mistral-7b-v0.3-bnb-4bit)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Override max sequence length"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        help="Dataset file within HuggingFace dataset"
    )
    parser.add_argument(
        "--local-file",
        type=str,
        help="Path to local JSONL file"
    )
    parser.add_argument(
        "--split-dataset",
        action="store_true",
        help="Create train/validation split"
    )

    # Training configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override per_device_train_batch_size"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        help="Override gradient_accumulation_steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max training steps (takes precedence over epochs)"
    )

    # Experiment tracking
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="kto-finetuning",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        help="W&B run name"
    )

    # Other options
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token for gated models"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Setup and validate without training"
    )

    args = parser.parse_args()

    # Setup environment
    setup_environment()

    # Get configuration
    print(f"Using {args.model_size.upper()} model configuration\n")
    config = get_config_by_size(args.model_size)

    # Apply command-line overrides
    if args.model_name:
        config.model.model_name = args.model_name
    if args.max_seq_length:
        config.model.max_seq_length = args.max_seq_length
        config.training.max_length = args.max_seq_length
        config.training.max_prompt_length = args.max_seq_length // 2

    if args.dataset_name:
        config.dataset.dataset_name = args.dataset_name
    if args.dataset_file:
        config.dataset.dataset_file = args.dataset_file

    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs

    if args.wandb:
        config.use_wandb = True
        config.wandb_project = args.wandb_project
        if args.wandb_run_name:
            config.wandb_run_name = args.wandb_run_name

    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=config.dataset.dataset_name if not args.local_file else None,
        data_files=config.dataset.dataset_file if not args.local_file else None,
        local_file=args.local_file,
        num_proc=config.dataset.num_proc,
        test_size=config.dataset.test_size,
        split_dataset=args.split_dataset
    )

    # Validate dataset
    if not validate_kto_dataset(train_dataset):
        print("✗ Dataset validation failed. Exiting.")
        return

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

    # Configure KTO training arguments
    training_args = KTOConfig(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        beta=config.training.beta,
        desirable_weight=config.training.desirable_weight,
        undesirable_weight=config.training.undesirable_weight,
        learning_rate=config.training.learning_rate,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type=config.training.lr_scheduler_type,
        max_length=config.training.max_length,
        max_prompt_length=config.training.max_prompt_length,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        fp16=not is_bfloat16_supported() if config.training.fp16 is False else config.training.fp16,
        bf16=is_bfloat16_supported() if config.training.bf16 is True else config.training.bf16,
        num_train_epochs=config.training.num_train_epochs if not args.max_steps else None,
        max_steps=args.max_steps if args.max_steps else -1,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        group_by_length=config.training.group_by_length,
        eval_strategy=config.training.eval_strategy if eval_dataset else "no",
        eval_steps=config.training.eval_steps if eval_dataset else None,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.wandb_run_name if config.use_wandb else None,
        seed=config.seed,
    )

    # Print training configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
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
    print(f"  Beta: {config.training.beta}")
    print(f"  Warmup ratio: {config.training.warmup_ratio}")
    print(f"  Max length: {config.training.max_length}")
    print(f"\nLoRA configuration:")
    print(f"  Rank: {config.lora.r}")
    print(f"  Alpha: {config.lora.lora_alpha}")
    print(f"  Dropout: {config.lora.lora_dropout}")
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
        print("✓ Dry run completed. Exiting without training.")
        return

    # Initialize callbacks
    callbacks = [
        MetricsTableCallback(log_every_n_steps=5),
        CheckpointMonitorCallback()
    ]

    # Initialize KTO Trainer
    print("Initializing KTO Trainer...")
    trainer = KTOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    print("✓ KTO trainer initialized with metrics tracking\n")

    # Start training
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    try:
        trainer_output = trainer.train()

        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final loss: {trainer_output.training_loss:.4f}")

        # Check final GPU memory
        print()
        check_gpu_memory()

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TRAINING FAILED")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nTroubleshooting:")
        print("  1. Check dataset has mixed True/False labels")
        print("  2. Reduce batch_size if OOM error")
        print("  3. Reduce max_length if OOM error")
        print("  4. Check GPU has sufficient memory")
        print("  5. Review logs above for specific error")
        raise

    # Save final model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    output_path = Path(config.training.output_dir) / "final_model"
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"✓ Model saved to: {output_path}")
    print("\nTo upload to HuggingFace:")
    print(f"  model.push_to_hub_merged('username/model-name', tokenizer, save_method='merged_16bit')")
    print("\n" + "=" * 60)
    print("✓ ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
