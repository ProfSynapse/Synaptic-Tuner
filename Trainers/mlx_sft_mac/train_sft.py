#!/usr/bin/env python3
"""
MLX SFT Training Script for Apple Silicon (M1/M2/M3/M4)
Uses mlx_lm's CLI for LoRA training.

Usage:
    python train_sft.py
    python train_sft.py --config custom_config.yaml
    python train_sft.py --iters 100  # Quick test
    python train_sft.py --dry-run    # Setup only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MLX SFT Training for Apple Silicon")

    # Configuration
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config file (default: config/config.yaml)")

    # Overrides
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--iters", type=int, help="Number of training iterations")
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length")

    # Dataset
    parser.add_argument("--dataset", type=str, help="Override dataset path")

    # Utility
    parser.add_argument("--dry-run", action="store_true", help="Setup only, don't train")
    parser.add_argument("--resume", type=str, help="Resume from adapter weights")

    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    import yaml

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = current_dir / config_file

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def prepare_dataset_for_mlx_lm(source_path: str, output_dir: Path, train_split: float = 0.9):
    """
    Convert our JSONL dataset to mlx_lm format.
    mlx_lm expects train.jsonl with 'text' field containing the full formatted conversation.
    """
    from data_loader import load_jsonl, sanitize_message, format_for_qwen

    # Load a tokenizer for formatting
    from mlx_lm import load
    _, tokenizer = load("mlx-community/Qwen3-0.6B-4bit")

    print(f"Preparing dataset for mlx_lm...")
    print(f"  Source: {source_path}")

    # Load data
    raw_data = load_jsonl(source_path)
    print(f"  Loaded {len(raw_data)} examples")

    # Filter for positive labels if present
    if raw_data and "label" in raw_data[0]:
        raw_data = [ex for ex in raw_data if ex.get("label", True) is True]
        print(f"  After filtering: {len(raw_data)} positive examples")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(raw_data)

    # Split - ensure at least 1 validation example if split < 1.0
    if train_split >= 1.0:
        train_data = raw_data
        valid_data = []
    else:
        split_idx = int(len(raw_data) * train_split)
        # Ensure at least 1 validation example
        split_idx = min(split_idx, len(raw_data) - 1)
        train_data = raw_data[:split_idx]
        valid_data = raw_data[split_idx:]

    print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Convert to mlx_lm format (with 'text' field)
    output_dir.mkdir(parents=True, exist_ok=True)

    def convert_example(example):
        """Convert a single example to mlx_lm 'text' format."""
        messages = example.get("messages") or example.get("conversations", [])
        messages = [sanitize_message(m) for m in messages]

        # Use tokenizer to format
        text = format_for_qwen(messages, tokenizer)
        return {"text": text}

    # Write train.jsonl
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for ex in train_data:
            converted = convert_example(ex)
            f.write(json.dumps(converted) + '\n')

    # Write valid.jsonl only if we have validation data
    valid_path = output_dir / "valid.jsonl"
    if valid_data:
        with open(valid_path, 'w', encoding='utf-8') as f:
            for ex in valid_data:
                converted = convert_example(ex)
                f.write(json.dumps(converted) + '\n')

    print(f"  Written to: {output_dir}")
    return train_path, valid_path, len(train_data), len(valid_data)


def print_banner():
    """Print startup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           MLX SFT Training System v1.0.0                     ║
    ║           Using mlx_lm LoRA CLI                              ║
    ║           Optimized for Apple Silicon                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_requirements():
    """Check system requirements."""
    import mlx.core as mx
    import psutil

    print("Checking system requirements...")

    # Check Metal
    try:
        x = mx.array([1.0])
        mx.eval(x)
        print("[OK] Metal GPU available")
    except Exception:
        print("[WARN] Metal GPU not available!")

    # Check RAM
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024**3)
    avail_gb = vm.available / (1024**3)
    print(f"[OK] System RAM: {total_gb:.1f} GB total, {avail_gb:.1f} GB available")

    if total_gb < 16:
        print("[WARN] Less than 16GB RAM - training may be slow")


def main():
    """Main training function."""
    args = parse_args()

    print_banner()
    check_requirements()

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Apply CLI overrides
    if args.batch_size:
        config['training']['per_device_batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.max_seq_length:
        config['training']['max_seq_length'] = args.max_seq_length
        config['model']['max_seq_length'] = args.max_seq_length
        config['data']['max_seq_length'] = args.max_seq_length
    if args.dataset:
        config['data']['dataset_path'] = args.dataset

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(config['output']['base_dir'])
    run_dir = base_output_dir / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining run directory: {run_dir}")

    # Resolve dataset path
    dataset_path = Path(config['data']['dataset_path'])
    if not dataset_path.is_absolute():
        dataset_path = current_dir / dataset_path
    config['data']['dataset_path'] = str(dataset_path)

    print(f"Dataset: {dataset_path}")

    # Prepare dataset for mlx_lm
    data_dir = run_dir / "data"
    train_path, valid_path, train_count, valid_count = prepare_dataset_for_mlx_lm(
        source_path=str(dataset_path),
        output_dir=data_dir,
        train_split=config['data']['train_split']
    )

    # Calculate iterations
    batch_size = config['training']['per_device_batch_size']
    num_epochs = config['training']['num_epochs']
    iters = args.iters if args.iters else (train_count // batch_size) * num_epochs

    # Print training configuration
    print("\n" + "=" * 60)
    print("SFT TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {train_count} train, {valid_count} valid examples")
    print(f"\nBatch configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iters}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Max sequence length: {config['data']['max_seq_length']}")
    print(f"\nLoRA configuration:")
    print(f"  Rank: {config['lora']['rank']}")
    print(f"  Alpha (scale): {config['lora']['alpha']}")
    print("=" * 60 + "\n")

    if args.dry_run:
        print("[OK] Dry run completed. Exiting without training.")
        return

    # Build mlx_lm command
    adapter_path = run_dir / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--train",
        "--model", config['model']['name'],
        "--data", str(data_dir),
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--learning-rate", str(config['training']['learning_rate']),
        "--adapter-path", str(adapter_path),
        "--save-every", str(config['training']['save_steps']),
        "--steps-per-report", str(config['training']['logging_steps']),
        "--steps-per-eval", str(config['training']['eval_steps']),
        "--max-seq-length", str(config['data']['max_seq_length']),
        "--num-layers", "16",  # Apply LoRA to 16 layers
        "--grad-checkpoint",  # Enable gradient checkpointing
    ]

    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"\nCommand: {' '.join(cmd)}\n")

    import time
    start_time = time.time()

    # Run training
    result = subprocess.run(cmd, cwd=str(current_dir))

    training_time = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[ERROR] Training failed with return code {result.returncode}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    # Save training lineage
    lineage = {
        "training_type": "SFT",
        "framework": "MLX",
        "infrastructure": "mlx_lm CLI",
        "timestamp": datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "model": {
            "base_model": config['model']['name'],
            "max_seq_length": config['model']['max_seq_length'],
        },
        "lora": {
            "rank": config['lora']['rank'],
            "alpha": config['lora']['alpha'],
        },
        "training": {
            "batch_size": batch_size,
            "iterations": iters,
            "learning_rate": config['training']['learning_rate'],
        },
        "dataset": {
            "path": str(dataset_path),
            "train_examples": train_count,
            "valid_examples": valid_count,
        },
        "results": {
            "training_time_seconds": round(training_time, 1),
            "training_time_formatted": f"{training_time // 3600:.0f}h {(training_time % 3600) // 60:.0f}m {training_time % 60:.0f}s",
        }
    }

    lineage_path = run_dir / "training_lineage.json"
    with open(lineage_path, 'w') as f:
        json.dump(lineage, f, indent=2)

    print(f"\n[OK] Training complete!")
    print(f"  Adapters saved to: {adapter_path}")
    print(f"  Lineage saved to: {lineage_path}")
    print(f"  Training time: {lineage['results']['training_time_formatted']}")


if __name__ == "__main__":
    main()
