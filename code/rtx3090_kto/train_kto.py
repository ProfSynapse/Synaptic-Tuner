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

# Load .env file for API keys (HF_TOKEN, WANDB_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unsloth import is_bfloat16_supported
from trl import KTOConfig, KTOTrainer

# Import custom KTO-S trainer (with SIGN correction)
from src.kto_s_trainer import KTOSTrainer

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
    create_reference_model,
    check_gpu_memory
)
from src.training_callbacks import MetricsTableCallback, CheckpointMonitorCallback, TwoStageLRCallback
from src.adaptive_memory import AdaptiveMemoryManager, get_adaptive_settings
from src.debug_logger import TrainingDebugger


def setup_wandb():
    """Auto-setup W&B if API key is in environment."""
    wandb_key = os.environ.get("WANDB_API_KEY")

    if not wandb_key:
        return False  # No key, W&B disabled

    try:
        import wandb

        # Login with API key from .env
        wandb.login(key=wandb_key, relogin=True, force=True)

        print("✓ W&B: Logged in automatically (using WANDB_API_KEY from .env)")
        return True

    except ImportError:
        print("⚠ W&B: API key found but wandb not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        print(f"⚠ W&B: Login failed - {e}")
        return False


def setup_environment():
    """Setup environment variables and configurations."""
    # Enable CUDA error debugging (optional, can slow down training)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Disable tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ============================================================================
    # CUDA MEMORY OPTIMIZATION: Reduce fragmentation
    # ============================================================================
    # expandable_segments reduces memory fragmentation by consolidating allocations
    # This can reduce memory usage by ~30% and prevent OOM errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Suppress verbose logging - we have our custom table
    import logging
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

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


def extract_previous_log_entries(checkpoint_path: str) -> list:
    """Extract log entries from a previous run when resuming from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory (e.g., "kto_output_rtx3090/20251114_135227/checkpoints/checkpoint-50")

    Returns:
        List of log entry dicts from the original run, up to the checkpoint step
    """
    import json
    import re
    from glob import glob

    # Parse checkpoint path to extract timestamp directory and step number
    checkpoint_path = Path(checkpoint_path)

    # Extract step number from checkpoint name (e.g., "checkpoint-50" -> 50)
    checkpoint_name = checkpoint_path.name
    step_match = re.search(r'checkpoint-(\d+)', checkpoint_name)
    if not step_match:
        print(f"⚠ Warning: Could not extract step number from checkpoint path: {checkpoint_path}")
        return []

    resume_step = int(step_match.group(1))

    # Navigate up to find the run directory (timestamp directory)
    # checkpoint_path is like: kto_output_rtx3090/20251114_135227/checkpoints/checkpoint-50
    # We want: kto_output_rtx3090/20251114_135227
    run_dir = checkpoint_path.parent.parent

    # Find log files in the logs subdirectory
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        print(f"⚠ Warning: Logs directory not found: {logs_dir}")
        return []

    # Find training log files (there may be multiple if resuming multiple times)
    log_files = list(logs_dir.glob("training_*.jsonl"))
    if not log_files:
        print(f"⚠ Warning: No log files found in: {logs_dir}")
        return []

    # Use the most recent log file (sorted by name, which includes timestamp)
    log_file = sorted(log_files)[-1]

    print(f"\n✓ Found previous run log: {log_file}")
    print(f"  Extracting entries from steps 0 to {resume_step}")

    # Read log entries up to the resume step
    previous_entries = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                step = entry.get('step', 0)

                # Include entries up to and including the resume step
                if step <= resume_step:
                    previous_entries.append(entry)
                else:
                    # We've passed the resume step, no need to read further
                    break

        print(f"  Extracted {len(previous_entries)} log entries\n")
        return previous_entries

    except Exception as e:
        print(f"⚠ Warning: Failed to read log file: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="KTO Training on RTX 3090")

    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=["3b", "7b", "13b", "20b", None],
        help="Model size preset (optional - if not provided, uses config defaults)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name (e.g., unsloth/mistral-7b-v0.3-bnb-4bit)"
    )

    # Friendly model selection shortcuts
    # 3B models (fast iteration)
    parser.add_argument("--qwen-3b", action="store_true", help="Use Qwen2.5 3B Instruct (fast iteration)")
    parser.add_argument("--llama-3b", action="store_true", help="Use Llama 3.2 3B Instruct (fast iteration)")

    # 7-8B models (production quality)
    parser.add_argument("--mistral-7b", action="store_true", help="Use Mistral 7B v0.3 (production quality)")
    parser.add_argument("--llama-8b", action="store_true", help="Use Llama 3.1 8B Instruct")
    parser.add_argument("--qwen-7b", action="store_true", help="Use Qwen2.5 7B Instruct")
    parser.add_argument("--magistral", action="store_true", help="Use Magistral Small 2509 (~7B)")
    parser.add_argument("--deepseek-7b", action="store_true", help="Use DeepSeek R1 Distill Qwen 7B (reasoning)")
    parser.add_argument("--qwen-vl-8b", action="store_true", help="Use Qwen3 VL 8B Instruct (vision-language)")
    parser.add_argument("--qwen-thinking-8b", action="store_true", help="Use Qwen3 VL 8B Thinking (reasoning + vision)")

    # 11-14B models (advanced)
    parser.add_argument("--llama-13b", action="store_true", help="Use Llama 2 13B (advanced)")
    parser.add_argument("--llama-vision-11b", action="store_true", help="Use Llama 3.2 11B Vision Instruct")
    parser.add_argument("--gemma-12b", action="store_true", help="Use Gemma 3 12B Instruct")
    parser.add_argument("--deepseek-14b", action="store_true", help="Use DeepSeek R1 Distill Qwen 14B (reasoning)")

    # 17-24B models (very large)
    parser.add_argument("--llama-scout-17b", action="store_true", help="Use Llama 4 Scout 17B (very large)")
    parser.add_argument("--gpt-20b", action="store_true", help="Use GPT-OSS 20B (very large)")
    parser.add_argument("--mistral-24b", action="store_true", help="Use Mistral Small 3.2 24B Instruct (extremely large)")

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
        "--adaptive-memory",
        action="store_true",
        help="Enable adaptive memory management (auto-adjust batch size based on VRAM)"
    )
    parser.add_argument(
        "--target-vram-util",
        type=float,
        default=0.80,
        help="Target VRAM utilization for adaptive memory (0.0-1.0, default: 0.80)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Override KTO beta parameter (controls KL divergence penalty)"
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

    # Two-stage learning rate schedule
    parser.add_argument(
        "--two-stage-lr",
        action="store_true",
        help="Enable two-stage learning rate schedule (reduces LR at specified step)"
    )
    parser.add_argument(
        "--lr-reduction-step",
        type=int,
        default=50,
        help="Step at which to reduce learning rate (default: 50)"
    )
    parser.add_argument(
        "--lr-reduction-factor",
        type=float,
        default=0.5,
        help="Factor to multiply LR by at reduction step (default: 0.5 = 50%% reduction)"
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
        "--resume-from-checkpoint",
        type=str,
        help="Path to checkpoint directory to resume training from"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Setup and validate without training"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging to diagnose freezes/hangs"
    )

    args = parser.parse_args()

    # Process friendly model selection flags
    model_map = {
        # 3B models
        'qwen_3b': ('3b', 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'),
        'llama_3b': ('3b', 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit'),

        # 7-8B models
        'mistral_7b': ('7b', 'unsloth/mistral-7b-v0.3-bnb-4bit'),
        'llama_8b': ('7b', 'unsloth/llama-3.1-8b-instruct-bnb-4bit'),
        'qwen_7b': ('7b', 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit'),
        'magistral': ('7b', 'unsloth/Magistral-Small-2509-unsloth-bnb-4bit'),
        'deepseek_7b': ('7b', 'unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit'),
        'qwen_vl_8b': ('7b', 'unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit'),
        'qwen_thinking_8b': ('7b', 'unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit'),

        # 11-14B models
        'llama_13b': ('13b', 'unsloth/llama-2-13b-bnb-4bit'),
        'llama_vision_11b': ('13b', 'unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit'),
        'gemma_12b': ('13b', 'unsloth/gemma-3-12b-it-unsloth-bnb-4bit'),
        'deepseek_14b': ('13b', 'unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit'),

        # 17-24B models
        'llama_scout_17b': ('20b', 'unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit'),
        'gpt_20b': ('20b', 'unsloth/gpt-oss-20b-unsloth-bnb-4bit'),
        'mistral_24b': ('20b', 'unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit'),
    }

    for flag, (size, model_name) in model_map.items():
        if getattr(args, flag):
            args.model_size = size
            args.model_name = model_name
            break

    # Setup environment
    setup_environment()

    # Auto-setup W&B if API key present in .env
    wandb_auto_enabled = setup_wandb()

    # Get configuration
    if args.model_size:
        print(f"Using {args.model_size.upper()} model preset configuration\n")
        config = get_config_by_size(args.model_size)
    else:
        print("Using config defaults from configs/training_config.py\n")
        config = Config()

    # Create timestamped run directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(config.training.output_dir)
    run_dir = base_output_dir / timestamp

    # Create subdirectories for this run
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Update config to use timestamped directories
    config.training.output_dir = str(checkpoints_dir)

    print(f"Training run directory: {run_dir}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Logs: {logs_dir}\n")

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
    # Apply adaptive memory management if requested
    if args.adaptive_memory:
        print("\n" + "="*60)
        print("ADAPTIVE MEMORY MANAGEMENT")
        print("="*60)
        adaptive_settings = get_adaptive_settings(
            model_size=args.model_size,
            target_utilization=args.target_vram_util
        )
        config.training.per_device_train_batch_size = adaptive_settings["batch_size"]
        config.training.gradient_accumulation_steps = adaptive_settings["gradient_accumulation"]
        if adaptive_settings.get("gradient_checkpointing"):
            config.training.gradient_checkpointing = True
        print(f"✓ Automatically adjusted settings:")
        print(f"  Batch size: {adaptive_settings['batch_size']}")
        print(f"  Gradient accumulation: {adaptive_settings['gradient_accumulation']}")
        print(f"  Effective batch size: {adaptive_settings['batch_size'] * adaptive_settings['gradient_accumulation']}")
        print(f"  Gradient checkpointing: {adaptive_settings.get('gradient_checkpointing', False)}")
        print("="*60 + "\n")
    # Allow manual overrides even with adaptive memory
    elif args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.beta:
        config.training.beta = args.beta
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs

    # Apply two-stage LR schedule overrides
    if args.two_stage_lr:
        config.training.use_two_stage_lr = True
    if args.lr_reduction_step:
        config.training.lr_reduction_step = args.lr_reduction_step
    if args.lr_reduction_factor:
        config.training.lr_reduction_factor = args.lr_reduction_factor

    # Auto-enable W&B if API key found or --wandb flag used
    if wandb_auto_enabled or args.wandb:
        config.use_wandb = True
        # Use sensible defaults for project/run name if not specified
        if args.wandb_project:
            config.wandb_project = args.wandb_project
        elif not hasattr(config, 'wandb_project') or not config.wandb_project:
            config.wandb_project = "kto-training"  # Default project name

        if args.wandb_run_name:
            config.wandb_run_name = args.wandb_run_name
        elif not hasattr(config, 'wandb_run_name') or not config.wandb_run_name:
            # Auto-generate run name: model-size-timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            config.wandb_run_name = f"{args.model_size}-{timestamp}"

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

    # Create reference model for KTO (frozen copy of base model, no LoRA)
    # For 7B+ models with limited VRAM, we let TRL handle reference model internally
    # This saves ~8GB VRAM by sharing weights between policy and reference model
    ref_model = None

    # Only create explicit reference model if requested via env var
    # This uses ~8GB extra VRAM but provides more stable KL computation
    import os
    if os.getenv("USE_EXPLICIT_REF_MODEL", "false").lower() == "true":
        print("\n⚠️  Creating explicit reference model (uses ~8GB extra VRAM)")
        ref_model = create_reference_model(
            model_name=config.model.model_name,
            max_seq_length=config.model.max_seq_length,
            dtype=config.model.dtype,
            load_in_4bit=config.model.load_in_4bit,
            hf_token=args.hf_token
        )
    else:
        print("\n✓ Using implicit reference model (TRL manages internally)")
        print("  Saves ~8GB VRAM by sharing weights with policy model")
        print("  To use explicit ref model: USE_EXPLICIT_REF_MODEL=true")

    # Apply LoRA adapters to policy model only (not reference)
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
        num_train_epochs=1 if args.max_steps else config.training.num_train_epochs,
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
    if config.training.use_two_stage_lr:
        reduced_lr = config.training.learning_rate * config.training.lr_reduction_factor
        print(f"  Two-stage LR: ENABLED")
        print(f"    - Steps 1-{config.training.lr_reduction_step}: {config.training.learning_rate:.2e}")
        print(f"    - Steps {config.training.lr_reduction_step+1}+: {reduced_lr:.2e} ({config.training.lr_reduction_factor:.1%} reduction)")
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

    # Setup debug logger if requested
    debugger = None
    if args.debug:
        print("\n" + "=" * 60)
        print("DEBUG MODE ENABLED")
        print("=" * 60)
        print(f"Debug log will be saved to: {logs_dir}/training_debug.log")
        print("This will show exactly where training freezes if issues occur")
        print("=" * 60 + "\n")
        debugger = TrainingDebugger(log_file=str(logs_dir / "training_debug.log"))

    # Extract previous log entries if resuming from checkpoint
    previous_log_entries = None
    if args.resume_from_checkpoint:
        previous_log_entries = extract_previous_log_entries(args.resume_from_checkpoint)

    # Initialize callbacks
    callbacks = [
        MetricsTableCallback(
            log_every_n_steps=5,
            output_dir=str(logs_dir),
            previous_log_entries=previous_log_entries
        ),
        CheckpointMonitorCallback()
    ]

    # Add two-stage LR callback if enabled
    if config.training.use_two_stage_lr:
        reduced_lr = config.training.learning_rate * config.training.lr_reduction_factor
        callbacks.append(
            TwoStageLRCallback(
                initial_lr=config.training.learning_rate,
                reduced_lr=reduced_lr,
                reduction_step=config.training.lr_reduction_step
            )
        )

    # Initialize KTO Trainer (with optional KTO-S)
    if config.training.use_kto_s:
        print("Initializing KTO-S Trainer (with SIGN correction)...")
        trainer = KTOSTrainer(
            model=model,
            ref_model=ref_model,  # Explicit reference model
            args=training_args,
            tokenizer=tokenizer,  # Use 'tokenizer' for TRL 0.11.4
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            use_sign_correction=True,  # Enable SIGN correction
        )
    else:
        print("Initializing Standard KTO Trainer...")
        print("⚠️  Warning: Standard KTO may have KL spikes with base models")
        trainer = KTOTrainer(
            model=model,
            ref_model=ref_model,  # Explicit reference model
            args=training_args,
            tokenizer=tokenizer,  # Use 'tokenizer' for TRL 0.11.4
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

    print("✓ KTO trainer initialized with metrics tracking")
    if ref_model is not None:
        print("✓ Explicit reference model provided for stable KL computation")
    else:
        print("✓ Using TRL's implicit reference model (shared base model)")
    print()

    # Start training
    print("=" * 60)
    if args.resume_from_checkpoint:
        print(f"RESUMING TRAINING FROM: {args.resume_from_checkpoint}")
    else:
        print("STARTING TRAINING")
    print("=" * 60 + "\n")

    try:
        if debugger:
            debugger.log_step_start(0)
            print("Debug: Training about to start...")

        trainer_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

        if debugger:
            debugger.log_step_end(trainer.state.global_step)

        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final loss: {trainer_output.training_loss:.4f}")

        # Check final GPU memory
        print()
        check_gpu_memory()

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("⚠ TRAINING INTERRUPTED BY USER")
        print("=" * 60)
        if debugger:
            debugger.log_exception(trainer.state.global_step if hasattr(trainer, 'state') else -1,
                                  KeyboardInterrupt("User interrupted"))
            debugger.close()
        raise
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TRAINING FAILED")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")

        if debugger:
            step = trainer.state.global_step if hasattr(trainer, 'state') else -1
            debugger.log_exception(step, e)
            debugger.close()
            print(f"\nDebug log saved to: {logs_dir}/training_debug.log")
            print("Check this file to see exactly where it failed")

        print("\nTroubleshooting:")
        print("  1. Check dataset has mixed True/False labels")
        print("  2. Reduce batch_size if OOM error")
        print("  3. Reduce max_length if OOM error")
        print("  4. Check GPU has sufficient memory")
        print("  5. Review logs above for specific error")
        if debugger:
            print(f"  6. Check debug log: {logs_dir}/training_debug.log")
        raise

    # Save final model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    output_path = run_dir / "final_model"
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"✓ Model saved to: {output_path}")
    print("\nTo upload to HuggingFace:")
    print(f"  model.push_to_hub_merged('username/model-name', tokenizer, save_method='merged_16bit')")
    if debugger:
        debugger.close()
        print(f"\nDebug log saved to: {logs_dir}/training_debug.log")

    print("\n" + "=" * 60)
    print("✓ ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
