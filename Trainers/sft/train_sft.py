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
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add repo root and src to path before imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Environment bootstrap — must run before importing torch/unsloth/transformers
from shared.env_bootstrap import init_trainer_env, suppress_transformers_logging

init_trainer_env()

import torch  # noqa: E402

from unsloth import is_bfloat16_supported  # noqa: E402

# Suppress transformers library-level logging after import
suppress_transformers_logging()

from transformers import Trainer
from trl import SFTConfig

from configs.config_loader import (
    get_3b_config,
    get_7b_config,
    get_13b_config,
    get_20b_config,
    load_config,
)
from src.data_loader import load_and_prepare_tokenized_dataset, print_dataset_samples
from src.model_loader import (
    load_model_and_tokenizer,
    apply_lora_adapters,
    check_gpu_memory
)
from src.training_callbacks import (
    MetricsTableCallback,
    CheckpointMonitorCallback,
    LiveDashboardCallback,
    suppress_training_logs,
    DASHBOARD_AVAILABLE,
)
from shared.cloud_artifacts import (
    HFBucketSyncCallback,
    build_manifest,
    build_run_paths,
    publish_final_model_to_hub,
    sync_directory_to_hf_bucket,
    write_manifest,
)
from shared.training_capacity import build_capacity_feature_row, capture_hardware_info, summarize_capacity_from_logs
from shared.training_utils import (
    setup_wandb,
    extract_previous_log_entries,
    save_training_lineage,
    build_base_lineage,
    apply_tier_preset,
)
from shared.experiment_tracking.lineage_enrichment import enrich_training_lineage

# Evolutionary training (optional)
try:
    from shared.evolutionary import EvolutionaryTrainerWrapper
    from shared.evolutionary.config import EvolutionaryConfig as EvoConfig
    EVOLUTIONARY_AVAILABLE = True
except ImportError:
    EVOLUTIONARY_AVAILABLE = False


def convert_to_evo_config(config_evo) -> "EvoConfig":
    """Convert config_loader EvolutionaryConfig to shared.evolutionary.EvolutionaryConfig."""
    if not EVOLUTIONARY_AVAILABLE:
        return None

    return EvoConfig(
        enabled=config_evo.enabled,
        num_candidates=config_evo.candidates,
        eval_batch_size=config_evo.eval_batch_size,
        validation_config_path=config_evo.validation_config,
        strategy=config_evo.strategy.type,
        noise_scale=config_evo.strategy.params.get('noise_scale', 0.1),
        scale_factors=config_evo.strategy.params.get('scale_factors', [0.5, 1.0, 1.5, 2.0]),
        selection_method=config_evo.selection.method,
        min_fitness_improvement=config_evo.selection.min_improvement,
        min_relative_improvement=config_evo.selection.min_relative_improvement,
        noise_floor_epsilon=config_evo.selection.noise_floor_epsilon,
        eval_frequency=config_evo.eval_frequency,
        warmup_steps=config_evo.warmup_steps,
        cache_baseline=config_evo.cache_baseline,
        log_candidates=config_evo.logging.candidates,
        log_selected=config_evo.logging.selected,
    )


# ============================================================================
# Model Compatibility Rules
# Extensible registry of model-family-specific configuration constraints.
# Each entry maps a detection function to a list of validation checks.
# Add new model families here as needed.
# ============================================================================

def _is_lfm2_family(model_name: str) -> bool:
    """Detect LiquidAI LFM2-family models (LFM2, LFM2.5, etc.)."""
    name_lower = model_name.lower()
    return "lfm" in name_lower or "liquidai" in name_lower


def _check_lfm2_4bit(config) -> Optional[str]:
    """Check if LFM2-family model is configured with incompatible 4-bit quantization."""
    if config.model.load_in_4bit:
        return (
            "LFM2.5 uses LIV convolution blocks incompatible with 4-bit quantization. "
            "Set load_in_4bit: false to avoid SIGABRT crash."
        )
    return None


# Known incompatible LoRA modules for LFM2-family models.
# These are standard transformer projection names that do not exist in LFM2's architecture.
_LFM2_INCOMPATIBLE_MODULES = {"o_proj", "gate_proj", "up_proj", "down_proj"}
_LFM2_CORRECT_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]


def _check_lfm2_lora_targets(config) -> Optional[str]:
    """Check if LFM2-family model has incompatible LoRA target modules."""
    configured = set(config.lora.target_modules)
    bad_modules = configured & _LFM2_INCOMPATIBLE_MODULES
    if bad_modules:
        return (
            f"LoRA target_modules contain modules incompatible with LFM2.5 architecture: "
            f"{sorted(bad_modules)}. "
            f"Use these instead: {_LFM2_CORRECT_MODULES}"
        )
    return None


# Registry: list of (detector_fn, [check_fn, ...]) tuples.
# To add a new model family, append a tuple with a detector and its checks.
MODEL_COMPATIBILITY_RULES = [
    (_is_lfm2_family, [_check_lfm2_4bit, _check_lfm2_lora_targets]),
]

UNSLOTH_ALLOWED_INIT_LORA_WEIGHTS = {"gaussian", "loftq", "corda"}
UNSLOTH_BLOCKED_INIT_LORA_WEIGHTS = {"pissa", "eva", "olora"}


def validate_model_compatibility(config) -> None:
    """Validate configuration against known model-family compatibility rules.

    Prints prominent warnings for any incompatible settings but does NOT raise
    exceptions, allowing the user to decide whether to proceed.

    Called in run() after config is loaded but before load_model_and_tokenizer().

    Args:
        config: The Config dataclass loaded from YAML / CLI.
    """
    model_name = config.model.model_name
    warnings_found = []

    for detector, checks in MODEL_COMPATIBILITY_RULES:
        if not detector(model_name):
            continue
        for check_fn in checks:
            warning = check_fn(config)
            if warning:
                warnings_found.append(warning)

    init_lora_weights = getattr(config.lora, "init_lora_weights", None)
    if init_lora_weights:
        normalized = str(init_lora_weights).strip().lower()
        if normalized in UNSLOTH_BLOCKED_INIT_LORA_WEIGHTS:
            warnings_found.append(
                f"init_lora_weights={init_lora_weights!r} is not supported by the current Unsloth path. "
                "EVA, PiSSA, and OLoRA require bypassing Unsloth and using PEFT directly."
            )
        elif normalized not in UNSLOTH_ALLOWED_INIT_LORA_WEIGHTS and normalized not in {"true", "false"}:
            warnings_found.append(
                f"init_lora_weights={init_lora_weights!r} may not be accepted by Unsloth. "
                "Expected one of gaussian, loftq, corda, true, or false."
            )

    if getattr(config.lora, "target_modules", None) == "all-linear":
        warnings_found.append(
            'target_modules="all-linear" requires the new Unsloth model path. '
            "On the legacy path this may break; use an explicit module list if you want the stable path."
        )

    if not warnings_found:
        return

    # Print a box that is impossible to miss
    border = "!" * 72
    print(f"\n{border}")
    print("!!  MODEL COMPATIBILITY WARNING" + " " * 37 + "!!")
    print(border)
    for warning in warnings_found:
        # Wrap long warnings at ~66 chars to fit inside the box
        words = warning.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 68:
                print(f"!!{line}")
                line = "  "
            line += word + " "
        if line.strip():
            print(f"!!{line}")
        print("!!")
    print(border + "\n")



def collate_prepared_sft_batch(features: list[dict[str, Any]], tokenizer) -> dict[str, torch.Tensor]:
    """Pad explicit tokenized SFT rows into a trainer-ready batch."""
    if not features:
        raise ValueError("Cannot collate an empty SFT batch.")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0

    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []

    for feature in features:
        feature_input_ids = list(feature["input_ids"])
        feature_attention_mask = list(feature["attention_mask"])
        feature_labels = list(feature["labels"])
        pad_len = max_len - len(feature_input_ids)
        if pad_len > 0:
            feature_input_ids = feature_input_ids + [pad_token_id] * pad_len
            feature_attention_mask = feature_attention_mask + [0] * pad_len
            feature_labels = feature_labels + [-100] * pad_len
        input_ids.append(feature_input_ids)
        attention_mask.append(feature_attention_mask)
        labels.append(feature_labels)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_training_lineage(
    config,
    train_dataset,
    eval_dataset,
    trainer,
    run_dir: Path,
    args: argparse.Namespace,
    training_time_seconds: Optional[float] = None,
    evolutionary_stats: Optional[Dict[str, Any]] = None,
    preprocessing_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build SFT training lineage using shared base + SFT-specific fields."""
    # Get dataset source info
    dataset_source = args.local_file or config.dataset.local_file
    if not dataset_source:
        dataset_source = f"{config.dataset.dataset_name}/{config.dataset.dataset_file}"

    lineage = build_base_lineage(
        training_type="SFT",
        model_info={
            "base_model": config.model.model_name,
            "max_seq_length": config.model.max_seq_length,
            "load_in_4bit": config.model.load_in_4bit,
            "dtype": str(config.model.dtype),
        },
        lora_info={
            "rank": config.lora.r,
            "alpha": config.lora.lora_alpha,
            "dropout": config.lora.lora_dropout,
            "target_modules": config.lora.target_modules,
            "bias": config.lora.bias,
        },
        training_info={
            "batch_size": config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "effective_batch_size": config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps,
            "learning_rate": config.training.learning_rate,
            "num_epochs": config.training.num_train_epochs,
            "max_steps": args.max_steps if args.max_steps else -1,
            "warmup_ratio": config.training.warmup_ratio,
            "lr_scheduler": config.training.lr_scheduler_type,
            "optimizer": config.training.optim,
            "max_grad_norm": config.training.max_grad_norm,
            "max_seq_length": config.training.max_seq_length,
            "packing": config.training.packing,
            "completion_only_loss": config.training.completion_only_loss,
            "gradient_checkpointing": config.training.gradient_checkpointing,
            "fp16": not torch.cuda.is_bf16_supported() if config.training.fp16 is False else config.training.fp16,
            "bf16": torch.cuda.is_bf16_supported() if config.training.bf16 is True else config.training.bf16,
            "seed": config.seed,
        },
        dataset_info={
            "source": dataset_source,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset) if eval_dataset else 0,
            "filter_desirable": config.dataset.filter_desirable,
        },
        run_dir=run_dir,
        trainer=trainer,
        training_time_seconds=training_time_seconds,
    )

    # SFT-specific extensions
    if preprocessing_metadata:
        lineage["dataset"]["preprocessing"] = dict(preprocessing_metadata)

    if hasattr(config, 'evolutionary') and config.evolutionary.enabled:
        lineage["evolutionary"] = {
            "enabled": True,
            "strategy": config.evolutionary.strategy.type,
            "candidates": config.evolutionary.candidates,
            "selection_method": config.evolutionary.selection.method,
            "eval_frequency": config.evolutionary.eval_frequency,
        }
        if evolutionary_stats:
            lineage["evolutionary"].update(
                {
                    "selection_events": evolutionary_stats.get("selection_events", 0),
                    "baseline_kept_count": evolutionary_stats.get("baseline_kept_count", 0),
                    "accepted_nonbaseline_count": evolutionary_stats.get("accepted_nonbaseline_count", 0),
                    "acceptance_rate": evolutionary_stats.get("acceptance_rate", 0.0),
                    "last_selected_candidate": evolutionary_stats.get("last_selected_candidate"),
                    "best_fitness_history": evolutionary_stats.get("best_fitness_history", []),
                    "events_path": evolutionary_stats.get("events_path"),
                }
            )

    return enrich_training_lineage(lineage, args=args)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SFT Training for RTX 3090")

    # Model configuration
    parser.add_argument("--model-size", type=str, choices=["3b", "7b", "13b", "20b"],
                       help="Model size preset (3b, 7b, 13b, or 20b)")
    parser.add_argument("--config", type=str,
                       help="Path to custom config file")
    parser.add_argument("--model-name", type=str,
                       help="Override Hugging Face model name/path")

    # Training parameters
    parser.add_argument("--batch-size", type=int,
                       help="Override per-device batch size")
    parser.add_argument("--save-steps", type=int,
                       help="Override checkpoint save frequency (steps)")
    parser.add_argument("--save-total-limit", type=int,
                       help="Override max checkpoints kept")
    parser.add_argument("--gradient-accumulation", type=int,
                       help="Override gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float,
                       help="Override learning rate")
    parser.add_argument("--seed", type=int,
                       help="Override the training random seed (config.seed)")
    parser.add_argument("--num-epochs", type=int,
                       help="Override number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum training steps (overrides epochs)")
    parser.add_argument("--max-seq-length", type=int,
                       help="Override maximum sequence length")
    parser.add_argument("--lora-r", type=int,
                       help="Override LoRA rank")
    parser.add_argument("--lora-alpha", type=int,
                       help="Override LoRA alpha")
    parser.add_argument("--lora-dropout", type=float,
                       help="Override LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        help="Comma-separated LoRA target modules override",
    )
    parser.add_argument("--use-dora", action="store_true",
                        help="Enable DoRA (Weight-Decomposed LoRA). Passes through to PEFT via Unsloth kwargs.")
    parser.add_argument("--use-rslora", action="store_true",
                        help="Enable rsLoRA (rank-stabilized scaling). Recommended at r>=128.")
    parser.add_argument("--init-lora-weights", type=str,
                        help="Set init_lora_weights (for example: gaussian, loftq, corda, eva, pissa, olora).")
    parser.add_argument("--evolutionary-enabled", action="store_true",
                        help="Enable experimental evolutionary gradient selection.")
    parser.add_argument("--evolutionary-candidates", type=int,
                        help="Override evolutionary candidate count.")
    parser.add_argument("--evolutionary-eval-batch-size", type=int,
                        help="Override evolutionary fitness eval batch size.")
    parser.add_argument("--evolutionary-validation-config", type=str,
                        help="Override evolutionary validation config path.")
    parser.add_argument("--evolutionary-strategy", choices=["gradient_noise", "antithetic_noise", "scale_variation", "combined"],
                        help="Override evolutionary candidate generation strategy.")
    parser.add_argument("--evolutionary-noise-scale", type=float,
                        help="Override evolutionary gradient noise scale.")
    parser.add_argument("--evolutionary-max-grad-norm", type=float,
                        help="Override evolutionary max gradient norm for candidate generation.")
    parser.add_argument("--evolutionary-scale-factors", type=str,
                        help="Comma-separated evolutionary scale factors.")
    parser.add_argument("--evolutionary-selection-method", choices=["best", "tournament", "proportional"],
                        help="Override evolutionary selection method.")
    parser.add_argument("--evolutionary-min-improvement", type=float,
                        help="Override evolutionary minimum fitness improvement.")
    parser.add_argument("--evolutionary-min-relative-improvement", type=float,
                        help="Override evolutionary minimum relative fitness improvement.")
    parser.add_argument("--evolutionary-noise-floor-epsilon", type=float,
                        help="Override evolutionary absolute acceptance floor.")
    parser.add_argument("--evolutionary-eval-frequency", type=int,
                        help="Override evolutionary evaluation frequency.")
    parser.add_argument("--evolutionary-warmup-steps", type=int,
                        help="Override evolutionary warmup steps.")
    parser.add_argument("--evolutionary-cache-baseline", action="store_true", dest="evolutionary_cache_baseline",
                        help="Enable evolutionary baseline caching.")
    parser.add_argument("--evolutionary-no-cache-baseline", action="store_false", dest="evolutionary_cache_baseline",
                        help="Disable evolutionary baseline caching.")
    parser.add_argument("--evolutionary-log-candidates", action="store_true", dest="evolutionary_log_candidates",
                        help="Enable evolutionary candidate logging.")
    parser.add_argument("--evolutionary-no-log-candidates", action="store_false", dest="evolutionary_log_candidates",
                        help="Disable evolutionary candidate logging.")
    parser.add_argument("--evolutionary-log-selected", action="store_true", dest="evolutionary_log_selected",
                        help="Enable evolutionary selection logging.")
    parser.add_argument("--evolutionary-no-log-selected", action="store_false", dest="evolutionary_log_selected",
                        help="Disable evolutionary selection logging.")
    parser.set_defaults(
        load_in_4bit=None,
        evolutionary_cache_baseline=None,
        evolutionary_log_candidates=None,
        evolutionary_log_selected=None,
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        dest="load_in_4bit",
        help="Force 4-bit model loading",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit model loading",
    )

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

    # Tier preset
    parser.add_argument("--tier", choices=["quick", "standard", "thorough"],
                       help="Preset complexity tier. Overrides individual LoRA/training hyperparams. "
                            "Explicit flags still override tier defaults.")

    # Utility
    parser.add_argument("--dry-run", action="store_true",
                       help="Setup only, don't train")
    parser.add_argument("--resume-from-checkpoint", type=str,
                       help="Resume from checkpoint path")
    parser.add_argument("--hf-token", type=str,
                       help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--output-root", type=str,
                       help="Override root directory for training outputs")
    parser.add_argument("--cloud-provider", type=str,
                       choices=["hf_jobs", "modal", "runpod"],
                       help="Cloud provider identifier for canonical cloud run layout")
    parser.add_argument("--artifact-backend", type=str,
                       choices=["hf_bucket", "modal_volume", "runpod_network_volume"],
                       help="Provider-native artifact backend")
    parser.add_argument("--artifact-bucket", type=str,
                       help="Hugging Face Bucket identifier used by HF Jobs")
    parser.add_argument("--artifact-prefix", type=str,
                       help="Bucket prefix for this run")
    parser.add_argument("--publish-final-model", action="store_true",
                       help="Publish final_model to Hugging Face Hub after training")
    parser.add_argument("--publish-target-repo", type=str,
                       help="Target Hugging Face model repo when publishing final_model")
    parser.add_argument("--run-timestamp", type=str,
                       help="Explicit run timestamp for canonical cloud run layout")

    # UI options
    parser.add_argument("--no-dashboard", action="store_true",
                       help="Disable live dashboard, use table output instead")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose library logs")

    return parser.parse_args(argv)


def run(args: argparse.Namespace):
    """Execute training with the provided CLI arguments."""
    run_metadata = {
        "train_size": None,
        "eval_size": None,
        "run_dir": None,
        "final_model_dir": None,
        "logs_dir": None,
        "manifest_path": None,
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

    # Apply tier preset (overrides base config, but explicit CLI flags override tier)
    if args.tier:
        _sft_tier_config_map = {
            "r": ("lora", "r"),
            "lora_alpha": ("lora", "lora_alpha"),
            "learning_rate": ("training", "learning_rate"),
            "num_train_epochs": ("training", "num_train_epochs"),
            "warmup_ratio": ("training", "warmup_ratio"),
            "batch_size": ("training", "per_device_train_batch_size"),
            "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
            "use_dora": ("lora", "use_dora"),
            "use_rslora": ("lora", "use_rslora"),
            "target_modules": ("lora", "target_modules"),
        }
        apply_tier_preset(
            config, args.tier, _sft_tier_config_map, args,
            configs_dir=Path(__file__).parent / "configs",
        )

    # Apply CLI overrides
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.save_steps is not None:
        config.training.save_steps = args.save_steps
    if args.save_total_limit is not None:
        config.training.save_total_limit = args.save_total_limit
    if args.gradient_accumulation:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    # is not None so seed=0 is honored (a truthy guard would silently drop the valid seed 0)
    if args.seed is not None:
        config.seed = args.seed
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs
    if args.max_seq_length:
        config.training.max_seq_length = args.max_seq_length
        config.model.max_seq_length = args.max_seq_length
    if args.lora_r is not None:
        config.lora.r = args.lora_r
    if args.lora_alpha is not None:
        config.lora.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        config.lora.lora_dropout = args.lora_dropout
    if args.model_name:
        config.model.model_name = args.model_name
    if args.load_in_4bit is not None:
        config.model.load_in_4bit = args.load_in_4bit
    if args.lora_target_modules:
        normalized_target_modules = args.lora_target_modules.strip()
        if normalized_target_modules == "all-linear":
            config.lora.target_modules = normalized_target_modules
        else:
            config.lora.target_modules = [
                module.strip()
                for module in normalized_target_modules.split(",")
                if module.strip()
            ]
    if args.use_dora:
        config.lora.use_dora = True
    if args.use_rslora:
        config.lora.use_rslora = True
    if args.init_lora_weights is not None:
        config.lora.init_lora_weights = args.init_lora_weights
    if args.evolutionary_enabled:
        config.evolutionary.enabled = True
    if args.evolutionary_candidates is not None:
        config.evolutionary.candidates = args.evolutionary_candidates
    if args.evolutionary_eval_batch_size is not None:
        config.evolutionary.eval_batch_size = args.evolutionary_eval_batch_size
    if args.evolutionary_validation_config is not None:
        config.evolutionary.validation_config = args.evolutionary_validation_config
    if args.evolutionary_strategy is not None:
        config.evolutionary.strategy.type = args.evolutionary_strategy
    if args.evolutionary_noise_scale is not None:
        config.evolutionary.strategy.params["noise_scale"] = args.evolutionary_noise_scale
    if args.evolutionary_max_grad_norm is not None:
        config.evolutionary.strategy.params["max_grad_norm"] = args.evolutionary_max_grad_norm
    if args.evolutionary_scale_factors:
        config.evolutionary.strategy.params["scale_factors"] = [
            float(value.strip())
            for value in args.evolutionary_scale_factors.split(",")
            if value.strip()
        ]
    if args.evolutionary_selection_method is not None:
        config.evolutionary.selection.method = args.evolutionary_selection_method
    if args.evolutionary_min_improvement is not None:
        config.evolutionary.selection.min_improvement = args.evolutionary_min_improvement
    if args.evolutionary_min_relative_improvement is not None:
        config.evolutionary.selection.min_relative_improvement = args.evolutionary_min_relative_improvement
    if args.evolutionary_noise_floor_epsilon is not None:
        config.evolutionary.selection.noise_floor_epsilon = args.evolutionary_noise_floor_epsilon
    if args.evolutionary_eval_frequency is not None:
        config.evolutionary.eval_frequency = args.evolutionary_eval_frequency
    if args.evolutionary_warmup_steps is not None:
        config.evolutionary.warmup_steps = args.evolutionary_warmup_steps
    if args.evolutionary_cache_baseline is not None:
        config.evolutionary.cache_baseline = args.evolutionary_cache_baseline
    if args.evolutionary_log_candidates is not None:
        config.evolutionary.logging.candidates = args.evolutionary_log_candidates
    if args.evolutionary_log_selected is not None:
        config.evolutionary.logging.selected = args.evolutionary_log_selected

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

    # Validate model compatibility BEFORE loading model
    validate_model_compatibility(config)

    repo_branch = os.environ.get("CLOUD_REPO_BRANCH")
    repo_commit = os.environ.get("CLOUD_REPO_COMMIT")
    artifact_identifier = args.artifact_bucket or os.environ.get("CLOUD_ARTIFACT_IDENTIFIER")

    # Create timestamped run directory (following KTO pattern)
    timestamp = args.run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_root) if args.output_root else Path(config.training.output_dir)
    if args.cloud_provider:
        run_paths = build_run_paths(
            base_output_dir=base_output_dir,
            provider=args.cloud_provider,
            method="sft",
            timestamp=timestamp,
            commit=repo_commit or "local",
        )
        run_dir = run_paths.run_dir
        checkpoints_dir = run_paths.checkpoints_dir
        logs_dir = run_paths.logs_dir
        final_model_path = run_paths.final_model_dir
        lineage_path = run_paths.lineage_path
        manifest_path = run_paths.manifest_path
        for path in (run_dir, checkpoints_dir, logs_dir):
            path.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(
            provider=args.cloud_provider,
            method="sft",
            artifact_backend=args.artifact_backend or "",
            artifact_identifier=artifact_identifier,
            run_paths=run_paths,
            repo_branch=repo_branch,
            repo_commit=repo_commit,
            publish_final_model=args.publish_final_model,
            publish_target_repo=args.publish_target_repo,
            status="running",
        )
        write_manifest(manifest_path, manifest)
        run_metadata["manifest_path"] = str(manifest_path)
    else:
        run_dir = base_output_dir / timestamp
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        final_model_path = run_dir / "final_model"
        lineage_path = run_dir / "training_lineage.json"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = None

    # Update config to use checkpoints directory (trainer saves here)
    config.training.output_dir = str(checkpoints_dir)

    print(f"Training run directory: {run_dir}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Logs: {logs_dir}\n")
    run_metadata.update(
        {
            "run_dir": str(run_dir),
            "final_model_dir": str(final_model_path),
            "logs_dir": str(logs_dir),
        }
    )

    # Load model and tokenizer FIRST (needed for packing preprocessing)
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        dtype=config.model.dtype,
        load_in_4bit=config.model.load_in_4bit,
        hf_token=args.hf_token
    )

    # Prefer the pretrained chat template when available; otherwise, apply a
    # family-specific fallback via Unsloth.
    from unsloth.chat_templates import get_chat_template

    model_name_lower = config.model.model_name.lower()
    if getattr(tokenizer, "chat_template", None):
        chat_template_name = "pretrained"
        print("✓ Using pretrained chat template from the loaded text encoder")
    else:
        if "qwen" in model_name_lower:
            chat_template_name = "chatml"
        elif "llama" in model_name_lower or "nemotron" in model_name_lower:
            chat_template_name = "llama-3"
        elif "mistral" in model_name_lower:
            chat_template_name = "mistral"
        elif "gemma" in model_name_lower:
            chat_template_name = "gemma"
        elif "phi" in model_name_lower:
            chat_template_name = "phi-3"
        elif "deepseek" in model_name_lower or "smollm" in model_name_lower:
            chat_template_name = "chatml"
        else:
            chat_template_name = "chatml"

        tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
        print(f"✓ Applied {chat_template_name} chat template via Unsloth")

    loss_mask_mode = "assistant_only" if config.training.completion_only_loss else "full_sequence"
    preprocessing_metadata = {
        "contract_version": 1,
        "dataset_representation": "tokenized",
        "loss_mask_mode": loss_mask_mode,
        "tool_call_mode": "render_text",
        "chat_template_source": chat_template_name,
        "packing": False,
    }

    # Materialize trainer-ready tokenized rows in-repo so cloud runs do not
    # depend on implicit TRL/Unsloth dataset preparation behavior.
    train_dataset, eval_dataset = load_and_prepare_tokenized_dataset(
        dataset_name=config.dataset.dataset_name if not config.dataset.local_file else None,
        data_files=config.dataset.dataset_file if not config.dataset.local_file else None,
        local_file=config.dataset.local_file,
        num_proc=config.dataset.num_proc,
        test_size=config.dataset.test_size,
        split_dataset=args.split_dataset or config.dataset.split_dataset,
        filter_desirable=config.dataset.filter_desirable,
        tokenizer=tokenizer,
        max_seq_length=config.training.max_seq_length,
        loss_mask_mode=loss_mask_mode,
    )
    run_metadata["train_size"] = len(train_dataset)
    run_metadata["eval_size"] = len(eval_dataset) if eval_dataset else None

    # Print samples
    print_dataset_samples(train_dataset, num_samples=2)

    # Apply LoRA adapters
    model = apply_lora_adapters(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
        random_state=config.lora.random_state,
        use_rslora=config.lora.use_rslora,
        use_dora=config.lora.use_dora,
        init_lora_weights=config.lora.init_lora_weights,
    )

    # Check initial GPU memory
    check_gpu_memory()

    # Initialize callbacks based on UI mode (need to know this before configuring trainer)
    # Dashboard is the default if available, use --no-dashboard to disable
    use_dashboard = DASHBOARD_AVAILABLE and not getattr(args, 'no_dashboard', False)
    use_quiet_mode = use_dashboard or args.quiet

    print(f"[UI] Dashboard available: {DASHBOARD_AVAILABLE}, using dashboard: {use_dashboard}")

    # Configure SFT training arguments
    sft_config_kwargs = {
        "output_dir": config.training.output_dir,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "max_grad_norm": config.training.max_grad_norm,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "max_seq_length": config.training.max_seq_length,
        "packing": False,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "optim": config.training.optim,
        "fp16": not is_bfloat16_supported() if config.training.fp16 is False else config.training.fp16,
        "bf16": is_bfloat16_supported() if config.training.bf16 is True else config.training.bf16,
        "num_train_epochs": 1 if args.max_steps else config.training.num_train_epochs,
        "max_steps": args.max_steps if args.max_steps else -1,
        "warmup_ratio": config.training.warmup_ratio,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "save_total_limit": config.training.save_total_limit,
        "dataloader_num_workers": config.training.dataloader_num_workers,
        "dataloader_pin_memory": config.training.dataloader_pin_memory,
        "eval_strategy": config.training.eval_strategy if eval_dataset else "no",
        "eval_steps": config.training.eval_steps if eval_dataset else None,
        "report_to": "wandb" if config.use_wandb else "none",
        "run_name": config.wandb_run_name if config.use_wandb else None,
        "seed": config.seed,
        "remove_unused_columns": False,
        # Disable tqdm when using dashboard (they conflict)
        "disable_tqdm": use_dashboard,
        # Suppress metrics dict logging (our callback handles display)
        "log_level": "warning" if use_dashboard else "info",
    }

    training_args = SFTConfig(**sft_config_kwargs)

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
    print("  Packing: False (explicit pre-encoded dataset path)")
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

    # Setup callbacks based on UI mode (use_dashboard/use_quiet_mode set earlier)
    if use_dashboard:
        print("[OK] Using live dashboard mode")
        callbacks = [
            LiveDashboardCallback(
                log_every_n_steps=config.training.logging_steps,
                output_dir=str(run_dir),
                training_type="sft",
                previous_log_entries=previous_log_entries
            ),
        ]
    else:
        callbacks = [
            MetricsTableCallback(
                log_every_n_steps=config.training.logging_steps,
                output_dir=str(run_dir),  # Pass run_dir, callback adds /logs
                previous_log_entries=previous_log_entries
            ),
            CheckpointMonitorCallback()
        ]
    if args.artifact_backend == "hf_bucket" and args.artifact_bucket and args.artifact_prefix:
        callbacks.append(
            HFBucketSyncCallback(
                run_dir=run_dir,
                bucket_id=args.artifact_bucket,
                prefix=args.artifact_prefix,
                token=args.hf_token,
                log_every_n_steps=config.training.logging_steps,
            )
        )

    # Apply log suppression for trainer initialization if quiet mode
    if use_quiet_mode:
        # Suppress verbose output during trainer creation and training
        import logging
        for name in ['unsloth', 'transformers', 'transformers.trainer', 'trl', 'peft', 'accelerate']:
            logging.getLogger(name).setLevel(logging.WARNING)

    # Extra suppression for dashboard mode - completely quiet the trainer's logging
    if use_dashboard:
        import logging
        # Set trainer to ERROR level to suppress metrics output (we handle it in callback)
        logging.getLogger('transformers.trainer').setLevel(logging.ERROR)

    print("Initializing trainer on explicit pre-encoded dataset...")
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": callbacks,
        "data_collator": lambda features: collate_prepared_sft_batch(features, tokenizer),
    }
    trainer = Trainer(**trainer_kwargs)

    # Remove PrinterCallback to stop the {'loss': ...} dict spam
    # Our LiveDashboardCallback or MetricsTableCallback handles display instead
    if use_dashboard:
        from transformers.trainer_callback import PrinterCallback
        trainer.remove_callback(PrinterCallback)

    print("[OK] Trainer initialized with explicit pre-encoded dataset contract")

    # Check if evolutionary training is enabled
    evo_wrapper = None
    if hasattr(config, 'evolutionary') and config.evolutionary.enabled:
        if not EVOLUTIONARY_AVAILABLE:
            print("[WARN] Evolutionary training requested but shared.evolutionary module not available")
        else:
            evo_config = convert_to_evo_config(config.evolutionary)
            evo_wrapper = EvolutionaryTrainerWrapper(
                trainer=trainer,
                config=evo_config,
                tokenizer=tokenizer,
                events_path=logs_dir / "evolutionary_events.jsonl",
            )
            print(f"[OK] Evolutionary training enabled:")
            print(f"     Strategy: {config.evolutionary.strategy.type}")
            print(f"     Candidates: {config.evolutionary.candidates}")
            print(f"     Selection: {config.evolutionary.selection.method}")
            if config.evolutionary.warmup_steps > 0:
                print(f"     Warmup: {config.evolutionary.warmup_steps} steps of standard training first")
    print()

    # Check memory before training
    check_gpu_memory()

    # Train the model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    if evo_wrapper:
        print("(Evolutionary gradient selection enabled)")
    if use_dashboard:
        print("(Live dashboard enabled)")
    print("=" * 60 + "\n")

    import time
    training_start_time = time.time()

    # Use evolutionary wrapper if enabled, otherwise standard training
    training_failed = False
    failure_message = None
    try:
        if evo_wrapper:
            evo_wrapper.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except Exception as exc:
        training_failed = True
        failure_message = str(exc)
        if manifest_path:
            write_manifest(
                manifest_path,
                build_manifest(
                    provider=args.cloud_provider or "local",
                    method="sft",
                    artifact_backend=args.artifact_backend or "",
                    artifact_identifier=artifact_identifier,
                    run_paths=run_paths,
                    repo_branch=repo_branch,
                    repo_commit=repo_commit,
                    publish_final_model=args.publish_final_model,
                    publish_target_repo=args.publish_target_repo,
                    status=f"failed: {failure_message}",
                ),
            )
            if args.artifact_backend == "hf_bucket" and args.artifact_bucket and args.artifact_prefix:
                try:
                    sync_directory_to_hf_bucket(
                        run_dir,
                        args.artifact_bucket,
                        args.artifact_prefix,
                        token=args.hf_token,
                    )
                except Exception as sync_exc:
                    print(f"[WARN] Failed to sync failed-run artifacts to HF bucket: {sync_exc}")
        raise

    training_end_time = time.time()
    training_time_seconds = training_end_time - training_start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    # Save final model
    print(f"\nSaving final model to: {final_model_path}")
    trainer.save_model(str(final_model_path))

    print(f"\n[OK] Training complete!")
    print(f"  Model saved to: {final_model_path}")
    print(f"  Logs saved to: {logs_dir}/")

    # Build and save training lineage
    print("\nBuilding training lineage...")
    lineage = build_training_lineage(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer=trainer,
        run_dir=run_dir,
        args=args,
        training_time_seconds=training_time_seconds,
        evolutionary_stats=evo_wrapper.get_stats() if evo_wrapper else None,
        preprocessing_metadata=preprocessing_metadata,
    )
    actual_lineage_path = save_training_lineage(lineage, run_dir)
    run_metadata["lineage_path"] = str(actual_lineage_path)

    # Post-training loss computation hook
    compute_losses_flag = getattr(config.training, "compute_losses", False) or getattr(args, "compute_losses", False)
    if compute_losses_flag:
        print("\nComputing per-example losses on training dataset...")
        try:
            from shared.experiment_tracking import compute_per_example_losses, save_losses
            from shared.experiment_tracking.lineage_enrichment import build_loss_lineage, write_json as write_lineage_json
            
            # Switch to eval mode
            import torch
            model.eval()
            
            # Unsloth for_inference to optimize inference speed
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
            
            dataset_path = config.data.train_dataset
            if not Path(dataset_path).exists():
                dataset_path = Path(_REPO_ROOT) / dataset_path
                
            losses = compute_per_example_losses(
                model=model,
                tokenizer=tokenizer,
                dataset_path=dataset_path,
                max_seq_length=config.training.max_seq_length,
                completion_only=config.training.completion_only_loss,
            )
            
            losses_path = logs_dir / "per_example_losses.jsonl"
            save_losses(losses, losses_path)
            loss_lineage = build_loss_lineage(
                dataset_path=dataset_path,
                output_root=logs_dir,
                loss_results=losses,
                completion_only=config.training.completion_only_loss,
                max_seq_length=config.training.max_seq_length,
                runtime_backend="unsloth",
            )
            write_lineage_json(logs_dir / "loss_lineage.json", loss_lineage)
            
            # Need to register this somewhere for adapters! 
            # Handled below in adapter conversion by passing path, or we can just let 
            # sft_lineage_to_run_record find it.
            print(f"[OK] Saved per-example losses to {losses_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to compute post-training losses: {e}")
            logging.getLogger(__name__).exception("Failed to compute losses")
            
    # Register in unified experiment tracking registry (best-effort)
    try:
        from shared.experiment_tracking.adapters import sft_lineage_to_run_record
        from shared.experiment_tracking.registry import RunRegistry

        is_cloud = getattr(args, "cloud_provider", None) is not None
        record = sft_lineage_to_run_record(lineage, str(run_dir), cloud=is_cloud)
        RunRegistry().register_run(record)
        logging.getLogger(__name__).info("Run registered: %s", record.run_id)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Unified tracking registration failed (non-fatal): %s", exc
        )

    if args.artifact_backend == "hf_bucket" and args.artifact_bucket and args.artifact_prefix:
        sync_directory_to_hf_bucket(run_dir, args.artifact_bucket, args.artifact_prefix, token=args.hf_token)

    if args.publish_final_model and args.publish_target_repo:
        if not args.hf_token:
            raise RuntimeError("Publishing final_model requires HF_TOKEN or --hf-token.")
        publish_final_model_to_hub(final_model_path, args.publish_target_repo, args.hf_token)

    if manifest_path:
        write_manifest(
            manifest_path,
            build_manifest(
                provider=args.cloud_provider,
                method="sft",
                artifact_backend=args.artifact_backend or "",
                artifact_identifier=artifact_identifier,
                run_paths=run_paths,
                repo_branch=repo_branch,
                repo_commit=repo_commit,
                publish_final_model=args.publish_final_model,
                publish_target_repo=args.publish_target_repo,
                status="completed",
            ),
        )
        if args.artifact_backend == "hf_bucket" and args.artifact_bucket and args.artifact_prefix:
            sync_directory_to_hf_bucket(run_dir, args.artifact_bucket, args.artifact_prefix, token=args.hf_token)

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
