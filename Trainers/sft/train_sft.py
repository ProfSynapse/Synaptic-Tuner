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
import json
import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Force UTF-8 output for Windows to handle unicode characters like ✓
if sys.platform == "win32":
    import io
    # Check if stdout/stderr are attached to a terminal or file (have buffer)
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load .env file for API keys (HF_TOKEN, WANDB_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

# Suppress verbose logging from transformers BEFORE importing it
# This prevents the metrics dict spam during training
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

# ============================================================================
# DISABLE TORCH.COMPILE - Required for VL models and WSL compatibility
# Must be set BEFORE importing unsloth
# ============================================================================
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_JIT'] = '0'

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

# Add shared to path for evolutionary module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unsloth import is_bfloat16_supported

# Now that unsloth is imported, suppress transformers logging
import transformers
transformers.logging.set_verbosity_warning()

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


def build_training_lineage(
    config,
    train_dataset,
    eval_dataset,
    trainer,
    run_dir: Path,
    args: argparse.Namespace,
    training_time_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """Build comprehensive training lineage for model cards and traceability.

    Args:
        config: Training configuration object
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (may be None)
        trainer: SFT trainer after training
        run_dir: Path to training run directory
        args: Command-line arguments
        training_time_seconds: Total training time in seconds

    Returns:
        Dictionary containing complete training lineage
    """
    hardware_info = capture_hardware_info(torch)

    # Get dataset source info
    dataset_source = args.local_file or config.dataset.local_file
    if not dataset_source:
        dataset_source = f"{config.dataset.dataset_name}/{config.dataset.dataset_file}"

    # Build lineage dictionary
    lineage = {
        "training_type": "SFT",
        "timestamp": datetime.now().isoformat(),
        "run_directory": str(run_dir),

        "model": {
            "base_model": config.model.model_name,
            "max_seq_length": config.model.max_seq_length,
            "load_in_4bit": config.model.load_in_4bit,
            "dtype": str(config.model.dtype),
        },

        "lora": {
            "rank": config.lora.r,
            "alpha": config.lora.lora_alpha,
            "dropout": config.lora.lora_dropout,
            "target_modules": config.lora.target_modules,
            "bias": config.lora.bias,
        },

        "training": {
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

        "dataset": {
            "source": dataset_source,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset) if eval_dataset else 0,
            "filter_desirable": config.dataset.filter_desirable,
        },

        "hardware": hardware_info,
        "capacity_profile": summarize_capacity_from_logs(run_dir / "logs"),

        "results": {},
    }

    # Add training results if available
    if hasattr(trainer, 'state') and trainer.state is not None:
        lineage["results"]["final_step"] = trainer.state.global_step
        lineage["results"]["total_epochs"] = trainer.state.epoch

        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            # Get the last logged loss
            for entry in reversed(trainer.state.log_history):
                if 'loss' in entry:
                    lineage["results"]["final_loss"] = entry['loss']
                    break

    if training_time_seconds:
        lineage["results"]["training_time_seconds"] = round(training_time_seconds, 1)
        lineage["results"]["training_time_formatted"] = f"{training_time_seconds // 3600:.0f}h {(training_time_seconds % 3600) // 60:.0f}m {training_time_seconds % 60:.0f}s"

    # Add evolutionary training info if available
    if hasattr(config, 'evolutionary') and config.evolutionary.enabled:
        lineage["evolutionary"] = {
            "enabled": True,
            "strategy": config.evolutionary.strategy.type,
            "candidates": config.evolutionary.candidates,
            "selection_method": config.evolutionary.selection.method,
            "eval_frequency": config.evolutionary.eval_frequency,
        }

    return lineage


def save_training_lineage(lineage: Dict[str, Any], run_dir: Path) -> Path:
    """Save training lineage to JSON file.

    Args:
        lineage: Training lineage dictionary
        run_dir: Path to training run directory

    Returns:
        Path to saved lineage file
    """
    lineage_path = run_dir / "training_lineage.json"

    with open(lineage_path, 'w', encoding='utf-8') as f:
        json.dump(lineage, f, indent=2, default=str)

    print(f"[OK] Training lineage saved to: {lineage_path}")

    feature_row = build_capacity_feature_row(lineage)
    if feature_row:
        features_path = run_dir / "capacity_features.json"
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(feature_row, f, indent=2, default=str)
        print(f"[OK] Capacity features saved to: {features_path}")

    return lineage_path


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

    # Apply proper chat template using Unsloth's get_chat_template
    # This is CRITICAL for VL models and ensures special tokens are handled correctly
    from unsloth.chat_templates import get_chat_template

    model_name_lower = config.model.model_name.lower()
    if "qwen" in model_name_lower:
        chat_template_name = "chatml"
    elif "llama" in model_name_lower:
        chat_template_name = "llama-3"
    elif "mistral" in model_name_lower:
        chat_template_name = "mistral"
    elif "gemma" in model_name_lower:
        chat_template_name = "gemma"
    elif "phi" in model_name_lower:
        chat_template_name = "phi-3"
    elif "deepseek" in model_name_lower:
        chat_template_name = "chatml"
    else:
        chat_template_name = "chatml"  # Default fallback

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
    print(f"✓ Applied {chat_template_name} chat template via Unsloth")

    # Load dataset (with preprocessing if packing is enabled)
    use_packing = config.training.packing
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=config.dataset.dataset_name if not config.dataset.local_file else None,
        data_files=config.dataset.dataset_file if not config.dataset.local_file else None,
        local_file=config.dataset.local_file,
        num_proc=config.dataset.num_proc,
        test_size=config.dataset.test_size,
        split_dataset=args.split_dataset or config.dataset.split_dataset,
        filter_desirable=config.dataset.filter_desirable,
        tokenizer=tokenizer if use_packing else None,
        apply_chat_template=use_packing  # Preprocess for packing support
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
        random_state=config.lora.random_state
    )

    # Check initial GPU memory
    check_gpu_memory()

    import json

    def render_tool_calls_to_content(tool_calls):
        """Convert tool_calls array to text content for chat template."""
        if not tool_calls:
            return ""

        rendered_parts = []
        for tc in tool_calls:
            # Handle OpenAI nested format: {"function": {"name": ..., "arguments": ...}}
            if "function" in tc and tc["function"]:
                func = tc["function"]
                name = func.get("name", "")
                args = func.get("arguments", "{}")
            else:
                # Handle flat format: {"name": ..., "arguments": ...}
                name = tc.get("name", "")
                args = tc.get("arguments", "{}")

            # Parse arguments if it's a string
            if isinstance(args, str):
                try:
                    args_obj = json.loads(args)
                except:
                    args_obj = args
            else:
                args_obj = args

            tool_call_obj = {"name": name, "arguments": args_obj}
            rendered_parts.append(
                f"<tool_call>\n{json.dumps(tool_call_obj, indent=2)}\n</tool_call>"
            )

        return "\n".join(rendered_parts)

    def sanitize_conversations(conversations):
        """Ensure all message fields are properly set and tool_calls are rendered to content."""
        sanitized = []
        for msg in conversations:
            new_msg = dict(msg)

            # Get existing content (or empty string if None)
            content = new_msg.get("content") or ""

            # If there are tool_calls, render them to content
            if "tool_calls" in new_msg and new_msg["tool_calls"]:
                tool_content = render_tool_calls_to_content(new_msg["tool_calls"])
                if tool_content:
                    content = f"{content}\n\n{tool_content}" if content else tool_content

            new_msg["content"] = content

            # Remove tool_calls since we've rendered them to content
            if "tool_calls" in new_msg:
                del new_msg["tool_calls"]

            sanitized.append(new_msg)
        return sanitized

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
            # Sanitize conversations to handle None content and tool_calls
            sanitized_msgs = sanitize_conversations(msgs)
            formatted.append(
                tokenizer.apply_chat_template(
                    sanitized_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return formatted

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
        "packing": use_packing,
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
        "dataset_kwargs": {"add_special_tokens": False},
        # Disable tqdm when using dashboard (they conflict)
        "disable_tqdm": use_dashboard,
        # Suppress metrics dict logging (our callback handles display)
        "log_level": "warning" if use_dashboard else "info",
    }

    # When packing is enabled, use preprocessed 'text' column
    if use_packing:
        sft_config_kwargs["dataset_text_field"] = "text"

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

    # Initialize SFT Trainer (NO ref_model needed!)
    print("Initializing SFT Trainer...")
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": callbacks,
    }

    # Only use formatting_func when packing is disabled
    # When packing is enabled, we use the preprocessed 'text' column instead
    if not use_packing:
        trainer_kwargs["formatting_func"] = format_chat_template

    trainer = SFTTrainer(**trainer_kwargs)

    # Remove PrinterCallback to stop the {'loss': ...} dict spam
    # Our LiveDashboardCallback or MetricsTableCallback handles display instead
    if use_dashboard:
        from transformers.trainer_callback import PrinterCallback
        trainer.remove_callback(PrinterCallback)

    print("[OK] SFT trainer initialized with metrics tracking")

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
                sync_directory_to_hf_bucket(run_dir, args.artifact_bucket, args.artifact_prefix, token=args.hf_token)
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
        training_time_seconds=training_time_seconds
    )
    actual_lineage_path = save_training_lineage(lineage, run_dir)
    run_metadata["lineage_path"] = str(actual_lineage_path)

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
