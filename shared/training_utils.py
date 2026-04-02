"""
shared/training_utils.py

Shared training utilities for SFT, KTO, and GRPO trainers.

Consolidates duplicated functions from train_sft.py, train_kto.py,
and train_grpo.py. Each function previously existed in two or three
trainer scripts with cosmetic or minor behavioral drift.

Used by: Trainers/sft/train_sft.py, Trainers/kto/train_kto.py,
         Trainers/grpo/train_grpo.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_wandb() -> bool:
    """Auto-setup W&B if WANDB_API_KEY is in environment.

    Returns True if W&B login succeeded, False otherwise.
    """
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        return False

    try:
        import wandb

        wandb.login(key=wandb_key, relogin=True, force=True)
        print("[OK] W&B: Logged in automatically (using WANDB_API_KEY from .env)")
        return True
    except ImportError:
        print("[WARN] W&B: API key found but wandb not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        print(f"[WARN] W&B: Login failed ({e})")
        return False


def extract_previous_log_entries(checkpoint_path: str) -> List[dict]:
    """Extract log entries from a previous run when resuming from checkpoint.

    Parses the checkpoint path to determine the resume step, finds the
    most recent training log file, and returns entries up to that step.

    This is the canonical implementation adopted from KTO's version, which
    provides step-filtered extraction (vs SFT's unfiltered glob approach).

    Args:
        checkpoint_path: Path to checkpoint directory
            (e.g., "output/20251114_135227/checkpoints/checkpoint-50")

    Returns:
        List of log entry dicts (empty list on failure)
    """
    checkpoint_path = Path(checkpoint_path)

    # Extract step number from checkpoint name (e.g., "checkpoint-50" -> 50)
    checkpoint_name = checkpoint_path.name
    step_match = re.search(r"checkpoint-(\d+)", checkpoint_name)
    if not step_match:
        print(f"[WARN] Could not extract step number from checkpoint path: {checkpoint_path}")
        return []

    resume_step = int(step_match.group(1))

    # Navigate up to find the run directory (timestamp directory)
    # checkpoint_path is like: output/20251114_135227/checkpoints/checkpoint-50
    # We want: output/20251114_135227
    run_dir = checkpoint_path.parent.parent

    # Find log files in the logs subdirectory
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        print(f"[WARN] Logs directory not found: {logs_dir}")
        return []

    # Find training log files (there may be multiple if resuming multiple times)
    log_files = list(logs_dir.glob("training_*.jsonl"))
    if not log_files:
        print(f"[WARN] No log files found in: {logs_dir}")
        return []

    # Use the most recent log file (sorted by name, which includes timestamp)
    log_file = sorted(log_files)[-1]

    print(f"\n[OK] Found previous run log: {log_file}")
    print(f"  Extracting entries from steps 0 to {resume_step}")

    # Read log entries up to the resume step
    previous_entries: List[dict] = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                step = entry.get("step", 0)

                # Include entries up to and including the resume step
                if step <= resume_step:
                    previous_entries.append(entry)
                else:
                    break

        print(f"  Extracted {len(previous_entries)} log entries\n")
        return previous_entries

    except Exception as e:
        print(f"[WARN] Failed to read log file: {e}")
        return []


def save_training_lineage(lineage: Dict[str, Any], run_dir: Path) -> Path:
    """Save training lineage to JSON file, plus capacity features.

    Args:
        lineage: Training lineage dictionary
        run_dir: Path to training run directory

    Returns:
        Path to saved lineage file
    """
    from shared.training_capacity import build_capacity_feature_row

    lineage_path = run_dir / "training_lineage.json"

    with open(lineage_path, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2, default=str)

    print(f"[OK] Training lineage saved to: {lineage_path}")

    feature_row = build_capacity_feature_row(lineage)
    if feature_row:
        features_path = run_dir / "capacity_features.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(feature_row, f, indent=2, default=str)
        print(f"[OK] Capacity features saved to: {features_path}")

    return lineage_path


def build_base_lineage(
    training_type: str,
    model_info: Dict[str, Any],
    lora_info: Dict[str, Any],
    training_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    run_dir: Path,
    trainer: Any,
    training_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the shared portion of training lineage.

    Constructs the common lineage structure used by all trainers.
    Callers merge trainer-specific fields (e.g., KTO beta, SFT evolutionary)
    via dict.update() on the returned structure.

    This follows the Open/Closed Principle: the shared function is closed
    for modification, but open for extension via the caller's merge.

    Args:
        training_type: "SFT", "KTO", or "GRPO"
        model_info: Dict with keys: base_model, max_seq_length, load_in_4bit, dtype
        lora_info: Dict with keys: rank, alpha, dropout, target_modules, bias
        training_info: Dict with keys: batch_size, gradient_accumulation_steps,
            effective_batch_size, learning_rate, num_epochs, max_steps,
            warmup_ratio, lr_scheduler, optimizer, max_grad_norm,
            gradient_checkpointing, fp16, bf16, seed
        dataset_info: Dict with keys: source, train_examples, eval_examples
            (plus any trainer-specific keys)
        run_dir: Path to training run directory
        trainer: Trainer object after training (for state extraction)
        training_time_seconds: Total training time in seconds

    Returns:
        Lineage dict with shared fields populated. Callers extend this.
    """
    from datetime import datetime

    import torch

    from shared.training_capacity import capture_hardware_info, summarize_capacity_from_logs

    hardware_info = capture_hardware_info(torch)

    lineage: Dict[str, Any] = {
        "training_type": training_type,
        "timestamp": datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "model": dict(model_info),
        "lora": dict(lora_info),
        "training": dict(training_info),
        "dataset": dict(dataset_info),
        "hardware": hardware_info,
        "capacity_profile": summarize_capacity_from_logs(run_dir / "logs"),
        "results": {},
    }

    # Add training results if available
    if hasattr(trainer, "state") and trainer.state is not None:
        lineage["results"]["final_step"] = trainer.state.global_step
        lineage["results"]["total_epochs"] = trainer.state.epoch

        if hasattr(trainer.state, "log_history") and trainer.state.log_history:
            for entry in reversed(trainer.state.log_history):
                if "loss" in entry:
                    lineage["results"]["final_loss"] = entry["loss"]
                    break

    if training_time_seconds:
        lineage["results"]["training_time_seconds"] = round(training_time_seconds, 1)
        hours = training_time_seconds // 3600
        minutes = (training_time_seconds % 3600) // 60
        seconds = training_time_seconds % 60
        lineage["results"]["training_time_formatted"] = f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"

    return lineage


def apply_tier_preset(
    config: Any,
    tier_name: str,
    tier_config_map: Dict[str, tuple],
    args: Any,
    configs_dir: Path,
) -> Dict[str, Any]:
    """Apply a tier preset from YAML config to the training configuration.

    Each trainer defines its own tier_config_map that maps tier YAML keys
    to (section, attribute) pairs on the config dataclass.

    Args:
        config: Training configuration dataclass
        tier_name: Name of the tier (e.g., "quick", "standard", "thorough")
        tier_config_map: Dict mapping tier key names to (section, attr) tuples
            on the config dataclass
        args: Command-line arguments namespace (for max_steps routing)
        configs_dir: Path to the configs directory containing tiers/ subdirectory

    Returns:
        The parsed tier_config dict (for logging)

    Raises:
        FileNotFoundError: If the tier YAML file does not exist
    """
    import yaml as _yaml

    tier_path = configs_dir / "tiers" / f"{tier_name}.yaml"
    if not tier_path.exists():
        raise FileNotFoundError(f"Tier config not found: {tier_path}")

    with open(tier_path) as f:
        tier_config = _yaml.safe_load(f)

    for key, value in tier_config.items():
        if key == "max_steps":
            # max_steps is handled via args, not config
            if getattr(args, "max_steps", None) is None:
                args.max_steps = value
        elif key in tier_config_map:
            section, attr = tier_config_map[key]
            setattr(getattr(config, section), attr, value)

    print(f"Applied '{tier_name}' tier preset: {tier_config}")
    return tier_config
