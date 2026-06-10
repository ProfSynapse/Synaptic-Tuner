#!/usr/bin/env python3
"""
DPO Training Script (mirrors Trainers/kto/train_kto.py)

Direct Preference Optimization with TRL's DPOTrainer/DPOConfig, built by
mirroring the KTO trainer so it inherits the same env bootstrap, model loader,
shared callbacks, cloud-artifact plumbing, and lineage/tracking. The deltas vs
KTO are: TRL DPOConfig/DPOTrainer in place of KTOConfig/KTOTrainer; a
prompt/chosen/rejected preference-pair data loader (no True/False interleaving);
no KTO-S sign-correction variant and no large-vocab forward monkey-patch (both
KTO-specific); and the DPO config surface (loss_type, no per-class weights).

Usage:
    python train_dpo.py --qwen3-4b --local-file ../../path/to/dpo_train.jsonl
    python train_dpo.py --model-name unsloth/Qwen3-8B-Instruct-bnb-4bit --dry-run

DEVIATION FROM KTO (flagged): the --dry-run exit is placed BEFORE model load
(KTO exits after loading the model). A dry-run here validates config + data +
preset resolution without downloading any model, which is the contract this
trainer's smoke test relies on.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src and repo root to path before imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Config + data + preset resolution are import-light (yaml/datasets only) and do
# not pull in torch/unsloth/trl, so a --dry-run can run without the ML stack.
from configs.config_loader import Config, load_config
from configs.model_presets import resolve_model_flag
from src.data_loader import (
    load_and_prepare_dataset,
    validate_dpo_dataset,
    print_dataset_samples,
)


def setup_environment():
    """Setup DPO-specific environment variables and print training banner."""
    import torch
    from unsloth import is_bfloat16_supported
    from shared.env_bootstrap import suppress_transformers_logging

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # CUDA MEMORY OPTIMIZATION: reduce fragmentation by consolidating allocations.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    suppress_transformers_logging()

    print("=" * 60)
    print("DPO TRAINING")
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


def build_training_lineage(
    config,
    train_dataset,
    eval_dataset,
    trainer,
    run_dir: Path,
    args: argparse.Namespace,
    training_time_seconds: Optional[float] = None,
    final_loss: Optional[float] = None
) -> Dict[str, Any]:
    """Build DPO training lineage using shared base + DPO-specific fields."""
    import torch
    from shared.training_utils import build_base_lineage
    from shared.experiment_tracking.lineage_enrichment import enrich_training_lineage

    dataset_source = args.local_file or config.dataset.local_file
    if not dataset_source:
        dataset_source = f"{config.dataset.dataset_name}/{config.dataset.dataset_file}"

    lineage = build_base_lineage(
        training_type="DPO",
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
            "max_length": config.training.max_length,
            "max_prompt_length": config.training.max_prompt_length,
            "gradient_checkpointing": config.training.gradient_checkpointing,
            "fp16": not torch.cuda.is_bf16_supported() if config.training.fp16 is False else config.training.fp16,
            "bf16": torch.cuda.is_bf16_supported() if config.training.bf16 is True else config.training.bf16,
            "seed": config.seed,
            # DPO-specific parameters
            "beta": config.training.beta,
            "loss_type": config.training.loss_type,
        },
        dataset_info={
            "source": dataset_source,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset) if eval_dataset else 0,
        },
        run_dir=run_dir,
        trainer=trainer,
        training_time_seconds=training_time_seconds,
    )

    if config.training.use_two_stage_lr:
        lineage["training"]["two_stage_lr"] = {
            "enabled": True,
            "initial_lr": config.training.learning_rate,
            "reduced_lr": config.training.learning_rate * config.training.lr_reduction_factor,
            "reduction_step": config.training.lr_reduction_step,
            "reduction_factor": config.training.lr_reduction_factor,
        }

    if final_loss is not None:
        lineage["results"]["final_loss"] = final_loss

    return enrich_training_lineage(lineage, args=args)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the DPO trainer argument parser.

    Factored out of main() so the CLI surface (model presets, LoRA budget knobs,
    cloud-artifact flags) can be exercised by the dry-run smoke test without
    importing the ML stack.
    """
    parser = argparse.ArgumentParser(description="DPO Training")

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
        help="Override model name (e.g., unsloth/Qwen3-8B-Instruct-bnb-4bit)"
    )

    # Friendly model selection shortcuts (dests match configs/model_presets.MODEL_MAP keys)
    # 3-4B models (fast iteration)
    parser.add_argument("--qwen-3b", action="store_true", help="Use Qwen2.5 3B Instruct (fast iteration)")
    parser.add_argument("--llama-3b", action="store_true", help="Use Llama 3.2 3B Instruct (fast iteration)")
    parser.add_argument("--qwen3-4b", action="store_true", help="Use Qwen3 4B Instruct (Phase 1 pilot pin)")

    # 7-8B models (production quality)
    parser.add_argument("--mistral-7b", action="store_true", help="Use Mistral 7B v0.3 (production quality)")
    parser.add_argument("--llama-8b", action="store_true", help="Use Llama 3.1 8B Instruct")
    parser.add_argument("--qwen-7b", action="store_true", help="Use Qwen2.5 7B Instruct")
    parser.add_argument("--qwen3-8b", action="store_true", help="Use Qwen3 8B Instruct (Phase 1 confirm pin)")

    # 11-14B models (advanced)
    parser.add_argument("--llama-13b", action="store_true", help="Use Llama 2 13B (advanced)")
    parser.add_argument("--gemma-12b", action="store_true", help="Use Gemma 3 12B Instruct")

    # 17-24B models (very large)
    parser.add_argument("--gpt-20b", action="store_true", help="Use GPT-OSS 20B (very large)")
    parser.add_argument("--mistral-24b", action="store_true", help="Use Mistral Small 3.2 24B Instruct (extremely large)")

    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length")

    # Dataset configuration
    parser.add_argument("--dataset-name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--dataset-file", type=str, help="Dataset file within HuggingFace dataset")
    parser.add_argument("--local-file", type=str, help="Path to local JSONL file (prompt/chosen/rejected)")
    parser.add_argument("--split-dataset", action="store_true", help="Create train/validation split")

    # Training configuration
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--output-root", type=str, help="Override root directory for training outputs")
    parser.add_argument(
        "--cloud-provider", type=str, choices=["hf_jobs", "modal", "runpod"],
        help="Cloud provider identifier for canonical cloud run layout"
    )
    parser.add_argument(
        "--artifact-backend", type=str,
        choices=["hf_bucket", "modal_volume", "runpod_network_volume"],
        help="Provider-native artifact backend"
    )
    parser.add_argument("--artifact-bucket", type=str, help="Hugging Face Bucket identifier used by HF Jobs")
    parser.add_argument("--artifact-prefix", type=str, help="Bucket prefix for this run")
    parser.add_argument("--publish-final-model", action="store_true", help="Publish final_model to Hugging Face Hub after training")
    parser.add_argument("--publish-target-repo", type=str, help="Target Hugging Face model repo when publishing final_model")
    parser.add_argument("--run-timestamp", type=str, help="Explicit run timestamp for canonical cloud run layout")
    parser.add_argument("--batch-size", type=int, help="Override per_device_train_batch_size")
    parser.add_argument("--gradient-accumulation", type=int, help="Override gradient_accumulation_steps")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--seed", type=int, help="Override the training random seed (config.seed)")
    parser.add_argument("--beta", type=float, help="Override DPO beta parameter (controls KL regularization strength)")
    parser.add_argument("--loss-type", type=str, help="Override DPO loss variant (default: sigmoid = vanilla DPO)")
    parser.add_argument("--num-epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--max-steps", type=int, help="Override max training steps (takes precedence over epochs)")

    # Two-stage learning rate schedule
    parser.add_argument("--two-stage-lr", action="store_true", help="Enable two-stage learning rate schedule (reduces LR at specified step)")
    parser.add_argument("--lr-reduction-step", type=int, default=50, help="Step at which to reduce learning rate (default: 50)")
    parser.add_argument("--lr-reduction-factor", type=float, default=0.5, help="Factor to multiply LR by at reduction step (default: 0.5)")

    # Experiment tracking
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="dpo-finetuning", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")

    # LoRA budget (same surface as KTO/SFT, so the recipe's LoRA budget flows
    # end-to-end rather than silently falling back to the trainer config default).
    parser.add_argument("--lora-r", type=int, help="Override LoRA rank (config.lora.r)")
    parser.add_argument("--lora-alpha", type=int, help="Override LoRA alpha (config.lora.lora_alpha)")
    parser.add_argument("--lora-dropout", type=float, help="Override LoRA dropout (config.lora.lora_dropout)")
    parser.add_argument("--lora-target-modules", type=str, help="Override LoRA target modules (comma-separated)")
    parser.add_argument("--init-lora-weights", type=str, help="Override LoRA weight initialization scheme")

    # LoRA technique variants (same budget surface as KTO/SFT)
    parser.add_argument("--use-dora", action="store_true", help="Enable DoRA (Weight-Decomposed LoRA). Passes through to PEFT via Unsloth kwargs.")
    parser.add_argument("--use-rslora", action="store_true", help="Enable rsLoRA (rank-stabilized scaling). Recommended at r>=128.")

    # Tier preset
    parser.add_argument(
        "--tier", choices=["quick", "standard", "thorough"],
        help="Preset complexity tier. Overrides individual LoRA/training hyperparams. "
             "Explicit flags still override tier defaults."
    )

    # Other options
    parser.add_argument("--hf-token", type=str, help="HuggingFace token for gated models")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Path to checkpoint directory to resume training from")
    parser.add_argument("--dry-run", action="store_true", help="Setup and validate config/data without training (no model download)")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging to diagnose freezes/hangs")

    return parser


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Resolve presets and apply CLI overrides onto the loaded config.

    Import-light (no ML stack) so the dry-run smoke test can verify preset
    resolution and override precedence. Mirrors KTO's override block minus the
    KTO-only knobs, and applies tier presets via the shared helper.
    """
    from shared.training_utils import apply_tier_preset

    # Resolve a friendly model flag (e.g. --qwen3-4b) to (size, repo)
    resolved = resolve_model_flag(args)
    if resolved is not None:
        args.model_size, args.model_name = resolved

    # Tier preset (overrides base config; explicit CLI flags override tier)
    if args.tier:
        _dpo_tier_config_map = {
            "r": ("lora", "r"),
            "lora_alpha": ("lora", "lora_alpha"),
            "use_dora": ("lora", "use_dora"),
            "use_rslora": ("lora", "use_rslora"),
            "target_modules": ("lora", "target_modules"),
            "learning_rate": ("training", "learning_rate"),
            "num_train_epochs": ("training", "num_train_epochs"),
            "warmup_ratio": ("training", "warmup_ratio"),
            "batch_size": ("training", "per_device_train_batch_size"),
            "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
        }
        apply_tier_preset(
            config, args.tier, _dpo_tier_config_map, args,
            configs_dir=Path(__file__).parent / "configs",
        )

    # LoRA technique CLI overrides (after tier, so they take precedence)
    if args.use_dora:
        config.lora.use_dora = True
    if args.use_rslora:
        config.lora.use_rslora = True

    # Command-line overrides
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

    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    # is not None so seed=0 and beta=0.0 are honored, not silently swapped for the
    # config default — the handler forwards explicit zeros (provenance: no silent override).
    if args.seed is not None:
        config.seed = args.seed
    if args.beta is not None:
        config.training.beta = args.beta
    if args.loss_type:
        config.training.loss_type = args.loss_type
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs

    # LoRA budget overrides. is not None so an explicit 0 (e.g. --lora-dropout 0.0)
    # is honored, not silently dropped — the recipe's LoRA budget is the SSOT.
    if args.lora_r is not None:
        config.lora.r = args.lora_r
    if args.lora_alpha is not None:
        config.lora.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        config.lora.lora_dropout = args.lora_dropout
    if args.lora_target_modules is not None:
        config.lora.target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if args.init_lora_weights is not None:
        config.lora.init_lora_weights = args.init_lora_weights

    if args.two_stage_lr:
        config.training.use_two_stage_lr = True
    if args.lr_reduction_step:
        config.training.lr_reduction_step = args.lr_reduction_step
    if args.lr_reduction_factor:
        config.training.lr_reduction_factor = args.lr_reduction_factor

    return config


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Environment bootstrap — must run before importing torch/unsloth/transformers.
    from shared.env_bootstrap import init_trainer_env
    init_trainer_env()

    setup_environment()

    # Auto-setup W&B if API key present in .env
    from shared.training_utils import setup_wandb
    wandb_auto_enabled = setup_wandb()

    print("Loading configuration from configs/config.yaml\n")
    config = load_config()
    config = apply_cli_overrides(config, args)

    # Auto-enable W&B if API key found or --wandb flag used
    if wandb_auto_enabled or args.wandb:
        config.use_wandb = True
        if args.wandb_project:
            config.wandb_project = args.wandb_project
        elif not getattr(config, "wandb_project", None):
            config.wandb_project = "dpo-training"
        if args.wandb_run_name:
            config.wandb_run_name = args.wandb_run_name
        elif not getattr(config, "wandb_run_name", None):
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            config.wandb_run_name = f"{args.model_size or 'dpo'}-{ts}"

    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")

    # Cloud-artifact run layout
    from datetime import datetime
    from shared.cloud_artifacts import build_manifest, build_run_paths, write_manifest

    repo_branch = os.environ.get("CLOUD_REPO_BRANCH")
    repo_commit = os.environ.get("CLOUD_REPO_COMMIT")
    artifact_identifier = args.artifact_bucket or os.environ.get("CLOUD_ARTIFACT_IDENTIFIER")

    timestamp = args.run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_root or args.output_dir or config.training.output_dir)
    manifest_path = None
    run_paths = None
    if args.cloud_provider:
        run_paths = build_run_paths(
            base_output_dir=base_output_dir,
            provider=args.cloud_provider,
            method="dpo",
            timestamp=timestamp,
            commit=repo_commit or "local",
        )
        run_dir = run_paths.run_dir
        checkpoints_dir = run_paths.checkpoints_dir
        logs_dir = run_paths.logs_dir
        output_path = run_paths.final_model_dir
        manifest_path = run_paths.manifest_path
        for path in (run_dir, checkpoints_dir, logs_dir):
            path.mkdir(parents=True, exist_ok=True)
        write_manifest(
            manifest_path,
            build_manifest(
                provider=args.cloud_provider,
                method="dpo",
                artifact_backend=args.artifact_backend or "",
                artifact_identifier=artifact_identifier,
                run_paths=run_paths,
                repo_branch=repo_branch,
                repo_commit=repo_commit,
                publish_final_model=args.publish_final_model,
                publish_target_repo=args.publish_target_repo,
                status="running",
            ),
        )
    else:
        run_dir = base_output_dir / timestamp
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        output_path = run_dir / "final_model"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

    config.training.output_dir = str(checkpoints_dir)
    print(f"Training run directory: {run_dir}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Logs: {logs_dir}\n")

    # Load dataset - prioritize local_file from args, then config, then HuggingFace
    local_file_path = args.local_file or config.dataset.local_file
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=config.dataset.dataset_name if not local_file_path else None,
        data_files=config.dataset.dataset_file if not local_file_path else None,
        local_file=local_file_path,
        num_proc=config.dataset.num_proc,
        test_size=config.dataset.test_size,
        split_dataset=args.split_dataset,
    )

    # Validate dataset (prompt/chosen/rejected structure). DPO is paired and
    # unweighted, so there is no interleaving step (contrast KTO).
    if not validate_dpo_dataset(train_dataset):
        print("✗ Dataset validation failed. Exiting.")
        return

    print_dataset_samples(train_dataset, num_samples=2)

    # Print training configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.model.model_name}")
    print(f"Output directory: {config.training.output_dir}")
    print(f"Dataset: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Validation: {len(eval_dataset)} examples")
    effective_batch = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    print(f"\nBatch configuration:")
    print(f"  Batch size: {config.training.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Beta: {config.training.beta}")
    print(f"  Loss type: {config.training.loss_type}")
    print(f"  Warmup ratio: {config.training.warmup_ratio}")
    print(f"  Max length: {config.training.max_length}")
    print(f"\nLoRA configuration:")
    print(f"  Rank: {config.lora.r}")
    print(f"  Alpha: {config.lora.lora_alpha}")
    print(f"  Dropout: {config.lora.lora_dropout}")
    print("=" * 60 + "\n")

    # DEVIATION FROM KTO: exit on dry-run BEFORE any model load, so a dry-run
    # never downloads a model. This is the smoke-test contract.
    if args.dry_run:
        print("✓ Dry run completed (config + data + presets validated). Exiting without model load or training.")
        return

    # ---- Heavy path: only reached for a real run (model load + training) ----
    import torch
    from unsloth import is_bfloat16_supported
    from trl import DPOConfig, DPOTrainer

    from src.model_loader import (
        load_model_and_tokenizer,
        apply_lora_adapters,
        create_reference_model,
        check_gpu_memory,
    )
    from src.training_callbacks import (
        LiveDashboardCallback,
        MetricsTableCallback,
        CheckpointMonitorCallback,
        TwoStageLRCallback,
        DASHBOARD_AVAILABLE,
        RICH_AVAILABLE,
    )
    from shared.cloud_artifacts import (
        HFBucketSyncCallback,
        publish_final_model_to_hub,
        sync_directory_to_hf_bucket,
    )
    from shared.training_utils import (
        extract_previous_log_entries,
        save_training_lineage,
    )

    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        dtype=config.model.dtype,
        load_in_4bit=config.model.load_in_4bit,
        hf_token=args.hf_token,
    )

    # Reference model: like KTO, default to TRL's implicit (shared-weight)
    # reference to save VRAM; create an explicit frozen copy only when asked.
    ref_model = None
    if os.getenv("USE_EXPLICIT_REF_MODEL", "false").lower() == "true":
        print("\n⚠️  Creating explicit reference model (uses extra VRAM)")
        ref_model = create_reference_model(
            model_name=config.model.model_name,
            max_seq_length=config.model.max_seq_length,
            dtype=config.model.dtype,
            load_in_4bit=config.model.load_in_4bit,
            hf_token=args.hf_token,
        )
    else:
        print("\n✓ Using implicit reference model (TRL manages internally)")

    # Apply LoRA adapters to policy model only (not reference)
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
    )

    check_gpu_memory()

    training_args = DPOConfig(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        beta=config.training.beta,
        loss_type=config.training.loss_type,
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

    previous_log_entries = None
    if args.resume_from_checkpoint:
        previous_log_entries = extract_previous_log_entries(args.resume_from_checkpoint)

    use_dashboard = DASHBOARD_AVAILABLE and RICH_AVAILABLE
    if use_dashboard:
        callbacks = [
            LiveDashboardCallback(
                log_every_n_steps=5,
                output_dir=str(run_dir),
                previous_log_entries=previous_log_entries,
            ),
        ]
    else:
        callbacks = [
            MetricsTableCallback(
                log_every_n_steps=5,
                output_dir=str(run_dir),
                previous_log_entries=previous_log_entries,
            ),
            CheckpointMonitorCallback(),
        ]
    if args.artifact_backend == "hf_bucket" and args.artifact_bucket and args.artifact_prefix:
        callbacks.append(
            HFBucketSyncCallback(
                run_dir=run_dir,
                bucket_id=args.artifact_bucket,
                prefix=args.artifact_prefix,
                token=args.hf_token,
                log_every_n_steps=5,
            )
        )
    if config.training.use_two_stage_lr:
        reduced_lr = config.training.learning_rate * config.training.lr_reduction_factor
        callbacks.append(
            TwoStageLRCallback(
                initial_lr=config.training.learning_rate,
                reduced_lr=reduced_lr,
                reduction_step=config.training.lr_reduction_step,
            )
        )

    print("Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    print("✓ DPO trainer initialized with metrics tracking")

    if use_dashboard:
        from transformers.trainer_callback import PrinterCallback
        trainer.remove_callback(PrinterCallback)

    print("=" * 60)
    if args.resume_from_checkpoint:
        print(f"RESUMING TRAINING FROM: {args.resume_from_checkpoint}")
    else:
        print("STARTING TRAINING")
    print("=" * 60 + "\n")

    training_start_time = time.time()
    final_loss = None
    try:
        trainer_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        final_loss = trainer_output.training_loss
        print("\n✓ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final loss: {final_loss:.4f}")
        check_gpu_memory()
    except Exception as e:
        print("\n✗ TRAINING FAILED")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        if manifest_path:
            write_manifest(
                manifest_path,
                build_manifest(
                    provider=args.cloud_provider or "local",
                    method="dpo",
                    artifact_backend=args.artifact_backend or "",
                    artifact_identifier=artifact_identifier,
                    run_paths=run_paths,
                    repo_branch=repo_branch,
                    repo_commit=repo_commit,
                    publish_final_model=args.publish_final_model,
                    publish_target_repo=args.publish_target_repo,
                    status=f"failed: {type(e).__name__}",
                ),
            )
            if args.artifact_backend == "hf_bucket" and args.artifact_bucket and args.artifact_prefix:
                sync_directory_to_hf_bucket(run_dir, args.artifact_bucket, args.artifact_prefix, token=args.hf_token)
        raise

    print("\nSAVING MODEL")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"✓ Model saved to: {output_path}")

    training_time_seconds = time.time() - training_start_time

    print("\nBuilding training lineage...")
    lineage = build_training_lineage(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer=trainer,
        run_dir=run_dir,
        args=args,
        training_time_seconds=training_time_seconds,
        final_loss=final_loss,
    )
    save_training_lineage(lineage, run_dir)

    # Register in unified experiment tracking registry (best-effort)
    try:
        from shared.experiment_tracking.adapters import dpo_lineage_to_run_record
        from shared.experiment_tracking.registry import RunRegistry

        is_cloud = getattr(args, "cloud_provider", None) is not None
        record = dpo_lineage_to_run_record(lineage, str(run_dir), cloud=is_cloud)
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
        publish_final_model_to_hub(output_path, args.publish_target_repo, args.hf_token)

    if manifest_path:
        write_manifest(
            manifest_path,
            build_manifest(
                provider=args.cloud_provider,
                method="dpo",
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

    print("\n" + "=" * 60)
    print("✓ ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
