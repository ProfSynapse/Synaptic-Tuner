#!/usr/bin/env python3
"""
GRPO / GSPO Training Script for RTX 3090 (24GB VRAM)

This trainer is YAML-driven (configs/config.yaml) and supports:
  - GRPO (default)
  - GSPO (set training.use_gspo=true)

Usage:
  python train_grpo.py
  python train_grpo.py --config ./configs/config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================================================
# GRPO is not supported on native Windows (use WSL2 or Linux)
# ============================================================================
if sys.platform == "win32":
    print("GRPO/GSPO training is not supported on native Windows.")
    print("Please use WSL2 (Ubuntu) or Linux.")
    sys.exit(1)

# ============================================================================
# Disable torch.compile (stability + compatibility)
# Must be set BEFORE importing torch/unsloth in many environments
# ============================================================================
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

# Load .env file for API keys (HF_TOKEN, WANDB_API_KEY)
try:  # noqa: E402
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unsloth import is_bfloat16_supported  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from configs.config_loader import load_config  # noqa: E402
from src.data_loader import load_raw_dataset, format_dataset_for_grpo, print_dataset_samples  # noqa: E402
from src.model_loader import (  # noqa: E402
    load_model_and_tokenizer,
    get_text_tokenizer,
    apply_lora_adapters,
    check_gpu_memory,
)
from src.rewards import build_combined_reward_function  # noqa: E402
from src.training_callbacks import MetricsTableCallback  # noqa: E402


def setup_wandb() -> bool:
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        return False
    try:
        import wandb  # type: ignore

        wandb.login(key=wandb_key, relogin=True, force=True)
        print("✓ W&B: Logged in automatically (using WANDB_API_KEY from .env)")
        return True
    except Exception as e:
        print(f"⚠ W&B: Login failed ({e})")
        return False


def _detect_chat_template(model_name: str) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "chatml"
    if "llama" in name:
        return "llama-3"
    if "mistral" in name:
        return "mistral"
    if "gemma" in name:
        return "gemma"
    if "phi" in name:
        return "phi-3"
    if "deepseek" in name:
        return "chatml"
    return "chatml"


def _build_grpo_config(config, checkpoints_dir: Path) -> GRPOConfig:
    import inspect

    bf16_supported = is_bfloat16_supported()
    bf16 = bool(config.training.bf16 and bf16_supported)
    fp16 = bool(config.training.fp16 and not bf16)
    if not bf16 and not fp16:
        fp16 = not bf16_supported

    args: Dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "per_device_train_batch_size": int(config.training.per_device_train_batch_size),
        "gradient_accumulation_steps": int(config.training.gradient_accumulation_steps),
        "num_generations": int(config.training.num_generations),
        "max_prompt_length": int(config.training.max_prompt_length),
        "max_completion_length": int(config.training.max_completion_length),
        "temperature": float(config.training.temperature),
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "warmup_ratio": float(config.training.warmup_ratio),
        "lr_scheduler_type": str(config.training.lr_scheduler_type),
        "optim": str(config.training.optim),
        "logging_steps": int(config.training.logging_steps),
        "save_steps": int(config.training.save_steps),
        "save_total_limit": int(config.training.save_total_limit),
        "num_train_epochs": int(config.training.num_train_epochs),
        "fp16": fp16,
        "bf16": bf16,
        "seed": int(config.seed),
        "report_to": str(config.training.report_to),
    }

    if int(config.training.max_steps) and int(config.training.max_steps) > 0:
        args["max_steps"] = int(config.training.max_steps)

    # GSPO toggle uses sequence-level importance sampling.
    if bool(config.training.use_gspo):
        args["importance_sampling_level"] = "sequence"

    # Optional pass-through args (only if supported by GRPOConfig)
    extra_args = config.training.extra_args or {}
    if not isinstance(extra_args, dict):
        raise TypeError("training.extra_args must be a mapping/dict")
    args.update(extra_args)

    # Filter to supported GRPOConfig kwargs to avoid version mismatches.
    sig = inspect.signature(GRPOConfig.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in args.items() if k in allowed}

    dropped = sorted(set(args.keys()) - set(filtered.keys()))
    if dropped:
        print(f"ℹ Dropping unsupported GRPOConfig args: {dropped}")

    return GRPOConfig(**filtered)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTX3090 GRPO/GSPO trainer (YAML-driven)")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (defaults to configs/config.yaml)")
    p.add_argument("--dry-run", action="store_true", help="Setup and validate, but do not train")
    p.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    # Minimal overrides (still config-first)
    p.add_argument("--model-name", type=str, default=None, help="Override model.model_name")
    p.add_argument("--dataset-name", type=str, default=None, help="Override dataset.dataset_name")
    p.add_argument("--dataset-file", type=str, default=None, help="Override dataset.dataset_file")
    p.add_argument("--local-file", type=str, default=None, help="Override dataset.local_file")
    p.add_argument("--use-gspo", action="store_true", help="Override training.use_gspo=true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)

    if args.model_name:
        config.model.model_name = args.model_name
    if args.dataset_name:
        config.dataset.dataset_name = args.dataset_name
    if args.dataset_file:
        config.dataset.dataset_file = args.dataset_file
    if args.local_file:
        config.dataset.local_file = args.local_file
    if args.use_gspo:
        config.training.use_gspo = True

    if config.wandb.enabled:
        config.use_wandb = setup_wandb()
        if config.use_wandb:
            config.training.report_to = "wandb"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(config.training.output_dir)
    run_dir = base_output_dir / timestamp

    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training run directory: {run_dir}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Logs: {logs_dir}\n")

    # Load dataset (raw, then formatted after chat template applied)
    raw_dataset = load_raw_dataset(
        dataset_name=config.dataset.dataset_name if not config.dataset.local_file else None,
        data_files=config.dataset.dataset_file if not config.dataset.local_file else None,
        local_file=config.dataset.local_file,
        num_proc=config.dataset.num_proc,
    )
    print_dataset_samples(raw_dataset, num_samples=2)

    # Load model + tokenizer/processor
    model, tok_or_proc, is_vl = load_model_and_tokenizer(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        dtype=config.model.dtype,
        load_in_4bit=config.model.load_in_4bit,
        hf_token=hf_token,
    )
    tokenizer = get_text_tokenizer(tok_or_proc)

    # Apply chat template (config override wins; else inferred)
    chat_template_name = config.model.chat_template or _detect_chat_template(config.model.model_name)
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
    print(f"✓ Applied {chat_template_name} chat template via Unsloth")

    # Apply LoRA
    model = apply_lora_adapters(
        model=model,
        is_vision_model=is_vl,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
        random_state=config.lora.random_state,
        use_rslora=getattr(config.lora, "use_rslora", False),
        use_dora=getattr(config.lora, "use_dora", False),
    )
    check_gpu_memory()

    # Format dataset for GRPO (prompt -> string using chat template)
    formatted_dataset = format_dataset_for_grpo(
        raw_dataset,
        tokenizer=tokenizer,
        prompt_column=config.dataset.prompt_column,
        num_proc=config.dataset.num_proc,
    )

    # Rewards
    reward_fn, reward_plan = build_combined_reward_function(
        rewards_config=config.rewards,
        base_dir=Path(__file__).parent,
    )
    print("\nReward configuration:")
    for item in reward_plan:
        print(f"  - {item['type']}: {item['name']} (weight={item['weight']})")

    # Training args
    training_args = _build_grpo_config(config, checkpoints_dir=checkpoints_dir)

    print("\n" + "=" * 60)
    print("GRPO TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Mode: {'GSPO' if config.training.use_gspo else 'GRPO'}")
    print(f"Model: {config.model.model_name}")
    print(f"Dataset: {len(formatted_dataset)} examples")
    print(f"Output: {checkpoints_dir}")
    print(f"Batch: {config.training.per_device_train_batch_size} x {config.training.gradient_accumulation_steps}")
    print(f"Generations per prompt: {config.training.num_generations}")
    print(f"Max prompt len: {config.training.max_prompt_length}")
    print(f"Max completion len: {config.training.max_completion_length}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Report to: {config.training.report_to}")
    print("=" * 60 + "\n")

    if args.dry_run:
        print("[OK] Dry run completed. Exiting without training.")
        return {"run_dir": str(run_dir), "dry_run": True}

    callbacks = [
        MetricsTableCallback(
            log_every_n_steps=config.training.logging_steps,
            output_dir=str(run_dir),
        )
    ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=callbacks,
    )

    print("Starting training...\n")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_model_path = run_dir / "final_model"
    final_model_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_path))
    try:
        tokenizer.save_pretrained(str(final_model_path))
    except Exception:
        pass

    print("\n[OK] Training complete!")
    print(f"  Final model: {final_model_path}")
    print(f"  Logs: {logs_dir}")
    return {"run_dir": str(run_dir), "final_model_dir": str(final_model_path)}


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
