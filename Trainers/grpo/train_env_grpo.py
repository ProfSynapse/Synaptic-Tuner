#!/usr/bin/env python3
"""Cloud-first environment-backed GRPO entrypoint.

This path is separate from the existing static projected-dataset GRPO trainer.
It is designed for a newer TRL/OpenEnv stack running inside an isolated
virtualenv on top of the Unsloth Docker image.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict

import yaml

# Add trainer src to path for direct execution.
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.env_dataset import (
    filter_env_rollout_dataset,
    format_dataset_for_env_grpo,
    load_env_rollout_dataset,
)
from src.env_rewards import build_env_reward_function
from src.env_rollout import build_prompt_registry, build_rollout_func
from src.env_runtime import build_cloud_bootstrap_commands, detect_openenv_runtime_support
from src.training_callbacks import DASHBOARD_AVAILABLE, RICH_AVAILABLE, LiveDashboardCallback, MetricsTableCallback


def _load_configured_tokenizer(model_cfg: Dict[str, Any], model_name: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer_cfg = model_cfg.get("tokenizer")
    if tokenizer_cfg is None:
        tokenizer_cfg = {}
    elif not isinstance(tokenizer_cfg, dict):
        raise ValueError("model.tokenizer must be a mapping when provided")

    tokenizer_name = str(
        tokenizer_cfg.get("path")
        or tokenizer_cfg.get("name")
        or model_cfg.get("tokenizer_path")
        or model_cfg.get("tokenizer_name")
        or model_name
    ).strip()
    if not tokenizer_name:
        raise RuntimeError("Tokenizer source resolved to an empty path/name")

    tokenizer_kwargs = dict(tokenizer_cfg.get("kwargs") or {})
    for key in ("trust_remote_code", "use_fast", "padding_side", "model_max_length"):
        if key in tokenizer_cfg and key not in tokenizer_kwargs:
            tokenizer_kwargs[key] = tokenizer_cfg[key]

    loader = str(tokenizer_cfg.get("loader") or model_cfg.get("tokenizer_loader") or "auto").strip().lower()
    if loader == "auto":
        return AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if loader in {"pretrained_fast", "pretrained-tokenizer-fast", "fast"}:
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    raise ValueError(f"Unsupported model.tokenizer.loader: {loader}")


def _build_peft_config(config: Dict[str, Any]):
    lora_cfg = config.get("lora") or config.get("peft")
    if not lora_cfg:
        return None
    if not isinstance(lora_cfg, dict):
        raise ValueError("lora/peft config must be a mapping when provided")
    if lora_cfg.get("enabled") is False:
        return None

    from peft import LoraConfig

    kwargs = {
        "r": int(lora_cfg.get("r", 16)),
        "lora_alpha": int(lora_cfg.get("lora_alpha", lora_cfg.get("alpha", 32))),
        "lora_dropout": float(lora_cfg.get("lora_dropout", lora_cfg.get("dropout", 0.0))),
        "bias": str(lora_cfg.get("bias", "none")),
        "task_type": str(lora_cfg.get("task_type", "CAUSAL_LM")),
    }
    target_modules = lora_cfg.get("target_modules")
    if target_modules:
        kwargs["target_modules"] = list(target_modules)
    for key in ("use_rslora", "use_dora", "modules_to_save"):
        if key in lora_cfg:
            kwargs[key] = lora_cfg[key]
    return LoraConfig(**kwargs)


def _build_grpo_trainer_class(base_trainer_cls):
    """Return a trainer class that honors rollout_func outside vLLM when configured."""

    class ConfiguredRolloutGRPOTrainer(base_trainer_cls):
        def _generate_single_turn(self, prompts: list):
            if self.rollout_func is None or self.use_vllm:
                return super()._generate_single_turn(prompts)

            outputs = self.rollout_func(prompts, self)
            if not isinstance(outputs, dict):
                raise RuntimeError("Configured rollout_func returned invalid output shape")

            required_fields = ("prompt_ids", "completion_ids", "logprobs")
            missing = [field for field in required_fields if field not in outputs]
            if missing:
                raise RuntimeError(f"Configured rollout_func missing required fields: {missing}")

            extra_fields = {key: value for key, value in outputs.items() if key not in required_fields}
            return (
                outputs["prompt_ids"],
                outputs["completion_ids"],
                outputs.get("logprobs"),
                extra_fields,
            )

    return ConfiguredRolloutGRPOTrainer


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    if config_path is None:
        raise ValueError("Env-GRPO requires an explicit --config path to a run-specific training YAML")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_config_path"] = str(Path(config_path).resolve())
    return config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud-first env-backed GRPO launcher")
    parser.add_argument("--config", type=str, required=True, help="Path to env-GRPO run config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/runtime/dataset and exit")
    parser.add_argument(
        "--print-cloud-bootstrap",
        action="store_true",
        help="Print shell commands for the isolated cloud runtime bootstrap",
    )
    parser.add_argument("--max-examples", type=int, default=0, help="Limit dataset rows during validation")
    parser.add_argument("--model-name", type=str, default=None, help="Override model.model_name")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override dataset.dataset_name")
    parser.add_argument("--dataset-file", type=str, default=None, help="Override dataset.dataset_file")
    parser.add_argument("--local-file", type=str, default=None, help="Override dataset.local_file")
    parser.add_argument("--output-dir", type=str, default=None, help="Override the run output directory")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.per_device_train_batch_size")
    parser.add_argument("--gradient-accumulation", type=int, default=None, help="Override training.gradient_accumulation_steps")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override training.learning_rate")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override training.num_train_epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override model.max_seq_length")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")

    model_cfg = config.get("model") or {}
    dataset_cfg = config.get("dataset") or {}
    training_cfg = config.get("training") or {}

    if args.model_name:
        model_cfg["model_name"] = args.model_name
    if args.dataset_name:
        dataset_cfg["dataset_name"] = args.dataset_name
    if args.dataset_file:
        dataset_cfg["dataset_file"] = args.dataset_file
    if args.local_file:
        dataset_cfg["local_file"] = args.local_file
    if args.batch_size is not None:
        training_cfg["per_device_train_batch_size"] = args.batch_size
    if args.gradient_accumulation is not None:
        training_cfg["gradient_accumulation_steps"] = args.gradient_accumulation
    if args.learning_rate is not None:
        training_cfg["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        training_cfg["num_train_epochs"] = args.num_epochs
    if args.max_steps is not None:
        training_cfg["max_steps"] = args.max_steps
    if args.max_seq_length is not None:
        model_cfg["max_seq_length"] = args.max_seq_length

    if args.print_cloud_bootstrap:
        runtime_cfg = ((config.get("env_training") or {}).get("runtime") or {})
        cloud_repo_root = str(runtime_cfg.get("repo_root_in_container") or "/workspace/repo")
        local_config_path = Path(str(config.get("_config_path") or "")).resolve()
        try:
            relative_config_path = local_config_path.relative_to(Path(__file__).resolve().parents[2])
        except ValueError:
            relative_config_path = Path("Trainers") / "grpo" / "configs" / local_config_path.name
        cloud_config_path = str(PurePosixPath(cloud_repo_root, *relative_config_path.parts))
        commands = build_cloud_bootstrap_commands(
            config,
            repo_root=cloud_repo_root,
            config_path=cloud_config_path,
        )
        print("\n".join(commands))
        return {"bootstrap_commands": commands}

    env_cfg = dict(config.get("env_training") or {})
    required_reviews = list((env_cfg.get("required_stage_reviews") or []))
    config_dir = Path(config["_config_path"]).parent if config.get("_config_path") else Path.cwd()
    local_file = dataset_cfg.get("local_file")
    if local_file:
        local_file = str((config_dir / str(local_file)).resolve())

    raw_dataset = load_env_rollout_dataset(
        dataset_name=dataset_cfg.get("dataset_name"),
        data_files=dataset_cfg.get("dataset_file"),
        local_file=local_file,
        num_proc=int(dataset_cfg.get("num_proc", 1)),
    )

    filtered_dataset = filter_env_rollout_dataset(
        raw_dataset,
        require_environment_passed=bool(env_cfg.get("require_environment_passed", True)),
        required_stage_reviews=required_reviews,
        require_environment_config=bool(env_cfg.get("require_environment_config", True)),
    )
    if args.max_examples and len(filtered_dataset) > args.max_examples:
        filtered_dataset = filtered_dataset.select(range(args.max_examples))
    formatted_dataset = format_dataset_for_env_grpo(
        filtered_dataset,
        prompt_message_roles=dataset_cfg.get("prompt_message_roles"),
        user_prompt_prefix=dataset_cfg.get("user_prompt_prefix"),
        user_prompt_suffix=dataset_cfg.get("user_prompt_suffix"),
    )

    runtime_support = detect_openenv_runtime_support()
    summary = {
        "raw_examples": len(raw_dataset),
        "filtered_examples": len(filtered_dataset),
        "formatted_examples": len(formatted_dataset),
        "prompt_message_roles": dataset_cfg.get("prompt_message_roles"),
        "runtime_support": runtime_support,
    }

    print("=" * 60)
    print("ENV-GRPO DRY RUN SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    if len(formatted_dataset) == 0:
        raise RuntimeError("No usable env rollout examples remained after filtering")

    sample = formatted_dataset[0]
    required_fields = ["prompt_messages", "resolved_environment_config", "task_context"]
    missing = [field for field in required_fields if not sample.get(field)]
    if missing:
        raise RuntimeError(f"Env-GRPO sample missing required fields: {missing}")

    if args.dry_run:
        return summary

    if not runtime_support.get("has_rollout_func"):
        raise RuntimeError(
            "Current runtime does not expose TRL rollout_func support. "
            "Use the isolated cloud runtime printed by --print-cloud-bootstrap."
        )

    model_name = str(model_cfg.get("model_name") or "").strip()
    if not model_name or model_name == "REPLACE_WITH_BUCKETED_SFT_MODEL":
        raise RuntimeError("Set model.model_name in the env-GRPO run config to the published bucketed SFT model repo")

    from trl import GRPOConfig, GRPOTrainer as BaseGRPOTrainer

    GRPOTrainer = _build_grpo_trainer_class(BaseGRPOTrainer)

    output_root = Path(training_cfg.get("output_dir") or "./env_grpo_output").resolve()
    run_dir = Path(args.output_dir).resolve() if args.output_dir else output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if env_cfg.get("log_trajectories") and not env_cfg.get("trajectory_log_path"):
        env_cfg["trajectory_log_path"] = str(logs_dir / "env_rollouts.jsonl")

    print("\n" + "=" * 60)
    print("ENV-GRPO TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Raw examples: {len(raw_dataset)}")
    print(f"Filtered examples: {len(filtered_dataset)}")
    print(f"Output: {run_dir}")
    print(
        "Batch: "
        f"{training_cfg.get('per_device_train_batch_size', 1)} x "
        f"{training_cfg.get('gradient_accumulation_steps', 1)}"
    )
    print(f"Generations per prompt: {training_cfg.get('num_generations', 4)}")
    print(f"Max prompt len: {training_cfg.get('max_prompt_length', 4096)}")
    print(f"Max completion len: {training_cfg.get('max_completion_length', 1024)}")
    print(f"Learning rate: {training_cfg.get('learning_rate', 5e-6)}")
    print("=" * 60 + "\n")

    tokenizer = _load_configured_tokenizer(model_cfg, model_name)
    peft_config = _build_peft_config(config)
    formatted_dataset = formatted_dataset.map(
        lambda ex: {
            **ex,
            "prompt": tokenizer.apply_chat_template(
                ex["prompt_messages"],
                tokenize=False,
                add_generation_prompt=True,
            ),
        },
        desc="Rendering chat prompts",
    )
    registry = build_prompt_registry(formatted_dataset)
    rollout_func = build_rollout_func(
        registry=registry,
        env_training_cfg=env_cfg,
    )
    reward_func = build_env_reward_function(config.get("rewards") or {})

    grpo_kwargs = {
        "output_dir": str(checkpoints_dir),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "num_generations": int(training_cfg.get("num_generations", 4)),
        "max_prompt_length": int(training_cfg.get("max_prompt_length", 4096)),
        "max_completion_length": int(training_cfg.get("max_completion_length", 1024)),
        "temperature": float(training_cfg.get("temperature", 0.9)),
        "learning_rate": float(training_cfg.get("learning_rate", 5e-6)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.05)),
        "lr_scheduler_type": str(training_cfg.get("lr_scheduler_type", "cosine")),
        "num_train_epochs": int(training_cfg.get("num_train_epochs", 1)),
        "max_steps": int(training_cfg.get("max_steps", -1)),
        "beta": float(training_cfg.get("beta", 0.04)),
        "logging_steps": int(training_cfg.get("logging_steps", 1)),
        "save_steps": int(training_cfg.get("save_steps", 25)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "report_to": str(training_cfg.get("report_to", "none")),
        "fp16": bool(training_cfg.get("fp16", False)),
        "bf16": bool(training_cfg.get("bf16", True)),
        "optim": str(training_cfg.get("optim", "adamw_torch")),
        "use_vllm": bool(training_cfg.get("use_vllm", False)),
        "vllm_mode": str(training_cfg.get("vllm_mode", "colocate")),
    }
    grpo_kwargs.update(dict(training_cfg.get("extra_args") or {}))
    allowed_grpo_args = set(inspect.signature(GRPOConfig.__init__).parameters) - {"self"}
    grpo_args = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in allowed_grpo_args})

    use_dashboard = DASHBOARD_AVAILABLE and RICH_AVAILABLE
    if use_dashboard:
        callbacks = [
            LiveDashboardCallback(
                log_every_n_steps=int(training_cfg.get("logging_steps", 1)),
                output_dir=str(run_dir),
            )
        ]
    else:
        callbacks = [
            MetricsTableCallback(
                log_every_n_steps=int(training_cfg.get("logging_steps", 1)),
                output_dir=str(run_dir),
            )
        ]

    trainer_kwargs = {
        "model": model_name,
        "processing_class": tokenizer,
        "reward_funcs": reward_func,
        "train_dataset": formatted_dataset,
        "rollout_func": rollout_func,
        "args": grpo_args,
        "callbacks": callbacks,
    }
    allowed_trainer_args = set(inspect.signature(GRPOTrainer.__init__).parameters) - {"self"}
    if peft_config is not None:
        if "peft_config" not in allowed_trainer_args:
            raise RuntimeError("Configured lora/peft block, but this TRL GRPOTrainer does not accept peft_config")
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)
    if use_dashboard:
        from transformers.trainer_callback import PrinterCallback

        trainer.remove_callback(PrinterCallback)
        print("✓ Using LiveDashboard for env-GRPO progress")

    print("Starting env-GRPO training...\n")
    trainer.train()

    final_model_dir = run_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    try:
        tokenizer.save_pretrained(str(final_model_dir))
    except Exception:
        pass

    print("\n[OK] Env-GRPO training complete!")
    print(f"  Final model: {final_model_dir}")
    print(f"  Logs: {logs_dir}")

    # ── Unified experiment tracking (best-effort) ──
    import logging as _logging

    try:
        from shared.experiment_tracking.adapters import register_grpo_run

        register_grpo_run(
            logs_dir,
            str(run_dir),
            model_name=model_name,
            dataset_source=dataset_cfg.get("local_file") or dataset_cfg.get("dataset_name"),
        )
    except Exception as _exc:
        _logging.getLogger(__name__).warning(
            "Unified tracking registration failed (non-fatal): %s", _exc
        )

    return {
        **summary,
        "run_dir": str(run_dir),
        "final_model_dir": str(final_model_dir),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
