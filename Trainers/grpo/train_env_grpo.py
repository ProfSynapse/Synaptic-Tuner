#!/usr/bin/env python3
"""Cloud-first environment-backed GRPO entrypoint.

This path is separate from the existing static projected-dataset GRPO trainer.
It is designed for a newer TRL/OpenEnv stack running inside an isolated
virtualenv on top of the Unsloth Docker image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

# Add trainer src to path for direct execution.
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.env_dataset import (
    filter_env_rollout_dataset,
    format_dataset_for_env_grpo_with_options,
    load_env_rollout_dataset,
)
from src.env_rewards import build_env_reward_function
from src.env_rollout import build_prompt_registry, build_rollout_func
from src.env_runtime import build_cloud_bootstrap_commands, detect_openenv_runtime_support
from src.training_callbacks import DASHBOARD_AVAILABLE, RICH_AVAILABLE, LiveDashboardCallback, MetricsTableCallback
from shared.training_utils import build_trainer_config, reject_unknown_config_keys


# Every YAML key this entrypoint and its env modules read. load_config refuses
# any other key so a typo cannot silently fall back to a default. Schema literal
# (see shared/training_utils.py): ``None`` is a leaf whose value is not walked,
# ``{...}`` a closed section. Readers outside this file: ``env_training`` rollout
# keys in src/env_rollout.py, ``env_training.runtime`` in src/env_runtime.py,
# ``env_training.prompt_augmentation`` in src/env_dataset.py, and ``rewards`` in
# src/env_rewards.py. tests/trainers/grpo/test_grpo_config_keys.py fails if this
# drifts from the keys the code reads.
ENV_GRPO_CONFIG_SCHEMA = {
    "model": {"model_name": None},
    "dataset": {
        "dataset_name": None,
        "dataset_file": None,
        "local_file": None,
        "num_proc": None,
    },
    "training": {
        "output_dir": None,
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
        "num_generations": None,
        "max_prompt_length": None,
        "max_completion_length": None,
        "temperature": None,
        "learning_rate": None,
        "weight_decay": None,
        "warmup_ratio": None,
        "lr_scheduler_type": None,
        "num_train_epochs": None,
        "max_steps": None,
        "beta": None,
        "logging_steps": None,
        "save_steps": None,
        "save_total_limit": None,
        "report_to": None,
        "fp16": None,
        "bf16": None,
        "optim": None,
        "use_vllm": None,
        "vllm_mode": None,
        "extra_args": None,
    },
    "lora": {
        "enabled": None,
        "r": None,
        "lora_alpha": None,
        "lora_dropout": None,
        "bias": None,
        "task_type": None,
        "target_modules": None,
    },
    "env_training": {
        "require_environment_passed": None,
        "require_environment_config": None,
        "required_stage_reviews": None,
        "debug_rollouts_path": None,
        "allow_transformers_rollout_func": None,
        "prompt_augmentation": {"system_append": None, "insert_if_missing": None},
        "env_backend": None,
        "max_turns": None,
        "max_tool_steps": None,
        "stop_on_text_response": None,
        "stop_on_environment_pass": None,
        "require_final_text_after_pass": None,
        "final_text_prompt": None,
        "continue_on_execution_error": None,
        "tool_result_format": None,
        "token_faithful": None,
        "context_token_policy": None,
        "runtime": {
            "repo_root_in_container": None,
            "isolated_venv_dir": None,
            "local_venv_dir": None,
            "project_pip_deps": None,
            "python_packages": None,
        },
    },
    "rewards": {
        "success_reward": None,
        "failure_penalty": None,
        "step_penalty": None,
        "max_tool_steps_penalty": None,
        "text_before_completion_penalty": None,
        "require_final_text_satisfied": None,
        "final_text_failure_penalty": None,
        "no_tool_call_penalty": None,
        "tool_call_parse_reward": None,
        "expected_tool_reward": None,
        "expected_tool_order_reward": None,
        "final_text_satisfied_reward": None,
        "max_progress_reward": None,
        "tool_text_mix_penalty": None,
        "placeholder_path_penalty": None,
        "invalid_wrapper_arg_penalty": None,
        "tool_status_rewards": None,
        "environment_issue_penalties": None,
        "wrapper_name": None,
        "wrapper_cli_field": None,
        "placeholder_patterns": None,
    },
}

# Internal GRPOConfig defaults whose acceptance depends on the installed TRL.
# Only these may be omitted when unsupported, and only when the YAML did not set
# them (a YAML-set value is always refused if unsupported):
#   max_prompt_length - removed from GRPOConfig in trl 0.28 (present in 0.24);
#                       older isolated runtimes still apply the 4096 default.
_VERSION_DEPENDENT_GRPO_DEFAULTS = frozenset({"max_prompt_length"})


def build_grpo_trainer_class(*, allow_transformers_rollout_func: bool):
    from trl import GRPOTrainer

    if not allow_transformers_rollout_func:
        return GRPOTrainer

    class TransformersRolloutGRPOTrainer(GRPOTrainer):
        def _generate_single_turn(self, prompts):
            if self.rollout_func is not None and not self.use_vllm:
                rollout = self.rollout_func(prompts, self)
                extra_fields = {
                    key: value
                    for key, value in rollout.items()
                    if key not in {"prompt_ids", "completion_ids", "logprobs"}
                }
                return (
                    rollout["prompt_ids"],
                    rollout["completion_ids"],
                    rollout.get("logprobs"),
                    extra_fields,
                )
            return super()._generate_single_turn(prompts)

    return TransformersRolloutGRPOTrainer


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = str(Path(__file__).parent / "configs" / "env_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    reject_unknown_config_keys(ENV_GRPO_CONFIG_SCHEMA, config, source=str(config_path))
    config["_config_path"] = str(Path(config_path).resolve())
    return config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud-first env-backed GRPO launcher")
    parser.add_argument("--config", type=str, default=None, help="Path to env GRPO config")
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
    parser.add_argument("--save-steps", type=int, default=None, help="Override training.save_steps")
    parser.add_argument("--save-total-limit", type=int, default=None, help="Override training.save_total_limit")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")

    model_cfg = config.get("model") or {}
    dataset_cfg = config.get("dataset") or {}
    training_cfg = config.get("training") or {}
    lora_cfg = config.get("lora") or {}

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
    if args.save_steps is not None:
        training_cfg["save_steps"] = args.save_steps
    if args.save_total_limit is not None:
        training_cfg["save_total_limit"] = args.save_total_limit

    if args.print_cloud_bootstrap:
        runtime_cfg = ((config.get("env_training") or {}).get("runtime") or {})
        cloud_repo_root = str(runtime_cfg.get("repo_root_in_container") or "/workspace/repo")
        cloud_config_path = str(Path(cloud_repo_root) / "Trainers" / "grpo" / "configs" / "env_config.yaml")
        commands = build_cloud_bootstrap_commands(
            config,
            repo_root=cloud_repo_root,
            config_path=cloud_config_path,
        )
        print("\n".join(commands))
        return {"bootstrap_commands": commands}

    env_cfg = config.get("env_training") or {}
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
    formatted_dataset = format_dataset_for_env_grpo_with_options(
        filtered_dataset,
        prompt_augmentation=env_cfg.get("prompt_augmentation"),
    )

    runtime_support = detect_openenv_runtime_support()
    summary = {
        "raw_examples": len(raw_dataset),
        "filtered_examples": len(filtered_dataset),
        "formatted_examples": len(formatted_dataset),
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
        raise RuntimeError("Set model.model_name in env_config.yaml to the published bucketed SFT model repo")

    from transformers import AutoTokenizer
    from trl import GRPOConfig

    output_root = Path(training_cfg.get("output_dir") or "./env_grpo_output").resolve()
    run_dir = Path(args.output_dir).resolve() if args.output_dir else output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    debug_rollouts_path = env_cfg.get("debug_rollouts_path")
    if debug_rollouts_path:
        env_cfg["debug_rollouts_path"] = str(Path(str(debug_rollouts_path).format(run_dir=run_dir)).resolve())
    rollout_func = build_rollout_func(
        registry=registry,
        env_training_cfg=env_cfg,
        runtime_support=runtime_support,
    )
    reward_func = build_env_reward_function(config.get("rewards") or {})
    trainer_cls = build_grpo_trainer_class(
        allow_transformers_rollout_func=bool(env_cfg.get("allow_transformers_rollout_func", False)),
    )

    peft_config = None
    if bool(lora_cfg.get("enabled", False)):
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
            target_modules=list(lora_cfg.get("target_modules") or []),
        )

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
    # Arguments the YAML (or a CLI override written into it) set; any of these
    # the installed GRPOConfig rejects raises instead of being dropped.
    grpo_origins = {name: f"training.{name}" for name in grpo_kwargs if name in training_cfg}
    extra_args = training_cfg.get("extra_args") or {}
    if not isinstance(extra_args, dict):
        raise TypeError("training.extra_args must be a mapping/dict")
    grpo_kwargs.update(extra_args)
    grpo_origins.update({name: f"training.extra_args.{name}" for name in extra_args})
    grpo_args = build_trainer_config(
        GRPOConfig,
        grpo_kwargs,
        origins=grpo_origins,
        version_dependent_defaults=_VERSION_DEPENDENT_GRPO_DEFAULTS,
    )

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

    trainer = trainer_cls(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=formatted_dataset,
        rollout_func=rollout_func,
        args=grpo_args,
        callbacks=callbacks,
        peft_config=peft_config,
    )
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
