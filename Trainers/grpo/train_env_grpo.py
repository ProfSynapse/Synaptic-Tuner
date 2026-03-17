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
from pathlib import Path
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


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = str(Path(__file__).parent / "configs" / "env_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
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
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")

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

    dataset_cfg = config.get("dataset") or {}
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
    formatted_dataset = format_dataset_for_env_grpo(filtered_dataset)

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

    model_name = str((config.get("model") or {}).get("model_name") or "").strip()
    if not model_name or model_name == "REPLACE_WITH_BUCKETED_SFT_MODEL":
        raise RuntimeError("Set model.model_name in env_config.yaml to the published bucketed SFT model repo")

    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

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
    rollout_func = build_rollout_func(
        registry=registry,
        env_training_cfg=env_cfg,
    )
    reward_func = build_env_reward_function(config.get("rewards") or {})

    training_cfg = config.get("training") or {}
    grpo_kwargs = {
        "output_dir": str(Path(training_cfg.get("output_dir") or "./env_grpo_output").resolve()),
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

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=formatted_dataset,
        rollout_func=rollout_func,
        args=grpo_args,
    )
    trainer.train()
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
