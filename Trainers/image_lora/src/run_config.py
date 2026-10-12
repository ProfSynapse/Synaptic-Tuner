"""
Run configuration for the image_lora method: config -> ai-toolkit job + estimate.

Location: Trainers/image_lora/src/run_config.py
Purpose:  Load configs/config.yaml (+ CLI overrides) and configs/model_registry.yaml,
          render the ai-toolkit job YAML, and compute the pre-launch cost
          estimate and the spend-cap timeout. Pure: no Modal or network calls.
Used by:  train_image_lora.py (plan/launch/run) and src/modal_runner.py.

The rendered job contains three placeholders the remote worker fills in:
``${MODEL_PATH}`` (pinned snapshot in the shared cache Volume), ``${DATASET_DIR}``
and ``${OUTPUT_DIR}`` (both on the per-run Volume).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TRAINER_DIR = Path(__file__).resolve().parent.parent
REGISTRY_PATH = TRAINER_DIR / "configs" / "model_registry.yaml"

OVERRIDE_PATHS = {
    "steps": ("training", "steps"),
    "rank": ("network", "rank"),
    "learning_rate": ("training", "learning_rate"),
    "gpu": ("modal", "gpu"),
    "sample_every": ("sample", "every"),
    "save_every": ("save", "every"),
}


def load_run_config(path: Path, overrides: dict[str, Any] | None = None,
                    registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for key, value in (overrides or {}).items():
        section, name = OVERRIDE_PATHS[key]
        config.setdefault(section, {})[name] = value
        if key == "rank":
            config["network"]["alpha"] = value      # keep alpha == rank (scale 1.0)
    with open(registry_path, encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    name = config.get("model", {}).get("registry_name")
    models = registry.get("models", {})
    if name not in models:
        raise ValueError(f"unknown model.registry_name {name!r}; known: {sorted(models)}")
    config["_model"] = dict(models[name], registry_name=name)
    return config


def load_dataset_info(dataset_dir: Path) -> dict[str, Any]:
    """Trigger, image count and sample prompts from a built dataset folder."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found; run build-dataset first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = sorted((dataset_dir / "images").glob("*.png"))
    missing = [p.name for p in images if not p.with_suffix(".txt").is_file()]
    if missing:
        raise ValueError(f"images without captions: {missing[:5]}")
    prompts_path = dataset_dir / "sample_prompts.yaml"
    prompts = []
    if prompts_path.is_file():
        prompts = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or []
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("sample_prompts.yaml must be a YAML list of strings")
    return {"trigger": manifest["trigger"], "images": len(images), "sample_prompts": prompts,
            "name": manifest.get("name", dataset_dir.name)}


def render_job(config: dict[str, Any], *, run_name: str, trigger: str,
               sample_prompts: list[str]) -> dict[str, Any]:
    """The ai-toolkit job dict (``job: extension`` / ``sd_trainer`` process)."""
    model, network = config["_model"], config["network"]
    train, save, sample = config["training"], config["save"], config["sample"]
    model_cfg = config.get("model", {})
    model_section: dict[str, Any] = {
        "name_or_path": "${MODEL_PATH}",
        "arch": model["arch"],
        "quantize": bool(model_cfg.get("quantize_transformer", False)),
        "quantize_te": bool(model_cfg.get("quantize_text_encoder", True)),
        "qtype_te": "qfloat8",
        "low_vram": bool(model_cfg.get("low_vram", False)),
    }
    if model_section["quantize"]:
        model_section["qtype"] = model_cfg.get("qtype", "qfloat8")
    process = {
        "type": "sd_trainer",
        "training_folder": "${OUTPUT_DIR}",
        "device": "cuda:0",
        "network": {"type": network["type"], "linear": int(network["rank"]),
                    "linear_alpha": int(network["alpha"])},
        "save": {"dtype": save.get("dtype", "bf16"), "save_every": int(save["every"]),
                 "max_step_saves_to_keep": int(save.get("keep", 8))},
        "datasets": [{
            "folder_path": "${DATASET_DIR}",
            "caption_ext": "txt",
            "caption_dropout_rate": float(train.get("caption_dropout_rate", 0.05)),
            "shuffle_tokens": False,
            "cache_latents_to_disk": True,
            "resolution": list(train["resolutions"]),
        }],
        "train": {
            "batch_size": int(train.get("batch_size", 1)),
            "steps": int(train["steps"]),
            "gradient_accumulation": int(train.get("gradient_accumulation", 1)),
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": bool(train.get("gradient_checkpointing", True)),
            "noise_scheduler": train.get("noise_scheduler", "flowmatch"),
            "timestep_type": train.get("timestep_type", "weighted"),
            "optimizer": train.get("optimizer", "adamw8bit"),
            "lr": float(train["learning_rate"]),
            "dtype": train.get("dtype", "bf16"),
            "cache_text_embeddings": bool(train.get("cache_text_embeddings", True)),
            "skip_first_sample": bool(sample.get("skip_first_sample", False)),
            "disable_sampling": not sample_prompts,
        },
        "model": model_section,
        "sample": {
            "sampler": train.get("noise_scheduler", "flowmatch"),
            "sample_every": int(sample["every"]),
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "prompts": list(sample_prompts),
            "neg": sample.get("negative", ""),
            "seed": int(sample.get("seed", 42)),
            "walk_seed": bool(sample.get("walk_seed", False)),
            "guidance_scale": float(sample["guidance_scale"]),
            "sample_steps": int(sample["steps"]),
        },
        "trigger_word": trigger,
    }
    if train.get("cache_text_embeddings", True):
        # ai-toolkit cannot inject a trigger into cached embeddings; captions
        # already start with it (build-dataset), so do not ask it to.
        process.pop("trigger_word")
    return {"job": "extension",
            "config": {"name": run_name, "process": [process]},
            "meta": {"name": "[name]", "version": "1.0", "method": "image_lora",
                     "base_model": f"{model['repo']}@{model.get('revision', 'main')}"}}


@dataclass
class Estimate:
    gpu: str
    hours: float
    usd_per_hour: float
    usd: float
    breakdown: dict[str, float] = field(default_factory=dict)


def estimate_cost(config: dict[str, Any], *, sample_prompt_count: int) -> Estimate:
    est, modal_cfg = config["estimate"], config["modal"]
    gpu = modal_cfg["gpu"]
    steps = int(config["training"]["steps"])
    every = int(config["sample"]["every"])
    rounds = steps // every + (0 if config["sample"].get("skip_first_sample") else 1)
    if steps % every:
        rounds += 1                                      # final sample at the last step
    train_s = steps * float(est["seconds_per_step"][gpu])
    sample_s = rounds * sample_prompt_count * float(est["seconds_per_sample_image"][gpu])
    overhead_s = float(est.get("overhead_minutes", 20)) * 60
    rates = est["usd_per_hour"]
    per_hour = (float(rates["gpu"][gpu]) + float(modal_cfg.get("cpu", 8)) * float(rates["cpu_core"])
                + float(modal_cfg.get("memory_mib", 65536)) / 1024 * float(rates["memory_gib"]))
    hours = (train_s + sample_s + overhead_s) / 3600
    return Estimate(gpu, round(hours, 2), round(per_hour, 3), round(hours * per_hour, 2),
                    {"train_hours": round(train_s / 3600, 2), "sample_hours": round(sample_s / 3600, 2),
                     "overhead_hours": round(overhead_s / 3600, 2), "sample_rounds": rounds})


def timeout_for_budget(estimate: Estimate, max_usd: float, *, ceiling_s: int = 24 * 3600) -> int:
    """Provider timeout so a runaway job cannot spend more than ``max_usd``."""
    if max_usd <= 0:
        raise ValueError("max_usd must be positive")
    if estimate.usd > max_usd:
        raise ValueError(f"estimate ${estimate.usd:.2f} exceeds the ${max_usd:.2f} cap; "
                         "lower steps/rank or raise the cap")
    return int(min(ceiling_s, math.floor(max_usd / estimate.usd_per_hour * 3600)))


@dataclass
class RunPlan:
    config: dict[str, Any]
    run_name: str
    job: dict[str, Any]
    estimate: Estimate
    dataset: dict[str, Any] | None

    @property
    def job_yaml(self) -> str:
        return yaml.safe_dump(self.job, sort_keys=False, allow_unicode=True, width=1000)

    def summary(self) -> dict[str, Any]:
        model = self.config["_model"]
        return {
            "method": "image_lora", "run_name": self.run_name,
            "base_model": f"{model['repo']} ({model['arch']})",
            "dataset": self.dataset, "gpu": self.config["modal"]["gpu"],
            "rank": self.config["network"]["rank"], "steps": self.config["training"]["steps"],
            "learning_rate": self.config["training"]["learning_rate"],
            "resolutions": self.config["training"]["resolutions"],
            "estimate": {"hours": self.estimate.hours, "usd": self.estimate.usd,
                         "usd_per_hour": self.estimate.usd_per_hour, **self.estimate.breakdown},
        }


def plan_run(config: dict[str, Any], *, dataset_dir: Path | None, run_name: str = "image_lora") -> RunPlan:
    dataset = load_dataset_info(dataset_dir) if dataset_dir else None
    trigger = dataset["trigger"] if dataset else "trigger phrase"
    prompts = dataset["sample_prompts"] if dataset else []
    job = render_job(copy.deepcopy(config), run_name=run_name, trigger=trigger, sample_prompts=prompts)
    estimate = estimate_cost(config, sample_prompt_count=len(prompts))
    return RunPlan(config, run_name, job, estimate, dataset)
