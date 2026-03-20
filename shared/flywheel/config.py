"""
shared/flywheel/config.py

FlywheelConfig dataclass and loader for the enterprise data flywheel pipeline.
All thresholds and backend settings live here. Loaded from YAML config file,
with sensible defaults for local single-GPU development.

Used by: all flywheel modules, services/proxy, tuner/handlers/flywheel_handler.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.utilities import load_yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("configs/flywheel/default.yaml")


@dataclass
class FlywheelConfig:
    """Configuration for the flywheel pipeline.

    Loaded from YAML config file. All thresholds have sensible defaults
    that can be overridden per-deployment.
    """

    # -- Scoring thresholds --
    sft_threshold: float = 0.8
    kto_min_threshold: float = 0.3
    ambiguous_min: float = 0.4
    ambiguous_max: float = 0.7
    max_errors_before_zero: int = 5
    scoring_method: str = "error_count"
    error_penalty_per_error: float = 0.1
    no_tool_call_score: float = 0.0

    # -- Non-tool-call handling --
    text_response_policy: str = "skip"

    # -- FitnessEvaluator config --
    fitness_config_path: str | None = None

    # -- GRPO --
    grpo_enabled: bool = True
    grpo_reward_scale: float = 1.0

    # -- Storage --
    catalog_backend: str = "sqlite"
    catalog_path: str = ".tracking/flywheel.db"
    catalog_url: str | None = None
    tenant_id: str | None = None
    log_dir: str = "inference_logs"
    datasets_dir: str = "Datasets/flywheel"

    # -- Logging proxy --
    proxy_port: int = 8080
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    proxy_timeout_seconds: float = 120.0
    flush_interval_seconds: float = 1.0

    # -- vLLM --
    vllm_adapter_name: str = "current-adapter"
    vllm_adapter_path: str | None = None
    vllm_base_model: str | None = None
    vllm_max_lora_rank: int = 64
    vllm_gpu_memory_utilization: float = 0.9
    vllm_enable_runtime_lora: bool = True

    # -- Retrain --
    retrain_mode: str = "gpu_mutex"
    retrain_trainer: str = "sft"
    min_new_examples: int = 500
    min_sft_examples: int = 100
    min_quality_score: float = 0.6
    min_days_since_last_cycle: int = 0

    # -- Cloud retrain --
    cloud_provider: str | None = None
    cloud_config_path: str | None = None

    # -- Log rotation --
    log_retention_days: int = 30
    compress_after_days: int = 7

    # -- Validation rules for FitnessEvaluator --
    validation_rules: list[dict] = field(default_factory=list)

    def to_fitness_config(self) -> dict[str, Any]:
        """Build FitnessEvaluator config dict from flywheel settings.

        If fitness_config_path is set, loads and returns the external YAML
        config directly. Otherwise, builds config from inline fields.

        An empty validations list means FitnessEvaluator runs zero validators
        and scores everything 1.0. The flywheel MUST either set
        fitness_config_path or provide validation_rules.
        """
        if self.fitness_config_path is not None:
            path = Path(self.fitness_config_path)
            if path.exists():
                return load_yaml(path)
            logger.warning(
                "Fitness config file not found: %s; using inline rules",
                self.fitness_config_path,
            )

        if not self.validation_rules:
            logger.warning(
                "No fitness_config_path or validation_rules set; "
                "FitnessEvaluator will score all responses 1.0"
            )

        return {
            "validations": self.validation_rules,
            "scoring": {
                "method": self.scoring_method,
                "no_tool_call_score": self.no_tool_call_score,
                "params": {
                    "max_errors_before_zero": self.max_errors_before_zero,
                    "penalty_per_error": self.error_penalty_per_error,
                },
            },
        }


def load_flywheel_config(
    config_path: str | Path | None = None,
) -> FlywheelConfig:
    """Load flywheel config from YAML file.

    Falls back to configs/flywheel/default.yaml if no path specified.
    Missing fields use dataclass defaults.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Populated FlywheelConfig instance.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.info(
            "Config file %s not found; using defaults", path,
        )
        return FlywheelConfig()

    raw = load_yaml(path)

    # Filter to known fields only (forward-compatible)
    known_fields = {f.name for f in FlywheelConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}

    return FlywheelConfig(**filtered)
