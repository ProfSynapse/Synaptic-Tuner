"""
shared/flywheel/experiment_config.py

Configuration dataclass for the autonomous experiment loop.
Defines hyperparameter search space, strategy, and runtime limits.

Used by: shared/flywheel/experiment_loop.py, tuner CLI
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.utilities import load_yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("configs/flywheel/experiment_loop.yaml")

_SHARED_OVERRIDE_FLAGS = {
    "learning_rate": "--learning-rate",
    "r": "--lora-r",
    "lora_alpha": "--lora-alpha",
    "lora_dropout": "--lora-dropout",
    "num_train_epochs": "--num-epochs",
    "batch_size": "--batch-size",
    "per_device_train_batch_size": "--batch-size",
    "gradient_accumulation_steps": "--gradient-accumulation",
    "seed": "--seed",
    "max_seq_length": "--max-seq-length",
    "max_steps": "--max-steps",
}

# Hyperparameters the experiment loop can vary (search_space keys and keys of
# the flat base config at base_config_path), mapped to the trainer CLI flag that
# applies them. A key without a flag on the chosen trainer is refused rather
# than written somewhere the trainer never reads.
TRAINER_OVERRIDE_FLAGS: Dict[str, Dict[str, str]] = {
    "sft": {
        **_SHARED_OVERRIDE_FLAGS,
        "evolutionary.candidates": "--evolutionary-candidates",
        "evolutionary.noise_scale": "--evolutionary-noise-scale",
        "evolutionary.max_grad_norm": "--evolutionary-max-grad-norm",
        "evolutionary.eval_frequency": "--evolutionary-eval-frequency",
        "evolutionary.warmup_steps": "--evolutionary-warmup-steps",
    },
    "kto": {**_SHARED_OVERRIDE_FLAGS, "beta": "--beta"},
}

# Boolean hyperparameters: True passes the flag; False passes nothing, which
# keeps the trainer config default (evolutionary is disabled in
# Trainers/sft/configs/config.yaml).
TRAINER_BOOLEAN_OVERRIDE_FLAGS: Dict[str, Dict[str, str]] = {
    "sft": {"evolutionary.enabled": "--evolutionary-enabled"},
    "kto": {},
}


def unsupported_override_keys(trainer_type: str, keys: List[str]) -> List[str]:
    """Return the keys the trainer has no CLI override flag for."""
    flags = TRAINER_OVERRIDE_FLAGS.get(trainer_type, {})
    booleans = TRAINER_BOOLEAN_OVERRIDE_FLAGS.get(trainer_type, {})
    return sorted(key for key in keys if key not in flags and key not in booleans)


@dataclass
class ExperimentConfig:
    """Configuration for the autonomous experiment loop.

    Controls how many experiments to run, what hyperparameters to search,
    and how to select the next configuration (random, LLM advisor,
    or LLM + LightGBM surrogate).
    """

    # -- Loop limits --
    max_experiments: int = 20
    max_steps_per_experiment: int = 200
    training_timeout_seconds: int = 7200

    # -- Trainer --
    trainer_type: str = "sft"
    base_config_path: str = ""
    dataset_path: str = ""

    # -- Evaluation --
    eval_scenario: str = ""
    eval_backend: str = "local"
    cloud_provider: str = "hf_jobs"
    local_min_vram_gb: int = 8

    # -- Search space: maps param name -> list of candidate values --
    search_space: Dict[str, List[Any]] = field(default_factory=dict)

    # -- Strategy --
    search_strategy: str = "llm_surrogate"
    surrogate_retrain_every: int = 5
    surrogate_phase_threshold: int = 10

    # -- LLM advisor --
    llm_backend: str = "openrouter"

    # -- Output --
    output_dir: str = "experiments/"

    # -- Natural-language instructions for LLM advisor --
    program_md: str = ""

    def validate(self) -> List[str]:
        """Validate configuration and return a list of issues found."""
        issues: List[str] = []

        if self.max_experiments < 1:
            issues.append("max_experiments must be >= 1")
        if self.max_steps_per_experiment < 1:
            issues.append("max_steps_per_experiment must be >= 1")
        if self.training_timeout_seconds < 60:
            issues.append("training_timeout_seconds must be >= 60")
        # Warn if timeout seems short for the configured max_steps
        # Assume ~10s per step as a rough heuristic
        estimated_min_seconds = self.max_steps_per_experiment * 10
        if (
            self.training_timeout_seconds < estimated_min_seconds
            and self.training_timeout_seconds >= 60
        ):
            issues.append(
                f"training_timeout_seconds ({self.training_timeout_seconds}s) "
                f"may be too short for {self.max_steps_per_experiment} steps "
                f"(estimated minimum: {estimated_min_seconds}s)"
            )
        if self.trainer_type not in ("sft", "kto"):
            issues.append(
                f"trainer_type must be 'sft' or 'kto', got '{self.trainer_type}'"
            )
        if self.search_strategy not in ("random", "llm_surrogate"):
            issues.append(
                f"search_strategy must be 'random' or 'llm_surrogate', "
                f"got '{self.search_strategy}'"
            )
        if self.surrogate_retrain_every < 1:
            issues.append("surrogate_retrain_every must be >= 1")
        if self.surrogate_phase_threshold < 1:
            issues.append("surrogate_phase_threshold must be >= 1")

        # Search space values must be non-empty lists
        for param, values in self.search_space.items():
            if not isinstance(values, list) or len(values) == 0:
                issues.append(
                    f"search_space['{param}'] must be a non-empty list"
                )
        unsupported = unsupported_override_keys(self.trainer_type, list(self.search_space))
        if self.trainer_type in TRAINER_OVERRIDE_FLAGS and unsupported:
            issues.append(
                f"search_space keys {unsupported} have no {self.trainer_type} "
                f"trainer CLI flag; supported: "
                f"{sorted({**TRAINER_OVERRIDE_FLAGS[self.trainer_type], **TRAINER_BOOLEAN_OVERRIDE_FLAGS[self.trainer_type]})}"
            )

        return issues

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        """Create config from a flat or nested dictionary (e.g. from YAML).

        Accepts a top-level ``experiment_loop`` key (as in the default YAML)
        or flat keys directly.
        """
        # Unwrap top-level key if present
        if "experiment_loop" in data:
            data = data["experiment_loop"]

        known_fields = {
            f.name for f in cls.__dataclass_fields__.values()
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary suitable for YAML dump."""
        from dataclasses import asdict

        return asdict(self)


def load_experiment_config(
    config_path: Optional[str | Path] = None,
) -> ExperimentConfig:
    """Load experiment loop config from YAML.

    Falls back to ``configs/flywheel/experiment_loop.yaml`` if no path given.
    Missing fields use dataclass defaults.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.info(
            "Experiment config %s not found; using defaults", path,
        )
        return ExperimentConfig()

    raw = load_yaml(path)
    return ExperimentConfig.from_dict(raw)
