"""
YAML run config for the decision trainer.

Location: Trainers/decision/decision_core/config.py
Used by:  train_decision.py.

Import-light (yaml + dataclasses only) so ``--dry-run`` works without torch.
Every section maps 1:1 to a dataclass; unknown keys are rejected so a typo in a
recipe fails loudly instead of silently training with a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .prompting import HEADERS as DEFAULT_HEADERS

@dataclass
class ModelSection:
    # Key into configs/model_registry.yaml (hf_id, revision, loader, LoRA targets...).
    registry_name: str = "qwen35-2b-base"
    # Alternative registry file; null uses Trainers/decision/configs/model_registry.yaml.
    registry_path: str | None = None
    readout: str = "pointer"
    pointer_dim: int = 256
    head_dropout: float = 0.05
    # Per-run overrides of the registry entry; null keeps the registry value.
    revision: str | None = None
    max_length: int | None = None
    torch_dtype: str | None = None
    attn_implementation: str | None = None


@dataclass
class PromptSection:
    # numbers -> "1." "2." ...; letters -> "A." "B." ... The letter_logits readout
    # reads these marker tokens, so each must be a single token.
    marker_style: str = "numbers"
    # One instruction line per question kind, rendered above the question text.
    # Saved with the checkpoint so serving renders exactly what training saw.
    headers: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HEADERS))


@dataclass
class LoraSection:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # null -> the registry entry's lora_target_modules.
    target_modules: list[str] | None = None


@dataclass
class DataSection:
    train_files: list[str] = field(default_factory=list)
    val_fraction: float = 0.03
    eval_files: list[str] = field(default_factory=list)
    # Separate calibration rows. When empty, `calibration.holdout_fraction` of the
    # eval rows (stratified by task) are carved off for calibration instead, so
    # temperatures are never fitted on the rows they are scored on.
    calibration_files: list[str] = field(default_factory=list)
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    seed: int = 0


@dataclass
class AugmentationSection:
    shuffle_options: bool = True
    reverse_score_prob: float = 0.5
    ordinal_smoothing: float = 0.1
    vary_instructions: bool = True


@dataclass
class LossSection:
    brier_weight: float = 0.0
    kl_frozen_weight: float = 0.3


@dataclass
class TrainingSection:
    num_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-4
    head_learning_rate: float = 1.0e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    logging_steps: int = 20
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    bf16: bool = True
    seed: int = 0


@dataclass
class CalibrationSection:
    enabled: bool = True
    holdout_fraction: float = 0.5
    # A kind with fewer calibration rows than this keeps temperature 1.0.
    min_rows_per_kind: int = 50


@dataclass
class EvaluationSection:
    enabled: bool = True
    # Re-render each eval row this many times with a random option order and
    # report how often the canonical answer changes.
    shuffle_trials: int = 2
    ece_bins: int = 15
    batch_size: int = 8


@dataclass
class OutputSection:
    output_root: str = "decision_output"


@dataclass
class DecisionRunConfig:
    model: ModelSection = field(default_factory=ModelSection)
    prompt: PromptSection = field(default_factory=PromptSection)
    lora: LoraSection = field(default_factory=LoraSection)
    data: DataSection = field(default_factory=DataSection)
    augmentation: AugmentationSection = field(default_factory=AugmentationSection)
    loss: LossSection = field(default_factory=LossSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    calibration: CalibrationSection = field(default_factory=CalibrationSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    output: OutputSection = field(default_factory=OutputSection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {f.name: f.default_factory for f in fields(DecisionRunConfig)}  # type: ignore[misc]


def _build_section(name: str, raw: Any) -> Any:
    section = _SECTIONS[name]()
    if raw is None:
        return section
    if not isinstance(raw, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    known = {f.name for f in fields(section)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown key(s) in config section {name!r}: {unknown}")
    for key, value in raw.items():
        setattr(section, key, value)
    return section


def load_run_config(path: str | Path) -> DecisionRunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ValueError(f"{path}: unknown config section(s) {unknown}")
    cfg = DecisionRunConfig(**{name: _build_section(name, raw.get(name)) for name in _SECTIONS})
    validate_run_config(cfg)
    return cfg


def validate_run_config(cfg: DecisionRunConfig) -> None:
    from .prompting import MARKER_STYLES
    from .constants import READOUTS
    from .examples import KINDS

    if cfg.model.readout not in READOUTS:
        raise ValueError(f"model.readout must be one of {READOUTS}, got {cfg.model.readout!r}")
    if cfg.prompt.marker_style not in MARKER_STYLES:
        raise ValueError(f"prompt.marker_style must be one of {MARKER_STYLES}")
    missing = sorted(set(KINDS) - set(cfg.prompt.headers))
    extra = sorted(set(cfg.prompt.headers) - set(KINDS))
    if missing or extra:
        raise ValueError(f"prompt.headers must define exactly {list(KINDS)} (missing {missing}, unknown {extra})")
    if not 0.0 <= cfg.augmentation.ordinal_smoothing < 1.0:
        raise ValueError("augmentation.ordinal_smoothing must be in [0, 1)")
    if not 0.0 <= cfg.calibration.holdout_fraction < 1.0:
        raise ValueError("calibration.holdout_fraction must be in [0, 1)")
    if cfg.loss.brier_weight < 0 or cfg.loss.kl_frozen_weight < 0:
        raise ValueError("loss weights must be non-negative")
