from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class DatasetSpec:
    source: str
    file: str
    hash: str = ""

    @property
    def identifier(self) -> str:
        if self.file:
            return f"{self.source}/{self.file}"
        return self.source


@dataclass
class TrainingStageSpec:
    model_name: str
    gpu: Optional[str] = None
    timeout_hours: Optional[float] = None
    image_profile: Optional[str] = None
    cloud_image: Optional[str] = None
    batch_size: Optional[int] = None
    gradient_accumulation: Optional[int] = None
    learning_rate: Optional[float] = None
    num_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    max_seq_length: Optional[int] = None
    load_in_4bit: Optional[bool] = None
    lora_target_modules: List[str] = field(default_factory=list)


@dataclass
class EvaluationStageSpec:
    enabled: bool = True
    preset: Optional[str] = "quick"
    scenarios: List[str] = field(default_factory=list)
    tags: Optional[str] = None
    gpu: Optional[str] = None
    timeout_hours: Optional[float] = None


@dataclass
class LossStageSpec:
    enabled: bool = True
    gpu: Optional[str] = None
    timeout_hours: Optional[float] = None
    max_seq_length: Optional[int] = None
    completion_only: bool = True


@dataclass
class FeaturesStageSpec:
    enabled: bool = True
    formats: List[str] = field(default_factory=lambda: ["jsonl", "csv"])


@dataclass
class ExperimentSpec:
    name: str
    provider: str
    method: str
    dataset: DatasetSpec
    training: TrainingStageSpec
    objective: str = ""
    evaluation: EvaluationStageSpec = field(default_factory=EvaluationStageSpec)
    loss: LossStageSpec = field(default_factory=LossStageSpec)
    features: FeaturesStageSpec = field(default_factory=FeaturesStageSpec)

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.name:
            issues.append("experiment.name is required")
        if self.provider not in {"hf_jobs", "modal", "runpod", "local"}:
            issues.append(f"unsupported provider '{self.provider}'")
        if self.method not in {"sft", "kto", "grpo"}:
            issues.append(f"unsupported method '{self.method}'")
        if not self.dataset.source:
            issues.append("experiment.dataset.source is required")
        if not self.training.model_name:
            issues.append("experiment.training.model_name is required")
        return issues

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        payload = data.get("experiment", data)
        dataset = DatasetSpec(**payload["dataset"])
        training = TrainingStageSpec(**payload["training"])
        evaluation = EvaluationStageSpec(**payload.get("evaluation", {}))
        loss = LossStageSpec(**payload.get("loss", {}))
        features = FeaturesStageSpec(**payload.get("features", {}))
        return cls(
            name=payload["name"],
            provider=payload["provider"],
            method=payload["method"],
            objective=payload.get("objective", ""),
            dataset=dataset,
            training=training,
            evaluation=evaluation,
            loss=loss,
            features=features,
        )


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    spec = ExperimentSpec.from_dict(raw)
    issues = spec.validate()
    if issues:
        raise ValueError("Experiment spec validation failed: " + "; ".join(issues))
    return spec
