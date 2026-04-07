from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml


EXPERIMENT_STAGES = ("training", "evaluation", "loss", "analysis", "recommendation")


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
    pip_packages: List[str] = field(default_factory=list)
    batch_size: Optional[int] = None
    gradient_accumulation: Optional[int] = None
    learning_rate: Optional[float] = None
    num_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    max_seq_length: Optional[int] = None
    load_in_4bit: Optional[bool] = None
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: Optional[str] = None
    lora_target_modules: List[str] | str = field(default_factory=list)
    evolutionary: "EvolutionaryStageSpec" = field(default_factory=lambda: EvolutionaryStageSpec())


@dataclass
class EvolutionaryStrategySpec:
    type: str = "gradient_noise"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionarySelectionSpec:
    method: str = "best"
    min_improvement: float = 0.0
    min_relative_improvement: float = 0.0
    noise_floor_epsilon: float = 1e-6


@dataclass
class EvolutionaryLoggingSpec:
    candidates: bool = True
    selected: bool = True


@dataclass
class EvolutionaryStageSpec:
    enabled: bool = False
    candidates: int = 4
    eval_batch_size: int = 2
    validation_config: Optional[str] = None
    strategy: EvolutionaryStrategySpec = field(default_factory=EvolutionaryStrategySpec)
    selection: EvolutionarySelectionSpec = field(default_factory=EvolutionarySelectionSpec)
    eval_frequency: int = 1
    warmup_steps: int = 0
    cache_baseline: bool = True
    logging: EvolutionaryLoggingSpec = field(default_factory=EvolutionaryLoggingSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvolutionaryStageSpec":
        payload = data or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            candidates=int(payload.get("candidates", 4)),
            eval_batch_size=int(payload.get("eval_batch_size", 2)),
            validation_config=payload.get("validation_config"),
            strategy=EvolutionaryStrategySpec(**payload.get("strategy", {})),
            selection=EvolutionarySelectionSpec(**payload.get("selection", {})),
            eval_frequency=int(payload.get("eval_frequency", 1)),
            warmup_steps=int(payload.get("warmup_steps", 0)),
            cache_baseline=bool(payload.get("cache_baseline", True)),
            logging=EvolutionaryLoggingSpec(**payload.get("logging", {})),
        )


@dataclass
class EvaluationStageSpec:
    enabled: bool = True
    preset: Optional[str] = "quick"
    scenarios: List[str] = field(default_factory=list)
    tags: Optional[str] = None
    runtime: Optional[str] = None
    image_profile: Optional[str] = None
    cloud_image: Optional[str] = None
    pip_packages: List[str] = field(default_factory=list)
    gpu: Optional[str] = None
    timeout_hours: Optional[float] = None


@dataclass
class LossStageSpec:
    enabled: bool = True
    gpu: Optional[str] = None
    timeout_hours: Optional[float] = None
    pip_packages: List[str] = field(default_factory=list)
    max_seq_length: Optional[int] = None
    completion_only: bool = True


@dataclass
class PostTrainingSpec:
    mode: str = "parallel"

    def validate(self) -> list[str]:
        if self.mode not in {"parallel", "same_job"}:
            return [f"unsupported post_training.mode '{self.mode}'"]
        return []


@dataclass
class FeaturesStageSpec:
    enabled: bool = True
    formats: List[str] = field(default_factory=lambda: ["jsonl", "csv"])


@dataclass
class ExecutionStageSpec:
    stages: List[str] = field(default_factory=lambda: list(EXPERIMENT_STAGES))
    from_stage: Optional[str] = None
    only_stage: Optional[str] = None
    skip_stages: List[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        issues: list[str] = []
        allowed = set(EXPERIMENT_STAGES)
        selected = list(self.stages or list(EXPERIMENT_STAGES))
        invalid = [stage for stage in selected if stage not in allowed]
        if invalid:
            issues.append(f"unsupported execution stage(s): {', '.join(sorted(set(invalid)))}")
        if self.only_stage and self.only_stage not in allowed:
            issues.append(f"unsupported execution.only_stage '{self.only_stage}'")
        if self.from_stage and self.from_stage not in allowed:
            issues.append(f"unsupported execution.from_stage '{self.from_stage}'")
        invalid_skip = [stage for stage in self.skip_stages if stage not in allowed]
        if invalid_skip:
            issues.append(f"unsupported execution.skip_stages value(s): {', '.join(sorted(set(invalid_skip)))}")
        if self.only_stage and self.from_stage:
            issues.append("execution.only_stage and execution.from_stage are mutually exclusive")
        if self.only_stage and self.only_stage in self.skip_stages:
            issues.append("execution.only_stage cannot also appear in execution.skip_stages")
        if not self.selected_stages():
            issues.append("execution stage selection resolves to an empty set")
        return issues

    def selected_stages(self) -> list[str]:
        selected = list(self.stages or list(EXPERIMENT_STAGES))
        if self.only_stage:
            selected = [self.only_stage]
        elif self.from_stage:
            if self.from_stage in selected:
                start = selected.index(self.from_stage)
                selected = selected[start:]
            else:
                selected = [self.from_stage]
        selected = [stage for stage in selected if stage not in set(self.skip_stages)]
        return [stage for stage in selected if stage in EXPERIMENT_STAGES]

    def includes(self, stage: str) -> bool:
        return stage in self.selected_stages()


@dataclass
class ExperimentSpec:
    name: str
    provider: str
    method: str
    dataset: DatasetSpec
    training: TrainingStageSpec
    objective: str = ""
    execution: ExecutionStageSpec = field(default_factory=ExecutionStageSpec)
    evaluation: EvaluationStageSpec = field(default_factory=EvaluationStageSpec)
    loss: LossStageSpec = field(default_factory=LossStageSpec)
    post_training: PostTrainingSpec = field(default_factory=PostTrainingSpec)
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
        issues.extend(self.execution.validate())
        issues.extend(self.post_training.validate())
        return issues

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        payload = data.get("experiment", data)
        required_keys = ("name", "provider", "method", "dataset", "training")
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise ValueError(
                f"Experiment spec missing required field(s): {', '.join(missing)}"
            )
        try:
            dataset = DatasetSpec(**payload["dataset"])
        except TypeError as exc:
            raise ValueError(f"Invalid experiment.dataset: {exc}") from exc
        training_payload = dict(payload["training"])
        evolutionary = EvolutionaryStageSpec.from_dict(training_payload.pop("evolutionary", {}))
        try:
            training = TrainingStageSpec(**training_payload, evolutionary=evolutionary)
        except TypeError as exc:
            raise ValueError(f"Invalid experiment.training: {exc}") from exc
        execution = ExecutionStageSpec(**payload.get("execution", {}))
        evaluation = EvaluationStageSpec(**payload.get("evaluation", {}))
        loss = LossStageSpec(**payload.get("loss", {}))
        post_training = PostTrainingSpec(**payload.get("post_training", {}))
        features = FeaturesStageSpec(**payload.get("features", {}))
        return cls(
            name=payload["name"],
            provider=payload["provider"],
            method=payload["method"],
            objective=payload.get("objective", ""),
            dataset=dataset,
            training=training,
            execution=execution,
            evaluation=evaluation,
            loss=loss,
            post_training=post_training,
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
