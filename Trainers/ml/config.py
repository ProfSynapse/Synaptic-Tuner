# Trainers/ml/config.py
# Pydantic v2 configuration models for ML training pipelines.
# Loaded from YAML, validated at parse time. Root model: TrainingConfig.
# Used by train.py orchestrator and all submodules (data, features, algorithms).

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(str, Enum):
    """Supported ML task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TaskConfig(BaseModel):
    """What to predict and how to measure it."""
    type: TaskType
    name: str = Field(..., min_length=1, description="Human-readable task name")
    target_column: str = Field(..., min_length=1)
    eval_metric: str = Field(
        default="f1_weighted",
        description=(
            "Primary optimization metric. "
            "Classification: accuracy, f1_weighted, roc_auc. "
            "Regression: rmse, mae, r2."
        ),
    )
    random_state: int = 42


class DataConfig(BaseModel):
    """Where to find data and how to split it."""
    train_path: str = Field(
        ..., description="Path to training data (CSV, Parquet, JSONL)"
    )
    test_path: Optional[str] = Field(
        default=None,
        description="Explicit test set. If None, split from train_path.",
    )
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    stratify: bool = Field(
        default=True,
        description="Stratified split for classification tasks",
    )

    @field_validator("train_path")
    @classmethod
    def train_path_must_exist(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"Training data not found: {v}")
        return v


class NumericFeaturesConfig(BaseModel):
    """Preprocessing for numeric columns."""
    columns: list[str]
    imputer: Literal["mean", "median", "none"] = "median"
    scaler: Literal["standard", "minmax", "robust", "none"] = "standard"


class CategoricalFeaturesConfig(BaseModel):
    """Preprocessing for categorical columns."""
    columns: list[str]
    encoder: Literal["onehot", "ordinal"] = "onehot"
    handle_unknown: Literal["ignore", "error"] = "ignore"


class FeaturesConfig(BaseModel):
    """Feature engineering specification. Phase 1: numeric + categorical only."""
    numeric: Optional[NumericFeaturesConfig] = None
    categorical: Optional[CategoricalFeaturesConfig] = None
    drop_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def at_least_one_feature_type(self) -> FeaturesConfig:
        if self.numeric is None and self.categorical is None:
            raise ValueError(
                "At least one feature type (numeric or categorical) must be specified"
            )
        return self


class AlgorithmConfig(BaseModel):
    """Algorithm selection and hyperparameters."""
    name: str = Field(
        default="lightgbm",
        description="Algorithm name from registry",
    )
    params: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    """Evaluation settings."""
    metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "f1_weighted"]
    )
    generate_plots: bool = False


class OutputConfig(BaseModel):
    """Output directory and serialization."""
    dir: str = "./ml_output"
    save_model: Literal["joblib", "both"] = "joblib"
    save_pipeline: bool = True


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""
    enabled: bool = False
    backend: Literal["mlflow", "local"] = "local"
    experiment_name: str = "default"
    tags: dict[str, str] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Root configuration — the contract between YAML and Python.

    Minimal valid config requires: task, data, features.
    All other sections have sensible defaults.
    """
    task: TaskConfig
    data: DataConfig
    features: FeaturesConfig
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
