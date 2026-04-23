# Architecture: ML Pipeline Foundation (Phase 0-1)

> Scope: Foundation infrastructure + Minimal Viable Trainer
> Phase 0: `shared/ml/`, `shared/experiment_tracking/`
> Phase 1: `Trainers/ml/` with LightGBM, numeric + categorical features
> Reference: [Approved plan](../plans/ml-pipeline-expansion-plan.md) | [Research](../preparation/ml-pipeline-expansion-research.md)

---

## 1. Module Dependency Diagram

```
                          tuner.py (CLI — Phase 4, out of scope)
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Trainers/ml/train.py   │  ◄── Main entry point (Phase 1)
                 │   (Training Orchestrator) │
                 └──────┬──────┬──────┬─────┘
                        │      │      │
           ┌────────────┘      │      └────────────┐
           ▼                   ▼                    ▼
┌──────────────────┐  ┌───────────────┐  ┌─────────────────────┐
│ Trainers/ml/     │  │ Trainers/ml/  │  │ Trainers/ml/        │
│ config.py        │  │ algorithms/   │  │ features/builder.py  │
│ (Pydantic)       │  │ registry.py   │  │ (ColumnTransformer)  │
└──────┬───────────┘  └───────┬───────┘  └──────────┬──────────┘
       │                      │                      │
       │              ┌───────┴───────┐              │
       │              │ lightgbm_     │              │
       │              │ wrapper.py    │              │
       │              └───────────────┘              │
       │                                             │
       ▼                                             ▼
┌──────────────────────────────────────────────────────────────┐
│                     shared/ (Phase 0)                         │
│                                                               │
│  ┌─────────────────┐  ┌──────────────────────────────────┐   │
│  │ shared/ml/      │  │ shared/experiment_tracking/       │   │
│  │  data_loaders   │  │  tracker.py (ABC)                │   │
│  │  metrics        │  │  mlflow_tracker.py               │   │
│  └────────┬────────┘  │  local_tracker.py (JSON fallback)│   │
│           │           └──────────────────────────────────┘   │
│           ▼                                                   │
│  ┌─────────────────┐                                         │
│  │ shared/         │                                         │
│  │  utilities/     │  ◄── Existing: yaml_loader, paths, env │
│  └─────────────────┘                                         │
└──────────────────────────────────────────────────────────────┘
```

**Dependency rules:**
- `Trainers/ml/` depends on `shared/ml/` and `shared/experiment_tracking/`
- `shared/ml/` depends only on `shared/utilities/` (existing)
- `shared/experiment_tracking/` has zero internal dependencies (standalone)
- No circular dependencies; arrows flow downward only

---

## 2. Interface Contracts

### Contract 1: TrainingConfig (Pydantic)

The root configuration model. Loaded from YAML, validated at parse time.

```python
"""Trainers/ml/config.py"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TaskConfig(BaseModel):
    """What to predict and how to measure it."""
    type: TaskType
    name: str = Field(..., min_length=1, description="Human-readable task name")
    target_column: str = Field(..., min_length=1)
    eval_metric: str = Field(
        default="f1_weighted",
        description="Primary optimization metric. "
        "Classification: accuracy, f1_weighted, roc_auc. "
        "Regression: rmse, mae, r2."
    )
    random_state: int = 42


class DataConfig(BaseModel):
    """Where to find data and how to split it."""
    train_path: str = Field(..., description="Path to training data (CSV, Parquet, JSONL)")
    test_path: Optional[str] = Field(
        default=None,
        description="Explicit test set. If None, split from train_path."
    )
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    stratify: bool = Field(
        default=True,
        description="Stratified split for classification tasks"
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
        description="Algorithm name from registry"
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
    """Root configuration — the contract between YAML and Python."""
    task: TaskConfig
    data: DataConfig
    features: FeaturesConfig
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
```

**Loading pattern:**
```python
import yaml
from Trainers.ml.config import TrainingConfig

with open("config.yaml") as f:
    raw = yaml.safe_load(f)

config = TrainingConfig(**raw)  # Validates on construction
```

**Design notes:**
- Phase 1 omits `tuning` (Optuna) and `cross_validation` sections — those are Phase 2+
- `FeaturesConfig` omits `text` — Phase 2+
- All fields have sensible defaults; minimal valid config requires only `task`, `data`, `features`
- `field_validator` on `train_path` gives immediate feedback on missing files
- The YAML config for SFT uses `yaml.safe_load` + manual `dict_to_dataclass` (see `Trainers/sft/configs/config_loader.py`). Pydantic replaces that entire conversion + validation layer.

---

### Contract 2: AlgorithmWrapper (ABC)

The Strategy pattern interface for pluggable algorithms.

```python
"""Trainers/ml/algorithms/registry.py"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sklearn.base import BaseEstimator


class AlgorithmWrapper(ABC):
    """Base class for all algorithm wrappers.

    Each wrapper knows how to create a scikit-learn compatible estimator
    for a given task type, and can report its capabilities and defaults.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry key (e.g., 'lightgbm')."""
        ...

    @property
    @abstractmethod
    def supports_classification(self) -> bool:
        ...

    @property
    @abstractmethod
    def supports_regression(self) -> bool:
        ...

    @property
    @abstractmethod
    def supports_gpu(self) -> bool:
        ...

    @abstractmethod
    def create_estimator(
        self,
        task_type: str,
        params: dict[str, Any],
    ) -> BaseEstimator:
        """Create a configured scikit-learn estimator.

        Args:
            task_type: "classification" or "regression"
            params: User-provided hyperparameters from config.
                    Merged over defaults (user params win).

        Returns:
            A scikit-learn compatible estimator (implements fit/predict).

        Raises:
            ValueError: If task_type is unsupported by this algorithm.
        """
        ...

    @abstractmethod
    def get_default_params(self, task_type: str) -> dict[str, Any]:
        """Return sensible default hyperparameters for this task type."""
        ...


# ---------------------------------------------------------------------------
# Registry: name -> wrapper class
# ---------------------------------------------------------------------------

_ALGORITHM_REGISTRY: dict[str, type[AlgorithmWrapper]] = {}


def register_algorithm(name: str):
    """Decorator to register an algorithm wrapper.

    Usage:
        @register_algorithm("lightgbm")
        class LightGBMWrapper(AlgorithmWrapper):
            ...
    """
    def decorator(cls: type[AlgorithmWrapper]) -> type[AlgorithmWrapper]:
        _ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator


def get_algorithm(name: str) -> AlgorithmWrapper:
    """Look up and instantiate an algorithm wrapper by name.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {available}"
        )
    return _ALGORITHM_REGISTRY[name]()


def list_algorithms() -> list[str]:
    """Return sorted list of registered algorithm names."""
    return sorted(_ALGORITHM_REGISTRY.keys())
```

**LightGBM wrapper (Phase 1 — sole algorithm):**
```python
"""Trainers/ml/algorithms/lightgbm_wrapper.py"""
from typing import Any

from sklearn.base import BaseEstimator

from .registry import AlgorithmWrapper, register_algorithm


@register_algorithm("lightgbm")
class LightGBMWrapper(AlgorithmWrapper):

    @property
    def name(self) -> str:
        return "lightgbm"

    @property
    def supports_classification(self) -> bool:
        return True

    @property
    def supports_regression(self) -> bool:
        return True

    @property
    def supports_gpu(self) -> bool:
        return True

    def get_default_params(self, task_type: str) -> dict[str, Any]:
        base = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "random_state": 42,
            "verbosity": -1,
        }
        if task_type == "classification":
            base["objective"] = "binary"
            base["metric"] = "binary_logloss"
        else:
            base["objective"] = "regression"
            base["metric"] = "rmse"
        return base

    def create_estimator(
        self, task_type: str, params: dict[str, Any]
    ) -> BaseEstimator:
        from lightgbm import LGBMClassifier, LGBMRegressor

        if task_type not in ("classification", "regression"):
            raise ValueError(f"Unsupported task type: {task_type}")

        merged = {**self.get_default_params(task_type), **params}

        if task_type == "classification":
            return LGBMClassifier(**merged)
        return LGBMRegressor(**merged)
```

**Registration mechanism:** Algorithm wrappers are discovered via import. The `Trainers/ml/algorithms/__init__.py` imports all wrapper modules:
```python
"""Trainers/ml/algorithms/__init__.py"""
from .registry import get_algorithm, list_algorithms, AlgorithmWrapper
from . import lightgbm_wrapper  # Triggers @register_algorithm
```

Future algorithms (Phase 2+) simply add a new wrapper file + import line.

---

### Contract 3: ExperimentTracker (ABC)

Facade over experiment tracking backends.

```python
"""shared/experiment_tracking/tracker.py"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking.

    Implementations: MLflowTracker, LocalTracker (JSON fallback).
    """

    @abstractmethod
    def set_experiment(self, experiment_name: str) -> None:
        """Create or select an experiment."""
        ...

    @abstractmethod
    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        """Context manager for a tracking run."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics."""
        ...

    @abstractmethod
    def log_artifact(self, local_path: str) -> None:
        """Log a file artifact."""
        ...
```

**LocalTracker (JSON fallback — always available):**
```python
"""shared/experiment_tracking/local_tracker.py"""
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from .tracker import ExperimentTracker


class LocalTracker(ExperimentTracker):
    """Writes tracking data to a JSON file in the output directory.

    This is the zero-dependency fallback. Always writes metrics.json
    regardless of whether MLflow is installed.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._experiment_name = "default"
        self._run_data: dict[str, Any] = {}

    def set_experiment(self, experiment_name: str) -> None:
        self._experiment_name = experiment_name

    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        self._run_data = {
            "experiment": self._experiment_name,
            "run_name": run_name,
            "started_at": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
        }
        try:
            yield
        finally:
            self._run_data["ended_at"] = datetime.now().isoformat()
            self._output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self._output_dir / "tracking.json"
            with open(out_path, "w") as f:
                json.dump(self._run_data, f, indent=2)

    def log_params(self, params: dict[str, Any]) -> None:
        self._run_data.setdefault("params", {}).update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._run_data.setdefault("metrics", {}).update(metrics)

    def log_artifact(self, local_path: str) -> None:
        self._run_data.setdefault("artifacts", []).append(local_path)
```

**MLflowTracker (optional — only if mlflow installed):**
```python
"""shared/experiment_tracking/mlflow_tracker.py"""
from contextlib import contextmanager
from typing import Any, Generator

from .tracker import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    """MLflow-backed experiment tracker.

    Requires: pip install mlflow
    """

    def __init__(self) -> None:
        import mlflow  # Fail fast if not installed
        self._mlflow = mlflow

    def set_experiment(self, experiment_name: str) -> None:
        self._mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        with self._mlflow.start_run(run_name=run_name):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        self._mlflow.log_artifact(local_path)
```

**Factory function:**
```python
"""shared/experiment_tracking/__init__.py"""
from .tracker import ExperimentTracker
from .local_tracker import LocalTracker


def create_tracker(backend: str = "local", output_dir: str = ".") -> ExperimentTracker:
    """Create an experiment tracker by backend name.

    Args:
        backend: "mlflow" or "local"
        output_dir: Used by LocalTracker for JSON output

    Returns:
        ExperimentTracker instance
    """
    if backend == "mlflow":
        try:
            from .mlflow_tracker import MLflowTracker
            return MLflowTracker()
        except ImportError:
            import warnings
            warnings.warn(
                "mlflow not installed, falling back to local JSON tracker. "
                "Install with: pip install mlflow"
            )
            return LocalTracker(output_dir)
    return LocalTracker(output_dir)
```

---

### Contract 4: build_preprocessor

Converts the `FeaturesConfig` into a scikit-learn `ColumnTransformer`.

```python
"""Trainers/ml/features/builder.py"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from Trainers.ml.config import FeaturesConfig


_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def build_preprocessor(config: FeaturesConfig) -> ColumnTransformer:
    """Build a ColumnTransformer from feature configuration.

    Args:
        config: Validated FeaturesConfig from YAML.

    Returns:
        Unfitted ColumnTransformer ready for pipeline.fit().

    Phase 1 scope: numeric + categorical.
    Phase 2 adds: text (TfidfVectorizer).
    """
    transformers: list[tuple[str, Pipeline | str, list[str]]] = []

    # --- Numeric pipeline ---
    if config.numeric is not None:
        steps = []
        nc = config.numeric
        if nc.imputer != "none":
            steps.append(("imputer", SimpleImputer(strategy=nc.imputer)))
        if nc.scaler != "none":
            steps.append(("scaler", _SCALERS[nc.scaler]()))
        if steps:
            transformers.append(("numeric", Pipeline(steps), nc.columns))
        else:
            transformers.append(("numeric", "passthrough", nc.columns))

    # --- Categorical pipeline ---
    if config.categorical is not None:
        cc = config.categorical
        imputer = SimpleImputer(strategy="constant", fill_value="__missing__")
        if cc.encoder == "onehot":
            encoder = OneHotEncoder(
                handle_unknown=cc.handle_unknown,
                sparse_output=False,
            )
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        transformers.append((
            "categorical",
            Pipeline([("imputer", imputer), ("encoder", encoder)]),
            cc.columns,
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop columns not in any transformer
    )
```

---

### Contract 5: Output Directory Structure

Every training run produces a timestamped directory with deterministic contents.

```
Trainers/ml/ml_output/
└── YYYYMMDD_HHMMSS/              # Timestamped run directory
    ├── config.yaml               # Frozen copy of the config that produced this run
    ├── model.joblib              # Serialized sklearn Pipeline (preprocessor + estimator)
    ├── metrics.json              # Evaluation metrics (always written)
    ├── schema.json               # Input feature schema (column names, dtypes, categories)
    ├── tracking.json             # LocalTracker output (if backend=local)
    └── logs/
        └── training.log          # Human-readable training log
```

**`metrics.json` schema:**
```json
{
  "task": {
    "type": "classification",
    "name": "customer_churn",
    "target_column": "churned",
    "eval_metric": "f1_weighted"
  },
  "algorithm": "lightgbm",
  "dataset": {
    "train_samples": 5634,
    "test_samples": 1409,
    "features": 15
  },
  "test_metrics": {
    "accuracy": 0.891,
    "f1_weighted": 0.847
  },
  "timestamp": "2026-03-13T15:30:00"
}
```

**`schema.json` schema:**
```json
{
  "columns": {
    "numeric": ["age", "tenure", "monthly_charges"],
    "categorical": ["contract_type", "payment_method"]
  },
  "target": "churned",
  "target_classes": ["0", "1"],
  "n_features_after_transform": 18
}
```

**Output directory creation:**
```python
"""Trainers/ml/output.py"""
import json
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_dir(base_dir: str) -> Path:
    """Create a timestamped run directory.

    Returns:
        Path to the new run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def save_run_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    pipeline: Any,
    metrics: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Save all standard artifacts to the run directory.

    Args:
        run_dir: Timestamped output directory.
        config: Raw config dict (for frozen YAML copy).
        pipeline: Fitted sklearn Pipeline.
        metrics: Evaluation results.
        schema: Feature schema info.
    """
    import joblib

    # Frozen config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Model
    joblib.dump(pipeline, run_dir / "model.joblib")

    # Metrics (always written — this is the zero-dependency tracking)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Schema
    with open(run_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)
```

---

## 3. File-by-File Specification (Phase 0 + 1)

### Phase 0: Foundation (`shared/`)

| File | Purpose | Public API |
|------|---------|------------|
| `shared/ml/__init__.py` | Package init | Re-exports `load_dataset`, `compute_metrics` |
| `shared/ml/data_loaders.py` | Multi-format dataset loading | `load_dataset(path: str) -> pd.DataFrame` — detects format from extension (.csv, .parquet, .jsonl) |
| `shared/ml/metrics.py` | Metric computation helpers | `compute_metrics(y_true, y_pred, y_proba, task_type, metric_names) -> dict[str, float]` |
| `shared/experiment_tracking/__init__.py` | Package init + factory | `create_tracker(backend, output_dir) -> ExperimentTracker` |
| `shared/experiment_tracking/tracker.py` | ABC definition | `ExperimentTracker` (see Contract 3) |
| `shared/experiment_tracking/local_tracker.py` | JSON fallback | `LocalTracker(output_dir)` |
| `shared/experiment_tracking/mlflow_tracker.py` | MLflow impl | `MLflowTracker()` |

**`shared/ml/data_loaders.py` — full signature:**
```python
def load_dataset(
    path: str | Path,
    *,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Load a dataset from CSV, Parquet, or JSONL.

    Detects format from file extension:
    - .csv -> pd.read_csv()
    - .parquet -> pd.read_parquet()
    - .jsonl -> line-by-line JSON -> pd.DataFrame

    Args:
        path: Path to dataset file.
        target_column: If provided, validates column exists.

    Returns:
        pandas DataFrame.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If format unsupported or target_column missing.
    """
```

**`shared/ml/metrics.py` — full signature:**
```python
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    task_type: str,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute requested metrics for a task type.

    Supported classification metrics: accuracy, f1_weighted, f1_macro,
        roc_auc, precision, recall, log_loss.
    Supported regression metrics: rmse, mae, r2, mape.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Predicted labels/values.
        y_proba: Predicted probabilities (classification only, optional).
        task_type: "classification" or "regression".
        metric_names: List of metric names to compute.

    Returns:
        Dict mapping metric name to computed value.

    Raises:
        ValueError: If metric name unknown for given task type.
    """
```

### Phase 1: ML Trainer (`Trainers/ml/`)

| File | Purpose | Public API |
|------|---------|------------|
| `Trainers/ml/__init__.py` | Package init | Empty (or version) |
| `Trainers/ml/train.py` | Main entry point | `main(config_path: str) -> Path` (returns run_dir); CLI via `if __name__ == "__main__"` |
| `Trainers/ml/config.py` | Pydantic config models | `TrainingConfig` and all sub-models (see Contract 1) |
| `Trainers/ml/pipeline_builder.py` | Assembles the full sklearn Pipeline | `build_pipeline(config: TrainingConfig) -> Pipeline` |
| `Trainers/ml/output.py` | Run directory + artifact saving | `create_run_dir(base_dir)`, `save_run_artifacts(...)` (see Contract 5) |
| `Trainers/ml/algorithms/__init__.py` | Registry exports + wrapper imports | `get_algorithm`, `list_algorithms`, `AlgorithmWrapper` |
| `Trainers/ml/algorithms/registry.py` | ABC + registry dict + decorator | See Contract 2 |
| `Trainers/ml/algorithms/lightgbm_wrapper.py` | LightGBM wrapper | `LightGBMWrapper` (auto-registered) |
| `Trainers/ml/features/__init__.py` | Package init | Re-exports `build_preprocessor` |
| `Trainers/ml/features/builder.py` | Config -> ColumnTransformer | `build_preprocessor(config: FeaturesConfig) -> ColumnTransformer` (see Contract 4) |
| `Trainers/ml/data/__init__.py` | Package init | Re-exports `load_and_split` |
| `Trainers/ml/data/splitter.py` | Train/test splitting | `load_and_split(config: TrainingConfig) -> tuple[DataFrame, DataFrame, Series, Series]` |
| `Trainers/ml/configs/templates/classification.yaml` | Starter template | Example config for classification with LightGBM |

**`Trainers/ml/pipeline_builder.py` — full signature:**
```python
from sklearn.pipeline import Pipeline
from Trainers.ml.config import TrainingConfig


def build_pipeline(config: TrainingConfig) -> Pipeline:
    """Construct a complete sklearn Pipeline from training config.

    Steps:
    1. Build preprocessor from features config (ColumnTransformer)
    2. Look up algorithm from registry
    3. Create estimator with merged params (defaults + user overrides)
    4. Return Pipeline([("preprocessor", ct), ("estimator", est)])

    Args:
        config: Validated TrainingConfig.

    Returns:
        Unfitted sklearn Pipeline.
    """
```

**`Trainers/ml/data/splitter.py` — full signature:**
```python
import pandas as pd

from Trainers.ml.config import TrainingConfig


def load_and_split(
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset and split into train/test.

    If config.data.test_path is provided, loads train and test separately.
    Otherwise, splits config.data.train_path using config.data.test_size.

    Stratification is applied when config.data.stratify is True
    and task type is classification.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
```

**`Trainers/ml/train.py` — orchestrator structure:**
```python
"""Main training entry point.

Usage:
    python -m Trainers.ml.train --config Trainers/ml/configs/templates/classification.yaml
"""
import argparse
import logging
import yaml
from pathlib import Path

from Trainers.ml.config import TrainingConfig
from Trainers.ml.data.splitter import load_and_split
from Trainers.ml.pipeline_builder import build_pipeline
from Trainers.ml.output import create_run_dir, save_run_artifacts
from shared.experiment_tracking import create_tracker
from shared.ml.metrics import compute_metrics

logger = logging.getLogger(__name__)


def main(config_path: str) -> Path:
    """Run the full training pipeline.

    1. Load and validate config (Pydantic)
    2. Load data and split
    3. Build sklearn Pipeline (preprocessor + estimator)
    4. Fit pipeline on training data
    5. Evaluate on test data
    6. Save artifacts to timestamped output directory
    7. Log to experiment tracker

    Returns:
        Path to the output run directory.
    """
    # 1. Config
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    config = TrainingConfig(**raw_config)
    logger.info(f"Training: {config.task.name} ({config.task.type.value})")

    # 2. Data
    X_train, X_test, y_train, y_test = load_and_split(config)
    logger.info(f"Data: {len(X_train)} train, {len(X_test)} test")

    # 3. Pipeline
    pipeline = build_pipeline(config)
    logger.info(f"Algorithm: {config.algorithm.name}")

    # 4. Fit
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            pass
    metrics = compute_metrics(
        y_test.values, y_pred, y_proba,
        config.task.type.value, config.evaluation.metrics,
    )
    logger.info(f"Metrics: {metrics}")

    # 6. Save
    run_dir = create_run_dir(config.output.dir)
    schema = _build_schema(X_train, y_train, config, pipeline)
    save_run_artifacts(run_dir, raw_config, pipeline, {
        "task": {"type": config.task.type.value, "name": config.task.name,
                 "target_column": config.task.target_column,
                 "eval_metric": config.task.eval_metric},
        "algorithm": config.algorithm.name,
        "dataset": {"train_samples": len(X_train),
                     "test_samples": len(X_test),
                     "features": X_train.shape[1]},
        "test_metrics": metrics,
    }, schema)

    # 7. Track
    tracker = create_tracker(
        config.tracking.backend if config.tracking.enabled else "local",
        str(run_dir),
    )
    tracker.set_experiment(config.tracking.experiment_name)
    with tracker.start_run(config.task.name):
        tracker.log_params({
            "algorithm": config.algorithm.name,
            "task_type": config.task.type.value,
            **{f"algo_{k}": v for k, v in config.algorithm.params.items()},
        })
        tracker.log_metrics(metrics)

    logger.info(f"Output: {run_dir}")
    return run_dir


def _build_schema(X_train, y_train, config, pipeline):
    """Build schema.json content."""
    schema = {
        "columns": {},
        "target": config.task.target_column,
        "n_features_after_transform": None,
    }
    if config.features.numeric:
        schema["columns"]["numeric"] = config.features.numeric.columns
    if config.features.categorical:
        schema["columns"]["categorical"] = config.features.categorical.columns
    if config.task.type.value == "classification":
        schema["target_classes"] = sorted(y_train.unique().astype(str).tolist())
    # n_features_after_transform populated after fit
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
            schema["n_features_after_transform"] = len(
                preprocessor.get_feature_names_out()
            )
    except Exception:
        pass
    return schema


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ML model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(args.config)
```

---

## 4. Data Flow

```
YAML config file
      │
      ▼
TrainingConfig(**yaml.safe_load(f))    ← Pydantic validates all fields
      │
      ├──► DataConfig.train_path
      │         │
      │         ▼
      │    load_dataset(path)           ← shared/ml/data_loaders.py
      │         │
      │         ▼
      │    load_and_split(config)       ← Trainers/ml/data/splitter.py
      │         │                         (train_test_split, stratify)
      │         ▼
      │    X_train, X_test, y_train, y_test
      │
      ├──► FeaturesConfig
      │         │
      │         ▼
      │    build_preprocessor(config.features)  ← ColumnTransformer
      │         │
      │         ▼
      │    Pipeline step 1: preprocessor
      │
      ├──► AlgorithmConfig.name
      │         │
      │         ▼
      │    get_algorithm(name)          ← registry lookup
      │         │
      │         ▼
      │    wrapper.create_estimator(task_type, params)
      │         │
      │         ▼
      │    Pipeline step 2: estimator
      │
      └──► build_pipeline(config) → Pipeline([preprocessor, estimator])
                  │
                  ▼
             pipeline.fit(X_train, y_train)
                  │
                  ▼
             pipeline.predict(X_test)
                  │
                  ▼
             compute_metrics(y_test, y_pred, y_proba, ...)
                  │
                  ▼
             save_run_artifacts(run_dir, ...)
                  │
                  ├── config.yaml (frozen)
                  ├── model.joblib
                  ├── metrics.json
                  └── schema.json
```

---

## 5. Integration Points with Existing Code

### `shared/utilities/yaml_loader.py`
- Already provides `load_yaml(file_path) -> dict`
- `Trainers/ml/train.py` can use this instead of raw `yaml.safe_load`, but the function is trivial — either approach works
- **Decision**: Use `yaml.safe_load` directly in train.py for zero coupling to the utility. The Pydantic model handles validation; the YAML loader adds no value

### `shared/utilities/paths.py`
- Provides `get_project_root()`, `get_trainer_root(name)`, `find_training_run(name, id)`
- **Integration**: `find_training_run` currently hardcodes `sft_output` and `kto_output` patterns. Phase 4 (CLI) will need to extend this or add an ML-specific variant
- **Phase 1**: No changes needed. `Trainers/ml/output.py` handles its own directory creation

### `shared/utilities/env.py`
- Loads `.env` for API keys
- **Integration**: Not needed for Phase 0-1 (ML training is local). Phase 4 may need it for MLflow remote tracking

### `shared/validation/`
- Existing validators (XML, JSON, YAML, regex, code) for dataset quality
- **Future integration** (Phase 4+): Validation results become features for the quality scorer. The `shared/ml/features.py` module (out of Phase 1 scope) will import from `shared/validation/` to extract schema error counts, tag presence, etc.
- **Phase 1**: No integration needed

### `Trainers/sft/configs/config_loader.py`
- Uses dataclasses + manual `dict_to_dataclass()` conversion
- **Relationship**: ML config uses Pydantic instead. These are independent — no shared config infrastructure. If the project later wants to unify, SFT/KTO configs could migrate to Pydantic, but that's out of scope
- **Key difference**: The SFT config loader has no validation beyond type conversion. Pydantic validates constraints (`test_size > 0`, `columns` non-empty, `train_path` exists)

### Import Path Convention
All modules assume the project root is on `PYTHONPATH`. Imports use dotted paths from root:
```python
from shared.ml.data_loaders import load_dataset
from shared.experiment_tracking import create_tracker
from Trainers.ml.config import TrainingConfig
from Trainers.ml.algorithms import get_algorithm
```
This matches the existing convention (e.g., `from shared.validation.validators import ...`).

---

## 6. Starter Template

```yaml
# Trainers/ml/configs/templates/classification.yaml
# Minimal classification config — LightGBM on tabular data
# Usage: python -m Trainers.ml.train --config Trainers/ml/configs/templates/classification.yaml

task:
  type: classification
  name: my_classifier
  target_column: target
  eval_metric: f1_weighted

data:
  train_path: Datasets/ml/my_data.csv
  test_size: 0.2
  stratify: true

features:
  numeric:
    columns: [feature_1, feature_2, feature_3]
    imputer: median
    scaler: standard
  categorical:
    columns: [category_a, category_b]
    encoder: onehot

algorithm:
  name: lightgbm
  params:
    n_estimators: 500
    learning_rate: 0.05

evaluation:
  metrics: [accuracy, f1_weighted, roc_auc]

output:
  dir: Trainers/ml/ml_output
```

---

## 7. Design Decisions Summary

| Decision | Choice | Why |
|----------|--------|-----|
| Config validation | Pydantic v2 | User-editable YAML needs rich validation with clear error messages; eliminates manual `dict_to_dataclass` boilerplate |
| Algorithm registry | ABC + `@register_algorithm` decorator + dict | Strategy pattern: enforces interface, enables config-driven lookup, extensible by adding files |
| Experiment tracking | ABC with LocalTracker fallback | Zero-dependency baseline; MLflow opt-in; tracker swappable without changing training code |
| Feature engineering | Config -> ColumnTransformer | Leverages sklearn's battle-tested Pipeline/ColumnTransformer; YAML-declarative for common cases |
| Output structure | Timestamped dirs under `Trainers/ml/ml_output/` | Mirrors SFT pattern (`sft_output/YYYYMMDD_HHMMSS/`); always produces `metrics.json` |
| Import convention | Dotted paths from project root | Matches existing `shared.*`, `Trainers.*` convention |
| Phase 1 scope | LightGBM only, numeric + categorical only | Smallest vertical slice that proves the architecture end-to-end |
