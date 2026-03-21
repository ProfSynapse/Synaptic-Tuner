import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class Experiment:
    """Experiment definition matching the local .tracking/experiments/{id}/ schema."""
    experiment_id: str
    name: str
    created_at: str
    dataset_path: str
    dataset_hash: str
    base_model_name: str
    run_ids: list[str] = field(default_factory=list)
    base_losses_path: str | None = None
    features_csv_path: str | None = None
    judge_scores_path: str | None = None
    status: str = "partial"
    provider: str = ""
    method: str = ""
    objective: str = ""
    spec_path: str | None = None
    training_run_id: str | None = None
    evaluation_run_id: str | None = None
    loss_run_id: str | None = None
    selected_run_id: str | None = None
    artifact_roots: dict[str, str] = field(default_factory=dict)
    derived_outputs: dict[str, str] = field(default_factory=dict)
    stage_statuses: dict[str, str] = field(default_factory=dict)
    hypothesis_context_path: str | None = None
    next_run_candidates_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experiment":
        """Deserialize from a dictionary, ignoring unknown fields."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

def create_experiment(
    name: str,
    dataset_path: str,
    dataset_hash: str,
    base_model_name: str,
    provider: str = "",
    method: str = "",
    objective: str = "",
    spec_path: str | None = None,
    base_dir: Path | str = ".tracking",
) -> Experiment:
    """Create a new experiment, write to disk, and return the metadata."""
    now = datetime.now(timezone.utc)
    # create timestamp exp_YYYYMMDD_HHMMSS
    timestamp_id = "exp_" + now.strftime("%Y%m%d_%H%M%S")
    
    experiment = Experiment(
        experiment_id=timestamp_id,
        name=name,
        created_at=now.isoformat(),
        dataset_path=dataset_path,
        dataset_hash=dataset_hash,
        base_model_name=base_model_name,
        provider=provider,
        method=method,
        objective=objective,
        spec_path=spec_path,
    )
    
    exp_dir = Path(base_dir) / "experiments" / timestamp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    save_experiment(experiment, base_dir=base_dir)
    return experiment

def save_experiment(experiment: Experiment, base_dir: Path | str = ".tracking") -> None:
    """Save an experiment.json to disk."""
    exp_dir = Path(base_dir) / "experiments" / experiment.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    exp_file = exp_dir / "experiment.json"
    with open(exp_file, "w", encoding="utf-8") as f:
        json.dump(experiment.to_dict(), f, indent=2)

def load_experiment(experiment_id: str, base_dir: Path | str = ".tracking") -> Experiment:
    """Load an experiment.json from disk."""
    exp_file = Path(base_dir) / "experiments" / experiment_id / "experiment.json"
    
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment file not found: {exp_file}")
        
    with open(exp_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return Experiment.from_dict(data)
