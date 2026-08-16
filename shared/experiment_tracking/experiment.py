import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.utilities.unique_ids import unique_prefixed_id

logger = logging.getLogger(__name__)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace one complete file using a unique same-directory temporary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))

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
    stage_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    hypothesis_context_path: str | None = None
    next_run_candidates_path: str | None = None
    source_lock_uri: str | None = None
    source_lock_sha256: str | None = None
    resolved_config_uri: str | None = None
    resolved_config_sha256: str | None = None

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
    timestamp_id = unique_prefixed_id("exp_", now=now)
    
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
    """Atomically save experiment.json without rewriting records during reads."""
    exp_dir = Path(base_dir) / "experiments" / experiment.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    exp_file = exp_dir / "experiment.json"
    payload = json.dumps(experiment.to_dict(), indent=2).encode("utf-8")
    _atomic_write_bytes(exp_file, payload)

def load_experiment(experiment_id: str, base_dir: Path | str = ".tracking") -> Experiment:
    """Load an experiment.json from disk."""
    exp_file = Path(base_dir) / "experiments" / experiment_id / "experiment.json"
    
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment file not found: {exp_file}")
        
    with open(exp_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return Experiment.from_dict(data)
