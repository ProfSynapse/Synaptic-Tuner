# Trainers/ml/output.py
# Run directory creation and artifact serialization.
# Mirrors the SFT trainer's timestamped output pattern (YYYYMMDD_HHMMSS/).
# Used by train.py to persist config, model, metrics, and schema after training.

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def create_run_dir(base_dir: str) -> Path:
    """Create a timestamped run directory.

    Structure:
        base_dir/YYYYMMDD_HHMMSS/
            logs/

    Args:
        base_dir: Parent directory for all runs (e.g., "Trainers/ml/ml_output").

    Returns:
        Path to the new run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    logger.info("Created run directory: %s", run_dir)
    return run_dir


def save_run_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    pipeline: Any,
    metrics: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Save all standard artifacts to the run directory.

    Produces:
        config.yaml  — frozen copy of the config that produced this run
        model.joblib — serialized sklearn Pipeline (preprocessor + estimator)
        metrics.json — evaluation results
        schema.json  — input feature schema (column names, dtypes, categories)

    Args:
        run_dir: Timestamped output directory.
        config: Raw config dict (for frozen YAML copy).
        pipeline: Fitted sklearn Pipeline.
        metrics: Evaluation results dict.
        schema: Feature schema info dict.
    """
    import joblib

    # Frozen config
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved config: %s", config_path)

    # Model
    model_path = run_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Saved model: %s", model_path)

    # Metrics (always written — this is the zero-dependency tracking)
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics: %s", metrics_path)

    # Schema
    schema_path = run_dir / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info("Saved schema: %s", schema_path)
