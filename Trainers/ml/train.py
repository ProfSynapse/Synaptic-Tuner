# Trainers/ml/train.py
# Main training entry point for the ML pipeline.
# Orchestrates: YAML config -> data loading -> feature building ->
# pipeline construction -> training -> evaluation -> artifact saving -> tracking.
#
# Usage:
#     python -m Trainers.ml.train --config Trainers/ml/configs/templates/classification.yaml

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from Trainers.ml.config import TrainingConfig
from Trainers.ml.data.splitter import load_and_split
from Trainers.ml.evaluation.metrics import evaluate_model
from Trainers.ml.evaluation.report import build_metrics_report
from Trainers.ml.output import create_run_dir, save_run_artifacts
from Trainers.ml.pipeline_builder import build_pipeline

logger = logging.getLogger(__name__)


def main(config_path: str) -> Path:
    """Run the full training pipeline.

    Steps:
    1. Load and validate config (Pydantic)
    2. Load data and split
    3. Build sklearn Pipeline (preprocessor + estimator)
    4. Fit pipeline on training data
    5. Evaluate on test data
    6. Save artifacts to timestamped output directory
    7. Log to experiment tracker

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Path to the output run directory.
    """
    # 1. Config
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    config = TrainingConfig(**raw_config)
    logger.info("Training: %s (%s)", config.task.name, config.task.type.value)

    # 2. Data
    X_train, X_test, y_train, y_test = load_and_split(config)
    logger.info("Data: %d train, %d test", len(X_train), len(X_test))

    # 3. Pipeline
    n_classes = int(y_train.nunique()) if config.task.type.value == "classification" else None
    pipeline = build_pipeline(config, n_classes=n_classes)
    logger.info("Algorithm: %s", config.algorithm.name)

    # 4. Fit
    logger.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    # 5. Evaluate
    metrics = evaluate_model(
        pipeline, X_test, y_test,
        config.task.type.value, config.evaluation.metrics,
    )

    # 6. Save
    run_dir = create_run_dir(config.output.dir)
    report = build_metrics_report(config, X_train, X_test, metrics)
    schema = _build_schema(X_train, y_train, config, pipeline)
    save_run_artifacts(run_dir, raw_config, pipeline, report, schema)

    # Optional plots
    if config.evaluation.generate_plots:
        from Trainers.ml.evaluation.plots import generate_plots
        import numpy as np

        y_pred = pipeline.predict(X_test)
        generate_plots(
            pipeline, X_test, y_test, np.asarray(y_pred),
            config.task.type.value, run_dir,
        )

    # 7. Track
    from shared.experiment_tracking import create_tracker

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

    logger.info("Output: %s", run_dir)
    return run_dir


def _build_schema(
    X_train, y_train, config: TrainingConfig, pipeline
) -> dict:
    """Build schema.json content describing the feature schema.

    Captures column names by type, target info, and post-transform feature count.
    """
    schema: dict = {
        "columns": {},
        "target": config.task.target_column,
        "n_features_after_transform": None,
    }
    if config.features.numeric:
        schema["columns"]["numeric"] = config.features.numeric.columns
    if config.features.categorical:
        schema["columns"]["categorical"] = config.features.categorical.columns
    if config.task.type.value == "classification":
        schema["target_classes"] = sorted(
            y_train.unique().astype(str).tolist()
        )

    # Populate n_features_after_transform from fitted preprocessor
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
            schema["n_features_after_transform"] = len(
                preprocessor.get_feature_names_out()
            )
    except Exception as e:
        logger.warning("Could not determine n_features_after_transform: %s", e)

    return schema


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an ML model",
        epilog="Example: python -m Trainers.ml.train --config config.yaml",
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(args.config)
