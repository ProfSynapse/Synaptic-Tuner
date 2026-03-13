# Trainers/ml/pipeline_builder.py
# Assembles a complete sklearn Pipeline from TrainingConfig.
# Wires preprocessor (ColumnTransformer) + estimator (from algorithm registry).
# Used by train.py as the central pipeline construction step.

from __future__ import annotations

import logging

from sklearn.pipeline import Pipeline

from Trainers.ml.algorithms import get_algorithm
from Trainers.ml.config import TrainingConfig
from Trainers.ml.features import build_preprocessor

logger = logging.getLogger(__name__)


def build_pipeline(
    config: TrainingConfig, n_classes: int | None = None,
) -> Pipeline:
    """Construct a complete sklearn Pipeline from training config.

    Steps:
    1. Build preprocessor from features config (ColumnTransformer)
    2. Look up algorithm from registry
    3. Create estimator with merged params (defaults + user overrides)
    4. Return Pipeline([("preprocessor", ct), ("estimator", est)])

    Args:
        config: Validated TrainingConfig.
        n_classes: Number of target classes (classification only).
                   Used to select binary vs multiclass objective.

    Returns:
        Unfitted sklearn Pipeline.
    """
    # Build feature preprocessor
    preprocessor = build_preprocessor(config.features)
    logger.info("Built preprocessor with %d transformer(s)", len(preprocessor.transformers))

    # Look up algorithm and create estimator
    wrapper = get_algorithm(config.algorithm.name)
    task_type = config.task.type.value

    if task_type == "classification" and not wrapper.supports_classification:
        raise ValueError(
            f"Algorithm '{wrapper.name}' does not support classification"
        )
    if task_type == "regression" and not wrapper.supports_regression:
        raise ValueError(
            f"Algorithm '{wrapper.name}' does not support regression"
        )

    estimator = wrapper.create_estimator(task_type, config.algorithm.params, n_classes)
    logger.info(
        "Created estimator: %s (%s)",
        wrapper.name, type(estimator).__name__,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])
