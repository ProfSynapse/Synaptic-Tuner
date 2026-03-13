# Trainers/ml/data/splitter.py
# Dataset loading and train/test splitting with optional stratification.
# Delegates to shared/ml/data_loaders for multi-format file loading.
# Used by train.py as the first step after config parsing.

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from Trainers.ml.config import TrainingConfig

logger = logging.getLogger(__name__)


def load_and_split(
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset and split into train/test.

    If config.data.test_path is provided, loads train and test separately.
    Otherwise, splits config.data.train_path using config.data.test_size.

    Stratification is applied when config.data.stratify is True
    and task type is classification.

    Args:
        config: Validated TrainingConfig.

    Returns:
        (X_train, X_test, y_train, y_test)

    Raises:
        ValueError: If target column is missing from dataset.
    """
    from shared.ml.data_loaders import load_dataset

    target = config.task.target_column
    drop_cols = config.features.drop_columns

    # Load training data
    train_df = load_dataset(config.data.train_path, target_column=target)
    logger.info(
        "Loaded training data: %d rows, %d columns",
        len(train_df), len(train_df.columns),
    )

    # Drop explicitly excluded columns (but keep target)
    cols_to_drop = [c for c in drop_cols if c in train_df.columns and c != target]
    if cols_to_drop:
        train_df = train_df.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)

    if config.data.test_path:
        # Separate test file provided
        test_df = load_dataset(config.data.test_path, target_column=target)
        if cols_to_drop:
            test_df = test_df.drop(
                columns=[c for c in cols_to_drop if c in test_df.columns]
            )

        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]
    else:
        # Split from single file
        X = train_df.drop(columns=[target])
        y = train_df[target]

        stratify_col = (
            y if config.data.stratify
            and config.task.type.value == "classification"
            else None
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.data.test_size,
            random_state=config.task.random_state,
            stratify=stratify_col,
        )

    logger.info(
        "Split: %d train, %d test", len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test
