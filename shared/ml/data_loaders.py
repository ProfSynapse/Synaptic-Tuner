"""
shared/ml/data_loaders.py

Multi-format dataset loading for the ML training pipeline. Supports CSV,
Parquet, JSONL, and Excel files with automatic format detection from
file extension.

Used by: Trainers/ml/data/splitter.py via load_dataset().
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

# Supported file extensions mapped to loader names (for error messages)
_SUPPORTED_FORMATS = {
    ".csv": "CSV",
    ".parquet": "Parquet",
    ".jsonl": "JSONL (line-delimited JSON)",
    ".xlsx": "Excel",
    ".xls": "Excel",
}


def load_dataset(
    path: Union[str, Path],
    *,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Load a dataset from CSV, Parquet, JSONL, or Excel.

    Detects format from file extension:
        - .csv     -> pd.read_csv()
        - .parquet -> pd.read_parquet()
        - .jsonl   -> line-by-line JSON -> pd.DataFrame
        - .xlsx/.xls -> pd.read_excel()

    Args:
        path: Path to the dataset file.
        target_column: If provided, validates that this column exists
                       in the loaded DataFrame.

    Returns:
        pandas DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file format is unsupported, the file is empty,
                    or the target_column is not found.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in _SUPPORTED_FORMATS:
        supported = ", ".join(sorted(_SUPPORTED_FORMATS.keys()))
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported: {supported}"
        )

    logger.info("Loading dataset from %s (%s)", file_path, _SUPPORTED_FORMATS[suffix])

    df = _load_by_format(file_path, suffix)

    if df.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    if target_column is not None and target_column not in df.columns:
        available = ", ".join(df.columns.tolist()[:20])
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {available}"
        )

    logger.info(
        "Loaded %d rows, %d columns from %s",
        len(df),
        len(df.columns),
        file_path.name,
    )
    return df


def _load_by_format(file_path: Path, suffix: str) -> pd.DataFrame:
    """Dispatch to the appropriate pandas loader based on file extension.

    Args:
        file_path: Path to the dataset file.
        suffix: Lowercase file extension (e.g., ".csv").

    Returns:
        pandas DataFrame.
    """
    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix == ".parquet":
        return pd.read_parquet(file_path)

    if suffix == ".jsonl":
        return _load_jsonl(file_path)

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)

    # Should not reach here due to pre-check, but be defensive
    raise ValueError(f"Unhandled format: {suffix}")


def _load_jsonl(file_path: Path) -> pd.DataFrame:
    """Load a JSONL file (one JSON object per line) into a DataFrame.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        pandas DataFrame with one row per JSON line.

    Raises:
        ValueError: If any line contains invalid JSON.
    """
    records: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {file_path}: {e}"
                ) from e

    return pd.DataFrame(records)
