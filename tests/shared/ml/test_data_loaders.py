"""Tests for shared/ml/data_loaders.py — multi-format dataset loading."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from shared.ml.data_loaders import load_dataset


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_loads_csv(self, classification_csv: Path):
        df = load_dataset(classification_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        assert "target" in df.columns

    def test_validates_target_column(self, classification_csv: Path):
        df = load_dataset(classification_csv, target_column="target")
        assert "target" in df.columns

    def test_rejects_missing_target_column(self, classification_csv: Path):
        with pytest.raises(ValueError, match="Target column 'nonexistent'"):
            load_dataset(classification_csv, target_column="nonexistent")


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

class TestLoadJSONL:
    def test_loads_jsonl(self, classification_jsonl: Path):
        df = load_dataset(classification_jsonl)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200

    def test_handles_blank_lines(self, tmp_path: Path):
        p = tmp_path / "with_blanks.jsonl"
        with open(p, "w") as f:
            f.write('{"a": 1, "b": 2}\n')
            f.write("\n")
            f.write('{"a": 3, "b": 4}\n')
        df = load_dataset(p)
        assert len(df) == 2

    def test_rejects_invalid_json_line(self, tmp_path: Path):
        p = tmp_path / "bad.jsonl"
        with open(p, "w") as f:
            f.write('{"a": 1}\n')
            f.write("not valid json\n")
        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            load_dataset(p)


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------

class TestLoadParquet:
    def test_loads_parquet(self, tmp_path: Path, classification_df: pd.DataFrame):
        p = tmp_path / "data.parquet"
        classification_df.to_parquet(p, index=False)
        df = load_dataset(p)
        assert len(df) == 200
        assert list(df.columns) == list(classification_df.columns)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestLoadErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset("/nonexistent/path/data.csv")

    def test_unsupported_format(self, tmp_path: Path):
        p = tmp_path / "data.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(p)

    def test_empty_csv(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("col_a,col_b\n")
        with pytest.raises(ValueError, match="Dataset is empty"):
            load_dataset(p)

    def test_empty_jsonl(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        with pytest.raises(ValueError, match="Dataset is empty"):
            load_dataset(p)
