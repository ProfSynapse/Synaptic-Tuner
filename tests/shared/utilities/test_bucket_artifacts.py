"""Tests for shared.utilities.bucket_artifacts — path building, tail, latest JSONL."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.utilities.bucket_artifacts import (
    _artifact_relative_path,
    build_artifact_path,
    latest_jsonl_record,
    tail_lines,
)


# ---------------------------------------------------------------------------
# build_artifact_path
# ---------------------------------------------------------------------------

class TestBuildArtifactPath:
    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="required"):
            build_artifact_path("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="required"):
            build_artifact_path("   ")

    def test_none_path_raises(self):
        with pytest.raises(ValueError, match="required"):
            build_artifact_path(None)

    def test_passthrough_hf_uri(self):
        uri = "hf://buckets/my-bucket/path/to/file"
        assert build_artifact_path(uri) == uri

    def test_passthrough_absolute_local_path(self):
        assert build_artifact_path("/tmp/model") == "/tmp/model"

    def test_passthrough_relative_dotslash(self):
        assert build_artifact_path("./results/out") == "./results/out"

    def test_passthrough_relative_dotdotslash(self):
        assert build_artifact_path("../up/file") == "../up/file"

    def test_relative_with_bucket_id(self):
        result = build_artifact_path("runs/sft/latest", bucket_id="my-bucket")
        assert result == "hf://buckets/my-bucket/runs/sft/latest"

    def test_relative_without_bucket_returns_as_is(self):
        assert build_artifact_path("runs/sft/latest") == "runs/sft/latest"

    def test_bucket_id_leading_slash_stripped(self):
        result = build_artifact_path("file.json", bucket_id="/bucket/")
        assert result == "hf://buckets/bucket/file.json"


# ---------------------------------------------------------------------------
# _artifact_relative_path
# ---------------------------------------------------------------------------

class TestArtifactRelativePath:
    def test_hf_bucket_uri_extracts_relative(self):
        rel = _artifact_relative_path("runs/sft/out", bucket_id="my-bucket")
        assert rel == Path("runs/sft/out")

    def test_hf_bucket_single_segment(self):
        rel = _artifact_relative_path("file.json", bucket_id="bucket")
        assert rel == Path("file.json")

    def test_local_relative_path(self):
        rel = _artifact_relative_path("local/path/file.txt")
        assert rel == Path("local/path/file.txt")


# ---------------------------------------------------------------------------
# tail_lines
# ---------------------------------------------------------------------------

class TestTailLines:
    def test_returns_last_n_lines(self):
        lines = ["a\n", "b\n", "c\n", "d\n", "e\n"]
        assert tail_lines(lines, 3) == ["c\n", "d\n", "e\n"]

    def test_count_zero_returns_all(self):
        lines = ["a\n", "b\n"]
        assert tail_lines(lines, 0) == ["a\n", "b\n"]

    def test_negative_count_returns_all(self):
        lines = ["a\n", "b\n"]
        assert tail_lines(lines, -1) == ["a\n", "b\n"]

    def test_count_exceeds_length(self):
        lines = ["a\n", "b\n"]
        assert tail_lines(lines, 100) == ["a\n", "b\n"]

    def test_empty_input(self):
        assert tail_lines([], 5) == []

    def test_works_with_generator(self):
        gen = (f"{i}\n" for i in range(10))
        result = tail_lines(gen, 2)
        assert result == ["8\n", "9\n"]


# ---------------------------------------------------------------------------
# latest_jsonl_record
# ---------------------------------------------------------------------------

class TestLatestJsonlRecord:
    def test_returns_last_record(self):
        lines = [
            '{"step": 1, "loss": 0.9}\n',
            '{"step": 2, "loss": 0.5}\n',
            '{"step": 3, "loss": 0.3}\n',
        ]
        record = latest_jsonl_record(lines)
        assert record == {"step": 3, "loss": 0.3}

    def test_skips_blank_lines(self):
        lines = ['{"a": 1}\n', "\n", "  \n", '{"a": 2}\n', "\n"]
        assert latest_jsonl_record(lines) == {"a": 2}

    def test_single_record(self):
        assert latest_jsonl_record(['{"ok": true}\n']) == {"ok": True}

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No JSONL records"):
            latest_jsonl_record([])

    def test_only_blanks_raises(self):
        with pytest.raises(ValueError, match="No JSONL records"):
            latest_jsonl_record(["\n", "  \n", "\n"])

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            latest_jsonl_record(["not valid json\n"])
