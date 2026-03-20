"""Tests for shared.flywheel.stager — DatasetStager JSONL assembly."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.flywheel.catalog import DatasetVersion, InferenceLogRecord, LogFilter
from shared.flywheel.config import FlywheelConfig
from shared.flywheel.stager import DatasetStager, StagingResult


def _make_record(
    log_id: str,
    source_file: str = "",
    line_number: int = 0,
    **kwargs,
) -> InferenceLogRecord:
    defaults = dict(
        timestamp="2026-01-15T12:00:00Z",
        model_id="test-model",
    )
    defaults.update(kwargs)
    return InferenceLogRecord(
        log_id=log_id,
        source_file=source_file,
        line_number=line_number,
        **defaults,
    )


def _write_log_file(path: Path, records: list[dict]) -> None:
    """Write multiple log content dicts as JSONL lines."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestStagingResult:
    """StagingResult dataclass defaults."""

    def test_defaults(self):
        r = StagingResult()
        assert r.version_id == ""
        assert r.sft_count == 0
        assert r.kto_pos_count == 0
        assert r.grpo_count == 0
        assert r.file_paths == {}


class TestDatasetStagerWrite:
    """DatasetStager writes correct JSONL formats for each training type."""

    def _make_stager(self, tmp_path, **config_kwargs):
        catalog = AsyncMock()
        catalog.find_logs = AsyncMock(return_value=[])
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)
        catalog.create_dataset_version = AsyncMock(return_value="v001")
        catalog.mark_used = AsyncMock()
        cfg = FlywheelConfig(**config_kwargs)
        return DatasetStager(catalog, cfg, datasets_dir=tmp_path / "datasets")

    @pytest.mark.asyncio
    async def test_sft_jsonl_format(self, tmp_path):
        """SFT output has conversations + label: true."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Hi"}],
            "response_content": "Hello!",
        }])

        sft_record = _make_record(
            "sft-1", source_file=str(log_file), line_number=0, tag="sft",
        )

        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [sft_record],  # sft query
            [],            # kto query
            [],            # grpo query
        ])

        with patch.object(stager, "_register_flywheel_cycle", return_value="run-1"):
            result = await stager.stage_dataset()

        assert result.sft_count == 1
        sft_path = Path(result.file_paths["sft"])
        assert sft_path.exists()

        with open(sft_path) as f:
            example = json.loads(f.readline())
        assert example["label"] is True
        assert example["conversations"][-1]["role"] == "assistant"
        assert example["conversations"][-1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_kto_jsonl_format(self, tmp_path):
        """KTO output has conversations + label field (true for positive, false for negative)."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [
            {"messages": [{"role": "user", "content": "Good"}], "response_content": "Great!"},
            {"messages": [{"role": "user", "content": "Bad"}], "response_content": "Wrong"},
        ])

        sft_record = _make_record(
            "kto-pos", source_file=str(log_file), line_number=0, tag="sft",
        )
        kto_record = _make_record(
            "kto-neg", source_file=str(log_file), line_number=1, tag="kto",
        )

        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [sft_record],   # sft query
            [kto_record],   # kto query
            [],             # grpo query
        ])

        with patch.object(stager, "_register_flywheel_cycle", return_value="run-1"):
            result = await stager.stage_dataset()

        assert result.kto_pos_count == 1
        assert result.kto_neg_count == 1

        kto_path = Path(result.file_paths["kto"])
        lines = kto_path.read_text().strip().splitlines()
        assert len(lines) == 2

        pos = json.loads(lines[0])
        neg = json.loads(lines[1])
        assert pos["label"] is True
        assert neg["label"] is False

    @pytest.mark.asyncio
    async def test_grpo_jsonl_format(self, tmp_path):
        """GRPO output has conversations + reward field."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Search for X"}],
            "response_content": "Found X",
        }])

        grpo_record = _make_record(
            "grpo-1", source_file=str(log_file), line_number=0,
            tag="grpo", fitness_score=0.7,
        )

        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [],             # sft query
            [],             # kto query
            [grpo_record],  # grpo query
        ])

        with patch.object(stager, "_register_flywheel_cycle", return_value="run-1"):
            result = await stager.stage_dataset()

        assert result.grpo_count == 1
        grpo_path = Path(result.file_paths["grpo"])
        example = json.loads(grpo_path.read_text().strip())
        assert "reward" in example
        assert example["reward"] == pytest.approx(0.7)
        assert "conversations" in example

    @pytest.mark.asyncio
    async def test_grpo_disabled_skips(self, tmp_path):
        """GRPO logs not staged when grpo_enabled=False."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "X"}],
            "response_content": "Y",
        }])

        grpo_record = _make_record(
            "grpo-skip", source_file=str(log_file), line_number=0,
            tag="grpo", fitness_score=0.6,
        )

        stager = self._make_stager(tmp_path, grpo_enabled=False)
        stager._catalog.find_logs = AsyncMock(side_effect=[
            [],              # sft
            [],              # kto
            [grpo_record],   # grpo
        ])

        with patch.object(stager, "_register_flywheel_cycle", return_value=""):
            result = await stager.stage_dataset()

        assert result.grpo_count == 0
        assert "grpo" not in result.file_paths

    @pytest.mark.asyncio
    async def test_no_logs_returns_empty_result(self, tmp_path):
        stager = self._make_stager(tmp_path)
        stager._catalog.find_logs = AsyncMock(return_value=[])

        result = await stager.stage_dataset()
        assert result.version_id == ""
        assert result.total_records == 0


class TestDatasetStagerVersioning:
    """DatasetStager version ID generation and dataset version creation."""

    def test_next_version_id_empty_dir(self, tmp_path):
        catalog = AsyncMock()
        stager = DatasetStager(catalog, FlywheelConfig(), datasets_dir=tmp_path / "ds")
        assert stager._next_version_id() == "v001"

    def test_next_version_id_increments(self, tmp_path):
        ds_dir = tmp_path / "ds"
        (ds_dir / "v001").mkdir(parents=True)
        (ds_dir / "v002").mkdir()

        catalog = AsyncMock()
        stager = DatasetStager(catalog, FlywheelConfig(), datasets_dir=ds_dir)
        assert stager._next_version_id() == "v003"

    @pytest.mark.asyncio
    async def test_creates_dataset_version_in_catalog(self, tmp_path):
        """stage_dataset creates a DatasetVersion entry in the catalog."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Hi"}],
            "response_content": "Hello!",
        }])

        sft_record = _make_record(
            "v-1", source_file=str(log_file), line_number=0, tag="sft",
        )

        catalog = AsyncMock()
        catalog.find_logs = AsyncMock(side_effect=[
            [sft_record], [], [],  # sft, kto, grpo
        ])
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)
        catalog.create_dataset_version = AsyncMock(return_value="v001")
        catalog.mark_used = AsyncMock()

        stager = DatasetStager(catalog, FlywheelConfig(), datasets_dir=tmp_path / "ds")

        with patch.object(stager, "_register_flywheel_cycle", return_value="run-1"):
            result = await stager.stage_dataset()

        catalog.create_dataset_version.assert_called_once()
        version_arg = catalog.create_dataset_version.call_args[0][0]
        assert isinstance(version_arg, DatasetVersion)
        assert version_arg.record_counts["sft"] == 1

    @pytest.mark.asyncio
    async def test_marks_logs_as_used(self, tmp_path):
        """stage_dataset calls mark_used with all consumed log IDs."""
        log_file = tmp_path / "logs.jsonl"
        _write_log_file(log_file, [{
            "messages": [{"role": "user", "content": "Hi"}],
            "response_content": "Hello!",
        }])

        record = _make_record(
            "mu-1", source_file=str(log_file), line_number=0, tag="sft",
        )

        catalog = AsyncMock()
        catalog.find_logs = AsyncMock(side_effect=[[record], [], []])
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)
        catalog.create_dataset_version = AsyncMock(return_value="v001")
        catalog.mark_used = AsyncMock()

        stager = DatasetStager(catalog, FlywheelConfig(), datasets_dir=tmp_path / "ds")

        with patch.object(stager, "_register_flywheel_cycle", return_value=""):
            await stager.stage_dataset()

        catalog.mark_used.assert_called_once()
        log_ids = catalog.mark_used.call_args[0][0]
        assert "mu-1" in log_ids


class TestContentHash:
    """DatasetStager._compute_content_hash produces consistent hashes."""

    def test_same_content_same_hash(self, tmp_path):
        catalog = AsyncMock()
        stager = DatasetStager(catalog, FlywheelConfig())

        f1 = tmp_path / "a.jsonl"
        f1.write_text('{"test": 1}\n')
        f2 = tmp_path / "b.jsonl"
        f2.write_text('{"test": 2}\n')

        h1 = stager._compute_content_hash({"sft": f1, "kto": f2})
        h2 = stager._compute_content_hash({"sft": f1, "kto": f2})
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        catalog = AsyncMock()
        stager = DatasetStager(catalog, FlywheelConfig())

        f1 = tmp_path / "a.jsonl"
        f1.write_text('{"test": 1}\n')
        f2 = tmp_path / "b.jsonl"
        f2.write_text('{"test": 2}\n')
        f3 = tmp_path / "c.jsonl"
        f3.write_text('{"test": 3}\n')

        h1 = stager._compute_content_hash({"sft": f1})
        h2 = stager._compute_content_hash({"sft": f3})
        assert h1 != h2
