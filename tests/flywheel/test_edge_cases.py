"""Edge-case tests for flywheel pipeline components.

Covers:
- Tagger: per-record error handling in tag_logs
- Stager: partial-file read (line_number exceeds file length)
- Cleaner: multi-batch pagination (while-True loop)
- SQLite catalog: concurrent read safety
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.flywheel.catalog import InferenceLogRecord, LogFilter
from shared.flywheel.config import FlywheelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(log_id: str, **kwargs) -> InferenceLogRecord:
    defaults = dict(
        timestamp="2026-01-15T12:00:00Z",
        model_id="test-model",
        source_file="test.jsonl",
        line_number=0,
    )
    defaults.update(kwargs)
    return InferenceLogRecord(log_id=log_id, **defaults)


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    """Write multiple JSON objects as JSONL lines."""
    with open(path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------------------
# Tagger: per-record error handling
# ---------------------------------------------------------------------------


class TestTaggerErrorHandling:
    """AutoTagger.tag_logs catches per-record exceptions."""

    @pytest.mark.asyncio
    async def test_single_record_error_does_not_abort_batch(self):
        """One record raising in _classify_by_rules -> error counted, others proceed."""
        from shared.flywheel.tagger import AutoTagger, TaggingResult

        good_record = _make_record(
            "good-1",
            fitness_score=0.9,
            tools_requested=True,
            tool_calls=[{"function": {"name": "f"}}],
        )
        bad_record = _make_record(
            "bad-1",
            fitness_score=0.5,
            tools_requested=True,
        )

        mock_catalog = AsyncMock()
        # First batch returns both records, second returns empty
        mock_catalog.find_logs = AsyncMock(
            side_effect=[[good_record, bad_record], []],
        )
        mock_catalog.update_tag = AsyncMock()

        cfg = FlywheelConfig()
        tagger = AutoTagger(mock_catalog, cfg)

        # Make _classify_by_rules fail on the bad record specifically
        original_classify = tagger._classify_by_rules

        def flaky_classify(record):
            if record.log_id == "bad-1":
                raise ValueError("unexpected score format")
            return original_classify(record)

        with patch.object(tagger, "_classify_by_rules", side_effect=flaky_classify):
            result = await tagger.tag_logs()

        assert result.errors == 1
        assert result.total_processed == 2
        # Good record should still have been tagged
        assert mock_catalog.update_tag.await_count >= 1

    @pytest.mark.asyncio
    async def test_catalog_update_tag_failure_counted_as_error(self):
        """When catalog.update_tag raises, it's counted as an error."""
        from shared.flywheel.tagger import AutoTagger

        record = _make_record(
            "err-1",
            fitness_score=0.95,
            tools_requested=True,
            tool_calls=[{"function": {"name": "f"}}],
        )

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[record], []])
        mock_catalog.update_tag = AsyncMock(side_effect=RuntimeError("DB write failed"))

        cfg = FlywheelConfig()
        tagger = AutoTagger(mock_catalog, cfg)
        result = await tagger.tag_logs()

        assert result.errors == 1

    @pytest.mark.asyncio
    async def test_unscored_records_skipped(self):
        """Records with fitness_score=None are silently skipped."""
        from shared.flywheel.tagger import AutoTagger

        unscored = _make_record("unscored-1", fitness_score=None)

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[unscored], []])
        mock_catalog.update_tag = AsyncMock()

        cfg = FlywheelConfig()
        tagger = AutoTagger(mock_catalog, cfg)
        result = await tagger.tag_logs()

        # Unscored records are skipped before total_processed increment
        assert result.total_processed == 0
        mock_catalog.update_tag.assert_not_awaited()


# ---------------------------------------------------------------------------
# Stager: _read_log_content with out-of-bounds line_number
# ---------------------------------------------------------------------------


class TestStagerPartialFile:
    """Stager handles missing/partial source files gracefully."""

    def test_line_number_exceeds_file_length_returns_none(self, tmp_path):
        """read_log_content returns None when line_number > file lines."""
        from shared.flywheel.utils import read_log_content

        log_file = tmp_path / "short.jsonl"
        _write_jsonl(log_file, [{"response_content": "line0"}])

        record = _make_record(
            "oob-1",
            source_file=str(log_file),
            line_number=5,  # File only has 1 line (index 0)
        )
        result = read_log_content(record)
        assert result is None

    def test_exact_last_line_readable(self, tmp_path):
        """Line at last index is readable."""
        from shared.flywheel.utils import read_log_content

        log_file = tmp_path / "multi.jsonl"
        _write_jsonl(log_file, [
            {"response_content": "first"},
            {"response_content": "second"},
            {"response_content": "third"},
        ])

        record = _make_record(
            "last-1",
            source_file=str(log_file),
            line_number=2,
        )
        result = read_log_content(record)
        assert result is not None
        assert result["response_content"] == "third"

    def test_nonexistent_file_returns_none(self):
        """Missing source file returns None."""
        from shared.flywheel.utils import read_log_content

        record = _make_record(
            "missing-1",
            source_file="/nonexistent/path/data.jsonl",
            line_number=0,
        )
        result = read_log_content(record)
        assert result is None

    def test_malformed_json_line_returns_none(self, tmp_path):
        """Corrupted JSON line returns None."""
        from shared.flywheel.utils import read_log_content

        log_file = tmp_path / "corrupt.jsonl"
        with open(log_file, "w") as f:
            f.write("not valid json\n")

        record = _make_record(
            "corrupt-1",
            source_file=str(log_file),
            line_number=0,
        )
        result = read_log_content(record)
        assert result is None

    def test_stager_write_sft_skips_unreadable_records(self, tmp_path):
        """_write_sft skips records whose content can't be read."""
        from shared.flywheel.stager import DatasetStager

        log_file = tmp_path / "data.jsonl"
        _write_jsonl(log_file, [
            {"messages": [{"role": "user", "content": "Q"}], "response_content": "A"},
        ])

        good_record = _make_record(
            "good-1", source_file=str(log_file), line_number=0,
        )
        bad_record = _make_record(
            "bad-1", source_file="/nonexistent/file.jsonl", line_number=0,
        )

        cfg = FlywheelConfig()
        catalog = AsyncMock()
        stager = DatasetStager(catalog, cfg, datasets_dir=tmp_path / "datasets")

        output_path = tmp_path / "sft_output.jsonl"
        count = stager._write_sft([good_record, bad_record], output_path)

        assert count == 1  # Only good record written
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Cleaner: multi-batch pagination
# ---------------------------------------------------------------------------


class TestCleanerPagination:
    """DataCleaner processes multiple batches in the while-True loop."""

    @pytest.mark.asyncio
    async def test_processes_multiple_batches(self, tmp_path):
        """Cleaner fetches batches until find_logs returns empty."""
        from shared.flywheel.cleaner import DataCleaner

        # Create source files for records
        records_batch1 = []
        records_batch2 = []
        for i in range(3):
            log_file = tmp_path / f"batch1_{i}.jsonl"
            _write_jsonl(log_file, [{"response_content": f"b1-{i}"}])
            records_batch1.append(
                _make_record(f"b1-{i}", source_file=str(log_file), line_number=0)
            )
        for i in range(2):
            log_file = tmp_path / f"batch2_{i}.jsonl"
            _write_jsonl(log_file, [{"response_content": f"b2-{i}"}])
            records_batch2.append(
                _make_record(f"b2-{i}", source_file=str(log_file), line_number=0)
            )

        mock_catalog = AsyncMock()
        # Three calls: batch1, batch2, empty (stop)
        mock_catalog.find_logs = AsyncMock(
            side_effect=[records_batch1, records_batch2, []],
        )
        mock_catalog.update_score = AsyncMock()

        cfg = FlywheelConfig()

        with patch("shared.flywheel.cleaner.FitnessEvaluator") as MockEval:
            from shared.validation.fitness import FitnessResult

            mock_evaluator = MagicMock()
            mock_evaluator.evaluate.return_value = FitnessResult(
                score=0.7, is_valid=True, errors=[], scoring_method="error_count",
            )
            MockEval.return_value = mock_evaluator

            cleaner = DataCleaner(mock_catalog, cfg)
            result = await cleaner.clean_logs(batch_size=3)

        assert result.total_processed == 5
        assert result.scored == 5
        assert mock_catalog.update_score.await_count == 5

    @pytest.mark.asyncio
    async def test_empty_catalog_returns_zero_result(self):
        """When no unscored logs exist, result is all zeros."""
        from shared.flywheel.cleaner import DataCleaner

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(return_value=[])

        cfg = FlywheelConfig()

        with patch("shared.flywheel.cleaner.FitnessEvaluator") as MockEval:
            MockEval.return_value = MagicMock()
            cleaner = DataCleaner(mock_catalog, cfg)
            result = await cleaner.clean_logs()

        assert result.total_processed == 0
        assert result.scored == 0
        assert result.errors == 0


# ---------------------------------------------------------------------------
# SQLite catalog: concurrent read safety via WAL
# ---------------------------------------------------------------------------


class TestSQLiteConcurrentReads:
    """SQLiteLogCatalog supports concurrent reads via WAL mode."""

    @pytest.mark.asyncio
    async def test_concurrent_reads_do_not_block(self, tmp_path):
        """Multiple concurrent find_logs calls complete without deadlock."""
        import asyncio

        from shared.flywheel.catalog import SQLiteLogCatalog

        db_path = tmp_path / "test_concurrent.db"
        catalog = SQLiteLogCatalog(db_path)
        await catalog.initialize()

        # Insert a few records
        for i in range(5):
            record = _make_record(
                f"concurrent-{i}",
                source_file="test.jsonl",
                line_number=i,
            )
            await catalog.insert_log(record)

        # Run concurrent reads
        async def read_logs():
            return await catalog.find_logs(LogFilter(limit=10))

        results = await asyncio.gather(
            read_logs(),
            read_logs(),
            read_logs(),
        )

        for result in results:
            assert len(result) == 5

        await catalog.close()

    @pytest.mark.asyncio
    async def test_wal_mode_is_set(self, tmp_path):
        """WAL journal mode is set on initialization."""
        from shared.flywheel.catalog import SQLiteLogCatalog

        db_path = tmp_path / "test_wal.db"
        catalog = SQLiteLogCatalog(db_path)
        await catalog.initialize()

        cursor = await catalog._conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row[0] == "wal"

        await catalog.close()

    @pytest.mark.asyncio
    async def test_insert_and_read_roundtrip(self, tmp_path):
        """Insert -> read roundtrip preserves record fields."""
        from shared.flywheel.catalog import SQLiteLogCatalog

        db_path = tmp_path / "test_roundtrip.db"
        catalog = SQLiteLogCatalog(db_path)
        await catalog.initialize()

        record = _make_record(
            "rt-1",
            model_id="my-model",
            source_file="data.jsonl",
            line_number=42,
            tool_calls=[{"function": {"name": "search"}}],
            tools_requested=True,
        )
        await catalog.insert_log(record)

        results = await catalog.find_logs(LogFilter(model_id="my-model"))
        assert len(results) == 1
        r = results[0]
        assert r.log_id == "rt-1"
        assert r.model_id == "my-model"
        assert r.line_number == 42

        await catalog.close()

    @pytest.mark.asyncio
    async def test_count_and_avg_score(self, tmp_path):
        """count_logs and avg_score work on scored records."""
        from shared.flywheel.catalog import SQLiteLogCatalog

        db_path = tmp_path / "test_agg.db"
        catalog = SQLiteLogCatalog(db_path)
        await catalog.initialize()

        for i, score in enumerate([0.5, 0.7, 0.9]):
            record = _make_record(
                f"agg-{i}", source_file="f.jsonl", line_number=i,
            )
            await catalog.insert_log(record)
            await catalog.update_score(f"agg-{i}", score, True, [])

        count = await catalog.count_logs(LogFilter())
        assert count == 3

        avg = await catalog.avg_score(LogFilter())
        assert abs(avg - 0.7) < 0.01

        await catalog.close()


# ---------------------------------------------------------------------------
# Tagger: ambiguous band without judge
# ---------------------------------------------------------------------------


class TestTaggerAmbiguousNoJudge:
    """When no judge is available, ambiguous records default to KTO."""

    @pytest.mark.asyncio
    async def test_ambiguous_defaults_to_kto_without_judge(self):
        """Score in ambiguous band (0.4-0.7) -> kto when no judge."""
        from shared.flywheel.tagger import AutoTagger

        record = _make_record(
            "amb-1",
            fitness_score=0.55,
            tools_requested=True,
            tool_calls=[],
            is_valid=False,
        )

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[record], []])
        mock_catalog.update_tag = AsyncMock()

        cfg = FlywheelConfig()
        tagger = AutoTagger(mock_catalog, cfg, judge=None)
        result = await tagger.tag_logs(use_judge=True)

        # Without judge, ambiguous defaults to kto
        mock_catalog.update_tag.assert_awaited_once()
        tag_call = mock_catalog.update_tag.call_args
        assert tag_call[0][1] == "kto"
        assert result.kto_count == 1
