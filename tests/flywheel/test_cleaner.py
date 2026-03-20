"""Tests for shared.flywheel.cleaner — DataCleaner scoring and PII."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.flywheel.catalog import InferenceLogRecord, LogFilter
from shared.flywheel.cleaner import CleaningResult, DataCleaner, NoOpPIIDetector
from shared.flywheel.config import FlywheelConfig


def _make_record(log_id: str, **kwargs) -> InferenceLogRecord:
    defaults = dict(
        timestamp="2026-01-15T12:00:00Z",
        model_id="test-model",
        source_file="test.jsonl",
        line_number=0,
    )
    defaults.update(kwargs)
    return InferenceLogRecord(log_id=log_id, **defaults)


def _write_log_content(path: Path, content: dict) -> None:
    """Write a single log content dict as line 0 of a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(content) + "\n")


class TestNoOpPIIDetector:
    """NoOpPIIDetector passes all text through unchanged."""

    def test_detect_returns_empty(self):
        d = NoOpPIIDetector()
        assert d.detect("John Doe, SSN 123-45-6789") == []

    def test_scrub_returns_input_unchanged(self):
        d = NoOpPIIDetector()
        text = "Some text with PII maybe"
        assert d.scrub(text) == text


class TestCleaningResult:
    """CleaningResult dataclass fields."""

    def test_default_values(self):
        r = CleaningResult()
        assert r.total_processed == 0
        assert r.scored == 0
        assert r.pii_scrubbed == 0
        assert r.errors == 0
        assert r.score_distribution == {}


class TestDataCleanerScoring:
    """DataCleaner evaluates logs via FitnessEvaluator."""

    @pytest.mark.asyncio
    async def test_scores_tool_call_response(self, tmp_path):
        """Tool-call response is scored by FitnessEvaluator."""
        log_file = tmp_path / "test.jsonl"
        log_content = {
            "response_content": "Here is the result",
            "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}],
            "finish_reason": "stop",
        }
        _write_log_content(log_file, log_content)

        record = _make_record(
            "tc-1",
            source_file=str(log_file),
            line_number=0,
            tool_calls=[{"function": {"name": "search"}}],
            tools_requested=True,
        )

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[record], []])
        mock_catalog.update_score = AsyncMock()

        cfg = FlywheelConfig()

        with patch(
            "shared.flywheel.cleaner.FitnessEvaluator"
        ) as MockEval:
            from shared.validation.fitness import FitnessResult

            mock_evaluator = MagicMock()
            mock_evaluator.evaluate.return_value = FitnessResult(
                score=0.9, is_valid=True, errors=[], scoring_method="error_count",
            )
            MockEval.return_value = mock_evaluator

            cleaner = DataCleaner(mock_catalog, cfg)
            result = await cleaner.clean_logs()

        assert result.total_processed == 1
        assert result.scored == 1
        mock_catalog.update_score.assert_called_once_with("tc-1", 0.9, True, [])

    @pytest.mark.asyncio
    async def test_scores_text_response(self, tmp_path):
        """Non-tool-call response still gets scored by FitnessEvaluator
        (text_response_policy is handled by tagger, not cleaner)."""
        log_file = tmp_path / "test.jsonl"
        log_content = {
            "response_content": "Just a text response",
            "tool_calls": [],
        }
        _write_log_content(log_file, log_content)

        record = _make_record(
            "text-1",
            source_file=str(log_file),
            line_number=0,
            tools_requested=False,
        )

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[record], []])
        mock_catalog.update_score = AsyncMock()

        cfg = FlywheelConfig(text_response_policy="sft")

        with patch(
            "shared.flywheel.cleaner.FitnessEvaluator"
        ) as MockEval:
            from shared.validation.fitness import FitnessResult

            mock_evaluator = MagicMock()
            # Non-tool-call gets no_tool_call_score from FitnessEvaluator
            mock_evaluator.evaluate.return_value = FitnessResult(
                score=0.0, is_valid=True, errors=[], scoring_method="error_count",
            )
            MockEval.return_value = mock_evaluator

            cleaner = DataCleaner(mock_catalog, cfg)
            result = await cleaner.clean_logs()

        assert result.scored == 1
        # Cleaner stores the score regardless of text_response_policy
        mock_catalog.update_score.assert_called_once_with("text-1", 0.0, True, [])

    @pytest.mark.asyncio
    async def test_missing_source_file_scores_zero(self):
        """Log with missing source file gets score=0.0, is_valid=False."""
        record = _make_record(
            "missing-1",
            source_file="/nonexistent/file.jsonl",
            line_number=0,
        )

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[record], []])
        mock_catalog.update_score = AsyncMock()

        cfg = FlywheelConfig()

        with patch(
            "shared.flywheel.cleaner.FitnessEvaluator"
        ) as MockEval:
            mock_evaluator = MagicMock()
            MockEval.return_value = mock_evaluator

            cleaner = DataCleaner(mock_catalog, cfg)
            result = await cleaner.clean_logs()

        assert result.scored == 1
        mock_catalog.update_score.assert_called_once()
        call_args = mock_catalog.update_score.call_args
        assert call_args[0][1] == 0.0  # score
        assert call_args[0][2] is False  # is_valid

    @pytest.mark.asyncio
    async def test_error_during_scoring_counted(self, tmp_path):
        """Exception during scoring is caught and counted as error."""
        log_file = tmp_path / "test.jsonl"
        _write_log_content(log_file, {"response_content": "x"})

        record = _make_record("err-1", source_file=str(log_file), line_number=0)

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[record], []])
        mock_catalog.update_score = AsyncMock(side_effect=RuntimeError("DB error"))

        cfg = FlywheelConfig()

        with patch(
            "shared.flywheel.cleaner.FitnessEvaluator"
        ) as MockEval:
            from shared.validation.fitness import FitnessResult

            mock_evaluator = MagicMock()
            mock_evaluator.evaluate.return_value = FitnessResult(
                score=0.5, is_valid=True, errors=[], scoring_method="error_count",
            )
            MockEval.return_value = mock_evaluator

            cleaner = DataCleaner(mock_catalog, cfg)
            result = await cleaner.clean_logs()

        assert result.errors == 1

    @pytest.mark.asyncio
    async def test_score_distribution_buckets(self, tmp_path):
        """Score distribution tracks buckets correctly."""
        records = []
        for i, score in enumerate([0.1, 0.5, 0.9]):
            log_file = tmp_path / f"test_{i}.jsonl"
            _write_log_content(log_file, {"response_content": "x"})
            records.append(
                _make_record(f"dist-{i}", source_file=str(log_file), line_number=0)
            )

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[records, []])
        mock_catalog.update_score = AsyncMock()

        cfg = FlywheelConfig()
        scores = iter([0.1, 0.5, 0.9])

        with patch(
            "shared.flywheel.cleaner.FitnessEvaluator"
        ) as MockEval:
            from shared.validation.fitness import FitnessResult

            mock_evaluator = MagicMock()
            mock_evaluator.evaluate.side_effect = [
                FitnessResult(score=s, is_valid=True, errors=[], scoring_method="error_count")
                for s in [0.1, 0.5, 0.9]
            ]
            MockEval.return_value = mock_evaluator

            cleaner = DataCleaner(mock_catalog, cfg)
            result = await cleaner.clean_logs()

        assert result.score_distribution.get("0.0-0.3", 0) == 1
        assert result.score_distribution.get("0.3-0.8", 0) == 1
        assert result.score_distribution.get("0.8-1.0", 0) == 1
