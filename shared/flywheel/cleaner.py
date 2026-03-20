"""
shared/flywheel/cleaner.py

DataCleaner: scores inference logs using FitnessEvaluator and applies PII
scrubbing. Reads unscored logs from the catalog, evaluates each with
FitnessEvaluator, and updates the catalog with fitness_score, is_valid,
and errors.

Used by: orchestrator.py (pipeline stage), CLI (manual cleaning)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from shared.validation.fitness import FitnessEvaluator, FitnessResult

from .catalog import InferenceLogRecord, LogCatalog, LogFilter
from .config import FlywheelConfig
from .utils import read_log_content

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII Detection (future feature -- interface defined now)
# ---------------------------------------------------------------------------

@dataclass
class PIIMatch:
    """A detected PII instance."""
    pii_type: str
    start: int
    end: int
    text: str


class PIIDetector(Protocol):
    """Interface for PII detection (future feature).

    v1 implementation is a no-op stub. Designed for extension
    with regex-based or ML-based detectors.
    """

    def detect(self, text: str) -> list[PIIMatch]: ...
    def scrub(self, text: str) -> str: ...


class NoOpPIIDetector:
    """v1 stub: passes all text through unchanged."""

    def detect(self, text: str) -> list[PIIMatch]:
        return []

    def scrub(self, text: str) -> str:
        return text


# ---------------------------------------------------------------------------
# CleaningResult
# ---------------------------------------------------------------------------

@dataclass
class CleaningResult:
    """Summary of a cleaning run."""
    total_processed: int = 0
    scored: int = 0
    pii_scrubbed: int = 0
    errors: int = 0
    score_distribution: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DataCleaner
# ---------------------------------------------------------------------------

class DataCleaner:
    """Scores inference logs using FitnessEvaluator and applies PII scrubbing.

    Reads unscored logs from the catalog, evaluates each with FitnessEvaluator,
    and updates the catalog with fitness_score, is_valid, and errors.

    Args:
        catalog: LogCatalog instance
        config: FlywheelConfig (controls scoring method, max_errors, etc.)
        pii_detector: PII detection implementation (default: NoOpPIIDetector)
    """

    def __init__(
        self,
        catalog: LogCatalog,
        config: FlywheelConfig,
        pii_detector: PIIDetector | None = None,
    ) -> None:
        self._catalog = catalog
        self._config = config
        self._pii = pii_detector or NoOpPIIDetector()

        # Build FitnessEvaluator from config
        fitness_cfg = config.to_fitness_config()
        self._evaluator = FitnessEvaluator(config=fitness_cfg)

    async def clean_logs(
        self,
        filters: LogFilter | None = None,
        batch_size: int = 100,
    ) -> CleaningResult:
        """Score and clean all unscored inference logs matching filters.

        Reads logs in batches, evaluates fitness, scrubs PII, and updates
        the catalog with results.

        Args:
            filters: Optional additional filters (default: unscored logs only)
            batch_size: Number of logs to process per batch

        Returns:
            CleaningResult with counts and score distribution
        """
        if filters is None:
            filters = LogFilter(unscored_only=True)
        else:
            filters.unscored_only = True

        filters.limit = batch_size
        result = CleaningResult()

        while True:
            logs = await self._catalog.find_logs(filters)
            if not logs:
                break

            for record in logs:
                result.total_processed += 1
                try:
                    fitness = self._evaluate_log(record)

                    # PII scrubbing (no-op in v1)
                    content = self._read_log_content(record)
                    if content:
                        scrubbed = self._pii.scrub(
                            content.get("response_content", "")
                        )
                        if scrubbed != content.get("response_content", ""):
                            result.pii_scrubbed += 1

                    # Update catalog with score
                    await self._catalog.update_score(
                        record.log_id,
                        fitness.score,
                        fitness.is_valid,
                        fitness.errors,
                    )
                    result.scored += 1

                    # Track score distribution
                    bucket = self._score_bucket(fitness.score)
                    result.score_distribution[bucket] = (
                        result.score_distribution.get(bucket, 0) + 1
                    )

                except Exception as exc:
                    logger.error(
                        "Error cleaning log %s: %s", record.log_id, exc,
                    )
                    result.errors += 1

        logger.info(
            "Cleaning complete: %d processed, %d scored, %d errors",
            result.total_processed, result.scored, result.errors,
        )
        return result

    def _evaluate_log(self, record: InferenceLogRecord) -> FitnessResult:
        """Evaluate a single inference log using FitnessEvaluator.

        Reconstructs model_output from the log's response_content and
        tool_calls, then passes it through the evaluator.

        IMPORTANT: FitnessEvaluator checks for tool calls in the response.
        If no tool calls are present, it returns no_tool_call_score (default 0.0).
        The DataCleaner does NOT filter by tools_requested here -- that logic
        belongs in the AutoTagger which checks the scoring path.
        """
        content = self._read_log_content(record)
        if not content:
            # Cannot read source file -- score as invalid
            return FitnessResult(
                score=0.0,
                is_valid=False,
                errors=["Source file not readable"],
                scoring_method="error",
            )

        # Reconstruct model output for FitnessEvaluator
        # If we have tool_calls in the response, build an OpenAI-style dict
        tool_calls = content.get("tool_calls", [])
        response_text = content.get("response_content", "")

        if tool_calls:
            model_output = {
                "choices": [{
                    "message": {
                        "content": response_text,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": content.get("finish_reason", "stop"),
                }],
            }
        else:
            model_output = response_text

        return self._evaluator.evaluate(model_output)

    @staticmethod
    def _read_log_content(record: InferenceLogRecord) -> dict | None:
        """Read full log content from the source JSONL file."""
        return read_log_content(record)

    @staticmethod
    def _score_bucket(score: float) -> str:
        """Map a fitness score to a distribution bucket."""
        if score < 0.3:
            return "0.0-0.3"
        if score < 0.8:
            return "0.3-0.8"
        return "0.8-1.0"
