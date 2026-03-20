"""Tests for shared.flywheel.tagger — AutoTagger rule-based classification."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from shared.flywheel.catalog import InferenceLogRecord, LogFilter
from shared.flywheel.config import FlywheelConfig
from shared.flywheel.tagger import AutoTagger, TaggedExample, TaggingResult


def _make_record(
    log_id: str,
    fitness_score: float | None = None,
    tools_requested: bool = True,
    tool_calls: list | None = None,
    is_valid: bool | None = None,
    **kwargs,
) -> InferenceLogRecord:
    defaults = dict(
        timestamp="2026-01-15T12:00:00Z",
        model_id="test-model",
        source_file="test.jsonl",
        line_number=0,
    )
    defaults.update(kwargs)
    return InferenceLogRecord(
        log_id=log_id,
        fitness_score=fitness_score,
        tools_requested=tools_requested,
        tool_calls=tool_calls or [],
        is_valid=is_valid,
        **defaults,
    )


class TestTaggedExample:
    """TaggedExample dataclass fields."""

    def test_fields(self):
        t = TaggedExample(log_id="x", tag="sft", fitness_score=0.9)
        assert t.log_id == "x"
        assert t.tag == "sft"
        assert t.label is None
        assert t.reward is None
        assert t.tag_source == "rule"


class TestAutoTaggerRules:
    """AutoTagger._classify_by_rules applies correct tag thresholds."""

    def _make_tagger(self, **config_kwargs):
        catalog = AsyncMock()
        cfg = FlywheelConfig(**config_kwargs)
        return AutoTagger(catalog, cfg)

    def test_high_score_tags_sft(self):
        tagger = self._make_tagger()
        record = _make_record("s1", fitness_score=0.85, tools_requested=True)
        assert tagger._classify_by_rules(record) == "sft"

    def test_exact_sft_threshold_tags_sft(self):
        tagger = self._make_tagger()
        record = _make_record("s2", fitness_score=0.8, tools_requested=True)
        assert tagger._classify_by_rules(record) == "sft"

    def test_low_score_tags_discard(self):
        tagger = self._make_tagger()
        record = _make_record("d1", fitness_score=0.1, tools_requested=True)
        assert tagger._classify_by_rules(record) == "discard"

    def test_below_kto_min_tags_discard(self):
        tagger = self._make_tagger()
        record = _make_record("d2", fitness_score=0.29, tools_requested=True)
        assert tagger._classify_by_rules(record) == "discard"

    def test_mid_range_with_valid_tools_tags_grpo(self):
        """Mid-range score with valid tool calls gets GRPO tag."""
        tagger = self._make_tagger()
        record = _make_record(
            "g1", fitness_score=0.5, tools_requested=True,
            tool_calls=[{"fn": "x"}], is_valid=True,
        )
        assert tagger._classify_by_rules(record) == "grpo"

    def test_mid_range_without_tool_calls_tags_ambiguous_or_kto(self):
        """Mid-range score without tool calls goes to ambiguous band."""
        tagger = self._make_tagger()
        record = _make_record(
            "a1", fitness_score=0.5, tools_requested=True,
            tool_calls=[], is_valid=False,
        )
        tag = tagger._classify_by_rules(record)
        assert tag == "ambiguous"

    def test_mid_range_outside_ambiguous_band_tags_kto(self):
        """Score between kto_min (0.3) and ambiguous_min (0.4) gets KTO."""
        tagger = self._make_tagger()
        record = _make_record(
            "k1", fitness_score=0.35, tools_requested=True,
            tool_calls=[], is_valid=False,
        )
        tag = tagger._classify_by_rules(record)
        assert tag == "kto"

    def test_unscored_returns_unscored(self):
        tagger = self._make_tagger()
        record = _make_record("u1", fitness_score=None, tools_requested=True)
        assert tagger._classify_by_rules(record) == "unscored"


class TestTextResponsePolicy:
    """Non-tool-call responses follow text_response_policy, not score thresholds."""

    def _make_tagger(self, policy: str):
        catalog = AsyncMock()
        cfg = FlywheelConfig(text_response_policy=policy)
        return AutoTagger(catalog, cfg)

    def test_skip_policy_discards(self):
        tagger = self._make_tagger("skip")
        record = _make_record("t1", fitness_score=0.0, tools_requested=False)
        assert tagger._classify_by_rules(record) == "discard"

    def test_sft_policy_tags_sft(self):
        tagger = self._make_tagger("sft")
        record = _make_record("t2", fitness_score=0.0, tools_requested=False)
        assert tagger._classify_by_rules(record) == "sft"

    def test_kto_policy_tags_kto(self):
        tagger = self._make_tagger("kto")
        record = _make_record("t3", fitness_score=0.0, tools_requested=False)
        assert tagger._classify_by_rules(record) == "kto"

    def test_policy_ignores_score_value(self):
        """Even with high score, non-tool requests follow policy, not thresholds."""
        tagger = self._make_tagger("skip")
        record = _make_record("t4", fitness_score=0.95, tools_requested=False)
        assert tagger._classify_by_rules(record) == "discard"


class TestGRPOEligibility:
    """AutoTagger._is_grpo_eligible checks all required conditions."""

    def _make_tagger(self, **config_kwargs):
        catalog = AsyncMock()
        cfg = FlywheelConfig(**config_kwargs)
        return AutoTagger(catalog, cfg)

    def test_eligible_when_all_conditions_met(self):
        tagger = self._make_tagger()
        record = _make_record(
            "ge1", fitness_score=0.7, tools_requested=True,
            tool_calls=[{"fn": "x"}], is_valid=True,
        )
        assert tagger._is_grpo_eligible(record) is True

    def test_not_eligible_when_no_tools_requested(self):
        tagger = self._make_tagger()
        record = _make_record(
            "ge2", fitness_score=0.7, tools_requested=False,
            tool_calls=[{"fn": "x"}], is_valid=True,
        )
        assert tagger._is_grpo_eligible(record) is False

    def test_not_eligible_when_no_tool_calls(self):
        tagger = self._make_tagger()
        record = _make_record(
            "ge3", fitness_score=0.7, tools_requested=True,
            tool_calls=[], is_valid=True,
        )
        assert tagger._is_grpo_eligible(record) is False

    def test_not_eligible_when_invalid(self):
        tagger = self._make_tagger()
        record = _make_record(
            "ge4", fitness_score=0.7, tools_requested=True,
            tool_calls=[{"fn": "x"}], is_valid=False,
        )
        assert tagger._is_grpo_eligible(record) is False

    def test_not_eligible_when_grpo_disabled(self):
        tagger = self._make_tagger(grpo_enabled=False)
        record = _make_record(
            "ge5", fitness_score=0.7, tools_requested=True,
            tool_calls=[{"fn": "x"}], is_valid=True,
        )
        assert tagger._is_grpo_eligible(record) is False

    def test_not_eligible_when_no_score(self):
        tagger = self._make_tagger()
        record = _make_record(
            "ge6", fitness_score=None, tools_requested=True,
            tool_calls=[{"fn": "x"}], is_valid=True,
        )
        assert tagger._is_grpo_eligible(record) is False


@pytest.mark.asyncio
class TestAutoTaggerTagLogs:
    """AutoTagger.tag_logs integration with catalog."""

    async def test_tags_sft_and_kto(self):
        """High and low scoring tool-call logs get appropriate tags."""
        high = _make_record("h1", fitness_score=0.9, tools_requested=True)
        low = _make_record("l1", fitness_score=0.1, tools_requested=True)

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[high, low], []])
        mock_catalog.update_tag = AsyncMock()

        cfg = FlywheelConfig()
        tagger = AutoTagger(mock_catalog, cfg)
        result = await tagger.tag_logs()

        assert result.sft_count == 1
        assert result.discard_count == 1
        assert result.total_processed == 2

    async def test_ambiguous_without_judge_defaults_to_kto(self):
        """Ambiguous-band logs default to KTO when no judge is available."""
        ambig = _make_record(
            "a1", fitness_score=0.5, tools_requested=True,
            tool_calls=[], is_valid=False,
        )

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[ambig], []])
        mock_catalog.update_tag = AsyncMock()

        cfg = FlywheelConfig()
        tagger = AutoTagger(mock_catalog, cfg, judge=None)
        result = await tagger.tag_logs()

        assert result.kto_count == 1

    async def test_grpo_count_tracked_orthogonally(self):
        """GRPO eligibility is tracked separately from SFT/KTO tags."""
        record = _make_record(
            "gp1", fitness_score=0.9, tools_requested=True,
            tool_calls=[{"fn": "x"}], is_valid=True,
        )

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[record], []])
        mock_catalog.update_tag = AsyncMock()

        cfg = FlywheelConfig()
        tagger = AutoTagger(mock_catalog, cfg)
        result = await tagger.tag_logs()

        assert result.sft_count == 1
        assert result.grpo_count == 1  # GRPO is orthogonal

    async def test_skips_unscored_records(self):
        """Unscored records are skipped (not counted in total_processed)."""
        unscored = _make_record("un1", fitness_score=None, tools_requested=True)

        mock_catalog = AsyncMock()
        mock_catalog.find_logs = AsyncMock(side_effect=[[unscored], []])

        cfg = FlywheelConfig()
        tagger = AutoTagger(mock_catalog, cfg)
        result = await tagger.tag_logs()

        assert result.total_processed == 0
