"""Tests for logits-only plugins (Min-P, Repetition Penalty).

These plugins only modify final logits and do not require vLLM Hook.
Tests use real torch tensors to verify the math.
"""

from __future__ import annotations

import pytest
import torch

from shared.inference.config import MinPConfig, RepetitionPenaltyConfig
from shared.inference.plugins.min_p import MinPPlugin
from shared.inference.plugins.repetition import RepetitionPenaltyPlugin


# ---------------------------------------------------------------------------
# MinPPlugin
# ---------------------------------------------------------------------------

class TestMinPPlugin:
    def test_name(self, min_p_config):
        plugin = MinPPlugin(min_p_config)
        assert plugin.name == "min_p"

    def test_masks_low_probability_tokens(self):
        plugin = MinPPlugin(MinPConfig(enabled=True, threshold=0.1))
        # Logits where one token is clearly dominant
        logits = torch.tensor([10.0, 0.0, -5.0, -10.0])
        result = plugin([], logits)

        # Token 0 (highest prob) should survive
        assert result[0] != float("-inf")
        # Very low probability tokens should be masked
        assert result[3] == float("-inf")

    def test_preserves_high_probability_tokens(self):
        plugin = MinPPlugin(MinPConfig(enabled=True, threshold=0.01))
        # Two tokens with similar high logits
        logits = torch.tensor([5.0, 4.9, -10.0])
        result = plugin([], logits)
        assert result[0] != float("-inf")
        assert result[1] != float("-inf")
        assert result[2] == float("-inf")

    def test_all_tokens_survive_with_low_threshold(self):
        plugin = MinPPlugin(MinPConfig(enabled=True, threshold=0.001))
        logits = torch.tensor([1.0, 0.9, 0.8, 0.7])
        result = plugin([], logits)
        # With very low threshold, all relatively similar tokens survive
        for i in range(4):
            assert result[i] != float("-inf")

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            MinPPlugin(MinPConfig(enabled=True, threshold=0.0))
        with pytest.raises(ValueError, match="threshold must be in"):
            MinPPlugin(MinPConfig(enabled=True, threshold=1.0))

    def test_output_shape_matches_input(self):
        plugin = MinPPlugin(MinPConfig(enabled=True, threshold=0.05))
        logits = torch.randn(1000)
        result = plugin([], logits)
        assert result.shape == logits.shape


# ---------------------------------------------------------------------------
# RepetitionPenaltyPlugin
# ---------------------------------------------------------------------------

class TestRepetitionPenaltyPlugin:
    def test_name(self, repetition_config):
        plugin = RepetitionPenaltyPlugin(repetition_config)
        assert plugin.name == "repetition_penalty"

    def test_penalizes_repeated_tokens(self):
        plugin = RepetitionPenaltyPlugin(
            RepetitionPenaltyConfig(enabled=True, penalty=2.0, window=10),
        )
        logits = torch.tensor([5.0, 3.0, 1.0, -1.0, -3.0])
        token_ids = [0, 1, 3]  # tokens 0, 1, 3 appeared recently

        result = plugin(token_ids, logits)

        # Positive logits divided by penalty
        assert result[0] == pytest.approx(2.5)  # 5.0 / 2.0
        assert result[1] == pytest.approx(1.5)  # 3.0 / 2.0
        # Negative logits multiplied by penalty
        assert result[3] == pytest.approx(-2.0)  # -1.0 * 2.0
        # Non-repeated tokens unchanged
        assert result[2] == pytest.approx(1.0)
        assert result[4] == pytest.approx(-3.0)

    def test_no_penalty_when_no_tokens(self):
        plugin = RepetitionPenaltyPlugin(
            RepetitionPenaltyConfig(enabled=True, penalty=2.0, window=10),
        )
        logits = torch.tensor([5.0, 3.0])
        result = plugin([], logits)
        assert torch.equal(result, logits)

    def test_penalty_one_is_noop(self):
        plugin = RepetitionPenaltyPlugin(
            RepetitionPenaltyConfig(enabled=True, penalty=1.0, window=10),
        )
        logits = torch.tensor([5.0, 3.0])
        result = plugin([0, 1], logits)
        assert torch.equal(result, logits)

    def test_window_limits_lookback(self):
        plugin = RepetitionPenaltyPlugin(
            RepetitionPenaltyConfig(enabled=True, penalty=2.0, window=3),
        )
        # Vocab size must cover all token IDs used
        logits = torch.tensor([5.0, 3.0, 1.0, 0.0, 0.0])
        # Token 0 appeared 10 tokens ago (outside window of 3)
        # Use token ID 4 as filler (within vocab bounds)
        token_ids = [0] + [4] * 10 + [1, 2]

        result = plugin(token_ids, logits)
        # Token 0 is outside the window — should NOT be penalized
        assert result[0] == pytest.approx(5.0)
        # Tokens 1, 2 are in the window — should be penalized
        assert result[1] == pytest.approx(1.5)  # 3.0 / 2.0
        assert result[2] == pytest.approx(0.5)  # 1.0 / 2.0

    def test_invalid_penalty_raises(self):
        with pytest.raises(ValueError, match="penalty must be >= 1.0"):
            RepetitionPenaltyPlugin(
                RepetitionPenaltyConfig(enabled=True, penalty=0.5),
            )

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="Window size must be >= 1"):
            RepetitionPenaltyPlugin(
                RepetitionPenaltyConfig(enabled=True, penalty=1.5, window=0),
            )

    def test_output_shape_matches_input(self):
        plugin = RepetitionPenaltyPlugin(
            RepetitionPenaltyConfig(enabled=True, penalty=1.5, window=32),
        )
        logits = torch.randn(50000)
        result = plugin(list(range(100)), logits)
        assert result.shape == logits.shape
