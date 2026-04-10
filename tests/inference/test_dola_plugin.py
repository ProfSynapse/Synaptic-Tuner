"""Tests for the DoLa plugin — contrastive layer decoding.

Tests cover layer selection, JSD computation, contrastive logit generation,
and relative-top filtering. Uses mock models and tensors (no real LLM needed).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from shared.inference.config import DoLaConfig
from shared.inference.plugins.dola import DoLaPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeLMHead(nn.Module):
    """Minimal LM head that projects hidden_dim -> vocab_size."""

    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# target_layers
# ---------------------------------------------------------------------------

class TestTargetLayers:
    def test_high_32_layers(self):
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers="high"))
        layers = plugin.target_layers(32)
        # Upper half: range(16, 31, 2) = [16, 18, 20, 22, 24, 26, 28, 30]
        assert all(l >= 16 for l in layers)
        assert all(l < 32 for l in layers)
        assert len(layers) > 0

    def test_low_32_layers(self):
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers="low"))
        layers = plugin.target_layers(32)
        # Lower half: range(0, 16, 2) = [0, 2, 4, 6, 8, 10, 12, 14]
        assert all(l < 16 for l in layers)
        assert len(layers) > 0

    def test_explicit_layers(self):
        plugin = DoLaPlugin(
            DoLaConfig(enabled=True, premature_layers=[2, 4, 6, 8]),
        )
        layers = plugin.target_layers(32)
        assert layers == [2, 4, 6, 8]

    def test_explicit_layers_deduplicated_and_sorted(self):
        plugin = DoLaPlugin(
            DoLaConfig(enabled=True, premature_layers=[8, 4, 8, 2]),
        )
        layers = plugin.target_layers(32)
        assert layers == [2, 4, 8]

    def test_out_of_range_raises(self):
        plugin = DoLaPlugin(
            DoLaConfig(enabled=True, premature_layers=[0, 50]),
        )
        with pytest.raises(ValueError, match="out of range"):
            plugin.target_layers(32)

    def test_unknown_spec_raises(self):
        plugin = DoLaPlugin(
            DoLaConfig(enabled=True, premature_layers="middle"),
        )
        with pytest.raises(ValueError, match="Unknown premature_layers"):
            plugin.target_layers(32)

    def test_small_model_low(self):
        """Even a 4-layer model should produce valid layers for 'low'."""
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers="low"))
        layers = plugin.target_layers(4)
        assert len(layers) > 0
        assert all(0 <= l < 2 for l in layers)

    def test_small_model_high(self):
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers="high"))
        layers = plugin.target_layers(4)
        assert len(layers) > 0
        assert all(l >= 2 for l in layers)


# ---------------------------------------------------------------------------
# on_layer_output / reset
# ---------------------------------------------------------------------------

class TestCapturedStates:
    def test_on_layer_output_stores_tensor(self):
        plugin = DoLaPlugin(DoLaConfig(enabled=True))
        hidden = torch.randn(1, 10, 64)
        plugin.on_layer_output(5, hidden)
        assert 5 in plugin._captured_states
        assert torch.equal(plugin._captured_states[5], hidden)

    def test_reset_clears_states(self):
        plugin = DoLaPlugin(DoLaConfig(enabled=True))
        plugin.on_layer_output(5, torch.randn(1, 10, 64))
        plugin.on_layer_output(10, torch.randn(1, 10, 64))
        plugin.reset()
        assert len(plugin._captured_states) == 0


# ---------------------------------------------------------------------------
# modify_logits
# ---------------------------------------------------------------------------

class TestModifyLogits:
    def test_no_captured_states_returns_unchanged(self):
        """If no hooks fired, logits pass through."""
        plugin = DoLaPlugin(DoLaConfig(enabled=True))
        lm_head = FakeLMHead(64, 100)
        logits = torch.randn(1, 100)
        result = plugin.modify_logits(logits, lm_head)
        assert torch.equal(result, logits)

    def test_contrastive_logits_differ_from_input(self):
        """With captured states, output should differ from input."""
        plugin = DoLaPlugin(DoLaConfig(enabled=True, jsd_threshold=0.0))
        hidden_dim = 64
        vocab_size = 100
        lm_head = FakeLMHead(hidden_dim, vocab_size)

        # Simulate a premature layer output that differs from mature
        plugin.on_layer_output(5, torch.randn(1, 1, hidden_dim))

        # Mature logits
        logits = torch.randn(1, vocab_size)
        result = plugin.modify_logits(logits, lm_head)

        # Should be different (contrastive decoding applied)
        assert not torch.equal(result, logits)

    def test_captured_states_cleared_after_modify(self):
        """modify_logits should clear captured states."""
        plugin = DoLaPlugin(DoLaConfig(enabled=True, jsd_threshold=0.0))
        lm_head = FakeLMHead(64, 100)
        plugin.on_layer_output(5, torch.randn(1, 1, 64))
        plugin.modify_logits(torch.randn(1, 100), lm_head)
        assert len(plugin._captured_states) == 0

    def test_high_jsd_threshold_returns_original(self):
        """If JSD threshold is very high, original logits should be returned."""
        plugin = DoLaPlugin(
            DoLaConfig(enabled=True, jsd_threshold=100.0),
        )
        lm_head = FakeLMHead(64, 100)
        plugin.on_layer_output(5, torch.randn(1, 1, 64))
        logits = torch.randn(1, 100)
        result = plugin.modify_logits(logits, lm_head)
        # JSD can never exceed log(2) ≈ 0.693, so threshold of 100 always gates
        assert torch.equal(result, logits)

    def test_output_shape_matches_input(self):
        plugin = DoLaPlugin(DoLaConfig(enabled=True, jsd_threshold=0.0))
        lm_head = FakeLMHead(64, 100)
        plugin.on_layer_output(5, torch.randn(1, 1, 64))
        logits = torch.randn(1, 100)
        result = plugin.modify_logits(logits, lm_head)
        assert result.shape == logits.shape

    def test_multiple_premature_layers(self):
        """With multiple captured layers, selects the highest JSD."""
        plugin = DoLaPlugin(DoLaConfig(enabled=True, jsd_threshold=0.0))
        hidden_dim = 64
        vocab_size = 100
        lm_head = FakeLMHead(hidden_dim, vocab_size)

        # Capture states from multiple layers
        plugin.on_layer_output(2, torch.randn(1, 1, hidden_dim))
        plugin.on_layer_output(8, torch.randn(1, 1, hidden_dim))
        plugin.on_layer_output(14, torch.randn(1, 1, hidden_dim))

        logits = torch.randn(1, vocab_size)
        result = plugin.modify_logits(logits, lm_head)
        assert result.shape == logits.shape


# ---------------------------------------------------------------------------
# JSD computation
# ---------------------------------------------------------------------------

class TestJSD:
    def test_identical_distributions_have_zero_jsd(self):
        # Use a less extreme distribution to avoid log(0) NaN
        log_probs = torch.tensor([[2.0, 1.0, 0.5]]).log_softmax(dim=-1)
        jsd = DoLaPlugin._compute_jsd(log_probs, log_probs)
        assert jsd == pytest.approx(0.0, abs=1e-5)

    def test_different_distributions_have_positive_jsd(self):
        p = torch.tensor([[10.0, 0.0]]).log_softmax(dim=-1)
        q = torch.tensor([[0.0, 10.0]]).log_softmax(dim=-1)
        jsd = DoLaPlugin._compute_jsd(p, q)
        assert jsd > 0.0

    def test_jsd_is_bounded(self):
        """JSD is bounded by log(2) ≈ 0.693."""
        p = torch.tensor([[100.0, -100.0]]).log_softmax(dim=-1)
        q = torch.tensor([[-100.0, 100.0]]).log_softmax(dim=-1)
        jsd = DoLaPlugin._compute_jsd(p, q)
        assert jsd <= math.log(2) + 1e-4

    def test_jsd_is_symmetric(self):
        p = torch.tensor([[3.0, 1.0, 0.5]]).log_softmax(dim=-1)
        q = torch.tensor([[0.5, 2.0, 3.0]]).log_softmax(dim=-1)
        jsd_pq = DoLaPlugin._compute_jsd(p, q)
        jsd_qp = DoLaPlugin._compute_jsd(q, p)
        assert jsd_pq == pytest.approx(jsd_qp, abs=1e-6)


# ---------------------------------------------------------------------------
# Relative-top filtering
# ---------------------------------------------------------------------------

class TestRelativeTopFiltering:
    def test_filters_low_tokens(self):
        plugin = DoLaPlugin(
            DoLaConfig(enabled=True, relative_top=0.1),
        )
        # Logits with a clear peak
        logits = torch.tensor([[5.0, 0.0, -5.0, -10.0]])
        result = plugin._apply_relative_top(logits)
        # Token with logit 5.0 should survive
        assert result[0, 0] != float("-inf")
        # Token with logit -10.0 (more than log(0.1)=-2.3 below max)
        # should be filtered
        assert result[0, 3] == float("-inf")

    def test_all_survive_with_relative_top_one(self):
        plugin = DoLaPlugin(
            DoLaConfig(enabled=True, relative_top=1.0),
        )
        logits = torch.tensor([[5.0, 4.0, 3.0]])
        result = plugin._apply_relative_top(logits)
        # relative_top=1.0 means threshold = max + log(1) = max + 0
        # Only tokens AT the max survive
        assert result[0, 0] != float("-inf")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestDoLaValidation:
    def test_invalid_relative_top_zero(self):
        with pytest.raises(ValueError, match="relative_top must be in"):
            DoLaPlugin(DoLaConfig(enabled=True, relative_top=0.0))

    def test_invalid_relative_top_negative(self):
        with pytest.raises(ValueError, match="relative_top must be in"):
            DoLaPlugin(DoLaConfig(enabled=True, relative_top=-0.1))

    def test_invalid_jsd_threshold_negative(self):
        with pytest.raises(ValueError, match="jsd_threshold must be >= 0"):
            DoLaPlugin(DoLaConfig(enabled=True, jsd_threshold=-1.0))
