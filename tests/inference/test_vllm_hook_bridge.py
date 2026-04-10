"""Tests for the vLLM Hook bridge — LayerHookManager with mock models.

These tests create a real (small) PyTorch transformer-like model and verify
that forward hooks correctly capture intermediate layer hidden states and
that plugins receive them during the forward pass.

This simulates what happens inside a vLLM server when inference plugins
are active.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from shared.inference.config import DoLaConfig
from shared.inference.hooks.vllm_hook_bridge import LayerHookManager
from shared.inference.plugins.dola import DoLaPlugin


# ---------------------------------------------------------------------------
# Fake transformer model (mimics LLaMA naming: model.layers.N)
# ---------------------------------------------------------------------------

class FakeTransformerLayer(nn.Module):
    """Single transformer layer with a linear projection."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Return tuple like real transformer layers: (hidden_states, past_kv)
        return self.proj(x), None


class FakeTransformerModel(nn.Module):
    """Minimal transformer model with LLaMA-style naming.

    Structure:
        model.layers.0 = FakeTransformerLayer
        model.layers.1 = FakeTransformerLayer
        ...
        lm_head = Linear(hidden_dim, vocab_size)
    """

    def __init__(self, num_layers: int, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [FakeTransformerLayer(hidden_dim) for _ in range(num_layers)],
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.model.layers:
            hidden, _ = layer(hidden)
        return self.lm_head(hidden)


class FakeGPT2Model(nn.Module):
    """Minimal model with GPT-2 naming (transformer.h.N)."""

    def __init__(self, num_layers: int, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(
            [FakeTransformerLayer(hidden_dim) for _ in range(num_layers)],
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.transformer.h:
            hidden, _ = layer(hidden)
        return self.lm_head(hidden)


# ---------------------------------------------------------------------------
# _extract_layer_index
# ---------------------------------------------------------------------------

class TestExtractLayerIndex:
    """Tests for LayerHookManager._extract_layer_index (static method)."""

    _extract = staticmethod(LayerHookManager._extract_layer_index)

    def test_llama_style(self):
        assert self._extract("model.layers.15") == 15

    def test_gpt2_style(self):
        assert self._extract("transformer.h.7") == 7

    def test_block_style(self):
        assert self._extract("encoder.block.3") == 3

    def test_no_match(self):
        assert self._extract("model.embed_tokens") is None

    def test_zero_index(self):
        assert self._extract("model.layers.0") == 0


# ---------------------------------------------------------------------------
# LayerHookManager construction
# ---------------------------------------------------------------------------

class TestLayerHookManagerInit:
    def test_collects_target_layers(self):
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[2, 4, 6]))
        manager = LayerHookManager([plugin], num_hidden_layers=32)
        assert manager._target_layers == {2, 4, 6}

    def test_empty_plugins_raises(self):
        with pytest.raises(ValueError, match="at least one plugin"):
            LayerHookManager([], num_hidden_layers=32)

    def test_zero_layers_raises(self):
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[0]))
        with pytest.raises(ValueError, match="num_hidden_layers must be >= 1"):
            LayerHookManager([plugin], num_hidden_layers=0)


# ---------------------------------------------------------------------------
# Hook registration on LLaMA-style model
# ---------------------------------------------------------------------------

class TestHookRegistrationLLaMA:
    def test_registers_hooks_on_target_layers(self):
        model = FakeTransformerModel(num_layers=8, hidden_dim=32, vocab_size=100)
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[1, 3, 5]))
        manager = LayerHookManager([plugin], num_hidden_layers=8)
        manager.register_hooks(model)
        assert manager.num_hooks == 3

    def test_missing_layers_raises(self):
        model = FakeTransformerModel(num_layers=4, hidden_dim=32, vocab_size=100)
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[0, 10]))
        manager = LayerHookManager([plugin], num_hidden_layers=16)
        with pytest.raises(RuntimeError, match="Could not find"):
            manager.register_hooks(model)

    def test_double_registration_raises(self):
        model = FakeTransformerModel(num_layers=8, hidden_dim=32, vocab_size=100)
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[1]))
        manager = LayerHookManager([plugin], num_hidden_layers=8)
        manager.register_hooks(model)
        with pytest.raises(RuntimeError, match="already registered"):
            manager.register_hooks(model)

    def test_remove_hooks(self):
        model = FakeTransformerModel(num_layers=8, hidden_dim=32, vocab_size=100)
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[1, 3]))
        manager = LayerHookManager([plugin], num_hidden_layers=8)
        manager.register_hooks(model)
        assert manager.num_hooks == 2
        manager.remove_hooks()
        assert manager.num_hooks == 0


# ---------------------------------------------------------------------------
# Hook registration on GPT-2-style model
# ---------------------------------------------------------------------------

class TestHookRegistrationGPT2:
    def test_registers_hooks_on_h_layers(self):
        model = FakeGPT2Model(num_layers=6, hidden_dim=32, vocab_size=100)
        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[0, 2, 4]))
        manager = LayerHookManager([plugin], num_hidden_layers=6)
        manager.register_hooks(model)
        assert manager.num_hooks == 3


# ---------------------------------------------------------------------------
# End-to-end: forward pass with hooks captures states
# ---------------------------------------------------------------------------

class TestEndToEndForwardPass:
    def test_hooks_capture_hidden_states(self):
        """Forward pass through model triggers hooks that capture states."""
        hidden_dim = 32
        vocab_size = 100
        model = FakeTransformerModel(num_layers=8, hidden_dim=hidden_dim, vocab_size=vocab_size)

        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[1, 3, 5]))
        manager = LayerHookManager([plugin], num_hidden_layers=8)
        manager.register_hooks(model)

        # Forward pass
        x = torch.randn(1, 1, hidden_dim)
        final_logits = model(x)

        # Plugin should have captured states from layers 1, 3, 5
        assert len(plugin._captured_states) == 3
        assert set(plugin._captured_states.keys()) == {1, 3, 5}

        # Each captured state should have the right shape
        for idx, state in plugin._captured_states.items():
            assert state.shape[-1] == hidden_dim

    def test_apply_plugins_modifies_logits(self):
        """After forward pass, apply_plugins should produce contrastive logits."""
        hidden_dim = 32
        vocab_size = 100
        model = FakeTransformerModel(num_layers=8, hidden_dim=hidden_dim, vocab_size=vocab_size)

        plugin = DoLaPlugin(
            DoLaConfig(enabled=True, premature_layers=[1, 3, 5], jsd_threshold=0.0),
        )
        manager = LayerHookManager([plugin], num_hidden_layers=8)
        manager.register_hooks(model)

        # Forward pass
        x = torch.randn(1, 1, hidden_dim)
        final_logits = model(x)

        # Apply contrastive decoding
        modified_logits = manager.apply_plugins(
            final_logits[:, -1:, :].squeeze(1) if final_logits.dim() == 3 else final_logits,
            model.lm_head,
        )

        assert modified_logits.shape[-1] == vocab_size

    def test_reset_clears_state_between_steps(self):
        """reset_plugins should clear captured states for next step."""
        hidden_dim = 32
        model = FakeTransformerModel(num_layers=8, hidden_dim=hidden_dim, vocab_size=100)

        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[1, 3]))
        manager = LayerHookManager([plugin], num_hidden_layers=8)
        manager.register_hooks(model)

        # First forward pass
        model(torch.randn(1, 1, hidden_dim))
        assert len(plugin._captured_states) == 2

        # Reset
        manager.reset_plugins()
        assert len(plugin._captured_states) == 0

        # Second forward pass
        model(torch.randn(1, 1, hidden_dim))
        assert len(plugin._captured_states) == 2

    def test_hooks_removed_after_cleanup(self):
        """After remove_hooks, forward pass should NOT capture states."""
        hidden_dim = 32
        model = FakeTransformerModel(num_layers=8, hidden_dim=hidden_dim, vocab_size=100)

        plugin = DoLaPlugin(DoLaConfig(enabled=True, premature_layers=[1, 3]))
        manager = LayerHookManager([plugin], num_hidden_layers=8)
        manager.register_hooks(model)

        # Remove hooks
        manager.remove_hooks()

        # Forward pass should NOT trigger hooks
        plugin.reset()
        model(torch.randn(1, 1, hidden_dim))
        assert len(plugin._captured_states) == 0


# ---------------------------------------------------------------------------
# Full pipeline: hook → capture → contrast → filter
# ---------------------------------------------------------------------------

class TestFullDoLaPipeline:
    """Integration test: full DoLa pipeline with a real (small) model."""

    def test_full_pipeline_produces_valid_logits(self):
        hidden_dim = 64
        vocab_size = 200
        num_layers = 16
        model = FakeTransformerModel(
            num_layers=num_layers, hidden_dim=hidden_dim, vocab_size=vocab_size,
        )

        plugin = DoLaPlugin(
            DoLaConfig(
                enabled=True,
                premature_layers="high",
                relative_top=0.1,
                jsd_threshold=0.0,
            ),
        )

        target_layers = plugin.target_layers(num_layers)
        manager = LayerHookManager([plugin], num_hidden_layers=num_layers)
        manager.register_hooks(model)

        # Forward pass
        x = torch.randn(1, 1, hidden_dim)
        final_logits = model(x)

        # Apply DoLa
        if final_logits.dim() == 3:
            final_logits = final_logits.squeeze(1)

        modified = manager.apply_plugins(final_logits, model.lm_head)

        # Should be a valid logits tensor
        assert modified.shape == (1, vocab_size)

        # With random weights the contrastive logits may all be close together,
        # so relative-top filtering may or may not mask tokens.  Just verify
        # the output is finite (no NaN) and at least one token is selectable.
        finite_mask = torch.isfinite(modified)
        assert finite_mask.any(), "At least one token must be selectable"

        # Cleanup
        manager.reset_plugins()
        manager.remove_hooks()

    def test_multiple_generation_steps(self):
        """Simulate multiple generation steps (token-by-token)."""
        hidden_dim = 32
        vocab_size = 50
        num_layers = 8
        model = FakeTransformerModel(
            num_layers=num_layers, hidden_dim=hidden_dim, vocab_size=vocab_size,
        )

        plugin = DoLaPlugin(
            DoLaConfig(enabled=True, premature_layers=[1, 3, 5], jsd_threshold=0.0),
        )
        manager = LayerHookManager([plugin], num_hidden_layers=num_layers)
        manager.register_hooks(model)

        # Simulate 5 generation steps
        for step in range(5):
            x = torch.randn(1, 1, hidden_dim)
            final_logits = model(x)
            if final_logits.dim() == 3:
                final_logits = final_logits.squeeze(1)

            modified = manager.apply_plugins(final_logits, model.lm_head)

            # Each step should produce valid output
            assert modified.shape == (1, vocab_size)
            assert not torch.isnan(modified).any()

            # Plugin states are cleared by modify_logits, but call reset for safety
            manager.reset_plugins()

        manager.remove_hooks()
