"""Tests for gradient clipping in evolutionary training."""
import math

import pytest
import torch

from shared.evolutionary.candidate_generator import CandidateGenerator


class TestClipGradients:
    """Tests for CandidateGenerator.clip_gradients()."""

    def test_clips_norms_above_threshold(self):
        """Gradients with norms above max_norm should be clipped."""
        # Create a gradient with norm = 10.0
        grad = torch.ones(10) * (10.0 / math.sqrt(10))
        assert abs(grad.norm().item() - 10.0) < 1e-4

        gradients = {"layer.weight": grad}
        clipped = CandidateGenerator.clip_gradients(gradients, max_norm=1.0)

        result_norm = clipped["layer.weight"].norm().item()
        assert abs(result_norm - 1.0) < 1e-5, (
            f"Expected norm ~1.0 after clipping, got {result_norm}"
        )

    def test_passes_through_norms_below_threshold(self):
        """Gradients with norms below max_norm should not be modified."""
        grad = torch.ones(10) * 0.01
        original_norm = grad.norm().item()

        gradients = {"layer.weight": grad}
        clipped = CandidateGenerator.clip_gradients(gradients, max_norm=1.0)

        result_norm = clipped["layer.weight"].norm().item()
        assert abs(result_norm - original_norm) < 1e-6, (
            f"Expected norm {original_norm} (unchanged), got {result_norm}"
        )

    def test_preserves_direction(self):
        """Clipped gradients should have the same direction as originals."""
        grad = torch.tensor([3.0, 4.0])  # norm = 5.0
        gradients = {"layer.weight": grad}
        clipped = CandidateGenerator.clip_gradients(gradients, max_norm=1.0)

        clipped_grad = clipped["layer.weight"]
        # Cosine similarity should be 1.0 (same direction)
        cosine_sim = torch.nn.functional.cosine_similarity(
            grad.unsqueeze(0), clipped_grad.unsqueeze(0)
        ).item()
        assert abs(cosine_sim - 1.0) < 1e-5, (
            f"Expected cosine similarity ~1.0, got {cosine_sim}"
        )

        # Verify magnitude is reduced
        assert clipped_grad.norm().item() < grad.norm().item()

    def test_zero_gradients_handled(self):
        """Zero gradients should not cause division by zero."""
        grad = torch.zeros(10)
        gradients = {"layer.weight": grad}

        # Should not raise
        clipped = CandidateGenerator.clip_gradients(gradients, max_norm=1.0)
        assert clipped["layer.weight"].norm().item() == 0.0

    def test_empty_gradients_dict(self):
        """Empty gradient dict should return empty dict."""
        clipped = CandidateGenerator.clip_gradients({}, max_norm=1.0)
        assert clipped == {}

    def test_multiple_parameters(self):
        """Clipping should work independently per parameter."""
        gradients = {
            "layer1.weight": torch.ones(10) * 10.0,  # norm >> 1.0
            "layer2.weight": torch.ones(10) * 0.01,  # norm << 1.0
        }
        max_norm = 1.0
        clipped = CandidateGenerator.clip_gradients(gradients, max_norm=max_norm)

        # layer1 should be clipped
        assert clipped["layer1.weight"].norm().item() <= max_norm + 1e-5
        # layer2 should be unchanged
        original_norm = gradients["layer2.weight"].norm().item()
        assert abs(clipped["layer2.weight"].norm().item() - original_norm) < 1e-6

    def test_exact_threshold_not_clipped(self):
        """Gradient with norm exactly at max_norm should not be clipped."""
        # Create a gradient with norm exactly 1.0
        grad = torch.zeros(10)
        grad[0] = 1.0
        assert abs(grad.norm().item() - 1.0) < 1e-6

        gradients = {"layer.weight": grad}
        clipped = CandidateGenerator.clip_gradients(gradients, max_norm=1.0)

        # Should be unchanged (not strictly greater)
        assert torch.allclose(clipped["layer.weight"], grad)

    def test_large_max_norm_no_clipping(self):
        """When max_norm is very large, nothing should be clipped."""
        grad = torch.ones(10) * 100.0
        original_norm = grad.norm().item()
        gradients = {"layer.weight": grad}

        clipped = CandidateGenerator.clip_gradients(gradients, max_norm=1e6)
        assert abs(clipped["layer.weight"].norm().item() - original_norm) < 1e-3
