"""Tests for GradientNoiseStrategy — noise injection and adaptive scaling."""

import torch

from shared.evolutionary.strategies.gradient_noise import GradientNoiseStrategy


class TestGradientNoiseStrategy:

    def _sample_gradients(self):
        return {"w": torch.ones(10) * 0.5, "b": torch.ones(3) * 0.1}

    def test_generate_correct_count(self):
        strategy = GradientNoiseStrategy(noise_scale=0.1, include_pure=True)
        candidates = strategy.generate_candidates(self._sample_gradients(), num_candidates=5)
        assert len(candidates) == 5

    def test_include_pure_false_all_noisy(self):
        strategy = GradientNoiseStrategy(noise_scale=0.5, include_pure=False)
        candidates = strategy.generate_candidates(self._sample_gradients(), num_candidates=3)
        assert len(candidates) == 3
        # All should have noise_scale > 0 in metadata
        for c in candidates:
            assert c.metadata["noise_scale"] > 0

    def test_include_pure_true_first_is_clean(self):
        grads = self._sample_gradients()
        strategy = GradientNoiseStrategy(noise_scale=0.5, include_pure=True)
        candidates = strategy.generate_candidates(grads, num_candidates=4)
        pure = candidates[0]
        for name in grads:
            assert torch.allclose(pure.gradients[name], grads[name])

    def test_noise_scale_zero_produces_unchanged(self):
        grads = self._sample_gradients()
        strategy = GradientNoiseStrategy(noise_scale=0.0, include_pure=False)
        candidates = strategy.generate_candidates(grads, num_candidates=2)
        for c in candidates:
            for name in grads:
                assert torch.allclose(c.gradients[name], grads[name])

    def test_max_grad_norm_caps_noise(self):
        """When raw gradient norm exceeds max_grad_norm, noise should be capped."""
        grads = {"w": torch.ones(10) * 100.0}  # norm ~316
        # With max_grad_norm=1.0, noise magnitude should be based on 1.0 not 316
        strategy = GradientNoiseStrategy(noise_scale=0.1, include_pure=False, max_grad_norm=1.0)
        candidates = strategy.generate_candidates(grads, num_candidates=10)
        # Noisy grads should be close to original since noise is capped
        for c in candidates:
            diff_norm = (c.gradients["w"] - grads["w"]).norm().item()
            # Noise ~ 0.1 * 1.0 * randn; 10-element randn norm is ~sqrt(10)~3.16
            # So diff should be much less than if we used raw norm (316 * 0.1 = 31.6)
            assert diff_norm < 10.0

    def test_adaptive_scale(self):
        strategy = GradientNoiseStrategy(
            noise_scale=1.0, include_pure=False, adaptive_scale=True
        )
        grads = {"w": torch.ones(10) * 10.0}
        candidates = strategy.generate_candidates(grads, num_candidates=3)
        assert len(candidates) == 3
        # The adaptive scale should be < noise_scale since avg_norm is large
        for c in candidates:
            assert c.metadata["noise_scale"] < 1.0

    def test_candidate_ids_are_sequential(self):
        strategy = GradientNoiseStrategy(include_pure=True)
        candidates = strategy.generate_candidates(self._sample_gradients(), num_candidates=4)
        ids = [c.id for c in candidates]
        assert ids == [0, 1, 2, 3]

    def test_gradient_shapes_preserved(self):
        grads = {"a": torch.randn(3, 4), "b": torch.randn(5)}
        strategy = GradientNoiseStrategy()
        candidates = strategy.generate_candidates(grads, num_candidates=2)
        for c in candidates:
            assert c.gradients["a"].shape == (3, 4)
            assert c.gradients["b"].shape == (5,)
