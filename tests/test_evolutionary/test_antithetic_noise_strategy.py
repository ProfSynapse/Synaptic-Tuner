"""Tests for AntitheticNoiseStrategy."""

import torch

from shared.evolutionary.strategies.antithetic_noise import AntitheticNoiseStrategy


class TestAntitheticNoiseStrategy:
    def _sample_gradients(self):
        return {"w": torch.ones(10) * 0.5, "b": torch.ones(3) * 0.1}

    def test_generate_correct_count(self):
        strategy = AntitheticNoiseStrategy(noise_scale=0.1, include_pure=True)
        candidates = strategy.generate_candidates(self._sample_gradients(), num_candidates=5)
        assert len(candidates) == 5

    def test_first_candidate_is_pure_when_enabled(self):
        grads = self._sample_gradients()
        strategy = AntitheticNoiseStrategy(noise_scale=0.2, include_pure=True)
        candidates = strategy.generate_candidates(grads, num_candidates=3)
        for name in grads:
            assert torch.allclose(candidates[0].gradients[name], grads[name])

    def test_mirrored_candidates_use_opposite_noise(self):
        grads = self._sample_gradients()
        strategy = AntitheticNoiseStrategy(noise_scale=0.2, include_pure=True)
        candidates = strategy.generate_candidates(grads, num_candidates=3)

        for name in grads:
            plus_delta = candidates[1].gradients[name] - grads[name]
            minus_delta = candidates[2].gradients[name] - grads[name]
            assert torch.allclose(plus_delta, -minus_delta)

    def test_candidate_metadata_marks_pair_and_variant(self):
        strategy = AntitheticNoiseStrategy(noise_scale=0.2, include_pure=True)
        candidates = strategy.generate_candidates(self._sample_gradients(), num_candidates=5)

        assert candidates[1].metadata["pair_index"] == 0
        assert candidates[1].metadata["variant"] == "plus"
        assert candidates[2].metadata["pair_index"] == 0
        assert candidates[2].metadata["variant"] == "minus"

    def test_zero_noise_produces_base_gradient(self):
        grads = self._sample_gradients()
        strategy = AntitheticNoiseStrategy(noise_scale=0.0, include_pure=False)
        candidates = strategy.generate_candidates(grads, num_candidates=2)
        for candidate in candidates:
            for name in grads:
                assert torch.allclose(candidate.gradients[name], grads[name])
