"""Tests for CandidateGenerator — generation, selection, and gradient helpers."""

import pytest
import torch

from shared.evolutionary.candidate_generator import CandidateGenerator
from shared.evolutionary.config import EvolutionaryConfig
from shared.evolutionary.strategies.base import GradientCandidate


class TestCandidateGeneration:
    """Verify generate() produces the expected number and shape of candidates."""

    def _make_generator(self, **overrides):
        defaults = dict(
            enabled=True,
            num_candidates=4,
            strategy="gradient_noise",
            noise_scale=0.1,
            max_grad_norm=1.0,
            validation_config={"dummy": True},
        )
        defaults.update(overrides)
        config = EvolutionaryConfig(**defaults)
        return CandidateGenerator(config)

    def _sample_gradients(self):
        return {
            "layer1.weight": torch.randn(4, 4),
            "layer2.bias": torch.randn(4),
        }

    def test_generates_correct_count(self):
        gen = self._make_generator(num_candidates=5)
        candidates = gen.generate(self._sample_gradients(), step=0)
        assert len(candidates) == 5

    def test_first_candidate_is_pure_gradient(self):
        gen = self._make_generator()
        grads = self._sample_gradients()
        candidates = gen.generate(grads, step=0)
        # First candidate should be the pure (unmodified) gradient
        pure = candidates[0]
        assert "Pure" in pure.description or pure.metadata.get("noise_scale") == 0.0
        for name in grads:
            assert torch.allclose(pure.gradients[name], grads[name])

    def test_noisy_candidates_differ_from_base(self):
        gen = self._make_generator(num_candidates=3, noise_scale=1.0)
        grads = self._sample_gradients()
        candidates = gen.generate(grads, step=0)
        # At least one noisy candidate should differ from base
        any_different = False
        for c in candidates[1:]:
            for name in grads:
                if not torch.allclose(c.gradients[name], grads[name]):
                    any_different = True
                    break
        assert any_different

    def test_candidate_shapes_match_input(self):
        gen = self._make_generator()
        grads = self._sample_gradients()
        candidates = gen.generate(grads, step=0)
        for c in candidates:
            for name, grad in grads.items():
                assert c.gradients[name].shape == grad.shape

    def test_antithetic_noise_strategy_generates_mirrored_pairs(self):
        gen = self._make_generator(strategy="antithetic_noise", num_candidates=5, noise_scale=0.2)
        grads = self._sample_gradients()
        candidates = gen.generate(grads, step=0)

        assert len(candidates) == 5
        assert candidates[0].metadata["variant"] == "pure"
        assert candidates[1].metadata["variant"] == "plus"
        assert candidates[2].metadata["variant"] == "minus"

        for name in grads:
            plus_delta = candidates[1].gradients[name] - grads[name]
            minus_delta = candidates[2].gradients[name] - grads[name]
            assert torch.allclose(plus_delta, -minus_delta)


class TestCandidateSelection:
    """Verify select_best picks correctly from evaluated candidates."""

    def _make_generator(self, selection_method="best"):
        config = EvolutionaryConfig(
            enabled=True,
            num_candidates=4,
            selection_method=selection_method,
            validation_config={"dummy": True},
        )
        return CandidateGenerator(config)

    def _make_candidates(self, fitness_scores):
        return [
            GradientCandidate(
                id=i,
                gradients={"w": torch.zeros(1)},
                description=f"candidate {i}",
                fitness=f,
            )
            for i, f in enumerate(fitness_scores)
        ]

    def test_best_selection(self):
        gen = self._make_generator("best")
        candidates = self._make_candidates([0.1, 0.9, 0.5, 0.3])
        best = gen.select_best(candidates)
        assert best.fitness == 0.9

    def test_no_fitness_raises(self):
        gen = self._make_generator("best")
        candidates = self._make_candidates([None, None])
        with pytest.raises(ValueError, match="No candidates have fitness"):
            gen.select_best(candidates)

    def test_tournament_selection_returns_valid(self):
        gen = self._make_generator("tournament")
        candidates = self._make_candidates([0.1, 0.9, 0.5, 0.3])
        best = gen.select_best(candidates)
        assert best.fitness is not None

    def test_proportional_selection_returns_valid(self):
        gen = self._make_generator("proportional")
        candidates = self._make_candidates([0.1, 0.9, 0.5, 0.3])
        best = gen.select_best(candidates)
        assert best.fitness is not None

    def test_proportional_zero_total_fitness(self):
        gen = self._make_generator("proportional")
        candidates = self._make_candidates([0.0, 0.0, 0.0])
        best = gen.select_best(candidates)
        assert best.fitness == 0.0

    def test_unknown_method_falls_back_to_best(self):
        gen = self._make_generator("nonexistent_method")
        candidates = self._make_candidates([0.1, 0.8, 0.3])
        best = gen.select_best(candidates)
        assert best.fitness == 0.8

    def test_single_candidate(self):
        gen = self._make_generator("best")
        candidates = self._make_candidates([0.5])
        assert gen.select_best(candidates).fitness == 0.5


class TestGradientHelpers:
    """Verify extract_gradients and apply_gradients work correctly."""

    def test_extract_gradients_from_model(self):
        model = torch.nn.Linear(3, 2)
        x = torch.randn(1, 3)
        loss = model(x).sum()
        loss.backward()

        grads = CandidateGenerator.extract_gradients(model)
        assert len(grads) > 0
        for name, grad in grads.items():
            assert isinstance(grad, torch.Tensor)

    def test_extract_gradients_skips_none(self):
        model = torch.nn.Linear(3, 2)
        # No backward call, so grads are None
        grads = CandidateGenerator.extract_gradients(model)
        assert grads == {}

    def test_apply_gradients_sets_grad(self):
        model = torch.nn.Linear(3, 2)
        fake_grads = {}
        for name, param in model.named_parameters():
            fake_grads[name] = torch.ones_like(param)

        CandidateGenerator.apply_gradients(model, fake_grads)
        for name, param in model.named_parameters():
            if name in fake_grads:
                assert torch.allclose(param.grad, fake_grads[name])

    def test_apply_gradients_overwrites_existing(self):
        model = torch.nn.Linear(3, 2)
        x = torch.randn(1, 3)
        loss = model(x).sum()
        loss.backward()

        new_grads = {}
        for name, param in model.named_parameters():
            new_grads[name] = torch.zeros_like(param)

        CandidateGenerator.apply_gradients(model, new_grads)
        for name, param in model.named_parameters():
            if name in new_grads:
                assert param.grad.abs().max().item() == 0.0
