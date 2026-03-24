"""Tests for EvolutionaryTrainerWrapper — init, train flow, evaluation, and state tracking."""

import json
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import pytest
import torch

from shared.evolutionary.config import EvolutionaryConfig
from shared.evolutionary.strategies.base import GradientCandidate


# ---------------------------------------------------------------------------
# Helpers: lightweight fakes for HuggingFace Trainer + tokenizer
# ---------------------------------------------------------------------------


@dataclass
class FakeTrainerArgs:
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.001


class FakeTrainer:
    """Minimal stand-in for HuggingFace Trainer."""

    def __init__(self, model=None, eval_dataset=None):
        self.model = model or _make_simple_model()
        self.args = FakeTrainerArgs()
        self.eval_dataset = eval_dataset
        self._training_step_called = 0

    def train(self, *args, **kwargs):
        """Simulate a training loop of 1 step."""
        if hasattr(self, "training_step") and callable(self.training_step):
            dummy_inputs = {"input_ids": torch.randint(0, 100, (1, 4))}
            self.training_step(self.model, dummy_inputs)
        return "train_result"

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Default training step (will be replaced by wrapper)."""
        self._training_step_called += 1
        x = torch.randn(1, 3)
        loss = model(x).sum()
        loss.backward()
        return loss.detach()

    def compute_loss_context_manager(self):
        import contextlib
        return contextlib.nullcontext()

    def compute_loss(self, model, inputs):
        x = torch.randn(1, 3)
        return model(x).sum()

    def get_train_dataloader(self):
        batch = {"input_ids": torch.randint(0, 100, (2, 4))}
        return [batch]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tokens, **kwargs):
        return "decoded text"


def _make_simple_model():
    """Create a minimal trainable model."""
    return torch.nn.Linear(3, 2)


def _make_config(**overrides):
    defaults = dict(
        enabled=True,
        num_candidates=3,
        strategy="gradient_noise",
        noise_scale=0.1,
        max_grad_norm=1.0,
        warmup_steps=0,
        eval_frequency=1,
        validation_config={"dummy": True},
        log_candidates=False,
        log_selected=False,
    )
    defaults.update(overrides)
    return EvolutionaryConfig(**defaults)


def _make_wrapper(**overrides):
    config = _make_config(**overrides)
    trainer = FakeTrainer()
    tokenizer = FakeTokenizer()
    from shared.evolutionary.trainer_wrapper import EvolutionaryTrainerWrapper
    return EvolutionaryTrainerWrapper(
        trainer=trainer, config=config, tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestEvolutionaryTrainerWrapperInit:

    def test_init_with_custom_fitness_evaluator(self):
        """Custom evaluator is used instead of creating one from config."""
        from shared.evolutionary.trainer_wrapper import EvolutionaryTrainerWrapper
        custom_evaluator = MagicMock()
        config = _make_config()
        wrapper = EvolutionaryTrainerWrapper(
            trainer=FakeTrainer(),
            config=config,
            tokenizer=FakeTokenizer(),
            fitness_evaluator=custom_evaluator,
        )
        assert wrapper.fitness_evaluator is custom_evaluator

    def test_init_with_validation_config_path(self, tmp_path):
        """Config path creates FitnessEvaluator from path."""
        from shared.evolutionary.trainer_wrapper import EvolutionaryTrainerWrapper
        config_file = tmp_path / "fitness.yaml"
        config_file.write_text("validations: []\n", encoding="utf-8")
        config = _make_config(
            validation_config=None,
            validation_config_path=str(config_file),
        )
        wrapper = EvolutionaryTrainerWrapper(
            trainer=FakeTrainer(), config=config, tokenizer=FakeTokenizer(),
        )
        assert wrapper.fitness_evaluator is not None

    def test_init_with_inline_validation_config(self):
        """Inline config dict creates FitnessEvaluator."""
        from shared.evolutionary.trainer_wrapper import EvolutionaryTrainerWrapper
        config = _make_config(validation_config={"validations": []})
        wrapper = EvolutionaryTrainerWrapper(
            trainer=FakeTrainer(), config=config, tokenizer=FakeTokenizer(),
        )
        assert wrapper.fitness_evaluator is not None

    def test_init_default_evaluator(self):
        """No config path or dict falls back to default evaluator."""
        from shared.evolutionary.trainer_wrapper import EvolutionaryTrainerWrapper
        config = _make_config(validation_config=None, validation_config_path=None)
        wrapper = EvolutionaryTrainerWrapper(
            trainer=FakeTrainer(), config=config, tokenizer=FakeTokenizer(),
        )
        assert wrapper.fitness_evaluator is not None

    def test_init_state_tracking(self):
        """Initial state tracking is zeroed out."""
        wrapper = _make_wrapper()
        assert wrapper.current_step == 0
        assert wrapper.current_micro_step == 0
        assert wrapper.best_fitness_history == []
        assert wrapper.candidate_stats == []

    def test_init_stores_eval_prompts(self):
        """Custom eval prompts are stored."""
        from shared.evolutionary.trainer_wrapper import EvolutionaryTrainerWrapper
        prompts = ["prompt 1", "prompt 2"]
        wrapper = EvolutionaryTrainerWrapper(
            trainer=FakeTrainer(),
            config=_make_config(),
            tokenizer=FakeTokenizer(),
            eval_prompts=prompts,
        )
        assert wrapper.eval_prompts == prompts

    def test_init_stores_events_path(self, tmp_path):
        events_path = tmp_path / "logs" / "evolutionary_events.jsonl"
        from shared.evolutionary.trainer_wrapper import EvolutionaryTrainerWrapper
        wrapper = EvolutionaryTrainerWrapper(
            trainer=FakeTrainer(),
            config=_make_config(),
            tokenizer=FakeTokenizer(),
            events_path=events_path,
        )
        assert wrapper.events_path == events_path
        assert events_path.parent.exists()

    def test_init_none_eval_prompts_defaults_to_empty(self):
        wrapper = _make_wrapper()
        assert wrapper.eval_prompts == []

    def test_config_validation_warnings_logged(self):
        """Config issues are logged as warnings during init."""
        from shared.evolutionary.trainer_wrapper import EvolutionaryTrainerWrapper
        # num_candidates=1 with enabled=True triggers a validation issue
        config = _make_config(num_candidates=1, validation_config={"dummy": True})
        with patch("shared.evolutionary.trainer_wrapper.logger") as mock_logger:
            EvolutionaryTrainerWrapper(
                trainer=FakeTrainer(), config=config, tokenizer=FakeTokenizer(),
            )
            mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# Train flow tests
# ---------------------------------------------------------------------------


class TestTrainFlow:

    def test_disabled_config_uses_standard_training(self):
        """When config.enabled is False, train delegates directly to trainer."""
        wrapper = _make_wrapper(enabled=False)
        result = wrapper.train()
        assert result == "train_result"

    def test_enabled_config_installs_evolutionary_step(self):
        """When enabled, train() replaces training_step then restores it."""
        wrapper = _make_wrapper()

        # Use a sentinel function so identity checks work on plain functions
        sentinel = lambda model, inputs, num_items_in_batch=None: None
        wrapper.trainer.training_step = sentinel

        # Stub trainer.train to avoid full pipeline (model(**eval_batch) fails
        # because our simple Linear model doesn't accept input_ids kwargs).
        # Instead, verify the mechanism: install replaces, then finally restores.
        train_called = [False]
        def fake_train(*a, **kw):
            # During train, the step should NOT be the sentinel
            train_called[0] = True
            assert wrapper.trainer.training_step is not sentinel
            return "ok"

        wrapper.trainer.train = fake_train
        wrapper.train()

        # After train() completes, original step should be restored
        assert train_called[0]
        assert wrapper.trainer.training_step is sentinel

    def test_install_and_uninstall_step(self):
        """_install replaces training_step; _uninstall restores it."""
        wrapper = _make_wrapper()
        sentinel = lambda model, inputs, num_items_in_batch=None: None
        wrapper.trainer.training_step = sentinel

        wrapper._install_evolutionary_step()
        assert wrapper.trainer.training_step is not sentinel
        assert wrapper._original_training_step is sentinel

        wrapper._uninstall_evolutionary_step()
        assert wrapper.trainer.training_step is sentinel

    def test_uninstall_noop_when_not_installed(self):
        """_uninstall does nothing if _original_training_step is None."""
        wrapper = _make_wrapper()
        sentinel = lambda model, inputs, num_items_in_batch=None: None
        wrapper.trainer.training_step = sentinel
        wrapper._uninstall_evolutionary_step()
        # _original_training_step is None, so training_step unchanged
        assert wrapper.trainer.training_step is sentinel

    def test_step_restored_on_training_error(self):
        """If training raises, the original step is still restored."""
        wrapper = _make_wrapper()
        sentinel = lambda model, inputs, num_items_in_batch=None: None
        wrapper.trainer.training_step = sentinel

        # Make trainer.train raise after step is installed
        def exploding_train(*a, **kw):
            raise RuntimeError("boom")

        wrapper.trainer.train = exploding_train

        with pytest.raises(RuntimeError, match="boom"):
            wrapper.train()

        # Original step should be restored via finally block
        assert wrapper.trainer.training_step is sentinel


# ---------------------------------------------------------------------------
# Evolutionary step logic tests
# ---------------------------------------------------------------------------


class TestEvolutionaryStep:

    def test_warmup_uses_original_step(self):
        """During warmup, original training step is used."""
        wrapper = _make_wrapper(warmup_steps=5)
        original_called = [0]
        real_original = wrapper.trainer.training_step

        def counting_step(model, inputs, num_items_in_batch=None):
            original_called[0] += 1
            return real_original(model, inputs, num_items_in_batch)

        wrapper._original_training_step = counting_step

        model = wrapper.trainer.model
        inputs = {"input_ids": torch.randint(0, 100, (1, 4))}

        # Steps 1-5 should use original
        for _ in range(5):
            wrapper._evolutionary_step(model, inputs)

        assert original_called[0] == 5
        assert wrapper.current_step == 5
        assert wrapper.current_micro_step == 5

    def test_eval_frequency_skips_evolutionary(self):
        """Non-evolutionary steps occur on non-matching eval_frequency."""
        wrapper = _make_wrapper(eval_frequency=3, warmup_steps=0)
        original_called = [0]
        real_original = wrapper.trainer.training_step

        def counting_step(model, inputs, num_items_in_batch=None):
            original_called[0] += 1
            return real_original(model, inputs, num_items_in_batch)

        wrapper._original_training_step = counting_step

        model = wrapper.trainer.model
        inputs = {"input_ids": torch.randint(0, 100, (1, 4))}

        # Steps 1, 2 skip evolutionary (not divisible by 3)
        wrapper._evolutionary_step(model, inputs)  # step 1 → original
        wrapper._evolutionary_step(model, inputs)  # step 2 → original
        assert original_called[0] == 2

    def test_step_increments_counter(self):
        """Each call increments current_step."""
        wrapper = _make_wrapper(warmup_steps=100)  # force warmup path for simplicity
        wrapper._original_training_step = wrapper.trainer.training_step
        model = wrapper.trainer.model
        inputs = {"input_ids": torch.randint(0, 100, (1, 4))}

        wrapper._evolutionary_step(model, inputs)
        wrapper._evolutionary_step(model, inputs)
        assert wrapper.current_step == 2
        assert wrapper.current_micro_step == 2

    def test_gradient_accumulation_uses_optimizer_step_semantics(self):
        """With grad accumulation, current_step tracks optimizer steps, not microsteps."""
        wrapper = _make_wrapper(warmup_steps=100)
        wrapper.trainer.args.gradient_accumulation_steps = 2
        wrapper._original_training_step = wrapper.trainer.training_step
        model = wrapper.trainer.model
        inputs = {"input_ids": torch.randint(0, 100, (1, 4))}

        wrapper._evolutionary_step(model, inputs)
        assert wrapper.current_micro_step == 1
        assert wrapper.current_step == 1

        wrapper._evolutionary_step(model, inputs)
        assert wrapper.current_micro_step == 2
        assert wrapper.current_step == 1

        wrapper._evolutionary_step(model, inputs)
        assert wrapper.current_micro_step == 3
        assert wrapper.current_step == 2

    def test_no_gradients_returns_loss(self):
        """When no gradients are computed, evolutionary selection is skipped."""
        wrapper = _make_wrapper()
        wrapper._original_training_step = wrapper.trainer.training_step

        model = _make_simple_model()
        inputs = {"input_ids": torch.randint(0, 100, (1, 4))}

        # Mock compute_loss to return a differentiable scalar, then
        # mock extract_gradients to return empty dict (simulating no grads)
        fake_loss = torch.tensor(1.0, requires_grad=True)
        wrapper.trainer.compute_loss = lambda m, i: fake_loss

        with patch(
            "shared.evolutionary.candidate_generator.CandidateGenerator.extract_gradients",
            return_value={},
        ):
            result = wrapper._evolutionary_step(model, inputs)

        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# Candidate evaluation tests
# ---------------------------------------------------------------------------


class TestCandidateEvaluation:

    def test_evaluate_by_gradient_norm(self):
        """Gradient-norm fallback assigns fitness based on RMS norm."""
        wrapper = _make_wrapper()
        candidates = [
            GradientCandidate(
                id=0,
                gradients={"w": torch.ones(5) * 0.1},
                description="small",
            ),
            GradientCandidate(
                id=1,
                gradients={"w": torch.ones(5) * 10.0},
                description="large",
            ),
        ]

        result = wrapper._evaluate_candidates_by_gradient_norm(candidates)

        assert len(result) == 2
        # Smaller gradient norm → higher fitness (closer to 1)
        assert result[0].fitness > result[1].fitness
        assert "rms_grad_norm" in result[0].metadata
        assert "rms_grad_norm" in result[1].metadata

    def test_evaluate_by_gradient_norm_sets_all_fitness(self):
        """All candidates get a fitness score assigned."""
        wrapper = _make_wrapper()
        candidates = [
            GradientCandidate(
                id=i,
                gradients={"w": torch.randn(3)},
                description=f"c{i}",
            )
            for i in range(4)
        ]
        result = wrapper._evaluate_candidates_by_gradient_norm(candidates)
        for c in result:
            assert c.fitness is not None
            assert 0.0 < c.fitness <= 1.0

    def test_evaluate_candidates_falls_back_to_norm(self):
        """When _get_eval_batch returns None, falls back to gradient norm."""
        wrapper = _make_wrapper()
        model = _make_simple_model()

        candidates = [
            GradientCandidate(
                id=0,
                gradients={"weight": torch.ones(2, 3) * 0.1, "bias": torch.ones(2) * 0.1},
                description="c0",
            ),
        ]

        with patch.object(wrapper, "_get_eval_batch", return_value=None):
            result = wrapper._evaluate_candidates(model, candidates)

        assert result[0].fitness is not None
        assert "rms_grad_norm" in result[0].metadata


# ---------------------------------------------------------------------------
# Eval batch rotation tests
# ---------------------------------------------------------------------------


class TestEvalBatchRotation:

    def test_get_eval_batch_from_train_dataloader(self):
        """Falls back to train_dataloader when no eval_dataset."""
        wrapper = _make_wrapper()
        batch = wrapper._get_eval_batch()
        assert batch is not None
        assert "input_ids" in batch

    def test_get_eval_batch_cycles(self):
        """After exhausting batches, cycles back to start."""
        wrapper = _make_wrapper()
        # First call initializes and gets first batch
        b1 = wrapper._get_eval_batch()
        assert b1 is not None

        # Second call triggers StopIteration → reset → first batch again
        b2 = wrapper._get_eval_batch()
        assert b2 is not None

    def test_get_eval_batch_returns_none_on_error(self):
        """Returns None when dataloader raises an error."""
        wrapper = _make_wrapper()
        wrapper.trainer.eval_dataset = None

        # Patch so both dataloader accessors raise
        with patch.object(
            wrapper.trainer, "get_train_dataloader",
            side_effect=RuntimeError("no dataloader"),
        ):
            batch = wrapper._get_eval_batch()
        assert batch is None


# ---------------------------------------------------------------------------
# Stats and logging tests
# ---------------------------------------------------------------------------


class TestStatsAndLogging:

    def test_get_stats_returns_structure(self):
        """get_stats returns expected keys."""
        wrapper = _make_wrapper()
        stats = wrapper.get_stats()
        assert "total_steps" in stats
        assert "best_fitness_history" in stats
        assert "candidate_stats" in stats
        assert "config" in stats
        assert stats["total_steps"] == 0
        assert stats["total_micro_steps"] == 0

    def test_log_candidates_records_stats(self):
        """_log_candidates appends to candidate_stats."""
        wrapper = _make_wrapper()
        wrapper.current_step = 42
        candidates = [
            GradientCandidate(id=0, gradients={}, description="c0", fitness=0.5),
            GradientCandidate(id=1, gradients={}, description="c1", fitness=0.8),
        ]
        wrapper._log_candidates(candidates)

        assert len(wrapper.candidate_stats) == 1
        assert wrapper.candidate_stats[0]["step"] == 42
        assert len(wrapper.candidate_stats[0]["candidates"]) == 2
        assert wrapper.candidate_stats[0]["candidates"][1]["fitness"] == 0.8

    def test_log_candidates_handles_none_fitness(self):
        """_log_candidates does not crash when fitness is None."""
        wrapper = _make_wrapper()
        wrapper.current_step = 1
        candidates = [
            GradientCandidate(id=0, gradients={}, description="c0", fitness=None),
        ]
        # Should not raise
        wrapper._log_candidates(candidates)
        assert len(wrapper.candidate_stats) == 1

    def test_emit_event_writes_jsonl(self, tmp_path):
        wrapper = _make_wrapper()
        wrapper.events_path = tmp_path / "evolutionary_events.jsonl"
        wrapper.current_step = 7

        wrapper._emit_event("candidate_selected", {"selected_candidate_id": 2, "selected_fitness": 0.9})

        lines = wrapper.events_path.read_text(encoding="utf-8").strip().splitlines()
        payload = json.loads(lines[-1])
        assert payload["event"] == "candidate_selected"
        assert payload["step"] == 7
        assert payload["selected_candidate_id"] == 2
        assert payload["selected_fitness"] == 0.9

    def test_get_eval_prompts_returns_custom(self):
        """When eval_prompts provided, returns them directly."""
        from shared.evolutionary.trainer_wrapper import EvolutionaryTrainerWrapper
        prompts = ["p1", "p2", "p3"]
        wrapper = EvolutionaryTrainerWrapper(
            trainer=FakeTrainer(),
            config=_make_config(),
            tokenizer=FakeTokenizer(),
            eval_prompts=prompts,
        )
        assert wrapper._get_eval_prompts() == prompts

    def test_get_eval_prompts_empty_when_no_data(self):
        """Returns empty list when no prompts or usable training data."""
        wrapper = _make_wrapper()
        wrapper.trainer.train_dataset = None
        result = wrapper._get_eval_prompts()
        assert result == []

    def test_get_stats_after_steps(self):
        """Stats reflect state after evolutionary steps."""
        wrapper = _make_wrapper()
        wrapper.current_step = 10
        wrapper.current_micro_step = 20
        wrapper.best_fitness_history = [0.5, 0.6, 0.7]
        wrapper.selection_events = 2
        wrapper.baseline_kept_count = 1
        wrapper.last_selected_candidate = {"step": 10, "id": 3}
        stats = wrapper.get_stats()
        assert stats["total_steps"] == 10
        assert stats["total_micro_steps"] == 20
        assert len(stats["best_fitness_history"]) == 3
        assert stats["selection_events"] == 2
        assert stats["baseline_kept_count"] == 1
        assert stats["last_selected_candidate"]["id"] == 3
