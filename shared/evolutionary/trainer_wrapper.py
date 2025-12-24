"""Evolutionary trainer wrapper for HuggingFace trainers."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager

import torch
from torch import nn

from shared.validation.fitness import FitnessEvaluator, FitnessResult
from .config import EvolutionaryConfig
from .candidate_generator import CandidateGenerator
from .strategies import GradientCandidate


logger = logging.getLogger(__name__)


class EvolutionaryTrainerWrapper:
    """
    Wraps a HuggingFace-style trainer with evolutionary gradient selection.

    This wrapper intercepts the training step to:
    1. Compute the standard gradient
    2. Generate multiple candidate modifications
    3. Evaluate each candidate using the fitness evaluator
    4. Apply only the best candidate

    Usage:
        from transformers import Trainer
        from shared.evolutionary import EvolutionaryTrainerWrapper, EvolutionaryConfig

        # Create your standard trainer
        trainer = Trainer(model=model, args=args, ...)

        # Wrap with evolutionary selection
        evo_config = EvolutionaryConfig(enabled=True, ...)
        evo_wrapper = EvolutionaryTrainerWrapper(
            trainer=trainer,
            config=evo_config,
            tokenizer=tokenizer,
        )

        # Train (this modifies the trainer's training step)
        evo_wrapper.train()
    """

    def __init__(
        self,
        trainer,  # HuggingFace Trainer or compatible
        config: EvolutionaryConfig,
        tokenizer,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        eval_prompts: Optional[List[str]] = None,
    ):
        """
        Initialize evolutionary wrapper.

        Args:
            trainer: The base trainer to wrap (e.g., SFTTrainer)
            config: Evolutionary training configuration
            tokenizer: Tokenizer for generating responses
            fitness_evaluator: Custom fitness evaluator (or created from config)
            eval_prompts: Prompts to use for fitness evaluation.
                         If None, uses samples from training data.
        """
        self.trainer = trainer
        self.config = config
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts or []

        # Create fitness evaluator
        if fitness_evaluator is not None:
            self.fitness_evaluator = fitness_evaluator
        elif config.validation_config_path:
            self.fitness_evaluator = FitnessEvaluator(
                config_path=config.validation_config_path
            )
        elif config.validation_config:
            self.fitness_evaluator = FitnessEvaluator(
                config=config.validation_config
            )
        else:
            # Default minimal evaluator
            self.fitness_evaluator = FitnessEvaluator()

        # Create candidate generator
        self.candidate_generator = CandidateGenerator(config)

        # State tracking
        self.current_step = 0
        self.best_fitness_history: List[float] = []
        self.candidate_stats: List[Dict[str, Any]] = []

        # Store original training step
        self._original_training_step = None

        # Validate config
        issues = config.validate()
        if issues:
            logger.warning(f"Evolutionary config issues: {issues}")

    def train(self, *args, **kwargs):
        """
        Train with evolutionary gradient selection.

        This wraps the trainer's train method to intercept training steps.
        """
        if not self.config.enabled:
            logger.info("Evolutionary training disabled, using standard training")
            return self.trainer.train(*args, **kwargs)

        logger.info(f"Starting evolutionary training with {self.config.num_candidates} candidates")
        logger.info(f"Strategy: {self.config.strategy}, Selection: {self.config.selection_method}")
        if self.config.warmup_steps > 0:
            logger.info(f"Warmup: {self.config.warmup_steps} steps of standard training first")

        # Install the evolutionary training step
        self._install_evolutionary_step()

        try:
            return self.trainer.train(*args, **kwargs)
        finally:
            # Restore original training step
            self._uninstall_evolutionary_step()

    def _install_evolutionary_step(self):
        """Replace trainer's training step with evolutionary version."""
        self._original_training_step = self.trainer.training_step

        def evolutionary_training_step(model, inputs, num_items_in_batch=None):
            return self._evolutionary_step(model, inputs, num_items_in_batch)

        self.trainer.training_step = evolutionary_training_step

    def _uninstall_evolutionary_step(self):
        """Restore original training step."""
        if self._original_training_step is not None:
            self.trainer.training_step = self._original_training_step

    def _evolutionary_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Execute one evolutionary training step.

        1. Forward + backward to get base gradient
        2. Generate candidate modifications
        3. Evaluate each candidate
        4. Apply best candidate's gradient
        """
        self.current_step += 1

        # Check if we're still in warmup phase
        if self.current_step <= self.config.warmup_steps:
            if self.current_step == self.config.warmup_steps:
                logger.info(f"Warmup complete at step {self.current_step}, enabling evolutionary selection")
            return self._original_training_step(model, inputs, num_items_in_batch)

        # Check if we should do evolutionary selection this step
        if self.current_step % self.config.eval_frequency != 0:
            # Just do normal training step
            return self._original_training_step(model, inputs, num_items_in_batch)

        # Step 1: Standard forward + backward
        model.train()

        # Use the trainer's compute_loss method
        with self.trainer.compute_loss_context_manager():
            loss = self.trainer.compute_loss(model, inputs)

        if self.trainer.args.gradient_accumulation_steps > 1:
            loss = loss / self.trainer.args.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Step 2: Extract base gradients
        base_gradients = CandidateGenerator.extract_gradients(model)

        if not base_gradients:
            logger.warning("No gradients computed, skipping evolutionary selection")
            return loss.detach()

        # Step 3: Generate candidates
        candidates = self.candidate_generator.generate(
            base_gradients,
            step=self.current_step,
        )

        # Step 4: Evaluate each candidate
        evaluated_candidates = self._evaluate_candidates(model, candidates)

        # Step 5: Select best
        best_candidate = self.candidate_generator.select_best(evaluated_candidates)

        # Check minimum improvement
        baseline_fitness = candidates[0].fitness if candidates else 0.0
        if (best_candidate.fitness - baseline_fitness) < self.config.min_fitness_improvement:
            # Use baseline gradient (no improvement found)
            best_candidate = candidates[0]

        # Step 6: Apply best candidate's gradients
        CandidateGenerator.apply_gradients(model, best_candidate.gradients)

        # Logging
        if self.config.log_candidates:
            self._log_candidates(candidates)

        if self.config.log_selected:
            logger.info(
                f"Step {self.current_step}: Selected candidate {best_candidate.id} "
                f"({best_candidate.description}) with fitness {best_candidate.fitness:.4f}"
            )

        self.best_fitness_history.append(best_candidate.fitness)

        return loss.detach()

    def _evaluate_candidates(
        self,
        model: nn.Module,
        candidates: List[GradientCandidate],
    ) -> List[GradientCandidate]:
        """
        Evaluate fitness of each candidate.

        For each candidate:
        1. Temporarily apply its gradients
        2. Take an optimizer step
        3. Generate responses
        4. Evaluate fitness
        5. Revert to original weights
        """
        model.eval()

        # Save original model state
        original_state = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        eval_prompts = self._get_eval_prompts()

        for candidate in candidates:
            try:
                # Apply candidate gradients temporarily
                CandidateGenerator.apply_gradients(model, candidate.gradients)

                # Simulate optimizer step (apply gradients to weights)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            # Simple SGD-like update for evaluation
                            lr = self.trainer.args.learning_rate
                            param.data.sub_(lr * param.grad)

                # Generate responses and evaluate fitness
                fitness_scores = []
                for prompt in eval_prompts[:self.config.eval_batch_size]:
                    response = self._generate_response(model, prompt)
                    result = self.fitness_evaluator.evaluate(response)
                    fitness_scores.append(result.score)

                # Average fitness
                candidate.fitness = sum(fitness_scores) / max(len(fitness_scores), 1)

            finally:
                # Restore original weights
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in original_state:
                            param.data.copy_(original_state[name])

        model.train()
        return candidates

    def _generate_response(
        self,
        model: nn.Module,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a response from the model."""
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def _get_eval_prompts(self) -> List[str]:
        """Get prompts for fitness evaluation."""
        if self.eval_prompts:
            return self.eval_prompts

        # Try to extract from training data
        # This is a simplified version - in practice you'd want
        # a dedicated eval set
        try:
            dataset = self.trainer.train_dataset
            prompts = []
            for i in range(min(self.config.eval_batch_size * 2, len(dataset))):
                item = dataset[i]
                if 'input_ids' in item:
                    prompt = self.tokenizer.decode(item['input_ids'][:256])
                    prompts.append(prompt)
                elif 'text' in item:
                    prompts.append(item['text'][:512])
            return prompts[:self.config.eval_batch_size]
        except Exception as e:
            logger.warning(f"Could not extract eval prompts: {e}")
            return []

    def _log_candidates(self, candidates: List[GradientCandidate]):
        """Log candidate fitness scores."""
        stats = {
            "step": self.current_step,
            "candidates": [
                {
                    "id": c.id,
                    "description": c.description,
                    "fitness": c.fitness,
                }
                for c in candidates
            ],
        }
        self.candidate_stats.append(stats)

        # Log summary
        fitnesses = [c.fitness for c in candidates if c.fitness is not None]
        if fitnesses:
            logger.debug(
                f"Step {self.current_step}: Candidate fitnesses: "
                f"min={min(fitnesses):.4f}, max={max(fitnesses):.4f}, "
                f"mean={sum(fitnesses)/len(fitnesses):.4f}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_steps": self.current_step,
            "best_fitness_history": self.best_fitness_history,
            "candidate_stats": self.candidate_stats,
            "config": self.config.to_dict(),
        }
