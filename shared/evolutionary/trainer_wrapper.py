"""Evolutionary trainer wrapper for HuggingFace trainers."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
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
        events_path: Optional[str | Path] = None,
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
        self.events_path = Path(events_path) if events_path else None
        if self.events_path:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)

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
        self.current_micro_step = 0
        self.best_fitness_history: List[float] = []
        self.candidate_stats: List[Dict[str, Any]] = []
        self.selection_events = 0
        self.baseline_kept_count = 0
        self.last_selected_candidate: Optional[Dict[str, Any]] = None

        # Store original training step
        self._original_training_step = None

        # Eval batch rotation - iterator that cycles through batches
        self._eval_dataloader = None
        self._eval_batch_iterator = None

        # Validate config
        issues = config.validate()
        if issues:
            logger.warning(f"Evolutionary config issues: {issues}")

    def _emit_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Persist a structured evolutionary event for bucket-backed inspection."""
        if not self.events_path:
            return
        event_payload = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": self.current_step,
        }
        if payload:
            event_payload.update(payload)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, ensure_ascii=False) + "\n")

    def _gradient_accumulation_steps(self) -> int:
        """Return the configured gradient accumulation factor."""
        grad_acc = getattr(self.trainer.args, "gradient_accumulation_steps", 1)
        try:
            return max(1, int(grad_acc))
        except (TypeError, ValueError):
            return 1

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
        self._emit_event(
            "evolutionary_enabled",
            {
                "strategy": self.config.strategy,
                "selection_method": self.config.selection_method,
                "num_candidates": self.config.num_candidates,
                "eval_frequency": self.config.eval_frequency,
                "warmup_steps": self.config.warmup_steps,
            },
        )

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
        self.current_micro_step += 1
        grad_acc = self._gradient_accumulation_steps()
        optimizer_step = ((self.current_micro_step - 1) // grad_acc) + 1
        at_optimizer_boundary = (self.current_micro_step % grad_acc) == 0
        self.current_step = optimizer_step

        # Debug: Log first few steps and when evolutionary kicks in
        if (
            at_optimizer_boundary
            and (optimizer_step <= 5 or optimizer_step == self.config.warmup_steps + 1)
        ):
            print(
                f"[EVO DEBUG] Step {optimizer_step}, micro_step={self.current_micro_step}, "
                f"warmup_steps={self.config.warmup_steps}, in_warmup={optimizer_step <= self.config.warmup_steps}"
            )

        # Check if we're still in warmup phase
        if optimizer_step <= self.config.warmup_steps:
            if at_optimizer_boundary and optimizer_step == self.config.warmup_steps:
                logger.info(f"Warmup complete at step {optimizer_step}, enabling evolutionary selection")
                self._emit_event(
                    "warmup_complete",
                    {
                        "warmup_steps": self.config.warmup_steps,
                    },
                )
            return self._original_training_step(model, inputs, num_items_in_batch)

        if not at_optimizer_boundary:
            return self._original_training_step(model, inputs, num_items_in_batch)

        # Check if we should do evolutionary selection this step
        if optimizer_step % self.config.eval_frequency != 0:
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

        # Clip gradients to prevent evaluation instability
        if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
            base_gradients = CandidateGenerator.clip_gradients(
                base_gradients, self.config.max_grad_norm
            )

        # Diagnostic logging for gradient norms
        if self.config.log_candidates:
            grad_norms = {name: grad.norm().item() for name, grad in base_gradients.items()}
            avg_norm = sum(grad_norms.values()) / max(len(grad_norms), 1)
            logger.info(
                f"Step {self.current_step}: avg_grad_norm={avg_norm:.4f}, "
                f"max_grad_norm={max(grad_norms.values()):.4f}, "
                f"num_params={len(grad_norms)}"
            )

        # Step 3: Generate candidates
        candidates = self.candidate_generator.generate(
            base_gradients,
            step=optimizer_step,
        )

        # Step 4: Evaluate each candidate
        evaluated_candidates = self._evaluate_candidates(model, candidates)

        # Step 5: Select best
        best_candidate = self.candidate_generator.select_best(evaluated_candidates)

        # Check minimum improvement
        baseline_fitness = candidates[0].fitness if candidates else 0.0
        baseline_kept = False
        if (best_candidate.fitness - baseline_fitness) < self.config.min_fitness_improvement:
            # Use baseline gradient (no improvement found)
            best_candidate = candidates[0]
            baseline_kept = True
            self.baseline_kept_count += 1

        # Step 6: Apply best candidate's gradients
        CandidateGenerator.apply_gradients(model, best_candidate.gradients)

        # Logging
        if self.config.log_candidates:
            self._log_candidates(candidates)

        fitnesses = [c.fitness for c in candidates]
        candidate_payload = [
            {
                "id": c.id,
                "description": c.description,
                "fitness": c.fitness,
                "metadata": dict(c.metadata),
            }
            for c in candidates
        ]
        self.selection_events += 1
        self.last_selected_candidate = {
            "step": optimizer_step,
            "id": best_candidate.id,
            "description": best_candidate.description,
            "fitness": best_candidate.fitness,
            "used_baseline": baseline_kept,
        }
        self._emit_event(
            "candidate_selected",
            {
                "candidate_count": len(candidates),
                "baseline_fitness": baseline_fitness,
                "selected_candidate_id": best_candidate.id,
                "selected_description": best_candidate.description,
                "selected_fitness": best_candidate.fitness,
                "used_baseline": baseline_kept,
                "min_improvement": self.config.min_fitness_improvement,
                "candidate_fitnesses": fitnesses,
                "candidates": candidate_payload,
            },
        )

        if self.config.log_selected:
            logger.info(
                f"Step {optimizer_step}: Selected candidate {best_candidate.id} "
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
        Evaluate fitness of each candidate using validation loss.

        For each candidate:
        1. Temporarily apply its gradients
        2. Compute loss on held-out validation batch (rotated each step)
        3. Use inverse loss as fitness (lower loss = higher fitness)
        4. Restore ALL modified weights

        This is Unsloth-compatible because:
        - Only clones trainable params (mostly LoRA, small)
        - Uses forward pass only, no generate()
        - Works with Unsloth's optimized training
        """
        # Get a ROTATED validation batch (different each step)
        eval_batch = self._get_eval_batch()
        if eval_batch is None:
            # Fallback to gradient-norm fitness if no eval batch
            return self._evaluate_candidates_by_gradient_norm(candidates)

        # Get list of param names we'll modify (from first candidate)
        # All candidates have same param names, just different values
        params_to_modify = set(candidates[0].gradients.keys()) if candidates else set()

        # Clone ALL params we're going to modify (not just LoRA)
        # This fixes the restore bug
        saved_state = {}
        for name, param in model.named_parameters():
            if name in params_to_modify:
                saved_state[name] = param.data.clone()

        lr = self.trainer.args.learning_rate

        for candidate in candidates:
            try:
                # Apply candidate gradients to weights temporarily
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in candidate.gradients:
                            # SGD-like update
                            param.data.sub_(lr * candidate.gradients[name])

                # Compute validation loss
                model.eval()
                with torch.no_grad():
                    outputs = model(**eval_batch)
                    val_loss = outputs.loss.item() if hasattr(outputs, 'loss') else outputs[0].item()

                # Fitness = inverse of loss (lower loss = higher fitness)
                # Normalize to reasonable range
                candidate.fitness = 1.0 / (1.0 + val_loss)
                candidate.metadata["val_loss"] = val_loss

            finally:
                # Restore ALL modified weights (not just LoRA)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in saved_state:
                            param.data.copy_(saved_state[name])

        model.train()
        return candidates

    def _evaluate_candidates_by_gradient_norm(
        self,
        candidates: List[GradientCandidate],
    ) -> List[GradientCandidate]:
        """Fallback: evaluate by gradient norm (fast, no forward pass)."""
        for candidate in candidates:
            total_norm = 0.0
            num_params = 0
            for name, grad in candidate.gradients.items():
                total_norm += grad.norm().item() ** 2
                num_params += 1
            rms_norm = (total_norm / max(num_params, 1)) ** 0.5
            candidate.fitness = 1.0 / (1.0 + rms_norm)
            candidate.metadata["rms_grad_norm"] = rms_norm
        return candidates

    def _get_eval_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a ROTATED batch for validation loss evaluation.

        Each call returns a different batch, cycling through the dataloader.
        This prevents overfitting to a single evaluation batch.
        """
        try:
            # Initialize dataloader and iterator if needed
            if self._eval_dataloader is None:
                if hasattr(self.trainer, 'get_eval_dataloader') and self.trainer.eval_dataset is not None:
                    self._eval_dataloader = self.trainer.get_eval_dataloader()
                elif hasattr(self.trainer, 'get_train_dataloader'):
                    self._eval_dataloader = self.trainer.get_train_dataloader()
                else:
                    return None
                self._eval_batch_iterator = iter(self._eval_dataloader)

            # Get next batch, cycling back to start if exhausted
            try:
                batch = next(self._eval_batch_iterator)
            except StopIteration:
                # Reset iterator and get first batch
                self._eval_batch_iterator = iter(self._eval_dataloader)
                batch = next(self._eval_batch_iterator)

            # Move to device
            device = next(self.trainer.model.parameters()).device
            return {k: v.to(device) if hasattr(v, 'to') else v
                    for k, v in batch.items()}

        except Exception as e:
            logger.warning(f"Could not get eval batch: {e}")
            return None

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
            "total_micro_steps": self.current_micro_step,
            "best_fitness_history": self.best_fitness_history,
            "candidate_stats": self.candidate_stats,
            "selection_events": self.selection_events,
            "baseline_kept_count": self.baseline_kept_count,
            "last_selected_candidate": self.last_selected_candidate,
            "events_path": str(self.events_path) if self.events_path else None,
            "config": self.config.to_dict(),
        }
