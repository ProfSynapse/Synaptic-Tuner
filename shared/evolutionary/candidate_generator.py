"""Candidate generator for evolutionary training."""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import torch

from .config import EvolutionaryConfig
from .strategies import get_strategy, GradientCandidate, BaseStrategy

if TYPE_CHECKING:
    from .strategies.base import BaseStrategy


class CandidateGenerator:
    """
    Generates candidate gradient updates for evolutionary selection.

    This class orchestrates the generation of multiple candidate
    gradient modifications using configurable strategies.

    Usage:
        generator = CandidateGenerator(config)

        # After computing gradients
        base_grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        candidates = generator.generate(base_grads, step=100)

        # Evaluate each candidate and pick the best
        for candidate in candidates:
            # Apply candidate.gradients temporarily and evaluate
            ...
    """

    def __init__(
        self,
        config: EvolutionaryConfig,
        strategy: Optional[BaseStrategy] = None,
    ):
        """
        Initialize candidate generator.

        Args:
            config: Evolutionary training config
            strategy: Optional custom strategy (overrides config.strategy)
        """
        self.config = config

        if strategy is not None:
            self.strategy = strategy
        else:
            # Create strategy from config
            self.strategy = get_strategy(
                name=config.strategy,
                noise_scale=config.noise_scale,
                max_grad_norm=config.max_grad_norm if config.max_grad_norm else 1.0,
                scale_factors=config.scale_factors,
            )

    def generate(
        self,
        base_gradients: Dict[str, torch.Tensor],
        step: int = 0,
    ) -> List[GradientCandidate]:
        """
        Generate candidate gradient updates.

        Args:
            base_gradients: Base gradients from backpropagation
            step: Current training step

        Returns:
            List of GradientCandidate objects
        """
        candidates = self.strategy.generate_candidates(
            base_gradients=base_gradients,
            num_candidates=self.config.num_candidates,
            step=step,
        )

        return candidates

    def select_best(self, candidates: List[GradientCandidate]) -> GradientCandidate:
        """
        Select the best candidate based on fitness scores.

        Args:
            candidates: List of evaluated candidates (with fitness set)

        Returns:
            The best candidate

        Raises:
            ValueError: If no candidates have fitness scores
        """
        # Filter candidates with fitness scores
        evaluated = [c for c in candidates if c.fitness is not None]

        if not evaluated:
            raise ValueError("No candidates have fitness scores. Call evaluate first.")

        if self.config.selection_method == "best":
            # Simple: pick highest fitness
            return max(evaluated, key=lambda c: c.fitness)

        elif self.config.selection_method == "tournament":
            # Tournament selection (useful for diversity)
            import random
            # Pick 2 random candidates, return the better one
            if len(evaluated) >= 2:
                a, b = random.sample(evaluated, 2)
                return a if a.fitness >= b.fitness else b
            return evaluated[0]

        elif self.config.selection_method == "proportional":
            # Probability proportional to fitness
            import random
            total_fitness = sum(c.fitness for c in evaluated)
            if total_fitness <= 0:
                return random.choice(evaluated)
            r = random.uniform(0, total_fitness)
            cumsum = 0
            for candidate in evaluated:
                cumsum += candidate.fitness
                if cumsum >= r:
                    return candidate
            return evaluated[-1]

        else:
            # Default to best
            return max(evaluated, key=lambda c: c.fitness)

    @staticmethod
    def clip_gradients(
        gradients: Dict[str, torch.Tensor],
        max_norm: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Clip per-parameter gradient norms before candidate generation.

        This prevents evaluation instability when early-training gradient norms
        are large (e.g., 100+). Each parameter's gradient is independently
        clipped to have at most ``max_norm`` L2 norm while preserving direction.

        Args:
            gradients: Dict mapping parameter names to gradient tensors
            max_norm: Maximum allowed L2 norm per parameter gradient

        Returns:
            Dict of clipped gradients (same keys, clipped values)
        """
        clipped = {}
        for name, grad in gradients.items():
            norm = grad.norm()
            if norm > max_norm:
                clipped[name] = grad * (max_norm / norm)
            else:
                clipped[name] = grad
        return clipped

    @staticmethod
    def extract_gradients(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Extract current gradients from model parameters.

        Args:
            model: PyTorch model with computed gradients

        Returns:
            Dict mapping parameter names to gradient tensors
        """
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients

    @staticmethod
    def apply_gradients(
        model: torch.nn.Module,
        gradients: Dict[str, torch.Tensor],
    ) -> None:
        """
        Apply gradients to model parameters.

        This replaces the current gradients with the provided ones.

        Args:
            model: PyTorch model
            gradients: Gradients to apply
        """
        for name, param in model.named_parameters():
            if name in gradients:
                if param.grad is None:
                    param.grad = gradients[name].clone()
                else:
                    param.grad.copy_(gradients[name])
