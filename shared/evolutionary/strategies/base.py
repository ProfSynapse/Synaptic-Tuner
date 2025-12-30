"""Base class for gradient modification strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch


@dataclass
class GradientCandidate:
    """
    A candidate gradient update.

    Represents one possible modification of the base gradient
    that could be applied to the model weights.
    """

    id: int
    """Unique identifier for this candidate."""

    gradients: Dict[str, torch.Tensor]
    """Modified gradients keyed by parameter name."""

    description: str
    """Human-readable description of this candidate."""

    metadata: Dict[str, Any] = None
    """Optional metadata (e.g., noise scale used, strategy name)."""

    fitness: Optional[float] = None
    """Fitness score (set after evaluation)."""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Base class for gradient modification strategies.

    Strategies take the base gradient computed by backpropagation
    and generate multiple candidate variants for evolutionary selection.
    """

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def generate_candidates(
        self,
        base_gradients: Dict[str, torch.Tensor],
        num_candidates: int,
        step: int = 0,
    ) -> List[GradientCandidate]:
        """
        Generate candidate gradient modifications.

        Args:
            base_gradients: Original gradients from backprop, keyed by param name
            num_candidates: Number of candidates to generate
            step: Current training step (some strategies adapt over time)

        Returns:
            List of GradientCandidate objects
        """
        pass

    def _clone_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clone gradients to avoid modifying originals."""
        return {name: grad.clone() for name, grad in gradients.items()}

    def _apply_to_all(
        self,
        gradients: Dict[str, torch.Tensor],
        modifier_fn,
    ) -> Dict[str, torch.Tensor]:
        """Apply a modifier function to all gradients."""
        return {
            name: modifier_fn(grad)
            for name, grad in gradients.items()
        }
