"""Scale variation strategy for evolutionary training."""
from __future__ import annotations

from typing import Dict, List

import torch

from .base import BaseStrategy, GradientCandidate


class ScaleVariationStrategy(BaseStrategy):
    """
    Vary gradient magnitude by different scaling factors.

    This strategy explores different learning rates by scaling
    the entire gradient uniformly. Useful when the computed
    gradient direction is good but magnitude is uncertain.

    Candidates include gradients scaled by each factor in scale_factors.
    """

    def __init__(
        self,
        scale_factors: List[float] = None,
    ):
        """
        Initialize scale variation strategy.

        Args:
            scale_factors: List of scaling factors to try.
                          Default: [0.5, 1.0, 1.5, 2.0]
        """
        super().__init__(name="scale_variation")
        self.scale_factors = scale_factors or [0.5, 1.0, 1.5, 2.0]

    def generate_candidates(
        self,
        base_gradients: Dict[str, torch.Tensor],
        num_candidates: int,
        step: int = 0,
    ) -> List[GradientCandidate]:
        """Generate candidates by scaling gradients."""
        candidates = []

        # Use as many scale factors as we need candidates
        factors_to_use = self.scale_factors[:num_candidates]

        # If we need more candidates than factors, interpolate
        while len(factors_to_use) < num_candidates:
            # Add intermediate scale factors
            new_factors = []
            for i in range(len(factors_to_use) - 1):
                new_factors.append((factors_to_use[i] + factors_to_use[i+1]) / 2)
            factors_to_use = sorted(set(factors_to_use + new_factors))[:num_candidates]

        for i, scale in enumerate(factors_to_use):
            scaled_gradients = {
                name: grad * scale
                for name, grad in base_gradients.items()
            }

            candidates.append(GradientCandidate(
                id=i,
                gradients=scaled_gradients,
                description=f"Scaled gradient (factor={scale:.2f})",
                metadata={
                    "strategy": self.name,
                    "scale_factor": scale,
                },
            ))

        return candidates
