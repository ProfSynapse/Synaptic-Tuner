"""Combined strategy mixing noise and scaling."""
from __future__ import annotations

from typing import Dict, List

import torch

from .base import BaseStrategy, GradientCandidate


class CombinedStrategy(BaseStrategy):
    """
    Combine noise addition and scale variation.

    This strategy generates candidates using a mix of approaches:
    - Pure gradient (baseline)
    - Scaled variants
    - Noisy variants
    - Scaled + noisy variants

    This provides more diverse exploration of the gradient space.
    """

    def __init__(
        self,
        noise_scale: float = 0.1,
        scale_factors: List[float] = None,
    ):
        """
        Initialize combined strategy.

        Args:
            noise_scale: Standard deviation of noise
            scale_factors: Scaling factors to try
        """
        super().__init__(name="combined")
        self.noise_scale = noise_scale
        self.scale_factors = scale_factors or [0.5, 1.0, 1.5]

    def generate_candidates(
        self,
        base_gradients: Dict[str, torch.Tensor],
        num_candidates: int,
        step: int = 0,
    ) -> List[GradientCandidate]:
        """Generate diverse candidates combining noise and scaling."""
        candidates = []

        # Always include pure gradient
        candidates.append(GradientCandidate(
            id=0,
            gradients=self._clone_gradients(base_gradients),
            description="Pure gradient",
            metadata={"strategy": self.name, "type": "pure"},
        ))

        remaining = num_candidates - 1
        if remaining <= 0:
            return candidates

        # Allocate remaining candidates
        # 1/3 scale, 1/3 noise, 1/3 combined
        num_scale = max(1, remaining // 3)
        num_noise = max(1, remaining // 3)
        num_combined = remaining - num_scale - num_noise

        # Scale variants
        for i, scale in enumerate(self.scale_factors[:num_scale]):
            scaled_gradients = {
                name: grad * scale
                for name, grad in base_gradients.items()
            }
            candidates.append(GradientCandidate(
                id=len(candidates),
                gradients=scaled_gradients,
                description=f"Scaled x{scale:.1f}",
                metadata={"strategy": self.name, "type": "scale", "factor": scale},
            ))

        # Noise variants
        for i in range(num_noise):
            noisy_gradients = {}
            for name, grad in base_gradients.items():
                noise = torch.randn_like(grad) * self.noise_scale * grad.norm()
                noisy_gradients[name] = grad + noise
            candidates.append(GradientCandidate(
                id=len(candidates),
                gradients=noisy_gradients,
                description=f"Noisy variant {i+1}",
                metadata={"strategy": self.name, "type": "noise"},
            ))

        # Combined variants (scale + noise)
        for i in range(num_combined):
            scale = self.scale_factors[i % len(self.scale_factors)]
            combined_gradients = {}
            for name, grad in base_gradients.items():
                scaled = grad * scale
                noise = torch.randn_like(grad) * self.noise_scale * scaled.norm()
                combined_gradients[name] = scaled + noise
            candidates.append(GradientCandidate(
                id=len(candidates),
                gradients=combined_gradients,
                description=f"Scaled x{scale:.1f} + noise",
                metadata={"strategy": self.name, "type": "combined", "factor": scale},
            ))

        return candidates[:num_candidates]
