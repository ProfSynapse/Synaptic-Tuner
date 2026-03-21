"""Gradient noise strategy for evolutionary training."""
from __future__ import annotations

from typing import Dict, List

import torch

from .base import BaseStrategy, GradientCandidate


class GradientNoiseStrategy(BaseStrategy):
    """
    Add Gaussian noise to gradients.

    This strategy explores the gradient landscape around the computed
    gradient by adding random perturbations. The idea is that the
    noisy variant might land in a better local optimum.

    Candidates include:
    - Pure gradient (no noise)
    - Multiple noisy variants with different noise samples
    """

    def __init__(
        self,
        noise_scale: float = 0.1,
        include_pure: bool = True,
        adaptive_scale: bool = False,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize noise strategy.

        Args:
            noise_scale: Standard deviation of noise relative to gradient norm
            include_pure: Include the original gradient as one candidate
            adaptive_scale: Adapt noise scale based on gradient magnitude
            max_grad_norm: Cap on gradient norm used for noise scaling.
                When raw gradient norms exceed this value, noise magnitude
                is based on max_grad_norm instead of the raw norm, preventing
                oversized perturbations in early training.
        """
        super().__init__(name="gradient_noise")
        self.noise_scale = noise_scale
        self.include_pure = include_pure
        self.adaptive_scale = adaptive_scale
        self.max_grad_norm = max_grad_norm

    def generate_candidates(
        self,
        base_gradients: Dict[str, torch.Tensor],
        num_candidates: int,
        step: int = 0,
    ) -> List[GradientCandidate]:
        """Generate candidates by adding Gaussian noise to gradients."""
        candidates = []

        # Candidate 0: Pure gradient (if enabled)
        if self.include_pure:
            candidates.append(GradientCandidate(
                id=0,
                gradients=self._clone_gradients(base_gradients),
                description="Pure gradient (no noise)",
                metadata={"strategy": self.name, "noise_scale": 0.0},
            ))
            num_noisy = num_candidates - 1
        else:
            num_noisy = num_candidates

        # Generate noisy variants
        for i in range(num_noisy):
            candidate_id = len(candidates)

            # Compute effective noise scale
            if self.adaptive_scale:
                # Scale noise based on gradient magnitude
                scale = self._compute_adaptive_scale(base_gradients)
            else:
                scale = self.noise_scale

            # Add noise to each gradient
            # Cap noise magnitude using max_grad_norm to prevent oversized
            # perturbations when raw gradient norms are large (early training)
            noisy_gradients = {}
            for name, grad in base_gradients.items():
                grad_norm = grad.norm()
                capped_norm = min(grad_norm, self.max_grad_norm) if self.max_grad_norm > 0 else grad_norm
                noise = torch.randn_like(grad) * scale * capped_norm
                noisy_gradients[name] = grad + noise

            candidates.append(GradientCandidate(
                id=candidate_id,
                gradients=noisy_gradients,
                description=f"Noisy variant {i+1} (scale={scale:.4f})",
                metadata={
                    "strategy": self.name,
                    "noise_scale": scale,
                    "variant_index": i,
                },
            ))

        return candidates

    def _compute_adaptive_scale(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Compute noise scale based on gradient statistics."""
        # Use average gradient norm to scale noise
        total_norm = 0.0
        num_params = 0
        for grad in gradients.values():
            total_norm += grad.norm().item()
            num_params += 1

        avg_norm = total_norm / max(num_params, 1)

        # Scale noise inversely with gradient magnitude
        # Large gradients = less noise, small gradients = more noise
        return self.noise_scale / (1.0 + avg_norm)
