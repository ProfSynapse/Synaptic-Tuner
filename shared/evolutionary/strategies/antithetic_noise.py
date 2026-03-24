"""Antithetic gradient-noise strategy for evolutionary training."""
from __future__ import annotations

from typing import Dict, List

import torch

from .base import BaseStrategy, GradientCandidate


class AntitheticNoiseStrategy(BaseStrategy):
    """
    Generate mirrored gradient-noise pairs.

    For each sampled noise tensor eps, evaluate both:
    - g + eps
    - g - eps

    This reduces variance relative to iid Gaussian candidates and makes the
    search around the baseline gradient more symmetric.
    """

    def __init__(
        self,
        noise_scale: float = 0.1,
        include_pure: bool = True,
        adaptive_scale: bool = False,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(name="antithetic_noise")
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
        candidates: List[GradientCandidate] = []

        if self.include_pure:
            candidates.append(
                GradientCandidate(
                    id=0,
                    gradients=self._clone_gradients(base_gradients),
                    description="Pure gradient (no noise)",
                    metadata={"strategy": self.name, "noise_scale": 0.0, "variant": "pure"},
                )
            )

        remaining = max(0, num_candidates - len(candidates))
        if remaining <= 0:
            return candidates

        if self.adaptive_scale:
            scale = self._compute_adaptive_scale(base_gradients)
        else:
            scale = self.noise_scale

        pair_count = (remaining + 1) // 2
        for pair_index in range(pair_count):
            noise_by_param: Dict[str, torch.Tensor] = {}
            for name, grad in base_gradients.items():
                grad_norm = grad.norm()
                capped_norm = min(grad_norm, self.max_grad_norm) if self.max_grad_norm > 0 else grad_norm
                noise_by_param[name] = torch.randn_like(grad) * scale * capped_norm

            plus_gradients = {
                name: grad + noise_by_param[name]
                for name, grad in base_gradients.items()
            }
            candidates.append(
                GradientCandidate(
                    id=len(candidates),
                    gradients=plus_gradients,
                    description=f"Antithetic + noise pair {pair_index + 1} (scale={scale:.4f})",
                    metadata={
                        "strategy": self.name,
                        "noise_scale": scale,
                        "pair_index": pair_index,
                        "variant": "plus",
                    },
                )
            )

            if len(candidates) >= num_candidates:
                break

            minus_gradients = {
                name: grad - noise_by_param[name]
                for name, grad in base_gradients.items()
            }
            candidates.append(
                GradientCandidate(
                    id=len(candidates),
                    gradients=minus_gradients,
                    description=f"Antithetic - noise pair {pair_index + 1} (scale={scale:.4f})",
                    metadata={
                        "strategy": self.name,
                        "noise_scale": scale,
                        "pair_index": pair_index,
                        "variant": "minus",
                    },
                )
            )

        return candidates[:num_candidates]

    def _compute_adaptive_scale(self, gradients: Dict[str, torch.Tensor]) -> float:
        total_norm = 0.0
        num_params = 0
        for grad in gradients.values():
            total_norm += grad.norm().item()
            num_params += 1
        avg_norm = total_norm / max(num_params, 1)
        return self.noise_scale / (1.0 + avg_norm)
