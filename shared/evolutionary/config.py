"""Configuration for evolutionary fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class EvolutionaryConfig:
    """
    Configuration for evolutionary training.

    This config controls the evolutionary selection process during training.
    It can be loaded from YAML or constructed programmatically.
    """

    enabled: bool = False
    """Whether to use evolutionary training. If False, standard training is used."""

    num_candidates: int = 4
    """Number of gradient candidates to generate per step."""

    eval_batch_size: int = 4
    """Batch size for fitness evaluation (can be smaller than training batch)."""

    validation_config_path: Optional[str] = None
    """Path to fitness validation config (same format as rubrics)."""

    validation_config: Optional[Dict[str, Any]] = None
    """Inline validation config (alternative to path)."""

    # Strategy configuration
    strategy: str = "gradient_noise"
    """Candidate generation strategy: 'gradient_noise', 'scale_variation', 'combined'."""

    noise_scale: float = 0.1
    """Scale of noise to add to gradients (for gradient_noise strategy)."""

    scale_factors: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    """Gradient scaling factors to try (for scale_variation strategy)."""

    # Selection configuration
    selection_method: str = "best"
    """How to select winner: 'best' (highest fitness), 'tournament', 'proportional'."""

    min_fitness_improvement: float = 0.0
    """Only apply candidate if it improves fitness by at least this much over baseline."""

    # Performance tuning
    eval_frequency: int = 1
    """Evaluate every N training steps (1 = every step, 2 = every other step)."""

    warmup_steps: int = 0
    """Steps to train normally before enabling evolutionary selection.
    Allows the model to learn basic patterns first. Recommended: 100-500 for SFT."""

    cache_baseline: bool = True
    """Cache baseline fitness to avoid recomputing."""

    # Logging
    log_candidates: bool = True
    """Log fitness of all candidates (useful for debugging)."""

    log_selected: bool = True
    """Log which candidate was selected."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionaryConfig":
        """Create config from dictionary (e.g., from YAML)."""
        return cls(
            enabled=data.get("enabled", False),
            num_candidates=data.get("candidates", data.get("num_candidates", 4)),
            eval_batch_size=data.get("eval_batch_size", 4),
            validation_config_path=data.get("validation_config"),
            strategy=data.get("strategy", {}).get("type", "gradient_noise"),
            noise_scale=data.get("strategy", {}).get("params", {}).get("noise_scale", 0.1),
            scale_factors=data.get("strategy", {}).get("params", {}).get("scale_factors", [0.5, 1.0, 1.5, 2.0]),
            selection_method=data.get("selection", {}).get("method", "best"),
            min_fitness_improvement=data.get("selection", {}).get("min_improvement", 0.0),
            eval_frequency=data.get("eval_frequency", 1),
            warmup_steps=data.get("warmup_steps", 0),
            cache_baseline=data.get("cache_baseline", True),
            log_candidates=data.get("logging", {}).get("candidates", True),
            log_selected=data.get("logging", {}).get("selected", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (e.g., for YAML serialization)."""
        return {
            "enabled": self.enabled,
            "candidates": self.num_candidates,
            "eval_batch_size": self.eval_batch_size,
            "validation_config": self.validation_config_path,
            "strategy": {
                "type": self.strategy,
                "params": {
                    "noise_scale": self.noise_scale,
                    "scale_factors": self.scale_factors,
                },
            },
            "selection": {
                "method": self.selection_method,
                "min_improvement": self.min_fitness_improvement,
            },
            "eval_frequency": self.eval_frequency,
            "warmup_steps": self.warmup_steps,
            "cache_baseline": self.cache_baseline,
            "logging": {
                "candidates": self.log_candidates,
                "selected": self.log_selected,
            },
        }

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if self.enabled:
            if self.num_candidates < 2:
                issues.append("num_candidates must be >= 2 for evolutionary training")

            if self.strategy not in ("gradient_noise", "scale_variation", "combined"):
                issues.append(f"Unknown strategy: {self.strategy}")

            if self.selection_method not in ("best", "tournament", "proportional"):
                issues.append(f"Unknown selection method: {self.selection_method}")

            if self.validation_config_path is None and self.validation_config is None:
                issues.append("validation_config or validation_config_path required when enabled")

        return issues
