"""
Surgery configuration and result dataclasses.

Location: shared/evolutionary/surgery/config.py
Purpose: Data containers for surgery configuration, per-operation results,
         and full pipeline results.
Used by: LoRASurgeon, all operation classes, surgery_handler, tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SurgeryConfig:
    """Configuration for LoRA weight surgery."""

    adapter_path: str = ""
    eval_scenario: str = ""
    eval_backend: str = "local"
    cloud_provider: str = "hf_jobs"
    local_min_vram_gb: int = 8
    min_improvement: float = 0.005
    operations: List[str] = field(
        default_factory=lambda: ["alpha_sweep", "layer_scaling", "module_ablation"]
    )
    output_dir: str = "surgery_results/"
    other_checkpoint_path: str = ""
    checkpoint_paths: List[str] = field(default_factory=list)
    checkpoint_scores: List[float] = field(default_factory=list)

    # Per-operation configs
    alpha_multipliers: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.25, 1.5, 2.0]
    )
    layer_scales: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.75, 1.25, 1.5]
    )
    dare_drop_rates: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5]
    )
    blend_ratios: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75]
    )
    svd_rank_fractions: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75]
    )

    @classmethod
    def from_yaml(cls, path: str) -> "SurgeryConfig":
        """Load config from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            SurgeryConfig populated from the YAML data.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required: pip install pyyaml") from exc

        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

        data = raw.get("surgery", raw)

        return cls(
            adapter_path=data.get("adapter_path", ""),
            eval_scenario=data.get("eval_scenario", ""),
            eval_backend=data.get("eval_backend", "local"),
            cloud_provider=data.get("cloud_provider", "hf_jobs"),
            local_min_vram_gb=data.get("local_min_vram_gb", 8),
            min_improvement=data.get("min_improvement", 0.005),
            operations=data.get("operations", ["alpha_sweep", "layer_scaling", "module_ablation"]),
            output_dir=data.get("output_dir", "surgery_results/"),
            other_checkpoint_path=data.get("other_checkpoint_path", ""),
            checkpoint_paths=data.get("checkpoint_paths", []),
            checkpoint_scores=data.get("checkpoint_scores", []),
            alpha_multipliers=data.get("alpha_sweep", {}).get(
                "multipliers", [0.5, 0.75, 1.25, 1.5, 2.0]
            ),
            layer_scales=data.get("layer_scaling", {}).get(
                "scales", [0.0, 0.5, 0.75, 1.25, 1.5]
            ),
            dare_drop_rates=data.get("dare", {}).get(
                "drop_rates", [0.1, 0.2, 0.3, 0.5]
            ),
            blend_ratios=data.get("checkpoint_interpolation", {}).get(
                "blend_ratios", [0.25, 0.5, 0.75]
            ),
            svd_rank_fractions=data.get("svd_rank_reduction", {}).get(
                "rank_fractions", [0.25, 0.5, 0.75]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return asdict(self)


@dataclass
class OperationResult:
    """Result of a single surgery operation."""

    operation: str
    variants_tried: int
    best_variant: str
    best_score: float
    improvement: float
    adapter_path: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurgeryResult:
    """Result of the full surgery pipeline."""

    baseline_score: float
    final_score: float
    total_improvement: float
    operations_applied: List[OperationResult] = field(default_factory=list)
    best_adapter_path: str = ""
    duration_seconds: float = 0.0
