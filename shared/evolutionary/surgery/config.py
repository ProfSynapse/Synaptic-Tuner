"""
Surgery configuration and result dataclasses.

Location: shared/evolutionary/surgery/config.py
Purpose: Data containers for surgery configuration, per-operation results,
         and full pipeline results. Operation-specific configs follow ISP:
         each operation receives only the parameters it needs.
Used by: LoRASurgeon, all operation classes, surgery_handler, tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Per-operation configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AlphaSweepConfig:
    """Config subset for the alpha_sweep operation."""

    multipliers: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.25, 1.5, 2.0]
    )


@dataclass
class LayerScalingConfig:
    """Config subset for the layer_scaling operation."""

    scales: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.75, 1.25, 1.5]
    )


@dataclass
class DAREConfig:
    """Config subset for the dare_drop_rescale operation."""

    drop_rates: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5]
    )


@dataclass
class CheckpointInterpolationConfig:
    """Config subset for the checkpoint_interpolation operation."""

    other_checkpoint_path: str = ""
    blend_ratios: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75]
    )


@dataclass
class SVDRankReductionConfig:
    """Config subset for the svd_rank_reduction operation."""

    rank_fractions: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75]
    )


@dataclass
class MetricsWeightedMergeConfig:
    """Config subset for the metrics_weighted_merge operation."""

    checkpoint_paths: List[str] = field(default_factory=list)
    checkpoint_scores: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level surgery configuration
# ---------------------------------------------------------------------------

@dataclass
class SurgeryConfig:
    """Configuration for LoRA weight surgery.

    Per-operation parameters are organized into typed config groups
    (``alpha_sweep_config``, ``layer_scaling_config``, etc.). Each
    operation accesses only its own config group via these attributes.

    For backward compatibility, the flat field names (``alpha_multipliers``,
    ``layer_scales``, etc.) are accepted in the constructor and as
    properties that delegate to the corresponding config group.
    """

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

    # Per-operation config groups
    alpha_sweep_config: AlphaSweepConfig = field(default_factory=AlphaSweepConfig)
    layer_scaling_config: LayerScalingConfig = field(default_factory=LayerScalingConfig)
    dare_config: DAREConfig = field(default_factory=DAREConfig)
    checkpoint_interpolation_config: CheckpointInterpolationConfig = field(
        default_factory=CheckpointInterpolationConfig
    )
    svd_rank_reduction_config: SVDRankReductionConfig = field(
        default_factory=SVDRankReductionConfig
    )
    metrics_weighted_merge_config: MetricsWeightedMergeConfig = field(
        default_factory=MetricsWeightedMergeConfig
    )

    # Maps deprecated flat kwarg → (config_group_attr, field_name)
    _FLAT_KWARGS: Dict[str, Tuple[str, str]] = field(
        default=None, init=False, repr=False, compare=False,  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        # _pending_flat is set by the __init__ wrapper when flat kwargs are
        # passed.  Apply them now that grouped configs are initialized.
        pending: Optional[Dict[str, Any]] = getattr(self, "_pending_flat", None)
        if pending:
            for flat_name, value in pending.items():
                cfg_attr, field_name = _FLAT_KWARG_MAP[flat_name]
                setattr(getattr(self, cfg_attr), field_name, value)
            del self._pending_flat

    # ------------------------------------------------------------------
    # Backward-compat properties — delegate to grouped configs
    # ------------------------------------------------------------------

    @property
    def alpha_multipliers(self) -> List[float]:
        return self.alpha_sweep_config.multipliers

    @alpha_multipliers.setter
    def alpha_multipliers(self, value: List[float]) -> None:
        self.alpha_sweep_config.multipliers = value

    @property
    def layer_scales(self) -> List[float]:
        return self.layer_scaling_config.scales

    @layer_scales.setter
    def layer_scales(self, value: List[float]) -> None:
        self.layer_scaling_config.scales = value

    @property
    def dare_drop_rates(self) -> List[float]:
        return self.dare_config.drop_rates

    @dare_drop_rates.setter
    def dare_drop_rates(self, value: List[float]) -> None:
        self.dare_config.drop_rates = value

    @property
    def blend_ratios(self) -> List[float]:
        return self.checkpoint_interpolation_config.blend_ratios

    @blend_ratios.setter
    def blend_ratios(self, value: List[float]) -> None:
        self.checkpoint_interpolation_config.blend_ratios = value

    @property
    def other_checkpoint_path(self) -> str:
        return self.checkpoint_interpolation_config.other_checkpoint_path

    @other_checkpoint_path.setter
    def other_checkpoint_path(self, value: str) -> None:
        self.checkpoint_interpolation_config.other_checkpoint_path = value

    @property
    def svd_rank_fractions(self) -> List[float]:
        return self.svd_rank_reduction_config.rank_fractions

    @svd_rank_fractions.setter
    def svd_rank_fractions(self, value: List[float]) -> None:
        self.svd_rank_reduction_config.rank_fractions = value

    @property
    def checkpoint_paths(self) -> List[str]:
        return self.metrics_weighted_merge_config.checkpoint_paths

    @checkpoint_paths.setter
    def checkpoint_paths(self, value: List[str]) -> None:
        self.metrics_weighted_merge_config.checkpoint_paths = value

    @property
    def checkpoint_scores(self) -> List[float]:
        return self.metrics_weighted_merge_config.checkpoint_scores

    @checkpoint_scores.setter
    def checkpoint_scores(self, value: List[float]) -> None:
        self.metrics_weighted_merge_config.checkpoint_scores = value

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
            alpha_sweep_config=AlphaSweepConfig(
                multipliers=data.get("alpha_sweep", {}).get(
                    "multipliers", [0.5, 0.75, 1.25, 1.5, 2.0]
                ),
            ),
            layer_scaling_config=LayerScalingConfig(
                scales=data.get("layer_scaling", {}).get(
                    "scales", [0.0, 0.5, 0.75, 1.25, 1.5]
                ),
            ),
            dare_config=DAREConfig(
                drop_rates=data.get("dare", {}).get(
                    "drop_rates", [0.1, 0.2, 0.3, 0.5]
                ),
            ),
            checkpoint_interpolation_config=CheckpointInterpolationConfig(
                other_checkpoint_path=data.get("other_checkpoint_path", ""),
                blend_ratios=data.get("checkpoint_interpolation", {}).get(
                    "blend_ratios", [0.25, 0.5, 0.75]
                ),
            ),
            svd_rank_reduction_config=SVDRankReductionConfig(
                rank_fractions=data.get("svd_rank_reduction", {}).get(
                    "rank_fractions", [0.25, 0.5, 0.75]
                ),
            ),
            metrics_weighted_merge_config=MetricsWeightedMergeConfig(
                checkpoint_paths=data.get("checkpoint_paths", []),
                checkpoint_scores=data.get("checkpoint_scores", []),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Flat-kwarg backward compatibility for SurgeryConfig.__init__
# ---------------------------------------------------------------------------

_FLAT_KWARG_MAP: Dict[str, Tuple[str, str]] = {
    "alpha_multipliers": ("alpha_sweep_config", "multipliers"),
    "layer_scales": ("layer_scaling_config", "scales"),
    "dare_drop_rates": ("dare_config", "drop_rates"),
    "blend_ratios": ("checkpoint_interpolation_config", "blend_ratios"),
    "other_checkpoint_path": ("checkpoint_interpolation_config", "other_checkpoint_path"),
    "svd_rank_fractions": ("svd_rank_reduction_config", "rank_fractions"),
    "checkpoint_paths": ("metrics_weighted_merge_config", "checkpoint_paths"),
    "checkpoint_scores": ("metrics_weighted_merge_config", "checkpoint_scores"),
}

# Wrap the dataclass-generated __init__ so callers can pass flat kwargs
# (e.g. ``SurgeryConfig(alpha_multipliers=[0.5])``).  The wrapper strips
# them out, forwards the rest to the real __init__, then __post_init__
# applies the flat values to the grouped config objects.
_dataclass_init = SurgeryConfig.__init__


def _surgery_config_init(self: SurgeryConfig, **kwargs: Any) -> None:
    flat = {k: kwargs.pop(k) for k in list(kwargs) if k in _FLAT_KWARG_MAP}
    if flat:
        object.__setattr__(self, "_pending_flat", flat)
    _dataclass_init(self, **kwargs)


SurgeryConfig.__init__ = _surgery_config_init  # type: ignore[assignment]


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
