"""
Surgery operations package.

Location: shared/evolutionary/surgery/operations/__init__.py
Purpose: Import all operation modules to trigger registration via the
         @register_operation decorator.
Used by: surgery/__init__.py imports this package to populate the registry.
"""

from .alpha_sweep import AlphaSweepOperation
from .attention_mlp_ablation import AttentionMLPAblationOperation
from .checkpoint_interpolation import CheckpointInterpolationOperation
from .dare_drop_rescale import DAREDropRescaleOperation
from .layer_scaling import LayerScalingOperation
from .metrics_weighted_merge import MetricsWeightedMergeOperation
from .module_ablation import ModuleAblationOperation
from .svd_rank_reduction import SVDRankReductionOperation

__all__ = [
    "AlphaSweepOperation",
    "AttentionMLPAblationOperation",
    "CheckpointInterpolationOperation",
    "DAREDropRescaleOperation",
    "LayerScalingOperation",
    "MetricsWeightedMergeOperation",
    "ModuleAblationOperation",
    "SVDRankReductionOperation",
]
