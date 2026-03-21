"""
LoRA Weight Surgery module.

Location: shared/evolutionary/lora_surgery.py
Purpose: Post-training weight surgery on LoRA adapters guided by eval scores
Used by: tuner CLI (surgery command), evolutionary pipeline

Implements eval-guided operations on trained LoRA adapters:
1. Alpha sweep - modify lora_alpha in adapter_config.json
2. Layer scaling - scale individual layer weights
3. Module ablation - zero module-type weights (q_proj, k_proj, etc.)
4. Checkpoint interpolation - blend two checkpoints at various ratios
5. DARE drop-and-rescale - randomly drop weights, rescale survivors
6. Metrics-weighted merge - merge N checkpoints weighted by eval scores
7. SVD rank reduction - compress LoRA via truncated SVD
8. Attention/MLP ablation - zero all attention vs all MLP LoRA weights

Each operation:
  1. Copies the adapter to a temp directory
  2. Modifies weights in the copy
  3. Evaluates the modified adapter
  4. Keeps improvements, discards regressions
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.eval_backend import EvalBackend, EvalResult

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from safetensors.torch import load_file as st_load_file, save_file as st_save_file
except ImportError:
    st_load_file = None  # type: ignore[assignment]
    st_save_file = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_dependencies() -> None:
    """Verify that required dependencies are available."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for LoRA surgery. Install with: pip install torch"
        )
    if st_load_file is None or st_save_file is None:
        raise ImportError(
            "safetensors is required for LoRA surgery. Install with: pip install safetensors"
        )


def _load_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    """Load adapter_config.json from an adapter directory."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with open(config_path, "r") as fh:
        return json.load(fh)


def _save_adapter_config(adapter_dir: str, config: Dict[str, Any]) -> None:
    """Save adapter_config.json to an adapter directory."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)


def _find_safetensor_files(adapter_dir: str) -> List[str]:
    """Find all safetensor weight files in an adapter directory."""
    files = []
    for fname in os.listdir(adapter_dir):
        if fname.endswith(".safetensors"):
            files.append(os.path.join(adapter_dir, fname))
    return sorted(files)


def _load_all_weights(adapter_dir: str) -> Dict[str, "torch.Tensor"]:
    """Load all weights from safetensor files in an adapter directory."""
    _check_dependencies()
    all_weights: Dict[str, torch.Tensor] = {}
    for fpath in _find_safetensor_files(adapter_dir):
        all_weights.update(st_load_file(fpath))
    return all_weights


def _save_all_weights(adapter_dir: str, weights: Dict[str, "torch.Tensor"]) -> None:
    """Save weights to a single safetensor file in the adapter directory."""
    _check_dependencies()
    out_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    st_save_file(weights, out_path)


def _copy_adapter(src: str, dst: str) -> str:
    """Copy an adapter directory, returning the destination path."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _get_layer_indices(weight_keys: List[str]) -> List[int]:
    """Extract unique layer indices from weight key names.

    Typical key pattern: ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``
    """
    indices = set()
    pattern = re.compile(r"layers\.(\d+)\.")
    for key in weight_keys:
        match = pattern.search(key)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _get_module_types(weight_keys: List[str]) -> List[str]:
    """Extract unique module type names (q_proj, k_proj, ...) from weight keys."""
    types = set()
    # Match the projection name before .lora_A or .lora_B
    pattern = re.compile(r"\.(\w+_proj)\.lora_[AB]")
    for key in weight_keys:
        match = pattern.search(key)
        if match:
            types.add(match.group(1))
    return sorted(types)


def _is_attention_key(key: str) -> bool:
    """Return True if key belongs to an attention module."""
    attention_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    for mod in attention_modules:
        if f".{mod}." in key:
            return True
    return False


def _is_mlp_key(key: str) -> bool:
    """Return True if key belongs to an MLP module."""
    mlp_modules = {"gate_proj", "up_proj", "down_proj"}
    for mod in mlp_modules:
        if f".{mod}." in key:
            return True
    return False


def _softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """Compute softmax over a list of floats with temperature scaling."""
    scaled = [v / temperature for v in values]
    max_val = max(scaled)
    exps = [math.exp(v - max_val) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# LoRASurgeon
# ---------------------------------------------------------------------------

class LoRASurgeon:
    """Performs eval-guided post-training weight surgery on LoRA adapters.

    The surgeon iterates through a configurable list of operations. Each
    operation produces one or more modified adapter variants. The variant
    with the highest eval score becomes the new baseline for the next
    operation, provided it exceeds the minimum improvement threshold.

    Args:
        adapter_path: Path to the LoRA adapter directory.
        eval_backend: Object implementing the ``EvalBackend`` protocol.
        eval_scenario: Scenario identifier passed to the eval backend.
        config: Surgery configuration.
    """

    # Map operation names to method names
    _OPERATION_METHODS = {
        "alpha_sweep": "alpha_sweep",
        "layer_scaling": "layer_scaling",
        "module_ablation": "module_ablation",
        "checkpoint_interpolation": "checkpoint_interpolation",
        "dare_drop_rescale": "dare_drop_rescale",
        "metrics_weighted_merge": "metrics_weighted_merge",
        "svd_rank_reduction": "svd_rank_reduction",
        "attention_mlp_ablation": "attention_mlp_ablation",
    }

    def __init__(
        self,
        adapter_path: str,
        eval_backend: EvalBackend,
        eval_scenario: str,
        config: SurgeryConfig,
    ) -> None:
        _check_dependencies()
        self.adapter_path = adapter_path
        self.eval_backend = eval_backend
        self.eval_scenario = eval_scenario
        self.config = config
        self._work_dir = os.path.join(config.output_dir, "_surgery_work")

    # ------------------------------------------------------------------
    # Context manager & cleanup
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "LoRASurgeon":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the temporary work directory created during surgery."""
        if os.path.isdir(self._work_dir):
            shutil.rmtree(self._work_dir)
            logger.info("Cleaned up work directory: %s", self._work_dir)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def run_surgery(self) -> SurgeryResult:
        """Run all enabled operations sequentially, keeping improvements.

        Returns:
            SurgeryResult summarising the full pipeline.
        """
        start_time = time.time()

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self._work_dir, exist_ok=True)

        try:
            baseline_score = await self._evaluate(self.adapter_path)
            best_score = baseline_score
            best_adapter = self.adapter_path
            operations_applied: List[OperationResult] = []

            logger.info(
                "Starting surgery on %s  baseline_score=%.4f",
                self.adapter_path,
                baseline_score,
            )

            for operation_name in self.config.operations:
                method_name = self._OPERATION_METHODS.get(operation_name)
                if method_name is None:
                    logger.warning("Unknown operation: %s, skipping", operation_name)
                    continue

                method = getattr(self, method_name, None)
                if method is None:
                    logger.warning("Method not found for operation: %s", operation_name)
                    continue

                logger.info("Running operation: %s", operation_name)

                try:
                    result = await self._run_operation(
                        method, operation_name, best_adapter, best_score
                    )
                except Exception:
                    logger.exception("Operation %s failed", operation_name)
                    continue

                if result.improvement > self.config.min_improvement:
                    best_score = result.best_score
                    best_adapter = result.adapter_path
                    operations_applied.append(result)
                    logger.info(
                        "Operation %s improved score: %.4f -> %.4f (+%.4f)",
                        operation_name,
                        best_score - result.improvement,
                        best_score,
                        result.improvement,
                    )
                else:
                    logger.info(
                        "Operation %s did not improve score (best=%.4f, improvement=%.4f < min=%.4f)",
                        operation_name,
                        result.best_score,
                        result.improvement,
                        self.config.min_improvement,
                    )

            # Copy best adapter to final output location
            final_path = os.path.join(self.config.output_dir, "best_adapter")
            if best_adapter != self.adapter_path:
                _copy_adapter(best_adapter, final_path)
            else:
                _copy_adapter(self.adapter_path, final_path)

            duration = time.time() - start_time

            surgery_result = SurgeryResult(
                baseline_score=baseline_score,
                final_score=best_score,
                total_improvement=best_score - baseline_score,
                operations_applied=operations_applied,
                best_adapter_path=final_path,
                duration_seconds=duration,
            )

            # Save report
            self._save_report(surgery_result)

            return surgery_result
        finally:
            self.cleanup()

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def alpha_sweep(
        self, adapter_path: str, baseline_score: float
    ) -> OperationResult:
        """Modify adapter_config.json lora_alpha, no weight changes.

        Tries several multiplier values for the current alpha and evaluates
        each variant.
        """
        original_config = _load_adapter_config(adapter_path)
        current_alpha = original_config.get("lora_alpha", 16)
        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"alpha_scores": {}}

        for mult in self.config.alpha_multipliers:
            new_alpha = int(round(current_alpha * mult))
            if new_alpha == current_alpha:
                continue

            variant_dir = os.path.join(
                self._work_dir, f"alpha_{new_alpha}"
            )
            _copy_adapter(adapter_path, variant_dir)

            variant_config = _load_adapter_config(variant_dir)
            variant_config["lora_alpha"] = new_alpha
            _save_adapter_config(variant_dir, variant_config)

            score = await self._evaluate(variant_dir)
            variants_tried += 1
            details["alpha_scores"][str(new_alpha)] = score

            logger.info("  alpha=%d  score=%.4f", new_alpha, score)

            if score > best_score:
                best_score = score
                best_variant = f"alpha={new_alpha}"
                best_path = variant_dir

        return OperationResult(
            operation="alpha_sweep",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )

    async def layer_scaling(
        self, adapter_path: str, baseline_score: float
    ) -> OperationResult:
        """Scale individual layers to find importance.

        For each layer index, tries several scale factors and keeps the
        combination that scores highest.
        """
        weights = _load_all_weights(adapter_path)
        layer_indices = _get_layer_indices(list(weights.keys()))

        if not layer_indices:
            return OperationResult(
                operation="layer_scaling",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_lora_layers_found"},
            )

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"layer_scores": {}}

        for layer_idx in layer_indices:
            layer_pattern = f"layers.{layer_idx}."
            for scale in self.config.layer_scales:
                variant_dir = os.path.join(
                    self._work_dir, f"layer{layer_idx}_scale{scale}"
                )
                _copy_adapter(adapter_path, variant_dir)

                modified_weights = dict(weights)
                for key, tensor in weights.items():
                    if layer_pattern in key:
                        modified_weights[key] = tensor * scale
                    else:
                        modified_weights[key] = tensor.clone()

                _save_all_weights(variant_dir, modified_weights)

                score = await self._evaluate(variant_dir)
                variants_tried += 1

                variant_label = f"layer{layer_idx}_scale{scale}"
                details["layer_scores"][variant_label] = score
                logger.info(
                    "  layer=%d scale=%.2f  score=%.4f", layer_idx, scale, score
                )

                if score > best_score:
                    best_score = score
                    best_variant = variant_label
                    best_path = variant_dir

        return OperationResult(
            operation="layer_scaling",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )

    async def module_ablation(
        self, adapter_path: str, baseline_score: float
    ) -> OperationResult:
        """Zero all weights of a module type one at a time, measure impact."""
        weights = _load_all_weights(adapter_path)
        module_types = _get_module_types(list(weights.keys()))

        if not module_types:
            return OperationResult(
                operation="module_ablation",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_module_types_found"},
            )

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"module_scores": {}}

        for mod_type in module_types:
            variant_dir = os.path.join(self._work_dir, f"ablate_{mod_type}")
            _copy_adapter(adapter_path, variant_dir)

            modified_weights = {}
            for key, tensor in weights.items():
                if f".{mod_type}." in key:
                    modified_weights[key] = torch.zeros_like(tensor)
                else:
                    modified_weights[key] = tensor.clone()

            _save_all_weights(variant_dir, modified_weights)

            score = await self._evaluate(variant_dir)
            variants_tried += 1
            details["module_scores"][mod_type] = score
            logger.info("  ablate %s  score=%.4f", mod_type, score)

            if score > best_score:
                best_score = score
                best_variant = f"ablate_{mod_type}"
                best_path = variant_dir

        return OperationResult(
            operation="module_ablation",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )

    async def checkpoint_interpolation(
        self, adapter_path: str, baseline_score: float
    ) -> OperationResult:
        """Blend two checkpoints at various ratios.

        Uses ``other_checkpoint_path`` from the config. If not provided,
        returns immediately with no improvement.
        """
        other_path = self.config.other_checkpoint_path

        if not other_path or not os.path.isdir(other_path):
            return OperationResult(
                operation="checkpoint_interpolation",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_other_checkpoint"},
            )

        weights_a = _load_all_weights(adapter_path)
        weights_b = _load_all_weights(other_path)

        # Only interpolate keys present in both
        common_keys = set(weights_a.keys()) & set(weights_b.keys())

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"blend_scores": {}}

        for ratio in self.config.blend_ratios:
            variant_dir = os.path.join(
                self._work_dir, f"blend_{ratio:.2f}"
            )
            _copy_adapter(adapter_path, variant_dir)

            blended: Dict[str, torch.Tensor] = {}
            for key in weights_a:
                if key in common_keys:
                    blended[key] = ratio * weights_a[key] + (1.0 - ratio) * weights_b[key]
                else:
                    blended[key] = weights_a[key].clone()

            _save_all_weights(variant_dir, blended)

            score = await self._evaluate(variant_dir)
            variants_tried += 1
            details["blend_scores"][str(ratio)] = score
            logger.info("  blend ratio=%.2f  score=%.4f", ratio, score)

            if score > best_score:
                best_score = score
                best_variant = f"blend_{ratio:.2f}"
                best_path = variant_dir

        return OperationResult(
            operation="checkpoint_interpolation",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )

    async def dare_drop_rescale(
        self, adapter_path: str, baseline_score: float
    ) -> OperationResult:
        """DARE: randomly drop weights and rescale survivors.

        For each drop rate p, creates a binary mask where each weight is
        kept with probability (1 - p), then rescales survivors by 1/(1-p)
        so the expected value is preserved.
        """
        weights = _load_all_weights(adapter_path)

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"dare_scores": {}}

        for drop_rate in self.config.dare_drop_rates:
            # Clamp drop_rate to [0.0, 0.99] to prevent division by zero
            if drop_rate >= 1.0:
                logger.warning(
                    "DARE drop_rate %.2f >= 1.0 would cause division by zero, "
                    "clamping to 0.99",
                    drop_rate,
                )
                drop_rate = 0.99
            drop_rate = max(0.0, drop_rate)

            variant_dir = os.path.join(
                self._work_dir, f"dare_{drop_rate:.2f}"
            )
            _copy_adapter(adapter_path, variant_dir)

            modified_weights: Dict[str, torch.Tensor] = {}
            for key, tensor in weights.items():
                mask = (torch.rand_like(tensor.float()) > drop_rate).to(tensor.dtype)
                modified_weights[key] = tensor * mask / (1.0 - drop_rate)

            _save_all_weights(variant_dir, modified_weights)

            score = await self._evaluate(variant_dir)
            variants_tried += 1
            details["dare_scores"][str(drop_rate)] = score
            logger.info("  dare drop=%.2f  score=%.4f", drop_rate, score)

            if score > best_score:
                best_score = score
                best_variant = f"dare_{drop_rate:.2f}"
                best_path = variant_dir

        return OperationResult(
            operation="dare_drop_rescale",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )

    async def metrics_weighted_merge(
        self, adapter_path: str, baseline_score: float
    ) -> OperationResult:
        """Merge N checkpoints weighted by their eval scores.

        Uses ``checkpoint_paths`` and ``checkpoint_scores`` from the config.
        """
        paths = self.config.checkpoint_paths
        scores = self.config.checkpoint_scores

        if len(paths) < 2 or len(paths) != len(scores):
            return OperationResult(
                operation="metrics_weighted_merge",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "insufficient_checkpoints_or_scores"},
            )

        # Load all checkpoint weights
        all_weights = [_load_all_weights(p) for p in paths]
        merge_weights = _softmax(scores, temperature=1.0)

        # Weighted sum
        merged: Dict[str, torch.Tensor] = {}
        reference_keys = list(all_weights[0].keys())

        for key in reference_keys:
            tensors = []
            for w_dict in all_weights:
                if key in w_dict:
                    tensors.append(w_dict[key])
            if len(tensors) != len(paths):
                # Key not present in all checkpoints, skip interpolation
                merged[key] = all_weights[0][key].clone()
                continue
            weighted = sum(w * t for w, t in zip(merge_weights, tensors))
            merged[key] = weighted

        variant_dir = os.path.join(self._work_dir, "metrics_merge")
        _copy_adapter(adapter_path, variant_dir)
        _save_all_weights(variant_dir, merged)

        score = await self._evaluate(variant_dir)

        return OperationResult(
            operation="metrics_weighted_merge",
            variants_tried=1,
            best_variant="metrics_merge",
            best_score=score,
            improvement=score - baseline_score,
            adapter_path=variant_dir if score > baseline_score else adapter_path,
            details={"merge_weights": merge_weights, "score": score},
        )

    async def svd_rank_reduction(
        self, adapter_path: str, baseline_score: float
    ) -> OperationResult:
        """Compress LoRA via truncated SVD.

        For each LoRA pair (A, B), computes W = B @ A, performs SVD, and
        truncates to a lower rank. Tries several rank fractions.
        """
        weights = _load_all_weights(adapter_path)
        adapter_config = _load_adapter_config(adapter_path)
        original_rank = adapter_config.get("r", adapter_config.get("lora_rank", 16))

        if original_rank <= 1:
            return OperationResult(
                operation="svd_rank_reduction",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "rank_too_small"},
            )

        # Group LoRA A/B pairs
        lora_pairs = _find_lora_pairs(weights)

        if not lora_pairs:
            return OperationResult(
                operation="svd_rank_reduction",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_lora_pairs_found"},
            )

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"svd_scores": {}}

        for fraction in self.config.svd_rank_fractions:
            new_rank = max(1, int(round(original_rank * fraction)))
            if new_rank >= original_rank:
                continue

            variant_dir = os.path.join(
                self._work_dir, f"svd_rank{new_rank}"
            )
            _copy_adapter(adapter_path, variant_dir)

            modified_weights = dict(weights)
            for prefix, (a_key, b_key) in lora_pairs.items():
                a_tensor = weights[a_key]  # shape: (r, in_features)
                b_tensor = weights[b_key]  # shape: (out_features, r)

                # Compose: W = B @ A -> (out_features, in_features)
                w = b_tensor.float() @ a_tensor.float()
                u, s, vh = torch.linalg.svd(w, full_matrices=False)

                # Truncate to new_rank
                u_trunc = u[:, :new_rank]
                s_trunc = s[:new_rank]
                vh_trunc = vh[:new_rank, :]

                # Reconstruct as new A and B
                # New A = sqrt(S) @ Vh  -> (new_rank, in_features)
                # New B = U @ sqrt(S)   -> (out_features, new_rank)
                sqrt_s = torch.diag(torch.sqrt(s_trunc))
                new_a = (sqrt_s @ vh_trunc).to(a_tensor.dtype)
                new_b = (u_trunc @ sqrt_s).to(b_tensor.dtype)

                modified_weights[a_key] = new_a
                modified_weights[b_key] = new_b

            _save_all_weights(variant_dir, modified_weights)

            # Update adapter_config with new rank
            variant_config = _load_adapter_config(variant_dir)
            variant_config["r"] = new_rank
            _save_adapter_config(variant_dir, variant_config)

            score = await self._evaluate(variant_dir)
            variants_tried += 1
            details["svd_scores"][str(new_rank)] = score
            logger.info("  svd rank=%d  score=%.4f", new_rank, score)

            if score > best_score:
                best_score = score
                best_variant = f"svd_rank{new_rank}"
                best_path = variant_dir

        return OperationResult(
            operation="svd_rank_reduction",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )

    async def attention_mlp_ablation(
        self, adapter_path: str, baseline_score: float
    ) -> OperationResult:
        """Zero all attention LoRA vs all MLP LoRA, measure which matters more."""
        weights = _load_all_weights(adapter_path)

        has_attn = any(_is_attention_key(k) for k in weights)
        has_mlp = any(_is_mlp_key(k) for k in weights)

        if not has_attn and not has_mlp:
            return OperationResult(
                operation="attention_mlp_ablation",
                variants_tried=0,
                best_variant="none",
                best_score=baseline_score,
                improvement=0.0,
                adapter_path=adapter_path,
                details={"reason": "no_attention_or_mlp_keys"},
            )

        best_score = baseline_score
        best_variant = "original"
        best_path = adapter_path
        variants_tried = 0
        details: Dict[str, Any] = {"ablation_scores": {}}

        # Ablate attention
        if has_attn:
            variant_dir = os.path.join(self._work_dir, "ablate_attention")
            _copy_adapter(adapter_path, variant_dir)

            modified = {}
            for key, tensor in weights.items():
                if _is_attention_key(key):
                    modified[key] = torch.zeros_like(tensor)
                else:
                    modified[key] = tensor.clone()
            _save_all_weights(variant_dir, modified)

            score = await self._evaluate(variant_dir)
            variants_tried += 1
            details["ablation_scores"]["zero_attention"] = score
            logger.info("  ablate attention  score=%.4f", score)

            if score > best_score:
                best_score = score
                best_variant = "ablate_attention"
                best_path = variant_dir

        # Ablate MLP
        if has_mlp:
            variant_dir = os.path.join(self._work_dir, "ablate_mlp")
            _copy_adapter(adapter_path, variant_dir)

            modified = {}
            for key, tensor in weights.items():
                if _is_mlp_key(key):
                    modified[key] = torch.zeros_like(tensor)
                else:
                    modified[key] = tensor.clone()
            _save_all_weights(variant_dir, modified)

            score = await self._evaluate(variant_dir)
            variants_tried += 1
            details["ablation_scores"]["zero_mlp"] = score
            logger.info("  ablate mlp  score=%.4f", score)

            if score > best_score:
                best_score = score
                best_variant = "ablate_mlp"
                best_path = variant_dir

        return OperationResult(
            operation="attention_mlp_ablation",
            variants_tried=variants_tried,
            best_variant=best_variant,
            best_score=best_score,
            improvement=best_score - baseline_score,
            adapter_path=best_path,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _evaluate(self, adapter_path: str) -> float:
        """Evaluate an adapter using the configured eval backend."""
        result = await self.eval_backend.run_eval(adapter_path, self.eval_scenario)
        return result.eval_score

    async def _run_operation(
        self,
        method,
        operation_name: str,
        adapter_path: str,
        baseline_score: float,
    ) -> OperationResult:
        """Run a single surgery operation."""
        return await method(adapter_path, baseline_score)

    def _save_report(self, result: SurgeryResult) -> None:
        """Save surgery results as a JSON report."""
        report_path = os.path.join(self.config.output_dir, "surgery_report.json")
        report_data = {
            "baseline_score": result.baseline_score,
            "final_score": result.final_score,
            "total_improvement": result.total_improvement,
            "best_adapter_path": result.best_adapter_path,
            "duration_seconds": result.duration_seconds,
            "operations_applied": [asdict(op) for op in result.operations_applied],
        }
        with open(report_path, "w") as fh:
            json.dump(report_data, fh, indent=2)
        logger.info("Surgery report saved to %s", report_path)


def _find_lora_pairs(
    weights: Dict[str, "torch.Tensor"],
) -> Dict[str, tuple]:
    """Find matching LoRA A/B weight pairs.

    Returns:
        Dict mapping a common prefix to (a_key, b_key) tuples.
    """
    a_keys = {}
    b_keys = {}
    for key in weights:
        if ".lora_A." in key:
            prefix = key.split(".lora_A.")[0]
            a_keys[prefix] = key
        elif ".lora_B." in key:
            prefix = key.split(".lora_B.")[0]
            b_keys[prefix] = key

    pairs = {}
    for prefix in a_keys:
        if prefix in b_keys:
            pairs[prefix] = (a_keys[prefix], b_keys[prefix])

    return pairs
