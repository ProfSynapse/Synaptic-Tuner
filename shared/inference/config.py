"""
shared/inference/config.py

Typed dataclass configuration for the vLLM inference plugin system.
Mirrors the YAML structure in configs/inference/default.yaml.
Each plugin section maps to a dedicated dataclass; the top-level
InferencePluginConfig composes them and provides YAML loading with
profile inheritance (``extends: default``).

Used by: shared/inference/registry.py, shared/inference/loader.py,
         tuner handlers, services/proxy.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from shared.utilities import load_yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path("configs/inference")


# ---------------------------------------------------------------------------
# Per-plugin configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DoLaConfig:
    """Config for Decoding by Contrasting Layers (DoLa).

    Layer-access plugin — requires vLLM Hook for intermediate activations.
    ``premature_layers`` can be ``"low"`` (lower half), ``"high"`` (upper half),
    or an explicit list of layer indices like ``[2, 4, 6, 8]``.
    """

    enabled: bool = False
    premature_layers: str | list[int] = "high"
    mature_layer: int | None = None
    relative_top: float = 0.1
    jsd_threshold: float = 0.0


@dataclass
class ActivationSteeringConfig:
    """Config for activation steering (representation engineering).

    Layer-access plugin — injects steering vectors at specified layers.
    """

    enabled: bool = False
    vectors_path: str | None = None
    scale: float = 1.0
    target_layers: list[int] = field(default_factory=list)


@dataclass
class RepetitionPenaltyConfig:
    """Config for token-level repetition penalty.

    Logits-only plugin — applies penalty to recently generated tokens.
    """

    enabled: bool = False
    penalty: float = 1.1
    window: int = 64


@dataclass
class MinPConfig:
    """Config for Min-P sampling.

    Logits-only plugin — suppresses tokens below a fraction of the
    top token's probability.
    """

    enabled: bool = False
    threshold: float = 0.05


@dataclass
class VLLMHookConfig:
    """Settings for the IBM vLLM Hook integration.

    Auto-enabled when any ``BaseLayerHookPlugin`` is active.
    """

    enabled: bool = False
    registry_port: int = 9090
    log_activations: bool = False
    activation_log_dir: str = "scratch/activations"


@dataclass
class InferenceOverrides:
    """Global inference parameter overrides.

    Applied after all plugin processing. ``None`` means "use the value
    from the incoming request".
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

@dataclass
class InferencePluginConfig:
    """Top-level configuration for the inference plugin system.

    Composes per-plugin config dataclasses.  Supports loading from a YAML
    file or from a named profile with ``extends`` inheritance.

    Examples::

        # Load directly from a YAML path
        cfg = InferencePluginConfig.from_yaml("configs/inference/default.yaml")

        # Load a named profile (merges on top of its base)
        cfg = InferencePluginConfig.from_profile("factual")

        # Build from a dict (e.g. from an API request)
        cfg = InferencePluginConfig.from_dict({"plugins": {"dola": {"enabled": True}}})
    """

    dola: DoLaConfig = field(default_factory=DoLaConfig)
    activation_steering: ActivationSteeringConfig = field(
        default_factory=ActivationSteeringConfig,
    )
    repetition_penalty: RepetitionPenaltyConfig = field(
        default_factory=RepetitionPenaltyConfig,
    )
    min_p: MinPConfig = field(default_factory=MinPConfig)
    vllm_hook: VLLMHookConfig = field(default_factory=VLLMHookConfig)
    inference: InferenceOverrides = field(default_factory=InferenceOverrides)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferencePluginConfig:
        """Build config from a dictionary (e.g. parsed YAML).

        Handles the ``plugins:`` nesting from the YAML schema, gracefully
        ignores unknown keys, and coerces types where needed.

        Args:
            data: Raw dictionary, typically from ``yaml.safe_load``.

        Returns:
            Fully populated ``InferencePluginConfig``.
        """
        plugins = data.get("plugins", {})

        dola = _build_sub_config(DoLaConfig, plugins.get("dola", {}))
        activation_steering = _build_sub_config(
            ActivationSteeringConfig, plugins.get("activation_steering", {}),
        )
        repetition_penalty = _build_sub_config(
            RepetitionPenaltyConfig, plugins.get("repetition_penalty", {}),
        )
        min_p = _build_sub_config(MinPConfig, plugins.get("min_p", {}))
        vllm_hook = _build_sub_config(
            VLLMHookConfig, data.get("vllm_hook", {}),
        )
        inference = _build_sub_config(
            InferenceOverrides, data.get("inference", {}),
        )

        return cls(
            dola=dola,
            activation_steering=activation_steering,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            vllm_hook=vllm_hook,
            inference=inference,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> InferencePluginConfig:
        """Load config from a YAML file.

        Args:
            path: Path to a YAML configuration file.

        Returns:
            Populated ``InferencePluginConfig``.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        raw = load_yaml(Path(path))
        return cls.from_dict(raw)

    @classmethod
    def from_profile(
        cls,
        profile_name: str,
        base_dir: str | Path = _DEFAULT_CONFIG_DIR,
    ) -> InferencePluginConfig:
        """Load a named profile, resolving ``extends`` inheritance.

        If the profile YAML contains ``extends: default`` (or another name),
        the base config is loaded first and the profile's values are merged
        on top.  Only fields specified in the profile override the base —
        all other fields retain their base values.

        Args:
            profile_name: Profile filename (without ``.yaml``) inside
                ``<base_dir>/profiles/``.
            base_dir: Root directory containing ``default.yaml`` and
                ``profiles/``.

        Returns:
            Merged ``InferencePluginConfig``.

        Raises:
            FileNotFoundError: If the profile YAML cannot be found.
        """
        base_dir = Path(base_dir)
        profile_path = base_dir / "profiles" / f"{profile_name}.yaml"
        profile_raw = load_yaml(profile_path)

        extends = profile_raw.pop("extends", None)
        if extends:
            base_path = base_dir / f"{extends}.yaml"
            base_raw = load_yaml(base_path)
            merged = _deep_merge(base_raw, profile_raw)
        else:
            merged = profile_raw

        return cls.from_dict(merged)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def needs_layer_hooks(self) -> bool:
        """Return ``True`` if any layer-access plugin is enabled."""
        return self.dola.enabled or self.activation_steering.enabled

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Produces the same nested structure expected by :meth:`from_dict`.
        """
        return {
            "plugins": {
                "dola": asdict(self.dola),
                "activation_steering": asdict(self.activation_steering),
                "repetition_penalty": asdict(self.repetition_penalty),
                "min_p": asdict(self.min_p),
            },
            "vllm_hook": asdict(self.vllm_hook),
            "inference": asdict(self.inference),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sub_config(cls: type, raw: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown keys.

    Args:
        cls: The dataclass type to instantiate.
        raw: Dictionary of values (may contain extra keys).

    Returns:
        Instance of *cls* with known fields populated.
    """
    if not raw:
        return cls()

    known_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    return cls(**filtered)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*.

    Nested dicts are merged recursively; all other values are replaced.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
