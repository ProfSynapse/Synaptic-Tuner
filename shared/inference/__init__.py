"""
shared/inference/

vLLM inference plugin framework. Provides typed configuration, abstract
base classes for two plugin types (layer-hook and logits-only), a registry
for discovering and instantiating enabled plugins, and a top-level loader
for resolving config files and named profiles.

Used by: services/proxy, tuner handlers, evaluator, CLI.
"""

from .base import BaseLayerHookPlugin, BaseLogitsPlugin
from .config import (
    ActivationSteeringConfig,
    DoLaConfig,
    InferenceOverrides,
    InferencePluginConfig,
    MinPConfig,
    RepetitionPenaltyConfig,
    VLLMHookConfig,
)
from .loader import load_inference_config
from .registry import PluginRegistry

__all__ = [
    # Config
    "DoLaConfig",
    "ActivationSteeringConfig",
    "RepetitionPenaltyConfig",
    "MinPConfig",
    "VLLMHookConfig",
    "InferenceOverrides",
    "InferencePluginConfig",
    "load_inference_config",
    # Base classes
    "BaseLayerHookPlugin",
    "BaseLogitsPlugin",
    # Registry
    "PluginRegistry",
]
