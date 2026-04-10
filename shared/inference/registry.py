"""
shared/inference/registry.py

Plugin registry that discovers and instantiates plugins from an
InferencePluginConfig.  Enabled plugins are lazily imported and
constructed during ``__init__`` so that only the dependencies of
active plugins need to be installed.

Used by: vLLM Hook integration, services/proxy, evaluator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import InferencePluginConfig

if TYPE_CHECKING:
    from .base import BaseLayerHookPlugin, BaseLogitsPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry that instantiates plugins from :class:`InferencePluginConfig`.

    Construction is eager: all enabled plugins are discovered and created
    during ``__init__``.  Individual plugin modules are imported lazily so
    that torch (or other heavy deps) is only pulled in when a plugin that
    needs it is actually enabled.

    Example::

        cfg = InferencePluginConfig.from_profile("factual")
        registry = PluginRegistry(cfg)
        for name in registry.active_plugin_names:
            print(f"Active: {name}")
    """

    def __init__(self, config: InferencePluginConfig) -> None:
        self.config = config
        self._layer_hook_plugins: list[BaseLayerHookPlugin] = []
        self._logits_plugins: list[BaseLogitsPlugin] = []
        self._discover_plugins()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_plugins(self) -> None:
        """Instantiate all enabled plugins from config.

        Each plugin is imported lazily from ``shared.inference.plugins.*``
        so only the required modules are loaded.
        """
        # --- Layer-access plugins (require vLLM Hook) ---

        if self.config.dola.enabled:
            from .plugins.dola import DoLaPlugin

            plugin = DoLaPlugin(self.config.dola)
            self._layer_hook_plugins.append(plugin)
            logger.info("Registered layer-hook plugin: %s", plugin.name)

        if self.config.activation_steering.enabled:
            from .plugins.activation_steering import ActivationSteeringPlugin

            plugin = ActivationSteeringPlugin(self.config.activation_steering)
            self._layer_hook_plugins.append(plugin)
            logger.info("Registered layer-hook plugin: %s", plugin.name)

        # --- Logits-only plugins (native vLLM LogitsProcessor) ---

        if self.config.repetition_penalty.enabled:
            from .plugins.repetition import RepetitionPenaltyPlugin

            plugin = RepetitionPenaltyPlugin(self.config.repetition_penalty)
            self._logits_plugins.append(plugin)
            logger.info("Registered logits plugin: %s", plugin.name)

        if self.config.min_p.enabled:
            from .plugins.min_p import MinPPlugin

            plugin = MinPPlugin(self.config.min_p)
            self._logits_plugins.append(plugin)
            logger.info("Registered logits plugin: %s", plugin.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def layer_hook_plugins(self) -> list[BaseLayerHookPlugin]:
        """All active plugins that require intermediate layer access."""
        return list(self._layer_hook_plugins)

    @property
    def logits_plugins(self) -> list[BaseLogitsPlugin]:
        """All active logits-only plugins."""
        return list(self._logits_plugins)

    @property
    def needs_layer_hooks(self) -> bool:
        """Return ``True`` if any active plugin requires layer hooks."""
        return len(self._layer_hook_plugins) > 0

    @property
    def active_plugin_names(self) -> list[str]:
        """Sorted list of names of all active plugins."""
        names = [p.name for p in self._layer_hook_plugins]
        names.extend(p.name for p in self._logits_plugins)
        return sorted(names)
