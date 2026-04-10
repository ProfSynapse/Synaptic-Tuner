"""Bridge between the inference plugin system and PyTorch forward hooks.

:class:`LayerHookManager` registers PyTorch ``register_forward_hook``
callbacks on numbered transformer layers so that
:class:`~shared.inference.base.BaseLayerHookPlugin` instances receive
intermediate hidden states during the forward pass.

This module does **not** depend on IBM vLLM Hook being installed -- it
provides a generic hook registration mechanism that works with any
``torch.nn.Module``-based transformer whose layers are accessible via
``named_modules()`` (e.g. ``model.layers.0``, ``transformer.h.15``).

Typical lifecycle::

    registry = PluginRegistry(config)
    manager = LayerHookManager(registry.layer_hook_plugins, num_hidden_layers=32)
    manager.register_hooks(model)

    # After the forward pass produces final_logits:
    logits = manager.apply_plugins(final_logits, lm_head)
    manager.reset_plugins()

    # Cleanup when done:
    manager.remove_hooks()
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.hooks import RemovableHook

    from shared.inference.base import BaseLayerHookPlugin

logger = logging.getLogger(__name__)

# Common naming patterns for numbered transformer layers.
# Each pattern captures the layer index as its last group.
# More specific patterns are listed first to avoid false matches.
_LAYER_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\.layers\.(\d+)$"),              # LLaMA, Qwen, Mistral
    re.compile(r"\.h\.(\d+)$"),                   # GPT-2, GPT-Neo
    re.compile(r"\.block\.(\d+)$"),               # T5, Flan
    re.compile(r"\.transformer\.h\.(\d+)$"),      # Some GPT variants
    re.compile(r"\.encoder\.layer\.(\d+)$"),      # BERT-style
    re.compile(r"\.decoder\.layers\.(\d+)$"),     # Decoder-only variants
]

# Fallback: any trailing .<digits> preceded by a known keyword.
# Used only when none of the explicit patterns match.
_GENERIC_LAYER_PATTERN: re.Pattern[str] = re.compile(
    r"(?:layers|blocks|h)\.(\d+)$"
)


class LayerHookManager:
    """Manages PyTorch forward hooks for :class:`BaseLayerHookPlugin` instances.

    Registers ``register_forward_hook`` callbacks on transformer layers to
    capture intermediate hidden states, then applies plugin logit
    modifications after the model's forward pass.

    Works with any model that has numbered transformer layers accessible
    via ``named_modules()`` (e.g. ``model.layers.0``, ``model.layers.15``).

    Parameters
    ----------
    plugins:
        List of layer-hook plugins that will receive hidden states and
        modify final logits.
    num_hidden_layers:
        Total number of hidden layers in the model.  Passed to each
        plugin's ``target_layers()`` to resolve layer indices.
    """

    def __init__(
        self,
        plugins: list[BaseLayerHookPlugin],
        num_hidden_layers: int,
    ) -> None:
        if not plugins:
            raise ValueError("LayerHookManager requires at least one plugin")
        if num_hidden_layers < 1:
            raise ValueError(
                f"num_hidden_layers must be >= 1, got {num_hidden_layers}"
            )

        self._plugins = list(plugins)
        self._num_hidden_layers = num_hidden_layers
        self._hooks: list[RemovableHook] = []

        # Build mapping: layer_idx -> list of plugins that target it.
        self._layer_to_plugins: dict[int, list[BaseLayerHookPlugin]] = {}
        self._target_layers: set[int] = set()

        for plugin in self._plugins:
            layers = plugin.target_layers(num_hidden_layers)
            self._target_layers.update(layers)
            for idx in layers:
                self._layer_to_plugins.setdefault(idx, []).append(plugin)

        logger.info(
            "LayerHookManager: %d plugin(s), hooking %d unique layer(s) "
            "out of %d total",
            len(self._plugins),
            len(self._target_layers),
            num_hidden_layers,
        )

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self, model: Module) -> None:
        """Register forward hooks on target layers of the model.

        Iterates ``model.named_modules()`` to find transformer layers by
        naming convention, then attaches a forward hook to each target
        layer.  The hook extracts hidden states from the module output and
        dispatches them to all plugins targeting that layer.

        Args:
            model: The full model (e.g. a HuggingFace ``PreTrainedModel``
                or a vLLM model wrapper).

        Raises:
            RuntimeError: If hooks are already registered (call
                :meth:`remove_hooks` first) or if target layers could not
                be found in the model.
        """
        if self._hooks:
            raise RuntimeError(
                "Hooks are already registered. Call remove_hooks() before "
                "re-registering."
            )

        hooked_layers: set[int] = set()

        for module_name, module in model.named_modules():
            layer_idx = self._extract_layer_index(module_name)
            if layer_idx is None or layer_idx not in self._target_layers:
                continue

            # Avoid double-hooking the same layer (could happen if nested
            # modules match at different depths).
            if layer_idx in hooked_layers:
                continue

            plugins_for_layer = self._layer_to_plugins[layer_idx]
            hook = module.register_forward_hook(
                self._make_hook_fn(layer_idx, plugins_for_layer),
            )
            self._hooks.append(hook)
            hooked_layers.add(layer_idx)

            logger.debug(
                "Registered forward hook on layer %d (%s) for %d plugin(s)",
                layer_idx,
                module_name,
                len(plugins_for_layer),
            )

        missing = self._target_layers - hooked_layers
        if missing:
            raise RuntimeError(
                f"Could not find model modules for target layer(s): "
                f"{sorted(missing)}. Searched all named_modules() using "
                f"standard transformer naming patterns. The model may use "
                f"an unsupported layer naming convention."
            )

        logger.info(
            "Successfully registered %d forward hook(s) on layers %s",
            len(self._hooks),
            sorted(hooked_layers),
        )

    # ------------------------------------------------------------------
    # Plugin application
    # ------------------------------------------------------------------

    def apply_plugins(self, final_logits: Tensor, lm_head: Module) -> Tensor:
        """Apply all layer-hook plugins to modify the final logits.

        Plugins are applied in the order they were passed to the
        constructor.  Each plugin's ``modify_logits`` receives the output
        of the previous plugin.

        Args:
            final_logits: Logits tensor of shape ``(batch, vocab_size)``
                produced by the model's LM head.
            lm_head: The model's LM head module.

        Returns:
            Modified logits tensor of the same shape.
        """
        logits = final_logits
        for plugin in self._plugins:
            logits = plugin.modify_logits(logits, lm_head)
        return logits

    def reset_plugins(self) -> None:
        """Reset all plugins between generation steps.

        Call this after sampling to clear any per-step state accumulated
        during the forward pass (e.g. captured hidden states).
        """
        for plugin in self._plugins:
            plugin.reset()

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks from the model.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if not self._hooks:
            return

        count = len(self._hooks)
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        logger.info("Removed %d forward hook(s)", count)

    @property
    def is_active(self) -> bool:
        """Return ``True`` if hooks are currently registered."""
        return len(self._hooks) > 0

    @property
    def num_hooks(self) -> int:
        """Number of currently registered hooks."""
        return len(self._hooks)

    @property
    def target_layer_indices(self) -> set[int]:
        """Set of all layer indices being hooked."""
        return set(self._target_layers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hook_fn(
        layer_idx: int,
        plugins: list[BaseLayerHookPlugin],
    ):
        """Create a forward hook callback for a specific layer.

        The returned callable has the signature expected by PyTorch's
        ``register_forward_hook``: ``(module, input, output) -> None``.

        The hook extracts the hidden-state tensor from the module output
        (handling both raw tensors and tuple outputs) and dispatches it
        to each plugin targeting this layer.

        Args:
            layer_idx: The zero-based layer index.
            plugins: Plugins to notify when this layer produces output.

        Returns:
            A hook callback function.
        """

        def hook_fn(module: Module, input: tuple, output) -> None:
            # Transformer layer outputs vary by architecture:
            #   - Some return a plain tensor (hidden_states)
            #   - Some return a tuple (hidden_states, attention_weights, ...)
            #   - Some return a tuple (hidden_states,) with length 1
            # In all cases, the hidden states are the first element.
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            for plugin in plugins:
                try:
                    plugin.on_layer_output(layer_idx, hidden_states)
                except Exception:
                    logger.exception(
                        "Plugin %r raised an error in on_layer_output "
                        "for layer %d",
                        plugin.name,
                        layer_idx,
                    )

        return hook_fn

    @staticmethod
    def _extract_layer_index(module_name: str) -> int | None:
        """Extract a layer index from a module name.

        Recognises common transformer naming conventions:

        - ``model.layers.15`` (LLaMA, Qwen, Mistral)
        - ``transformer.h.7`` (GPT-2, GPT-Neo)
        - ``encoder.block.3`` (T5, Flan)
        - ``decoder.layers.5``

        Falls back to a generic pattern matching ``layers.N``, ``blocks.N``,
        or ``h.N`` at the end of the name.

        Args:
            module_name: Fully qualified module name from
                ``model.named_modules()``.

        Returns:
            Zero-based layer index, or ``None`` if the name does not
            match any known pattern.
        """
        # Try explicit patterns first
        for pattern in _LAYER_NAME_PATTERNS:
            match = pattern.search(module_name)
            if match:
                return int(match.group(1))

        # Fallback: generic pattern
        match = _GENERIC_LAYER_PATTERN.search(module_name)
        if match:
            return int(match.group(1))

        return None
