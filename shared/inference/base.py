"""
shared/inference/base.py

Abstract base classes for the two plugin types in the inference plugin system:

- **BaseLayerHookPlugin** — needs intermediate layer activations (via IBM
  vLLM Hook). Captures hidden states from hooked layers, then modifies the
  final logits before sampling.
- **BaseLogitsPlugin** — operates only on the final logits tensor. Compatible
  with vLLM's native ``LogitsProcessor`` interface and does not require the
  vLLM Hook extension.

All torch imports are behind ``TYPE_CHECKING`` so the module can be imported
without torch installed (for config-only usage, testing, etc.).

Used by: plugin implementations under shared/inference/plugins/,
         shared/inference/registry.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


class BaseLayerHookPlugin(ABC):
    """Plugin requiring intermediate layer access (via IBM vLLM Hook).

    Implementations capture hidden states from specific transformer layers,
    then modify the final logits before the sampling step.  This enables
    techniques like DoLa (contrastive decoding) and activation steering
    that need access to intermediate representations.

    Lifecycle per generation step:
        1. ``target_layers`` — declare which layers to hook.
        2. ``on_layer_output`` — called for each hooked layer with its output.
        3. ``modify_logits`` — called once with the final logits tensor.
        4. ``reset`` — called after sampling to clear per-step state.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name (e.g. ``'dola'``)."""

    @abstractmethod
    def target_layers(self, num_hidden_layers: int) -> list[int]:
        """Return layer indices to hook, given the model's total layer count.

        Args:
            num_hidden_layers: Number of hidden layers in the model.

        Returns:
            Sorted list of zero-based layer indices to intercept.
        """

    @abstractmethod
    def on_layer_output(self, layer_idx: int, hidden_states: Tensor) -> None:
        """Called when a hooked layer produces output.

        Store state internally for later use in :meth:`modify_logits`.

        Args:
            layer_idx: Zero-based index of the layer that produced the output.
            hidden_states: Tensor of shape ``(batch, seq_len, hidden_dim)``.
        """

    @abstractmethod
    def modify_logits(self, final_logits: Tensor, lm_head: Module) -> Tensor:
        """Modify final logits before sampling using captured layer outputs.

        Args:
            final_logits: Logits tensor of shape ``(batch, vocab_size)``
                produced by the language model head.
            lm_head: The model's language model head module, in case the
                plugin needs to project intermediate hidden states.

        Returns:
            Modified logits tensor of the same shape.
        """

    def reset(self) -> None:
        """Reset internal state between generation steps.

        Override if the plugin accumulates per-step state in
        :meth:`on_layer_output`.  The default implementation is a no-op.
        """


class BaseLogitsPlugin(ABC):
    """Plugin that only modifies final logits (vLLM native LogitsProcessor).

    Implementations receive the generated token IDs and the raw logits,
    and return modified logits.  This is compatible with vLLM's built-in
    ``LogitsProcessor`` interface and does not require the vLLM Hook
    extension.

    Examples: repetition penalty, Min-P filtering, temperature scaling.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name (e.g. ``'min_p'``)."""

    @abstractmethod
    def __call__(self, token_ids: list[int], logits: Tensor) -> Tensor:
        """Modify logits. Standard ``LogitsProcessor`` interface.

        Args:
            token_ids: Token IDs generated so far in this sequence.
            logits: Raw logits tensor of shape ``(vocab_size,)``.

        Returns:
            Modified logits tensor of the same shape.
        """
