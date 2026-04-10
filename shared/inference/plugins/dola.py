"""Decoding by Contrasting Layers (DoLa) plugin.

DoLa improves factuality by contrasting the output distributions of a mature
(final) layer with premature (earlier) layers.  The intuition is that factual
knowledge emerges in later transformer layers, so amplifying the difference
between early and late representations suppresses hallucinated content.

Algorithm per token:
    1. Capture hidden states from premature layers via forward hooks.
    2. Project each premature hidden state through the LM head to obtain
       premature logits.
    3. Compute the Jensen-Shannon divergence (JSD) between the mature
       (final) and each premature distribution.
    4. Select the premature layer with the highest mean JSD (the one that
       *most disagrees* with the mature layer).
    5. Subtract the selected premature log-probs from the mature log-probs
       to produce contrastive logits.
    6. Apply relative-top filtering to suppress low-confidence tokens.

If no premature layer exceeds ``jsd_threshold``, the original logits are
returned unchanged (the model is confident and contrastive decoding is
unnecessary).

Reference: Chuang et al., "DoLa: Decoding by Contrasting Layers Improves
Factuality in Large Language Models" (ICLR 2024).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from shared.inference.base import BaseLayerHookPlugin
from shared.inference.config import DoLaConfig

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

logger = logging.getLogger(__name__)


class DoLaPlugin(BaseLayerHookPlugin):
    """Layer-hook plugin implementing DoLa contrastive decoding.

    Parameters
    ----------
    config:
        A :class:`DoLaConfig` specifying premature layer selection strategy,
        relative-top filtering threshold, and JSD gating threshold.
    """

    def __init__(self, config: DoLaConfig) -> None:
        self._premature_layers_spec = config.premature_layers
        self._mature_layer = config.mature_layer
        self._relative_top = config.relative_top
        self._jsd_threshold = config.jsd_threshold

        # Captured hidden states: layer_idx -> tensor (batch, seq_len, hidden_dim)
        self._captured_states: dict[int, Tensor] = {}

        if self._relative_top <= 0.0 or self._relative_top > 1.0:
            raise ValueError(
                f"relative_top must be in (0, 1], got {self._relative_top}"
            )
        if self._jsd_threshold < 0.0:
            raise ValueError(
                f"jsd_threshold must be >= 0, got {self._jsd_threshold}"
            )

        logger.debug(
            "DoLaPlugin initialised (premature_layers=%r, relative_top=%.3f, "
            "jsd_threshold=%.4f)",
            self._premature_layers_spec,
            self._relative_top,
            self._jsd_threshold,
        )

    # ------------------------------------------------------------------
    # BaseLayerHookPlugin interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "dola"

    def target_layers(self, num_hidden_layers: int) -> list[int]:
        """Resolve the premature layer specification to concrete indices.

        ``"low"`` selects even-spaced layers in the lower half,
        ``"high"`` selects even-spaced layers in the upper half,
        and an explicit ``list[int]`` is used as-is after validation.

        Args:
            num_hidden_layers: Total number of hidden layers in the model.

        Returns:
            Sorted list of zero-based layer indices to hook.

        Raises:
            ValueError: If *premature_layers* is an unrecognised string or
                contains out-of-range indices.
        """
        spec = self._premature_layers_spec

        if isinstance(spec, list):
            # Validate explicit indices
            for idx in spec:
                if idx < 0 or idx >= num_hidden_layers:
                    raise ValueError(
                        f"Premature layer index {idx} out of range "
                        f"[0, {num_hidden_layers - 1}]"
                    )
            layers = sorted(set(spec))
        elif spec == "low":
            half = num_hidden_layers // 2
            layers = list(range(0, half, 2))
        elif spec == "high":
            half = num_hidden_layers // 2
            layers = list(range(half, num_hidden_layers - 1, 2))
        else:
            raise ValueError(
                f"Unknown premature_layers spec: {spec!r}. "
                f"Expected 'low', 'high', or a list of ints."
            )

        if not layers:
            raise ValueError(
                f"Premature layer spec {spec!r} resolved to an empty list "
                f"for a model with {num_hidden_layers} layers."
            )

        logger.debug(
            "DoLa target layers resolved: %s (from spec=%r, model layers=%d)",
            layers,
            spec,
            num_hidden_layers,
        )

        return layers

    def on_layer_output(self, layer_idx: int, hidden_states: Tensor) -> None:
        """Store the hidden states from a hooked premature layer.

        Args:
            layer_idx: Zero-based index of the layer that produced output.
            hidden_states: Tensor of shape ``(batch, seq_len, hidden_dim)``.
        """
        self._captured_states[layer_idx] = hidden_states

    def modify_logits(self, final_logits: Tensor, lm_head: Module) -> Tensor:
        """Apply DoLa contrastive decoding to the final logits.

        Selects the premature layer whose distribution diverges most from
        the mature distribution (highest JSD), then subtracts its log-probs
        from the mature log-probs.  If no layer exceeds the JSD threshold,
        returns the original logits unchanged.

        Args:
            final_logits: Logits of shape ``(batch, vocab_size)`` from
                the model's LM head.
            lm_head: The LM head module for projecting premature hidden
                states to vocabulary space.

        Returns:
            Modified logits of the same shape.
        """
        import torch

        if not self._captured_states:
            logger.debug("DoLa: no captured states, returning logits unchanged")
            return final_logits

        try:
            return self._contrast_logits(final_logits, lm_head)
        except Exception:
            logger.exception("DoLa: error in contrastive decoding, returning original logits")
            return final_logits
        finally:
            self._captured_states.clear()

    def reset(self) -> None:
        """Clear captured hidden states between generation steps."""
        self._captured_states.clear()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _contrast_logits(
        self, final_logits: Tensor, lm_head: Module,
    ) -> Tensor:
        """Core DoLa algorithm: select best premature layer and contrast.

        Args:
            final_logits: Mature logits ``(batch, vocab_size)``.
            lm_head: LM head module.

        Returns:
            Contrastive logits or original logits if JSD is below threshold.
        """
        import torch
        import torch.nn.functional as F

        # Step 1: Compute mature log-probabilities
        mature_log_probs = F.log_softmax(final_logits, dim=-1)

        best_jsd = -1.0
        best_premature_log_probs: Tensor | None = None

        # Step 2: Find the premature layer with highest JSD
        for layer_idx, hidden_states in self._captured_states.items():
            premature_log_probs = self._project_to_logits(
                hidden_states, lm_head,
            )
            jsd_value = self._compute_jsd(mature_log_probs, premature_log_probs)

            if jsd_value > best_jsd:
                best_jsd = jsd_value
                best_premature_log_probs = premature_log_probs

        # Step 3: Gate on JSD threshold
        if best_jsd <= self._jsd_threshold or best_premature_log_probs is None:
            logger.debug(
                "DoLa: best JSD %.6f <= threshold %.6f, returning original logits",
                best_jsd,
                self._jsd_threshold,
            )
            return final_logits

        logger.debug("DoLa: best premature JSD=%.6f, applying contrastive decoding", best_jsd)

        # Step 4: Contrastive logits = mature_log_probs - premature_log_probs
        contrastive_logits = mature_log_probs - best_premature_log_probs

        # Step 5: Apply relative-top filtering
        contrastive_logits = self._apply_relative_top(contrastive_logits)

        return contrastive_logits

    def _project_to_logits(
        self, hidden_states: Tensor, lm_head: Module,
    ) -> Tensor:
        """Project premature hidden states through the LM head.

        Uses only the last token position (matching vLLM's single-token
        generation convention) and runs in ``no_grad`` mode since we do
        not need gradients for inference-time contrastive decoding.

        Args:
            hidden_states: Tensor of shape ``(batch, seq_len, hidden_dim)``.
            lm_head: The model's language model head module.

        Returns:
            Log-softmax of premature logits, shape ``(batch, vocab_size)``.
        """
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            # Use last token position to match the mature logits
            if hidden_states.dim() == 3:
                last_hidden = hidden_states[:, -1:, :]  # (batch, 1, hidden_dim)
            else:
                # Already (batch, hidden_dim) or (1, hidden_dim)
                last_hidden = hidden_states

            premature_logits = lm_head(last_hidden)

            # Squeeze the seq_len dimension if present
            if premature_logits.dim() == 3:
                premature_logits = premature_logits.squeeze(1)  # (batch, vocab_size)

            return F.log_softmax(premature_logits, dim=-1)

    @staticmethod
    def _compute_jsd(
        log_probs_p: Tensor, log_probs_q: Tensor,
    ) -> float:
        """Compute mean Jensen-Shannon Divergence between two distributions.

        Both inputs should be log-probabilities (output of ``log_softmax``).

        .. math::

            M = 0.5 (P + Q)
            JSD(P \\| Q) = 0.5 KL(P \\| M) + 0.5 KL(Q \\| M)

        Args:
            log_probs_p: Log-probs from the mature layer ``(batch, vocab_size)``.
            log_probs_q: Log-probs from a premature layer ``(batch, vocab_size)``.

        Returns:
            Mean JSD across the batch (scalar float).
        """
        import torch

        with torch.no_grad():
            # Convert to probabilities for computing the mixture M
            p = log_probs_p.exp()
            q = log_probs_q.exp()

            # Mixture distribution M = 0.5 * (P + Q)
            m = 0.5 * (p + q)
            log_m = m.log()

            # KL(P || M) = sum(P * (log P - log M))
            kl_p_m = (p * (log_probs_p - log_m)).sum(dim=-1)
            # KL(Q || M) = sum(Q * (log Q - log M))
            kl_q_m = (q * (log_probs_q - log_m)).sum(dim=-1)

            # JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
            jsd = 0.5 * kl_p_m + 0.5 * kl_q_m

            # Clamp to avoid numerical issues (JSD is non-negative)
            jsd = jsd.clamp(min=0.0)

            return jsd.mean().item()

    def _apply_relative_top(self, logits: Tensor) -> Tensor:
        """Apply relative-top filtering to suppress low-confidence tokens.

        Tokens whose logit falls below ``max_logit + log(relative_top)`` are
        set to ``-inf``.  This is analogous to top-p but operates in log-space
        relative to the peak, providing a cleaner cutoff for contrastive
        logits which can have unusual distributions.

        Args:
            logits: Contrastive logits ``(batch, vocab_size)``.

        Returns:
            Filtered logits of the same shape.
        """
        import torch

        # Compute the cutoff: tokens must be within log(relative_top) of
        # the maximum logit in each batch element.
        log_threshold = math.log(self._relative_top)
        max_logits = logits.max(dim=-1, keepdim=True).values
        cutoff = max_logits + log_threshold

        # Mask tokens below the cutoff
        mask = logits < cutoff
        logits = logits.masked_fill(mask, float("-inf"))

        return logits
