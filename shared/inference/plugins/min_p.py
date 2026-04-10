"""Min-P sampling logits plugin.

Min-P is a dynamic nucleus-style filter that removes low-probability tokens
relative to the *maximum* probability in the distribution.  For each token
whose softmax probability is below ``max_prob * threshold``, the logit is
set to ``-inf`` so that downstream sampling never selects it.

This adapts automatically to the model's confidence: when the model is
certain (high max-prob) the cutoff is strict; when the model is uncertain
(low max-prob) the cutoff relaxes, preserving diversity.

Reference: https://arxiv.org/abs/2407.01082
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from shared.inference.base import BaseLogitsPlugin

if TYPE_CHECKING:
    from torch import Tensor

    from shared.inference.config import MinPConfig

logger = logging.getLogger(__name__)


class MinPPlugin(BaseLogitsPlugin):
    """Zero-out tokens whose probability falls below a dynamic threshold.

    Parameters
    ----------
    config:
        A ``MinPConfig`` with *threshold* (fraction of max probability
        below which tokens are masked).  Must be in (0, 1).
    """

    def __init__(self, config: MinPConfig) -> None:
        if not 0.0 < config.threshold < 1.0:
            raise ValueError(
                f"Min-P threshold must be in (0, 1), got {config.threshold}"
            )

        self._threshold: float = config.threshold

        logger.debug(
            "MinPPlugin initialised (threshold=%.4f)",
            self._threshold,
        )

    # ------------------------------------------------------------------
    # BaseLogitsPlugin interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "min_p"

    def __call__(self, token_ids: list[int], logits: Tensor) -> Tensor:
        """Mask logits whose softmax probability is below the dynamic cutoff.

        The cutoff equals ``max(softmax(logits)) * threshold``.  Masked
        logits are set to ``-inf`` so they receive zero probability after
        a subsequent softmax.

        *token_ids* is accepted for interface compatibility but not used
        by this plugin.
        """
        import torch

        # Compute probabilities via softmax (numerically stable).
        probs = torch.softmax(logits, dim=-1)

        # Dynamic cutoff: fraction of the peak probability.
        max_prob = probs.max()
        cutoff = max_prob * self._threshold

        # Build a mask of tokens to *keep* (prob >= cutoff).
        keep_mask = probs >= cutoff

        # Replace masked logits with -inf.
        logits = logits.masked_fill(~keep_mask, float("-inf"))

        return logits
