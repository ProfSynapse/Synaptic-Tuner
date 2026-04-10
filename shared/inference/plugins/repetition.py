"""Windowed repetition-penalty logits plugin.

Applies a multiplicative penalty to tokens that have already appeared in a
recent sliding window of generated token IDs.  The behaviour matches the
HuggingFace ``repetition_penalty`` convention:

* logit > 0 → divide by *penalty*
* logit < 0 → multiply by *penalty*

This discourages the model from repeating tokens while leaving the rest of
the distribution untouched.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from shared.inference.base import BaseLogitsPlugin

if TYPE_CHECKING:
    from torch import Tensor

    from shared.inference.config import RepetitionPenaltyConfig

logger = logging.getLogger(__name__)


class RepetitionPenaltyPlugin(BaseLogitsPlugin):
    """Penalise recently-generated tokens so the model avoids repetition.

    Parameters
    ----------
    config:
        A ``RepetitionPenaltyConfig`` with *penalty* (multiplicative factor,
        ≥ 1.0) and *window* (how many recent tokens to consider).
    """

    def __init__(self, config: RepetitionPenaltyConfig) -> None:
        if config.penalty < 1.0:
            raise ValueError(
                f"Repetition penalty must be >= 1.0, got {config.penalty}"
            )
        if config.window < 1:
            raise ValueError(
                f"Window size must be >= 1, got {config.window}"
            )

        self._penalty: float = config.penalty
        self._window: int = config.window

        logger.debug(
            "RepetitionPenaltyPlugin initialised (penalty=%.3f, window=%d)",
            self._penalty,
            self._window,
        )

    # ------------------------------------------------------------------
    # BaseLogitsPlugin interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "repetition_penalty"

    def __call__(self, token_ids: list[int], logits: Tensor) -> Tensor:
        """Apply windowed repetition penalty to *logits* in-place.

        Only tokens that appear in the last ``window`` entries of
        *token_ids* are affected.  A penalty of 1.0 is a no-op.
        """
        import torch

        if not token_ids or self._penalty == 1.0:
            return logits

        # Slice the window — take at most the last `window` tokens.
        window_ids = token_ids[-self._window :]

        # De-duplicate so each token is penalised once per call, and
        # convert to a tensor index for efficient gather/scatter.
        unique_ids = torch.tensor(
            list(set(window_ids)),
            dtype=torch.long,
            device=logits.device,
        )

        # Gather the logits for the tokens that appeared in the window.
        selected_logits = logits[unique_ids]

        # Apply the HuggingFace-style repetition penalty:
        #   positive logits  → divide by penalty  (make less attractive)
        #   negative logits  → multiply by penalty (make more repulsive)
        penalised = torch.where(
            selected_logits > 0,
            selected_logits / self._penalty,
            selected_logits * self._penalty,
        )

        # Scatter the penalised values back.
        logits[unique_ids] = penalised

        return logits
