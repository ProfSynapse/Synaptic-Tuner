"""
Readouts that turn torso hidden states into one logit per option.

Location: Trainers/decision/decision_core/readouts.py
Used by:  modeling.py, losses.py.

- ``pointer`` (strands-decider v19): a ~1M-parameter attention score between
  the ``<answer>`` hidden state (query) and each option line's last-token hidden
  state (key). No per-slot parameters, so the option count is unbounded.
- ``letter_logits`` (OpenJev-style): no new parameters. Option k's logit is the
  LM's own next-token logit for option k's marker ("1".."9", "A".."Z") at the
  ``<answer>`` position, computed from only those rows of the output embedding.

Both feed the same masked softmax, loss, and temperature calibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Large negative rather than -inf keeps softmax finite under bf16 autocast.
MASK_VALUE = -1e4


class PointerHead(nn.Module):
    """logit_k = <q(norm(h_answer)), k(norm(h_option_k))> / sqrt(dim), in fp32."""

    def __init__(self, hidden_size: int, dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.q = nn.Linear(hidden_size, dim)
        self.k = nn.Linear(hidden_size, dim)
        self.scale = dim ** -0.5

    def forward(self, answer: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """answer [B, d], options [B, K, d] -> logits [B, K]."""
        q = self.q(self.dropout(self.norm(answer))).unsqueeze(-1)
        k = self.k(self.dropout(self.norm(options)))
        return (k @ q).squeeze(-1) * self.scale


def gather_positions(hidden: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """hidden [B, L, d] + index [B, K] (-1 = padding) -> [B, K, d]."""
    idx = index.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    return hidden.gather(1, idx)


def gather_answer(hidden: torch.Tensor, answer_index: torch.Tensor) -> torch.Tensor:
    """hidden [B, L, d] + answer_index [B] -> [B, d]."""
    return hidden[torch.arange(hidden.size(0), device=hidden.device), answer_index]


def marker_logits(
    answer: torch.Tensor,
    output_weight: torch.Tensor,
    marker_ids: torch.Tensor,
    width: int,
    output_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """LM logits of the first ``width`` marker tokens at the answer position -> [B, width]."""
    if width > marker_ids.numel():
        raise ValueError(
            f"letter_logits readout has {marker_ids.numel()} single-token markers but the batch "
            f"needs {width}; drop or split rows with more options"
        )
    ids = marker_ids[:width].to(output_weight.device)
    w = output_weight.index_select(0, ids).to(torch.float32)
    logits = answer.to(torch.float32) @ w.T
    if output_bias is not None:
        logits = logits + output_bias.index_select(0, ids).to(torch.float32)
    return logits


def mask_logits(logits: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
    valid = torch.arange(logits.size(-1), device=logits.device).unsqueeze(0) < n_options.unsqueeze(1)
    return logits.masked_fill(~valid, MASK_VALUE)


def masked_log_softmax(logits: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
    """Log-softmax over each row's first n_options entries; padded slots get ~0 mass."""
    return F.log_softmax(mask_logits(logits.float(), n_options), dim=-1)


def per_row_temperature(kinds: torch.Tensor, temperature_by_kind: dict[str, float],
                        default: float = 1.0) -> torch.Tensor:
    """Temperature tensor [B] from each row's kind id (index into examples.KINDS)."""
    from .examples import KINDS

    table = torch.tensor(
        [float(temperature_by_kind.get(k, default)) for k in KINDS],
        dtype=torch.float32,
        device=kinds.device,
    )
    return table[kinds]
