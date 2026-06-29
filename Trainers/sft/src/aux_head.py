"""
Auxiliary scalar readout head for SFT training.

Location: Trainers/sft/src/aux_head.py

Summary
-------
A small, generic, optional ``nn.Module`` that reads one hidden-layer activation
from a base model and emits a single scalar in ``[0, 1]`` per row. It is trained
by a proper-scoring loss against a per-row target while the base model stays
frozen (Phase A). The head is portable: it is saved as a standalone sidecar
artifact (``aux_head.safetensors`` + ``aux_head_config.json``) and reloaded for
inference independently of the base checkpoint.

How it is used
--------------
- ``aux_head_trainer.AuxHeadTrainer`` constructs an ``AuxHead`` (input_dim =
  base hidden size), freezes the base, and calls ``reduce_hidden_states`` +
  ``compute_aux_head_loss`` inside its ``compute_loss`` override.
- ``train_sft.run`` wires the head behind ``config.aux_head.enabled`` and calls
  ``save_aux_head`` after training.
- Downstream callers reload via ``load_aux_head`` and score inputs with
  ``infer_aux_scalar``.

This module is intentionally task-neutral: the target is whatever per-row
column the config names. No decision policy / threshold is baked in here — how
the scalar is consumed is the caller's concern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


VALID_HEAD_TYPES = ("linear", "mlp")
VALID_LOSSES = ("bce", "brier")
VALID_OUT_ACTIVATIONS = ("sigmoid", "identity")


class AuxHead(nn.Module):
    """A tiny scalar readout head over a single hidden-state vector.

    Args:
        input_dim: Width of the hidden state the head reads (base hidden size).
        head_type: ``"linear"`` (default; a single ``nn.Linear(input_dim, 1)``)
            or ``"mlp"`` (stacked ``Linear`` + ``GELU`` blocks per ``hidden_dims``
            then a final ``Linear(..., 1)``).
        hidden_dims: Hidden widths for the ``mlp`` head. Ignored for ``linear``.
        out_activation: ``"sigmoid"`` (default) squashes the logit into ``[0, 1]``;
            ``"identity"`` returns the raw logit (callers that want logits).

    forward(hidden) -> Tensor:
        ``hidden`` has shape ``[batch, input_dim]`` (already reduced over the
        sequence). The pulled hidden state is cast to the head's parameter dtype
        first, so a bf16/fp16 base can feed an fp32 head safely. Returns a
        ``[batch]`` tensor; with ``out_activation="sigmoid"`` every element lies
        in ``[0, 1]``.
    """

    def __init__(
        self,
        input_dim: int,
        head_type: str = "linear",
        hidden_dims: Sequence[int] = (),
        out_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"AuxHead input_dim must be positive, got {input_dim}.")
        if head_type not in VALID_HEAD_TYPES:
            raise ValueError(f"Unknown head_type {head_type!r}; expected one of {VALID_HEAD_TYPES}.")
        if out_activation not in VALID_OUT_ACTIVATIONS:
            raise ValueError(
                f"Unknown out_activation {out_activation!r}; expected one of {VALID_OUT_ACTIVATIONS}."
            )

        self.input_dim = int(input_dim)
        self.head_type = head_type
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.out_activation = out_activation

        if head_type == "linear":
            self.net: nn.Module = nn.Linear(self.input_dim, 1)
        else:  # "mlp"
            layers: list[nn.Module] = []
            prev = self.input_dim
            for width in self.hidden_dims:
                layers.append(nn.Linear(prev, width))
                layers.append(nn.GELU())
                prev = width
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype — the dtype the pulled hidden state is cast to."""
        return next(self.parameters()).dtype

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Cast the pulled hidden state to the head's dtype (bf16 base -> fp32 head).
        hidden = hidden.to(self.dtype)
        logits = self.net(hidden).squeeze(-1)  # [batch, 1] -> [batch]
        if self.out_activation == "sigmoid":
            return torch.sigmoid(logits)
        return logits  # "identity"


def resolve_hidden_size(model: Any) -> int:
    """Resolve the base model's hidden size for the aux_head input dim.

    The model returned by the PEFT/Unsloth wrap usually proxies ``.config`` to the
    base, but not always — fall back to the base model's config, then to the input
    embedding width.

    This lives in the unsloth-free head module (not the ``train_sft`` entry point,
    which imports unsloth at module load) so all three fallback branches are
    importable and unit-testable. ``train_sft.run`` imports and calls it.
    """
    config_obj = getattr(model, "config", None)
    hidden_size = getattr(config_obj, "hidden_size", None)
    if hidden_size is None:
        base = getattr(model, "base_model", None)
        base_config = getattr(base, "config", None)
        hidden_size = getattr(base_config, "hidden_size", None)
    if hidden_size is None:
        embeddings = model.get_input_embeddings()
        hidden_size = getattr(embeddings, "embedding_dim", None) or embeddings.weight.shape[1]
    return int(hidden_size)


def reduce_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    token_position: Union[str, int] = "last",
) -> torch.Tensor:
    """Reduce a ``[batch, seq, hidden]`` tensor to ``[batch, hidden]``.

    The SFT collator right-pads, so the last *real* token per row is
    ``attention_mask.sum(1) - 1`` (NOT ``seq_len - 1``).

    Args:
        hidden: ``[batch, seq, hidden]`` hidden states for one layer.
        attention_mask: ``[batch, seq]`` 1/0 mask (1 = real token, right-padded).
        token_position: ``"last"`` (last non-pad token), ``"mean"`` (mask-weighted
            mean over real tokens), or an int index into the sequence.

    Returns:
        ``[batch, hidden]``.
    """
    if hidden.dim() != 3:
        raise ValueError(f"reduce_hidden_states expects [batch, seq, hidden]; got shape {tuple(hidden.shape)}.")

    batch_size = hidden.size(0)
    arange = torch.arange(batch_size, device=hidden.device)

    if token_position == "last":
        lengths = attention_mask.to(hidden.device).long().sum(dim=1)
        last_idx = (lengths - 1).clamp_min(0)  # all-zero mask is impossible, but cheap insurance
        return hidden[arange, last_idx]

    if token_position == "mean":
        mask = attention_mask.to(hidden.device).to(hidden.dtype).unsqueeze(-1)  # [batch, seq, 1]
        summed = (hidden * mask).sum(dim=1)  # [batch, hidden]
        counts = mask.sum(dim=1).clamp_min(1.0)  # [batch, 1]
        return summed / counts

    if isinstance(token_position, int):
        seq_len = hidden.size(1)
        idx = token_position if token_position >= 0 else seq_len + token_position
        if not 0 <= idx < seq_len:
            raise ValueError(f"token_position index {token_position} out of range for seq_len {seq_len}.")
        return hidden[:, idx, :]

    raise ValueError(
        f"Unsupported token_position {token_position!r}; expected 'last', 'mean', or an int index."
    )


def compute_aux_head_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Proper-scoring loss between predicted probabilities and per-row targets.

    Computed in fp32 with autocast disabled: ``F.binary_cross_entropy`` is on the
    autocast block-list and would raise inside a mixed-precision region, and fp32
    is numerically safer for the score regardless.

    Args:
        pred: ``[batch]`` predicted probabilities in ``[0, 1]``.
        target: ``[batch]`` per-row targets in ``[0, 1]`` (soft) or ``{0, 1}``.
        loss_type: ``"bce"`` (binary cross-entropy) or ``"brier"`` (MSE on prob).
    """
    if loss_type not in VALID_LOSSES:
        raise ValueError(f"Unknown loss {loss_type!r}; expected one of {VALID_LOSSES}.")

    device_type = pred.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        pred_f = pred.float()
        target_f = target.float()
        if loss_type == "bce":
            return F.binary_cross_entropy(pred_f, target_f)
        return F.mse_loss(pred_f, target_f)  # "brier"


def save_aux_head(
    aux_head: AuxHead,
    output_dir: Union[str, Path],
    *,
    layer: int,
    token_position: Union[str, int],
    loss: str,
) -> Path:
    """Persist the head weights + the resolved config as a portable sidecar.

    Writes ``aux_head.safetensors`` (state_dict) and ``aux_head_config.json``
    (everything ``load_aux_head`` needs to reconstruct the module and everything
    inference needs to read the right layer/token). Neither SFT save path
    serializes an attached head, so this sidecar is the only persistence.
    """
    from safetensors.torch import save_file

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {key: value.detach().contiguous().cpu() for key, value in aux_head.state_dict().items()}
    save_file(state, str(out_dir / "aux_head.safetensors"))

    resolved = {
        "input_dim": aux_head.input_dim,
        "head_type": aux_head.head_type,
        "hidden_dims": list(aux_head.hidden_dims),
        "out_activation": aux_head.out_activation,
        "layer": layer,
        "token_position": token_position,
        "loss": loss,
    }
    (out_dir / "aux_head_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return out_dir


def load_aux_head(run_dir: Union[str, Path], base_model: Optional[Any] = None) -> AuxHead:
    """Reconstruct an :class:`AuxHead` from a sidecar and load its weights.

    Args:
        run_dir: Directory holding ``aux_head.safetensors`` + ``aux_head_config.json``.
        base_model: Optional — when given, the head is moved onto the base model's
            device/dtype so it can be fed hidden states directly. Reconstruction
            itself does not depend on it (the head is standalone/portable).
    """
    from safetensors.torch import load_file

    src = Path(run_dir)
    config_path = src / "aux_head_config.json"
    weights_path = src / "aux_head.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"aux_head sidecar incomplete in {src}: expected aux_head_config.json + aux_head.safetensors."
        )

    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    head = AuxHead(
        input_dim=resolved["input_dim"],
        head_type=resolved.get("head_type", "linear"),
        hidden_dims=tuple(resolved.get("hidden_dims", []) or []),
        out_activation=resolved.get("out_activation", "sigmoid"),
    )
    head.load_state_dict(load_file(str(weights_path)))

    if base_model is not None:
        try:
            ref = next(base_model.parameters())
            head = head.to(device=ref.device, dtype=ref.dtype)
        except StopIteration:
            pass
    return head


def read_aux_head_resolved_config(run_dir: Union[str, Path]) -> dict:
    """Return the sidecar's resolved config dict (layer, token_position, loss, ...)."""
    return json.loads((Path(run_dir) / "aux_head_config.json").read_text(encoding="utf-8"))


def infer_aux_scalar(
    model: Any,
    aux_head: AuxHead,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
    token_position: Union[str, int] = "last",
) -> torch.Tensor:
    """Generic inference hook: base + head + input -> per-row scalar in ``[0, 1]``.

    Runs the base with ``output_hidden_states=True``, reduces ``hidden_states[layer]``
    at ``token_position``, and applies the head. No decision policy/threshold is
    applied — that is the caller's concern.

    Returns ``[batch]`` (one scalar per row).
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[layer]
            reduced = reduce_hidden_states(hidden, attention_mask, token_position)
            return aux_head(reduced)
    finally:
        if was_training:
            model.train()
