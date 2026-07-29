"""
Forward-hook intervention engine.

Two intervention laws act on a single decoder layer's output hidden state h:

  additive push:      h' = h + alpha * d
  erase-and-write:    h' = h - (h . c) c + (gain * sigma) c

The additive push shifts the residual stream along a unit direction d by a
per-row strength alpha. The erase-and-write law removes whatever the current
row projects onto a unit direction c and writes a commanded coordinate
gain*sigma in its place, so the post-write projection equals gain*sigma exactly
regardless of the pre-write value, while the orthogonal complement is untouched.

Both laws support:
  - per-row selection: only rows the caller marks active are edited; inactive
    rows pass through unchanged (a strength of zero, or no gain, is a no-op by
    default).
  - an explicit active_override bool tensor that replaces the default
    value-based active detection, so a caller can force a row to be edited even
    when its strength/gain is exactly zero -- the "apply erase_write with a
    zero setpoint" (ablate) case, distinct from a true no-op. A NaN component
    is always excluded from the active set regardless of the override, since
    writing a NaN setpoint would corrupt the hidden state.
  - per-batch-element strength: alpha / gain may be a scalar or a length-batch
    vector.
  - position policies: which token columns are edited (anchor / anchor_onward /
    final / answer_window).
  - readback: the caller can measure the realized projection to confirm the
    commanded edit landed and that off-target rows did not move.

The final-position policy resolves each row's true last non-pad token from the
attention mask and handles left AND right padding identically. Directions are
stored float32 and cast to the hidden dtype at edit time; the hidden tensor is
cloned before any in-place write so no autograd view alias is mutated.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch


# Decoder-layer container attribute paths, tried in order. Covers the common
# causal-LM and multimodal-wrapper layouts.
_LAYER_PATHS = (
    "model.layers",
    "language_model.model.layers",
    "model.decoder.layers",
    "transformer.h",
    "model.model.layers",
    # transformers 5.x multimodal wrappers (e.g. Gemma4ForConditionalGeneration)
    # nest the text decoder directly under model.language_model, with sibling
    # vision/audio towers; this path names the text stack unambiguously.
    "model.language_model.layers",
)


def get_decoder_layer(model, layer_idx: int):
    """Return the decoder block at layer_idx for common architectures.

    Raises AttributeError if no known layer container is found.
    """
    for path in _LAYER_PATHS:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj[layer_idx]
    raise AttributeError(
        "Could not locate decoder layers on this model; tried: "
        + ", ".join(_LAYER_PATHS)
    )


def resolve_final_positions(
    batch: int,
    seq_len: int,
    attention_mask: Optional[torch.Tensor] = None,
    explicit: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a length-batch LongTensor of each row's target column.

    Priority: explicit positions (wrapped modulo seq_len) > derived from the
    attention mask > fallback to seq_len - 1 for every row.

    The mask derivation flips the mask and takes the argmax so it finds the last
    real token under both left and right padding. A fully padded row clamps to
    seq_len - 1.
    """
    dev = device if device is not None else (
        attention_mask.device if attention_mask is not None else torch.device("cpu")
    )
    if explicit is not None:
        pos = explicit.to(dev).long() % seq_len
        return pos
    if attention_mask is not None:
        mask = attention_mask.to(dev)
        flipped = torch.flip(mask, dims=[1])
        last_from_right = torch.argmax(flipped.long(), dim=1)
        pos = (seq_len - 1) - last_from_right
        pos = pos.clamp(min=0, max=seq_len - 1).long()
        return pos
    return torch.full((batch,), seq_len - 1, dtype=torch.long, device=dev)


def _strength_per_row(
    strength: Union[float, Sequence[float], torch.Tensor],
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast a scalar / length-1 / length-batch strength to length batch."""
    if isinstance(strength, torch.Tensor):
        t = strength.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t.expand(batch).clone()
        if t.numel() == 1:
            return t.reshape(1).expand(batch).clone()
        if t.numel() != batch:
            raise ValueError(
                f"strength length {t.numel()} does not match batch {batch}"
            )
        return t
    if isinstance(strength, (list, tuple)):
        if len(strength) == 1:
            return torch.full((batch,), float(strength[0]), device=device, dtype=dtype)
        if len(strength) != batch:
            raise ValueError(
                f"strength length {len(strength)} does not match batch {batch}"
            )
        return torch.tensor(list(strength), device=device, dtype=dtype)
    return torch.full((batch,), float(strength), device=device, dtype=dtype)


def _column_mask_for_policy(
    policy: str,
    seq_len: int,
    anchor_index: Optional[int],
    window_start: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Return a length-seq_len bool column mask for shared-position policies.

    Used by anchor, anchor_onward, and answer_window; final is per-row and handled
    separately.

    answer_window is a genuinely narrowed window, not an alias of anchor_onward:
    it steers only the columns at and after window_start, which the caller sets to
    the first generated (visible) token index so the prompt is excluded. It
    requires an explicit window_start and refuses to silently fall back to column
    zero, so a misconfigured answer_window fails loudly rather than steering the
    whole prompt. To also exclude a leading thinking or reasoning span, advance
    window_start past that span; the engine cannot locate a thinking boundary on
    its own because that marker is tokenizer-specific.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if policy == "anchor":
        idx = anchor_index if anchor_index is not None else seq_len - 1
        idx = idx % seq_len
        mask[idx] = True
    elif policy == "anchor_onward":
        start = (window_start or 0) % seq_len if window_start else 0
        mask[start:] = True
    elif policy == "answer_window":
        if window_start is None:
            raise ValueError(
                "answer_window requires an explicit window_start (the first "
                "generated-token index); it does not default to the full sequence"
            )
        start = max(0, min(window_start, seq_len))
        mask[start:] = True
    else:
        raise ValueError(f"policy {policy!r} is not a shared-column policy")
    return mask


def _resolve_force_active(
    force_active: Union[bool, Sequence[bool], torch.Tensor, None],
    batch: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Broadcast a scalar / length-1 / length-batch force_active to an override tensor.

    A plain False (or None) keeps the default value-based active detection (see
    _resolve_active): returns None. A plain True is the historical single-pass
    semantics -- every row in the call is force-active -- and returns an
    all-True tensor. A per-row bool sequence/tensor (a batched pass mixing
    active and inactive rows within one generate() call) is broadcast/validated
    like _strength_per_row and returned as the override directly: the caller
    (e.g. a batched steer pass) is expected to have already folded each row's
    own force-active decision into that tensor, so this is a full replacement
    of the default check, not a union with it.
    """
    if force_active is None or force_active is False:
        return None
    if force_active is True:
        return torch.ones(batch, dtype=torch.bool, device=device)
    t = torch.as_tensor(force_active, dtype=torch.bool, device=device)
    if t.numel() == 1:
        return t.reshape(1).expand(batch).clone()
    if t.numel() != batch:
        raise ValueError(
            f"force_active length {t.numel()} does not match batch {batch}"
        )
    return t


def _resolve_active(
    value_row: torch.Tensor, active_override: Optional[torch.Tensor]
) -> torch.Tensor:
    """Active-row mask: the override if given, else "value is nonzero".

    A NaN component is always excluded, whether or not an override was given,
    so a caller can never accidentally command a NaN write.
    """
    if active_override is not None:
        active = active_override.to(dtype=torch.bool, device=value_row.device)
    else:
        active = value_row != 0.0
    return active & (~torch.isnan(value_row))


def additive_push(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    alpha_row: torch.Tensor,
    columns: torch.Tensor,
    per_row: bool,
    final_positions: Optional[torch.Tensor] = None,
    active_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply h += alpha * d in place on a cloned hidden tensor.

    direction is unit-norm, cast to hidden dtype/device by the caller.
    columns is a bool column mask (shared policies) unless per_row is True, in
    which case final_positions gives each row's single target column.
    Rows with alpha == 0 are skipped, unless active_override marks them active
    (see _resolve_active).
    """
    batch, seq_len, hidden_dim = hidden.shape
    d = direction
    if per_row:
        active = _resolve_active(alpha_row, active_override)
        if active.any():
            rows = torch.nonzero(active, as_tuple=False).squeeze(1)
            cols = final_positions[rows]
            delta = alpha_row[rows].unsqueeze(1) * d.unsqueeze(0)
            hidden[rows, cols, :] = hidden[rows, cols, :] + delta
        return hidden
    # add is (batch, 1, hidden_dim): per-row strength times the direction. It
    # broadcasts over the masked columns, so the number of selected columns need
    # not be baked into its shape.
    add = alpha_row.view(batch, 1, 1) * d.view(1, 1, hidden_dim)
    hidden[:, columns, :] = hidden[:, columns, :] + add
    return hidden


def erase_and_write(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    setpoint_row: torch.Tensor,
    gain_row: torch.Tensor,
    columns: torch.Tensor,
    per_row: bool,
    final_positions: Optional[torch.Tensor] = None,
    active_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply h' = h - (h.c)c + setpoint*c in place on a cloned hidden tensor.

    direction (c) is unit-norm, cast to hidden dtype/device by the caller.
    setpoint_row[b] is the commanded projection for row b (typically gain*sigma).
    By default gain_row selects which rows are active (gain == 0 or NaN is a
    no-op row). Pass active_override to force specific rows active regardless
    of their gain value -- this is the only way to express "erase the
    projection and write a zero setpoint" (ablate) as distinct from "leave
    this row untouched" (a true no-op), since a plain gain of 0.0 would
    otherwise be skipped identically to an unselected row. A NaN gain is
    always excluded even under an override.
    """
    batch, seq_len, hidden_dim = hidden.shape
    c = direction
    active = _resolve_active(gain_row, active_override)
    if per_row:
        if active.any():
            rows = torch.nonzero(active, as_tuple=False).squeeze(1)
            cols = final_positions[rows]
            sub = hidden[rows, cols, :]
            proj = (sub @ c).unsqueeze(-1)
            sp = setpoint_row[rows].unsqueeze(-1)
            hidden[rows, cols, :] = sub - proj * c + sp * c
        return hidden
    # shared columns: apply the law per active row across the masked columns
    if not active.any():
        return hidden
    rows = torch.nonzero(active, as_tuple=False).squeeze(1)
    col_idx = torch.nonzero(columns, as_tuple=False).squeeze(1)
    for b in rows.tolist():
        sub = hidden[b, col_idx, :]
        proj = (sub @ c).unsqueeze(-1)
        hidden[b, col_idx, :] = sub - proj * c + setpoint_row[b] * c
    return hidden


class InterventionHook:
    """A forward hook that edits a decoder layer's output hidden state.

    law:      "additive" or "erase_write".
    direction: unit-norm 1-D tensor (float32); cast to hidden dtype at call time.
    strength:  alpha (additive) or gain (erase_write); scalar or length-batch.
    sigma:     scale for erase_write (setpoint = gain * sigma); ignored otherwise.
    position:  "anchor" | "anchor_onward" | "final" | "answer_window".
    attention_mask / final_positions / anchor_index / window_start: position inputs.
    force_active: when True, every row in the batch is treated as active for
      this call regardless of its resolved strength/gain value (a NaN
      component is still excluded). This is how a caller expresses "apply
      erase_write with a zero setpoint" (ablate) as opposed to a strength of
      zero meaning no-op (baseline); see MechInterp/cell.py's write_at_zero.
      May also be a per-row bool sequence/tensor so a single batched call can
      mix active and inactive rows (see _resolve_force_active); a length-1
      value broadcasts to every row like strength does.

    When measure_readback is True the realized projection of each edited row onto
    the direction is stored in last_readback after each call.
    """

    def __init__(
        self,
        law: str,
        direction: torch.Tensor,
        strength: Union[float, Sequence[float], torch.Tensor] = 1.0,
        sigma: float = 1.0,
        position: str = "final",
        attention_mask: Optional[torch.Tensor] = None,
        final_positions: Optional[torch.Tensor] = None,
        anchor_index: Optional[int] = None,
        window_start: Optional[int] = None,
        measure_readback: bool = False,
        force_active: Union[bool, Sequence[bool], torch.Tensor] = False,
    ):
        if law not in ("additive", "erase_write"):
            raise ValueError(f"unknown law {law!r}")
        if position not in ("anchor", "anchor_onward", "final", "answer_window"):
            raise ValueError(f"unknown position policy {position!r}")
        self.law = law
        self.direction = direction.detach().to(torch.float32)
        self.strength = strength
        self.sigma = float(sigma)
        self.position = position
        self.attention_mask = attention_mask
        self.final_positions = final_positions
        self.anchor_index = anchor_index
        self.window_start = window_start
        self.measure_readback = measure_readback
        self.force_active = force_active
        self.active = True
        self.last_readback: Optional[dict] = None

    def __call__(self, module, inputs, output):
        if not self.active:
            return output

        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        rest = output[1:] if is_tuple else ()

        hidden = hidden.clone()
        batch, seq_len, hidden_dim = hidden.shape
        device, dtype = hidden.device, hidden.dtype
        c = self.direction.to(device=device, dtype=dtype)

        per_row = self.position == "final"
        if per_row:
            final_pos = resolve_final_positions(
                batch,
                seq_len,
                attention_mask=self.attention_mask,
                explicit=self.final_positions,
                device=device,
            )
            columns = torch.zeros(seq_len, dtype=torch.bool, device=device)
        else:
            final_pos = None
            columns = _column_mask_for_policy(
                self.position, seq_len, self.anchor_index, self.window_start, device
            )

        active_override = _resolve_force_active(self.force_active, batch, device)

        if self.law == "additive":
            alpha = _strength_per_row(self.strength, batch, device, dtype)
            hidden = additive_push(
                hidden, c, alpha, columns, per_row, final_pos,
                active_override=active_override,
            )
            gain_for_readback = alpha
        else:
            gain = _strength_per_row(self.strength, batch, device, dtype)
            setpoint = gain * self.sigma
            hidden = erase_and_write(
                hidden, c, setpoint, gain, columns, per_row, final_pos,
                active_override=active_override,
            )
            gain_for_readback = gain

        if self.measure_readback:
            self.last_readback = self._readback(
                hidden, c, per_row, final_pos, columns, gain_for_readback, active_override
            )

        if is_tuple:
            return (hidden,) + rest
        return hidden

    def _readback(self, hidden, c, per_row, final_pos, columns, gain_row, active_override=None) -> dict:
        """Measure realized projection onto the direction (float64) after the edit.

        Returns per-row commanded vs measured projection for active rows and the
        mean absolute projection of inactive rows (off-target parity check).
        """
        c64 = c.detach().to(torch.float64)
        batch = hidden.shape[0]
        if per_row:
            active = _resolve_active(gain_row, active_override)
            rows = torch.nonzero(active, as_tuple=False).squeeze(1)
            inactive = torch.nonzero(~active, as_tuple=False).squeeze(1)
            measured = []
            for b in rows.tolist():
                col = int(final_pos[b].item())
                measured.append(
                    float(hidden[b, col, :].to(torch.float64) @ c64)
                )
            off = []
            for b in inactive.tolist():
                col = int(final_pos[b].item())
                off.append(abs(float(hidden[b, col, :].to(torch.float64) @ c64)))
        else:
            col_idx = torch.nonzero(columns, as_tuple=False).squeeze(1)
            first_col = int(col_idx[0].item()) if col_idx.numel() else hidden.shape[1] - 1
            active = _resolve_active(gain_row, active_override)
            rows = torch.nonzero(active, as_tuple=False).squeeze(1)
            inactive = torch.nonzero(~active, as_tuple=False).squeeze(1)
            measured = [
                float(hidden[b, first_col, :].to(torch.float64) @ c64)
                for b in rows.tolist()
            ]
            off = [
                abs(float(hidden[b, first_col, :].to(torch.float64) @ c64))
                for b in inactive.tolist()
            ]
        if self.law == "erase_write":
            commanded = [float(gain_row[b].item()) * self.sigma for b in rows.tolist()]
        else:
            commanded = [None for _ in rows.tolist()]
        return {
            "law": self.law,
            "active_rows": rows.tolist(),
            "commanded": commanded,
            "measured": measured,
            "offtarget_abs_mean": (sum(off) / len(off)) if off else 0.0,
            "offtarget_abs_max": max(off) if off else 0.0,
        }


class GenerationInterventionController:
    """Wraps an InterventionHook to gate prefill vs decode during generate().

    Register the controller (not the raw hook) once. Call begin_pass(...) before
    each generate() to set the active law/strength for that pass, and reset()
    after. The controller counts forward calls to distinguish the prefill step
    (seq_len == prompt_len, call 1) from single-token decode steps (seq_len == 1).

    modes:
      "anchor"     edit only the prefill step at the last prompt token; the edit
                   propagates through the KV cache to all later tokens.
      "gen_stream" skip prefill, then edit every decode step at its single token.
      "off"        pass through.
    """

    def __init__(self, hook: InterventionHook):
        self.hook = hook
        self.mode = "off"
        self._nth_call = 0

    def begin_pass(
        self,
        mode: str,
        strength: Union[float, Sequence[float], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        force_active: Union[bool, Sequence[bool], torch.Tensor] = False,
    ) -> None:
        """Arm the hook for one generate() call.

        force_active passes through to the hook: set it when the caller's arm
        resolution decided a row should be edited even at a zero strength/gain
        (the ablate case). For a single-row pass this is a plain bool. For a
        batched pass driving several rows through one generate() call, pass a
        per-row bool tensor instead: strength is already per-row-capable (see
        _strength_per_row), and force_active follows the same broadcasting
        rule (see _resolve_force_active) so one pass can mix active and
        inactive rows correctly.
        """
        if mode not in ("anchor", "gen_stream", "off"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.hook.strength = strength
        self.hook.attention_mask = attention_mask
        self.hook.force_active = force_active
        self._nth_call = 0

    def reset(self) -> None:
        self.mode = "off"
        self._nth_call = 0
        self.hook.force_active = False
        self.hook.active = False

    def __call__(self, module, inputs, output):
        self._nth_call += 1
        if self.mode == "off":
            self.hook.active = False
            return output

        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        seq_len = hidden.shape[1]
        is_prefill = self._nth_call == 1 and seq_len > 1

        if self.mode == "anchor":
            if is_prefill:
                self.hook.active = True
                self.hook.position = "anchor"
                self.hook.anchor_index = None
            else:
                self.hook.active = False
        elif self.mode == "gen_stream":
            if is_prefill:
                self.hook.active = False
            else:
                self.hook.active = True
                self.hook.position = "anchor_onward"
                self.hook.window_start = 0

        return self.hook(module, inputs, output)
