"""
Hidden-state extraction during generation.

For each input row the extractor generates a completion, then runs a single
clean forward pass over the prompt plus completion (with the KV cache disabled)
and slices hidden states at configurable token positions across a configurable
layer range. Captured tensors are written to safetensors, one file per row and
position family, with a manifest describing the run.

Position families:
  anchor        the last prompt token (pre-generation read).
  first_visible the first generated token.
  answer_end    the last content token of the completion (post-generation read).
  every_k       every k-th generated token, stacked.

Capturing at anchor and answer_end together gives the dual pre/post read used by
many readout studies. Positions are resolved against the concatenated
prompt+completion sequence so the same forward pass serves every family.

The tensors are stored float32 and contiguous so downstream probe fitting reads
them without surprises. Nothing here assumes a particular tokenizer, chat
template, or research question; the caller supplies a render function and a
content-end resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch


@dataclass
class PositionSpec:
    """Which token positions to capture and over which layers.

    families: subset of {"anchor", "first_visible", "answer_end", "every_k"}.
    every_k:  stride for the every_k family (ignored otherwise).
    layers:   explicit list of hidden_states indices, or None for all layers.
    """

    families: Sequence[str] = field(default_factory=lambda: ["anchor", "answer_end"])
    every_k: int = 4
    layers: Optional[Sequence[int]] = None


_VALID_FAMILIES = {"anchor", "first_visible", "answer_end", "every_k"}


def resolve_capture_positions(
    spec: PositionSpec,
    prompt_len: int,
    content_end: int,
    seq_total: int,
) -> dict[str, list[int]]:
    """Map each requested family to a list of token indices in the full sequence.

    prompt_len is the number of prompt tokens; content_end is the index of the
    last content token in the concatenated sequence; seq_total is its length.
    Indices out of range are dropped so a short completion never raises.
    """
    for fam in spec.families:
        if fam not in _VALID_FAMILIES:
            raise ValueError(f"unknown position family {fam!r}")
    positions: dict[str, list[int]] = {}
    if "anchor" in spec.families:
        positions["anchor"] = [max(0, prompt_len - 1)]
    if "first_visible" in spec.families:
        idx = prompt_len
        positions["first_visible"] = [idx] if idx <= content_end else []
    if "answer_end" in spec.families:
        positions["answer_end"] = [content_end] if content_end >= 0 else []
    if "every_k" in spec.families:
        k = max(1, spec.every_k)
        idxs = list(range(prompt_len, content_end + 1, k))
        positions["every_k"] = [i for i in idxs if 0 <= i < seq_total]
    return positions


def _layer_indices(spec: PositionSpec, n_hidden_states: int) -> list[int]:
    if spec.layers is None:
        return list(range(n_hidden_states))
    return [li for li in spec.layers if 0 <= li < n_hidden_states]


def extract_rows(
    model,
    tokenizer,
    rows: list[dict],
    render_fn: Callable[[dict], str],
    content_end_fn: Callable[[torch.Tensor, int, "object"], int],
    spec: PositionSpec,
    out_dir: str | Path,
    max_new_tokens: int = 48,
    row_key_fn: Callable[[dict], str] = lambda r: str(r.get("row_key", r.get("id"))),
    save_tensors: bool = True,
    device: Optional[str] = None,
) -> dict:
    """Generate + capture hidden states for each row; write safetensors + manifest.

    Args:
      render_fn(row) -> prompt string (caller applies any chat template).
      content_end_fn(full_ids, prompt_len, tokenizer) -> index of last content
        token in the concatenated sequence, or a value < prompt_len if the row
        produced no usable content.
      spec: which positions/layers to capture.
      out_dir: destination directory for <row>__<family>.safetensors + manifest.

    Returns the manifest dict.
    """
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device or (next(model.parameters()).device if save_tensors else "cpu")

    row_records = []
    hidden_dim = None
    n_hidden_states = None

    for row in rows:
        prompt = render_fn(row)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        prompt_len = input_ids.shape[1]

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
        )
        full = gen.sequences[0]
        seq_total = full.shape[0]
        content_end = int(content_end_fn(full, prompt_len, tokenizer))
        answered = content_end >= prompt_len

        rec = {
            "row_key": row_key_fn(row),
            "prompt_len": int(prompt_len),
            "seq_total": int(seq_total),
            "content_end": int(content_end),
            "answered": bool(answered),
            "answer_text": tokenizer.decode(
                full[prompt_len:], skip_special_tokens=True
            ),
        }

        if answered and save_tensors:
            fwd_ids = full[: content_end + 1].unsqueeze(0)
            attn = torch.ones_like(fwd_ids)
            with torch.no_grad():
                out = model(
                    input_ids=fwd_ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hs = out.hidden_states
            n_hidden_states = len(hs)
            hidden_dim = hs[0].shape[-1]
            layers = _layer_indices(spec, n_hidden_states)
            positions = resolve_capture_positions(
                spec, prompt_len, content_end, content_end + 1
            )
            safe_key = rec["row_key"].replace("::", "__").replace("|", "_").replace("/", "_")
            for family, idxs in positions.items():
                if not idxs:
                    continue
                tensors = {}
                for li in layers:
                    cols = [
                        hs[li][0, p, :].float().cpu().contiguous()
                        for p in idxs
                        if p < hs[li].shape[1]
                    ]
                    if cols:
                        tensors[f"L{li}"] = torch.stack(cols, dim=0)
                if tensors:
                    save_file(
                        tensors, str(out_dir / f"{safe_key}__{family}.safetensors")
                    )
            rec["captured_families"] = list(positions.keys())

        row_records.append(rec)

    manifest = {
        "n_rows": len(rows),
        "n_answered": sum(1 for r in row_records if r["answered"]),
        "hidden_dim": hidden_dim,
        "n_hidden_states": n_hidden_states,
        "position_families": list(spec.families),
        "every_k": spec.every_k,
        "layers": spec.layers if spec.layers is not None else "all",
        "max_new_tokens": max_new_tokens,
        "persist_dtype": "float32",
        "decode": "greedy",
        "rows": row_records,
    }
    import json

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
