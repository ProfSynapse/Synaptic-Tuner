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

import contextlib
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

from MechInterp.extraction.persistence import CaptureJournal, file_info, row_key, safe_key


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
    content_end_fn: Optional[Callable[[torch.Tensor, int, "object"], int]],
    spec: PositionSpec,
    out_dir: str | Path,
    max_new_tokens: int = 48,
    row_key_fn: Callable[[dict], str] = row_key,
    save_tensors: bool = True,
    device: Optional[str] = None,
    capture_sites: Optional[Sequence] = None,
    manifest_extra: Optional[dict] = None,
    use_cache: bool = False,
    forward_only: bool = False,
    resume: bool = False,
    run_identity: Optional[dict] = None,
) -> dict:
    """Generate + capture hidden states for each row; write safetensors + manifest.

    Args:
      render_fn(row) -> prompt string (caller applies any chat template).
      content_end_fn(full_ids, prompt_len, tokenizer) -> index of last content
        token in the concatenated sequence, or a value < prompt_len if the row
        produced no usable content.
      spec: which positions/layers to capture.
      out_dir: destination directory for <row>__<family>.safetensors + manifest.
      capture_sites: optional explicit module sites (see sites.py). Captured
        during the same clean forward pass at the selected position families
        only, written to <row>__<family>.sites.safetensors (a separate file so
        residual readers never see them). Absent = residual extraction only.
      manifest_extra: extra top-level manifest keys (e.g. a weight-patch receipt).
      use_cache: value passed as ``use_cache`` to the teacher-forced capture
        forward. Default False (the historical behavior). Set True for models
        whose cache-less forward is not equivalent to their normal one (e.g.
        hybrid or KV-sharing architectures).
      forward_only: skip generation entirely. Only the ``anchor`` family is
        valid (the last prompt token); ``content_end_fn`` is not used and
        ``max_new_tokens`` is ignored. Rows record ``answer_text == ""``.
      resume: persist a per-row journal and resume the same run. Requires an
        explicit run_identity. Without resume, out_dir must be empty.
      run_identity: source identity used by the journal (CLI supplies hashes of
        local source files or a pinned Hub revision and callable file hashes).

    Returns the manifest dict.
    """
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device or (next(model.parameters()).device if save_tensors else "cpu")

    site_specs = None
    if capture_sites:
        from MechInterp.extraction.sites import resolve_sites, to_site_specs

        site_specs = to_site_specs(capture_sites)
        resolve_sites(model, site_specs)  # fail before any generation

    if forward_only:
        if list(spec.families) != ["anchor"]:
            raise ValueError("forward_only extraction supports exactly the 'anchor' family")
    elif content_end_fn is None:
        raise ValueError("content_end_fn is required unless forward_only is set")

    # Validate keys and rendered inputs before any generation or artifact writes.
    keys = [row_key_fn(row) for row in rows]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("extraction row keys must be non-empty strings")
    names = [safe_key(key) for key in keys]
    if len(set(keys)) != len(keys) or len({name.casefold() for name in names}) != len(names):
        raise ValueError("extraction requires unique row keys and unique file names")
    prompts = [render_fn(row) for row in rows]
    if resume and not run_identity:
        raise ValueError("resumable extraction requires an explicit run_identity")
    identity = {
        "schema_version": "mechinterp-capture-journal/v1",
        "run": run_identity,
        "inputs": [{"row_key": key, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
                   for key, prompt in zip(keys, prompts)],
        "families": list(spec.families), "layers": spec.layers, "every_k": spec.every_k,
        "max_new_tokens": max_new_tokens, "use_cache": use_cache,
        "forward_only": forward_only, "save_tensors": save_tensors,
        "sites": [vars(site) for site in site_specs] if site_specs else [],
        "manifest_extra": manifest_extra,
    }
    row_records = []
    hidden_dim = None
    n_hidden_states = None

    with CaptureJournal(out_dir, identity, resume=resume) as journal:
        for row, key, prompt, filename_key in zip(rows, keys, prompts, names):
            if key in journal.records:
                rec = journal.records[key]
                row_records.append(rec)
                hidden_dim = rec.get("hidden_dim") or hidden_dim
                n_hidden_states = rec.get("n_hidden_states") or n_hidden_states
                continue
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(dev)
            prompt_len = input_ids.shape[1]

            if forward_only:
                full = input_ids[0]
                seq_total = full.shape[0]
                content_end = prompt_len - 1
                answered = True
            else:
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
                "row_key": key,
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
                site_capture = None
                site_positions: dict[str, list[int]] = {}
                if site_specs:
                    from MechInterp.extraction.sites import SiteCapture

                    site_positions = resolve_capture_positions(
                        spec, prompt_len, content_end, content_end + 1
                    )
                    union = sorted({p for idxs in site_positions.values() for p in idxs})
                    if union:
                        site_capture = SiteCapture(model, site_specs)
                with site_capture if site_capture is not None else contextlib.nullcontext():
                    if site_capture is not None:
                        site_capture.begin(union)
                    with torch.no_grad():
                        out = model(
                            input_ids=fwd_ids,
                            attention_mask=attn,
                            output_hidden_states=True,
                            use_cache=use_cache,
                        )
                hs = out.hidden_states
                n_hidden_states = len(hs)
                hidden_dim = hs[0].shape[-1]
                layers = _layer_indices(spec, n_hidden_states)
                positions = resolve_capture_positions(
                    spec, prompt_len, content_end, content_end + 1
                )
                rec["capture_files"] = {}
                rec["hidden_dim"] = hidden_dim
                rec["n_hidden_states"] = n_hidden_states
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
                        path = out_dir / f"{filename_key}__{family}.safetensors"
                        save_file(tensors, str(path))
                        rec["capture_files"][family] = file_info(path)
                rec["captured_families"] = list(positions.keys())
                if site_capture is not None:
                    rec["site_captures"] = _save_site_captures(
                        site_capture, site_positions, out_dir, filename_key
                    )

            journal.record(rec)
            row_records.append(rec)

        manifest = {
            "schema_version": "mechinterp-capture/v2",
            "complete": True,
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
        # Only a non-default choice is recorded, so default manifests keep their bytes.
        if use_cache:
            manifest["capture_use_cache"] = True
        if forward_only:
            manifest["forward_only"] = True
        if site_specs:
            manifest["capture_sites"] = [
                {"key": c.key, "layer": c.layer, "module": c.module, "tensor": c.tensor}
                for c in site_specs
            ]
            manifest["capture_site_layer_convention"] = "decoder_block"
            manifest["capture_site_persist_dtype"] = "float32"
        if manifest_extra:
            manifest.update(manifest_extra)

        journal.finalize(manifest)
        return manifest


def _save_site_captures(site_capture, family_positions, out_dir: Path, safe_key: str) -> dict:
    """Write each family's selected-position site activations; return metadata."""
    from safetensors.torch import save_file

    results = site_capture.results()
    meta = site_capture.metadata()
    row_of = {p: i for i, p in enumerate(site_capture.positions)}
    summary: dict[str, dict] = {}
    for family, idxs in family_positions.items():
        if not idxs:
            continue
        sel = [row_of[p] for p in idxs]
        tensors = {key: t[sel].contiguous() for key, t in results.items()}
        file_meta = {
            "layer_convention": "decoder_block",
            "family": family,
            "positions": json.dumps(list(idxs)),
        }
        save_file(
            tensors,
            str(out_dir / f"{safe_key}__{family}.sites.safetensors"),
            metadata=file_meta,
        )
        summary[family] = {
            **file_info(out_dir / f"{safe_key}__{family}.sites.safetensors"),
            "positions": list(idxs),
            "sites": {
                key: {
                    "shape": list(t.shape),
                    "persist_dtype": "float32",
                    "source_dtype": meta[key]["source_dtype"],
                }
                for key, t in tensors.items()
            },
        }
    return summary
