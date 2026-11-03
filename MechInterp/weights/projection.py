"""Bounded numeric reads of checkpoint tensors for ``weights-inspect``.

Answers questions like "what is the dot product of each of these selected
columns of a 2-D tensor with this configured vector" without loading the model
and without emitting unbounded arrays. Exactly one tensor is read at a time (its
size is capped), only the selected lines are converted to float64, and the total
number of emitted values is capped by the request. Nothing here knows what the
tensor, vector or indices mean.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from MechInterp.weights.checkpoint import LocalCheckpoint
from MechInterp.weights.errors import WeightPatchError
from MechInterp.weights.patch import buffer_to_tensor
from MechInterp.weights.safetensors_io import (
    FLOAT_TORCH_NAMES,
    file_sha256,
    read_header,
    read_region,
    strict_json_loads,
)
from MechInterp.weights.spec import InspectRequest, ProjectionRequest, VectorSource

_MAX_VECTOR_FILE_BYTES = 64 * 1024 * 1024
_MAX_DIRECTION_JSON_BYTES = 8 * 1024 * 1024
_DIRECTION_SCHEMA = "mechinterp-direction/v1"


def _pinned_file(path: str, expected_sha: str, limit: int, what: str) -> Path:
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise WeightPatchError("path", f"{what} must be an existing regular, non-symlink file")
    if p.stat().st_size > limit:
        raise WeightPatchError("budget", f"{what} exceeds {limit} bytes")
    if file_sha256(p) != expected_sha:
        raise WeightPatchError("payload", f"{what} sha256 does not match its pin")
    return p


def load_vector(src: VectorSource) -> tuple[torch.Tensor, dict]:
    """(float64 1-D vector, provenance dict)."""
    if src.inline is not None:
        vec = torch.tensor(src.inline, dtype=torch.float64)
        info = {"source": "inline", "sha256": None}
    elif src.safetensors is not None:
        ref = src.safetensors
        p = _pinned_file(ref.path, ref.sha256, _MAX_VECTOR_FILE_BYTES, "vector file")
        sf = read_header(p)
        entry = sf.entries.get(ref.key)
        if entry is None:
            raise WeightPatchError("payload", f"vector file has no array {ref.key!r}")
        if entry.torch_dtype not in FLOAT_TORCH_NAMES or len(entry.shape) != 1:
            raise WeightPatchError("payload", "vector array must be a 1-D float array")
        vec = buffer_to_tensor(read_region(sf, entry), entry.torch_dtype, entry.shape).to(torch.float64)
        info = {"source": "safetensors", "sha256": ref.sha256}
    else:
        ref = src.direction_json
        p = _pinned_file(ref.path, ref.sha256, _MAX_DIRECTION_JSON_BYTES, "direction file")
        doc = strict_json_loads(p.read_bytes())
        values = doc.get("vector") if isinstance(doc, dict) else None
        if (
            not isinstance(doc, dict)
            or doc.get("schema_version") != _DIRECTION_SCHEMA
            or not isinstance(values, list)
            or not values
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        ):
            raise WeightPatchError("payload", f"direction file must be a {_DIRECTION_SCHEMA} document with a numeric vector")
        vec = torch.tensor(values, dtype=torch.float64)
        info = {"source": "direction_json", "sha256": ref.sha256}
    if not bool(torch.isfinite(vec).all()):
        raise WeightPatchError("nonfinite", "vector contains NaN/Inf")
    info["length"] = int(vec.numel())
    info["l2_norm"] = float(torch.linalg.vector_norm(vec).item())
    return vec, info


def _indices(values, dim: int, label: str, pid: str) -> torch.Tensor:
    if values is None:
        return torch.arange(dim, dtype=torch.int64)
    idx = torch.tensor(values, dtype=torch.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= dim:
        raise WeightPatchError("payload", f"projection {pid}: {label} index out of range [0, {dim})")
    if int(torch.unique(idx).numel()) != idx.numel():
        raise WeightPatchError("payload", f"projection {pid}: {label} are not distinct")
    return idx


def _count(values, dim: int) -> int:
    return dim if values is None else len(values)


def run_projections(request: InspectRequest, ckpt: LocalCheckpoint) -> list[dict]:
    results: list[dict] = []
    emitted = 0
    for req in request.projections:
        results.append(_one(req, request, ckpt, emitted))
        emitted += results[-1]["n_values"]
    return results


def _one(req: ProjectionRequest, request: InspectRequest, ckpt: LocalCheckpoint, emitted: int) -> dict:
    if req.tensor not in ckpt.tensor_to_file:
        raise WeightPatchError("binding", f"projection {req.id}: tensor {req.tensor!r} is not in the checkpoint")
    sf, entry = ckpt.entry(req.tensor)
    if entry.torch_dtype not in FLOAT_TORCH_NAMES or len(entry.shape) != 2:
        raise WeightPatchError("payload", f"projection {req.id}: {req.tensor} must be a 2-D float tensor")
    if entry.nbytes > request.max_tensor_bytes:
        raise WeightPatchError("budget", f"projection {req.id}: tensor exceeds max_tensor_bytes")
    n_rows, n_cols = entry.shape
    line_dim, pos_dim = (n_cols, n_rows) if req.axis == "cols" else (n_rows, n_cols)

    n_lines = _count(req.indices, line_dim)
    n_pos = _count(req.other_indices, pos_dim)
    if req.kind == "slice":
        n_values = n_lines * n_pos
    else:
        n_values = n_lines * (2 if req.include_norms else 1)
    if emitted + n_values > request.max_output_values:
        raise WeightPatchError(
            "budget",
            f"projection {req.id}: output would exceed max_output_values {request.max_output_values}",
        )

    lines = _indices(req.indices, line_dim, "indices", req.id)
    pos = _indices(req.other_indices, pos_dim, "other_indices", req.id)

    raw = read_region(sf, entry)
    tensor_sha = hashlib.sha256(raw).hexdigest()
    w = buffer_to_tensor(raw, entry.torch_dtype, entry.shape)
    if req.axis == "cols":
        sub = w.index_select(1, lines).index_select(0, pos).to(torch.float64).T  # [lines, positions]
    else:
        sub = w.index_select(0, lines).index_select(1, pos).to(torch.float64)
    if not bool(torch.isfinite(sub).all()):
        raise WeightPatchError("nonfinite", f"projection {req.id}: selected values contain NaN/Inf")

    out: dict = {
        "id": req.id,
        "tensor": req.tensor,
        "kind": req.kind,
        "axis": req.axis,
        "dtype": entry.torch_dtype,
        "shape": [n_rows, n_cols],
        "tensor_sha256": tensor_sha,
        "n_lines": n_lines,
        "n_positions": n_pos,
        "n_values": n_values,
        "indices": lines.tolist() if req.indices is not None else "all",
    }
    if req.other_indices is not None:
        out["other_indices"] = pos.tolist()
    if req.kind == "slice":
        out["values"] = sub.tolist()
        return out

    vec, info = load_vector(req.vector)
    if vec.numel() != n_pos:
        raise WeightPatchError(
            "payload", f"projection {req.id}: vector length {vec.numel()} != line length {n_pos}"
        )
    out["vector"] = info
    dots = sub @ vec
    norms = torch.linalg.vector_norm(sub, dim=1)
    if req.kind == "dot":
        out["values"] = dots.tolist()
    else:
        vnorm = float(info["l2_norm"])
        if vnorm == 0.0:
            raise WeightPatchError("payload", f"projection {req.id}: cosine with a zero vector")
        out["values"] = [
            None if float(n) == 0.0 else float(d) / (float(n) * vnorm)
            for d, n in zip(dots.tolist(), norms.tolist())
        ]
    if req.include_norms:
        out["line_norms"] = norms.tolist()
    return out
