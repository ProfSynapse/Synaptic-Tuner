"""Pure patch planning: validate a spec against base tensors, compute new bytes.

``build_plan`` never mutates anything. It reads each targeted base tensor through
a caller-supplied ``read_tensor``, checks it against the spec's pinned dtype /
shape / byte hash, applies the operations to private copies, and enforces the
declared finite budget on the REALIZED result (entries whose stored value
actually changed, per-tensor relative Frobenius norm of the realized delta).
Any violation raises before a single byte is written anywhere, so application
(in memory or to disk) is all-or-nothing by construction.

Zero-patch semantics: an operation only replaces entries whose new value (after
casting to the tensor dtype) differs numerically from the old one, so a
zero-valued patch leaves every byte of the tensor identical and is a true no-op.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

from MechInterp.weights.errors import WeightPatchError
from MechInterp.weights.safetensors_io import (
    FLOAT_TORCH_NAMES,
    file_sha256,
    read_header,
    read_region,
)
from MechInterp.weights.spec import Operation, WeightPatchSpec

if sys.byteorder != "little":  # pragma: no cover - safetensors is little-endian
    raise ImportError("weight patching requires a little-endian host")

_CHUNK_ELEMENTS = 4 * 1024 * 1024
_DIRECTION_NORM_TOL = 1e-3


def torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


# --------------------------------------------------------------------------
# Byte-exact hashing and (de)serialization
# --------------------------------------------------------------------------


def tensor_bytes_view(t: torch.Tensor):
    """Flat uint8 numpy view of a CPU tensor's stored dtype bytes."""
    flat = t.detach().to("cpu").contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy()


def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of the tensor's raw little-endian dtype bytes (bf16 included)."""
    flat = t.detach().to("cpu").contiguous().reshape(-1)
    h = hashlib.sha256()
    if flat.numel():
        step = max(1, _CHUNK_ELEMENTS)
        for i in range(0, flat.numel(), step):
            h.update(flat[i : i + step].view(torch.uint8).numpy())
    return h.hexdigest()


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    flat = t.detach().to("cpu").contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""


def buffer_to_tensor(buf: bytearray, dtype_name: str, shape: tuple[int, ...]) -> torch.Tensor:
    dtype = getattr(torch, dtype_name)
    if len(buf) == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(buf, dtype=dtype).reshape(shape)


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item()) if t.numel() else True


def _sq_diff_sum(a: torch.Tensor, b: Optional[torch.Tensor]) -> float:
    fa = a.reshape(-1)
    fb = b.reshape(-1) if b is not None else None
    total = 0.0
    for i in range(0, fa.numel(), _CHUNK_ELEMENTS):
        x = fa[i : i + _CHUNK_ELEMENTS].to(torch.float64)
        if fb is not None:
            x = x - fb[i : i + _CHUNK_ELEMENTS].to(torch.float64)
        total += float((x * x).sum().item())
    return total


# --------------------------------------------------------------------------
# Payload
# --------------------------------------------------------------------------


@dataclass
class Payload:
    sha256: str
    arrays: dict[str, torch.Tensor]


def _op_array_names(spec: WeightPatchSpec) -> set[str]:
    return {name for op in spec.operations for name in op.arrays.values()}


def load_payload(spec: WeightPatchSpec) -> Payload:
    """Verify and load the safetensors payload named by the spec."""
    path = Path(spec.payload.path)
    try:
        if not path.is_file():
            raise WeightPatchError("path", "payload file does not exist")
        if path.is_symlink():
            raise WeightPatchError("path", "payload must not be a symlink")
        size = path.stat().st_size
    except OSError:
        raise WeightPatchError("path", "payload file is not accessible") from None
    if size > spec.budget.max_payload_bytes:
        raise WeightPatchError("budget", "payload exceeds budget.max_payload_bytes")
    if file_sha256(path) != spec.payload.sha256:
        raise WeightPatchError("payload", "payload sha256 does not match payload.sha256")
    sf = read_header(path)
    needed = _op_array_names(spec)
    missing = needed - set(sf.entries)
    if missing:
        raise WeightPatchError("payload", f"payload lacks arrays: {sorted(missing)[:3]}")
    extra = set(sf.entries) - needed
    if extra:
        raise WeightPatchError(
            "payload", f"payload holds arrays no operation references: {sorted(extra)[:3]}"
        )
    arrays: dict[str, torch.Tensor] = {}
    for name in sorted(needed):
        entry = sf.entries[name]
        if entry.torch_dtype not in FLOAT_TORCH_NAMES and entry.torch_dtype not in {"int32", "int64"}:
            raise WeightPatchError("payload", f"array {name!r} has unsupported dtype {entry.torch_dtype}")
        t = buffer_to_tensor(read_region(sf, entry), entry.torch_dtype, entry.shape)
        if t.is_floating_point() and not _is_finite(t):
            raise WeightPatchError("nonfinite", f"payload array {name!r} contains NaN/Inf")
        arrays[name] = t
    return Payload(spec.payload.sha256, arrays)


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


def _resolve_indices(
    op: Operation, role: str, payload: Payload, dim: int, max_count: int
) -> Optional[torch.Tensor]:
    inline = getattr(op, role)
    if inline is not None:
        idx = torch.tensor(inline, dtype=torch.int64) if inline else None
        if idx is None:
            raise WeightPatchError("payload", f"op {op.id}: {role} must not be empty")
    elif role in op.arrays:
        raw = payload.arrays[op.arrays[role]]
        if raw.dtype not in (torch.int32, torch.int64) or raw.ndim != 1 or raw.numel() == 0:
            raise WeightPatchError("payload", f"op {op.id}: {role} array must be a non-empty 1-D integer array")
        idx = raw.to(torch.int64)
    else:
        return None
    if idx.numel() > max_count:
        raise WeightPatchError("budget", f"op {op.id}: {role} has {idx.numel()} indices (> max_index_count)")
    if int(idx.min()) < 0 or int(idx.max()) >= dim:
        raise WeightPatchError("payload", f"op {op.id}: {role} index out of range [0, {dim})")
    if int(torch.unique(idx).numel()) != idx.numel():
        raise WeightPatchError("payload", f"op {op.id}: {role} indices are not distinct")
    return idx


def _vector(op: Operation, role: str, payload: Payload, length: int) -> torch.Tensor:
    t = payload.arrays[op.arrays[role]]
    if t.ndim != 1 or t.numel() != length:
        raise WeightPatchError(
            "payload", f"op {op.id}: array {role!r} must be 1-D of length {length}, got {tuple(t.shape)}"
        )
    return t


def _gather(w: torch.Tensor, rows, cols) -> torch.Tensor:
    sub = w
    if rows is not None:
        sub = sub.index_select(0, rows)
    if cols is not None:
        sub = sub.index_select(1, cols)
    return sub.clone() if sub is w else sub


def _scatter(w: torch.Tensor, rows, cols, new: torch.Tensor) -> None:
    if rows is not None and cols is not None:
        w[rows[:, None], cols[None, :]] = new
    elif rows is not None:
        w[rows] = new
    elif cols is not None:
        w[:, cols] = new
    else:
        w.copy_(new)


def _apply_operation(
    w: torch.Tensor, op: Operation, payload: Payload, max_index_count: int
) -> None:
    """Apply one operation to ``w`` (a private working copy) in place."""
    if w.ndim != 2:
        raise WeightPatchError("payload", f"op {op.id}: {op.tensor} is {w.ndim}-D; operations need a 2-D tensor")
    out_dim, in_dim = w.shape
    rows = _resolve_indices(op, "rows", payload, out_dim, max_index_count)
    cols = _resolve_indices(op, "cols", payload, in_dim, max_index_count)
    n_rows = out_dim if rows is None else rows.numel()
    n_cols = in_dim if cols is None else cols.numel()
    compute = torch.float64 if w.dtype == torch.float64 else torch.float32
    old = _gather(w, rows, cols)

    if op.op == "add_rank_one":
        u = _vector(op, "u", payload, n_rows).to(compute)
        v = _vector(op, "v", payload, n_cols).to(compute)
        new = (old.to(compute) + float(op.scale) * torch.outer(u, v)).to(w.dtype)
    elif op.op == "project":
        d_len = out_dim if op.axis == "output" else in_dim
        d = _vector(op, "direction", payload, d_len).to(torch.float64)
        norm = float(torch.linalg.vector_norm(d).item())
        if norm == 0.0:
            raise WeightPatchError("payload", f"op {op.id}: direction has zero norm")
        if op.normalize:
            d = d / norm
        elif abs(norm - 1.0) > _DIRECTION_NORM_TOL:
            raise WeightPatchError(
                "payload", f"op {op.id}: direction norm {norm:.6g} is not 1 (set normalize: true)"
            )
        d = d.to(compute)
        block = old.to(compute)
        if op.axis == "output":
            new = (block - float(op.scale) * torch.outer(d, d @ block)).to(w.dtype)
        else:
            new = (block - float(op.scale) * torch.outer(block @ d, d)).to(w.dtype)
    else:  # replace_rows / replace_cols
        values = payload.arrays[op.arrays["values"]]
        expected = (n_rows, in_dim) if op.op == "replace_rows" else (out_dim, n_cols)
        if tuple(values.shape) != expected:
            raise WeightPatchError(
                "payload", f"op {op.id}: values must have shape {list(expected)}, got {list(values.shape)}"
            )
        new = values.to(w.dtype)

    if not _is_finite(new):
        raise WeightPatchError("nonfinite", f"op {op.id}: result contains NaN/Inf")
    new = torch.where(new != old, new, old)  # unchanged values keep their exact bytes
    _scatter(w, rows, cols, new)


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


@dataclass
class TensorPlan:
    name: str
    dtype: str
    shape: tuple[int, ...]
    before_sha256: str
    after_sha256: str
    entries_changed: int
    delta_frobenius: float
    base_frobenius: float
    relative_frobenius: float
    op_ids: list[str]
    new_tensor: Optional[torch.Tensor] = field(default=None, repr=False)

    @property
    def modified(self) -> bool:
        return self.entries_changed > 0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "entries_changed": self.entries_changed,
            "delta_frobenius": self.delta_frobenius,
            "base_frobenius": self.base_frobenius,
            "relative_frobenius": self.relative_frobenius,
            "operations": list(self.op_ids),
        }


@dataclass
class PatchPlan:
    spec_sha256: str
    payload_sha256: str
    revision: str
    targeted: dict[str, TensorPlan]

    @property
    def modified(self) -> dict[str, TensorPlan]:
        return {n: t for n, t in self.targeted.items() if t.modified}

    @property
    def is_noop(self) -> bool:
        return not self.modified

    @property
    def total_entries_changed(self) -> int:
        return sum(t.entries_changed for t in self.targeted.values())

    def patch_digest(self) -> str:
        """Lineage id: spec + payload + exact before/after bytes of every targeted tensor."""
        doc = {
            "kind": "mechinterp-weight-patch-digest/v1",
            "spec_sha256": self.spec_sha256,
            "payload_sha256": self.payload_sha256,
            "revision": self.revision,
            "tensors": [
                [n, t.before_sha256, t.after_sha256] for n, t in sorted(self.targeted.items())
            ],
        }
        return hashlib.sha256(
            json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def receipt(self) -> dict:
        return {
            "schema_version": "mechinterp-weight-patch-receipt/v1",
            "patch_digest": self.patch_digest(),
            "spec_sha256": self.spec_sha256,
            "payload_sha256": self.payload_sha256,
            "revision": self.revision,
            "noop": self.is_noop,
            "total_entries_changed": self.total_entries_changed,
            "tensors": [t.summary() for _, t in sorted(self.targeted.items())],
        }


def build_plan(
    spec: WeightPatchSpec,
    spec_sha256: str,
    payload: Payload,
    read_tensor: Callable[[str], torch.Tensor],
) -> PatchPlan:
    """Validate and compute the patch result without mutating anything."""
    budget = spec.budget
    targeted: dict[str, TensorPlan] = {}
    for name in spec.targeted_tensors():
        pin = spec.base.tensors[name]
        base = read_tensor(name)
        if torch_dtype_name(base.dtype) != pin.dtype or list(base.shape) != pin.shape:
            raise WeightPatchError(
                "base_hash_mismatch",
                f"{name}: base is {torch_dtype_name(base.dtype)}{list(base.shape)}, "
                f"spec pins {pin.dtype}{pin.shape}",
            )
        if tensor_sha256(base) != pin.sha256:
            raise WeightPatchError("base_hash_mismatch", f"{name}: base tensor bytes differ from the pin")
        if not _is_finite(base):
            raise WeightPatchError("nonfinite", f"{name}: base tensor contains NaN/Inf")

        work = base.clone()
        op_ids = []
        for op in spec.operations:
            if op.tensor == name:
                _apply_operation(work, op, payload, budget.max_index_count)
                op_ids.append(op.id)
        if not _is_finite(work):
            raise WeightPatchError("nonfinite", f"{name}: patched tensor contains NaN/Inf")

        changed = int((work != base).sum().item())
        delta = _sq_diff_sum(work, base) ** 0.5
        norm = _sq_diff_sum(base, None) ** 0.5
        if delta == 0.0:
            rel = 0.0
        elif norm == 0.0:
            rel = float("inf")
        else:
            rel = delta / norm
        if rel > budget.max_relative_frobenius:
            raise WeightPatchError(
                "budget",
                f"{name}: relative Frobenius change {rel:.6g} exceeds "
                f"max_relative_frobenius {budget.max_relative_frobenius:.6g}",
            )
        targeted[name] = TensorPlan(
            name=name,
            dtype=pin.dtype,
            shape=tuple(base.shape),
            before_sha256=pin.sha256,
            after_sha256=tensor_sha256(work) if changed else pin.sha256,
            entries_changed=changed,
            delta_frobenius=delta,
            base_frobenius=norm,
            relative_frobenius=rel,
            op_ids=op_ids,
            new_tensor=work if changed else None,
        )

    plan = PatchPlan(spec_sha256, payload.sha256, spec.base.revision, targeted)
    if len(plan.modified) > budget.max_modified_tensors:
        raise WeightPatchError("budget", "modified tensor count exceeds max_modified_tensors")
    if plan.total_entries_changed > budget.max_modified_entries:
        raise WeightPatchError(
            "budget",
            f"{plan.total_entries_changed} modified entries exceed max_modified_entries "
            f"{budget.max_modified_entries}",
        )
    return plan
