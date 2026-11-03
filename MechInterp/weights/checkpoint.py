"""Inspect and bind a LOCAL safetensors checkpoint (no download, no pickle).

A checkpoint is either one ``.safetensors`` file or a directory holding
``model.safetensors.index.json`` + its shards (or a single ``model.safetensors``).
Everything is validated structurally before any byte is trusted:

  * shard names in the index must be plain file names (no separators, no ``..``),
  * no symlinks unless the spec opts in, and then only to regular files,
  * every shard header must parse strictly (see safetensors_io.read_header),
  * the index ``weight_map`` and the shard headers must agree exactly
    (no missing, extra, or duplicated tensors).

Revision binding. A bare revision string cannot be authenticated from file
bytes, so the binding has two independent legs: (1) exact content pins
(``base.index_sha256``/``base.files_sha256`` here, plus per-tensor byte hashes in
the spec), and (2) local revision evidence when the layout carries it (a Hugging
Face snapshot directory name, or ``.cache/huggingface/download/*.metadata``
commit lines). Any contradicting evidence refuses; with no evidence the content
pins must be present.
"""

from __future__ import annotations

import fnmatch
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from MechInterp.weights.errors import WeightPatchError
from MechInterp.weights.safetensors_io import (
    SafetensorsFile,
    TensorEntry,
    file_sha256,
    hash_file_with_regions,
    read_header,
    strict_json_loads,
)

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"
_REV40 = re.compile(r"^[0-9a-f]{40}$")
_MAX_INDEX_BYTES = 64 * 1024 * 1024


@dataclass
class LocalCheckpoint:
    root: Path  # directory containing the weight files
    files: dict[str, SafetensorsFile]  # file name -> parsed header
    tensor_to_file: dict[str, str]
    index_path: Optional[Path]
    index_total_size: Optional[int]
    is_single_file_input: bool
    _hash_cache: dict[str, tuple[str, dict[str, str]]] = field(default_factory=dict)

    def entry(self, tensor: str) -> tuple[SafetensorsFile, TensorEntry]:
        fname = self.tensor_to_file.get(tensor)
        if fname is None:
            raise WeightPatchError("binding", f"tensor {tensor!r} is not in this checkpoint")
        sf = self.files[fname]
        return sf, sf.entries[tensor]

    def file_hashes(self, fname: str) -> tuple[str, dict[str, str]]:
        """(whole-file sha256, per-tensor region sha256), computed once."""
        if fname not in self._hash_cache:
            self._hash_cache[fname] = hash_file_with_regions(self.files[fname])
        return self._hash_cache[fname]

    def tensor_sha256(self, tensor: str) -> str:
        fname = self.tensor_to_file[tensor]
        return self.file_hashes(fname)[1][tensor]

    def index_or_file_sha256(self) -> str:
        if self.index_path is not None:
            return file_sha256(self.index_path)
        (only,) = self.files.values()
        return self.file_hashes(only.path.name)[0]

    def total_tensor_bytes(self) -> int:
        return sum(e.nbytes for sf in self.files.values() for e in sf.entries.values())


def _check_regular(path: Path, root_real: Path, allow_symlinks: bool, label: str) -> None:
    try:
        st = os.lstat(path)
    except OSError:
        raise WeightPatchError("path", f"{label}: file does not exist") from None
    if stat.S_ISLNK(st.st_mode):
        if not allow_symlinks:
            raise WeightPatchError(
                "path", f"{label}: symlinked weight files are refused (base.allow_symlinked_files)"
            )
        try:
            target = path.resolve(strict=True)
        except OSError:
            raise WeightPatchError("path", f"{label}: dangling symlink") from None
        if not target.is_file():
            raise WeightPatchError("path", f"{label}: symlink target is not a regular file")
        return
    if not stat.S_ISREG(st.st_mode):
        raise WeightPatchError("path", f"{label}: not a regular file")
    # A regular leaf can still sit behind a symlinked directory component.
    if not allow_symlinks and path.resolve(strict=True).parent != root_real:
        raise WeightPatchError("path", f"{label}: resolves outside the checkpoint directory")


def _plain_name(name: str) -> bool:
    return (
        isinstance(name, str)
        and name not in {"", ".", ".."}
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
        and not name.startswith(".")
        and name.endswith(".safetensors")
    )


def open_checkpoint(path: str | Path, *, allow_symlinked_files: bool = False) -> LocalCheckpoint:
    p = Path(path)
    if not p.exists():
        raise WeightPatchError("path", "checkpoint path does not exist")
    if p.is_file():
        root = p.parent
        root_real = root.resolve(strict=True)
        if not _plain_name(p.name):
            raise WeightPatchError("path", "a checkpoint file must be a plain .safetensors name")
        _check_regular(p, root_real, allow_symlinked_files, p.name)
        sf = read_header(p)
        return LocalCheckpoint(
            root, {p.name: sf}, {name: p.name for name in sf.entries}, None, None, True
        )
    if not p.is_dir():
        raise WeightPatchError("path", "checkpoint path is neither a file nor a directory")
    root = p
    root_real = root.resolve(strict=True)

    index_path = root / INDEX_NAME
    single_path = root / SINGLE_NAME
    has_index = os.path.lexists(index_path)
    has_single = os.path.lexists(single_path)
    if has_index and has_single:
        raise WeightPatchError(
            "index", f"both {INDEX_NAME} and {SINGLE_NAME} exist; layout is ambiguous"
        )
    if not has_index and not has_single:
        raise WeightPatchError("index", f"no {INDEX_NAME} or {SINGLE_NAME} in checkpoint directory")

    if has_single:
        _check_regular(single_path, root_real, allow_symlinked_files, SINGLE_NAME)
        sf = read_header(single_path)
        return LocalCheckpoint(
            root, {SINGLE_NAME: sf}, {n: SINGLE_NAME for n in sf.entries}, None, None, False
        )

    _check_regular(index_path, root_real, allow_symlinked_files, INDEX_NAME)
    if index_path.stat().st_size > _MAX_INDEX_BYTES:
        raise WeightPatchError("index", f"{INDEX_NAME} is too large")
    index = strict_json_loads(index_path.read_bytes())
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise WeightPatchError("index", f"{INDEX_NAME}: weight_map must be an object")
    weight_map = index["weight_map"]
    if not weight_map:
        raise WeightPatchError("index", f"{INDEX_NAME}: weight_map is empty")
    total_size = None
    meta = index.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("total_size"), int) and not isinstance(
        meta.get("total_size"), bool
    ):
        total_size = meta["total_size"]
    for tensor, fname in weight_map.items():
        if not isinstance(tensor, str) or not tensor:
            raise WeightPatchError("index", f"{INDEX_NAME}: invalid tensor name")
        if not _plain_name(fname):
            raise WeightPatchError(
                "index", f"{INDEX_NAME}: shard name for {tensor!r} must be a plain .safetensors file name"
            )

    files: dict[str, SafetensorsFile] = {}
    for fname in sorted(set(weight_map.values())):
        fpath = root / fname
        _check_regular(fpath, root_real, allow_symlinked_files, fname)
        files[fname] = read_header(fpath)

    for tensor, fname in weight_map.items():
        if tensor not in files[fname].entries:
            raise WeightPatchError(
                "index", f"{INDEX_NAME} maps {tensor!r} to {fname}, which does not contain it"
            )
    for fname, sf in files.items():
        extra = [t for t in sf.entries if weight_map.get(t) != fname]
        if extra:
            raise WeightPatchError(
                "index", f"{fname} holds tensors not mapped to it by the index: {extra[:3]}"
            )
    return LocalCheckpoint(root, files, dict(weight_map), index_path, total_size, False)


def revision_evidence(ckpt: LocalCheckpoint) -> list[dict]:
    """Local, non-network evidence about which revision this directory holds."""
    evidence: list[dict] = []
    root_real = ckpt.root.resolve(strict=True)
    if root_real.parent.name == "snapshots" and _REV40.match(root_real.name):
        evidence.append({"source": "hf_snapshot_directory", "value": root_real.name})
    for fname in sorted(ckpt.files):
        meta = ckpt.root / ".cache" / "huggingface" / "download" / f"{fname}.metadata"
        try:
            if meta.is_file() and meta.stat().st_size <= 4096:
                first = meta.read_text(encoding="utf-8", errors="strict").splitlines()
                if first and _REV40.match(first[0].strip()):
                    evidence.append(
                        {"source": f"hf_download_metadata:{fname}", "value": first[0].strip()}
                    )
        except (OSError, UnicodeDecodeError):
            continue
    return evidence


def verify_binding(spec, ckpt: LocalCheckpoint) -> dict:
    """Verify the spec's source declaration against this local checkpoint.

    Refuses on any contradiction. Returns a report of what was proven.
    """
    base = spec.base
    report: dict = {"revision": base.revision, "model_id": base.model_id}

    content_pinned = False
    if base.index_sha256 is not None:
        actual = ckpt.index_or_file_sha256()
        if actual != base.index_sha256:
            raise WeightPatchError("binding", "checkpoint index/file sha256 does not match base.index_sha256")
        content_pinned = True
    report["index_sha256"] = ckpt.index_or_file_sha256() if content_pinned else None

    if base.files_sha256 is not None:
        if set(base.files_sha256) != set(ckpt.files):
            raise WeightPatchError(
                "binding",
                "base.files_sha256 must list exactly the checkpoint's weight files "
                f"(declared {sorted(base.files_sha256)}, found {sorted(ckpt.files)})",
            )
        for fname, expected in base.files_sha256.items():
            if ckpt.file_hashes(fname)[0] != expected:
                raise WeightPatchError("binding", f"file sha256 mismatch for {fname}")
        content_pinned = True

    evidence = revision_evidence(ckpt)
    for item in evidence:
        item["matches"] = item["value"] == base.revision
    report["revision_evidence"] = evidence
    if any(not item["matches"] for item in evidence):
        raise WeightPatchError(
            "binding", "local revision evidence contradicts the declared base.revision"
        )
    if not content_pinned and not evidence:
        raise WeightPatchError(
            "binding",
            "cannot bind the declared revision: provide base.index_sha256 or base.files_sha256, "
            "or use a checkpoint layout that carries revision evidence",
        )

    for name, pin in base.tensors.items():
        if name not in ckpt.tensor_to_file:
            raise WeightPatchError("binding", f"targeted tensor {name!r} is not in the checkpoint")
        _, entry = ckpt.entry(name)
        if entry.torch_dtype != pin.dtype or list(entry.shape) != pin.shape:
            raise WeightPatchError(
                "base_hash_mismatch",
                f"{name}: checkpoint is {entry.torch_dtype}{list(entry.shape)}, "
                f"spec pins {pin.dtype}{pin.shape}",
            )
        if ckpt.tensor_sha256(name) != pin.sha256:
            raise WeightPatchError("base_hash_mismatch", f"{name}: base tensor bytes differ from the pin")
    report["content_pinned"] = content_pinned
    report["targeted_tensors_verified"] = len(base.tensors)
    return report


def inspect_checkpoint(request, ckpt: Optional[LocalCheckpoint] = None) -> dict:
    """Build the ``weights-inspect`` report (JSON-serializable)."""
    ckpt = ckpt or open_checkpoint(
        request.checkpoint, allow_symlinked_files=request.allow_symlinked_files
    )
    dtypes: dict[str, int] = {}
    for sf in ckpt.files.values():
        for e in sf.entries.values():
            dtypes[e.torch_dtype] = dtypes.get(e.torch_dtype, 0) + 1
    total = ckpt.total_tensor_bytes()
    report: dict = {
        "schema_version": request.schema_version,
        "layout": "single_file" if ckpt.index_path is None else "sharded_index",
        "n_files": len(ckpt.files),
        "n_tensors": len(ckpt.tensor_to_file),
        "tensor_bytes": total,
        "dtype_counts": dict(sorted(dtypes.items())),
        "index_total_size_declared": ckpt.index_total_size,
        "index_total_size_matches": (
            None if ckpt.index_total_size is None else ckpt.index_total_size == total
        ),
        "revision_evidence": revision_evidence(ckpt),
        "files": [],
        "tensors": [],
    }
    for fname, sf in sorted(ckpt.files.items()):
        item = {"name": fname, "size": sf.size, "n_tensors": len(sf.entries)}
        if request.hash_files:
            item["sha256"] = ckpt.file_hashes(fname)[0]
        report["files"].append(item)
    if ckpt.index_path is not None:
        report["index_sha256"] = file_sha256(ckpt.index_path)

    names = sorted(
        n
        for n in ckpt.tensor_to_file
        if any(fnmatch.fnmatchcase(n, pat) for pat in request.tensors)
    )
    report["n_tensors_matched"] = len(names)
    report["listing_truncated"] = len(names) > request.max_listed_tensors
    snippet: dict[str, dict] = {}
    for name in names[: request.max_listed_tensors]:
        sf, entry = ckpt.entry(name)
        item = {
            "name": name,
            "file": sf.path.name,
            "dtype": entry.torch_dtype,
            "shape": list(entry.shape),
            "nbytes": entry.nbytes,
        }
        if request.hash_tensors:
            item["sha256"] = ckpt.tensor_sha256(name)
            snippet[name] = {"dtype": entry.torch_dtype, "shape": list(entry.shape), "sha256": item["sha256"]}
        report["tensors"].append(item)
    if snippet:
        report["base_tensors_snippet"] = snippet
    if request.projections:
        from MechInterp.weights.projection import run_projections

        report["projections"] = run_projections(request, ckpt)
    return report
