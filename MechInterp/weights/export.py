"""Create-only export of an exact modified checkpoint plus a lineage manifest.

Because patching never changes a tensor's dtype or shape, the output keeps every
shard's header and layout byte-for-byte: each shard is copied and ONLY the byte
ranges of modified tensors are overwritten. Untouched tensors (and untouched
shards entirely) are therefore preserved exactly, and this is verified by
re-hashing the output against the base. The checkpoint is staged in a sibling
directory and renamed into place only after verification; an existing destination
is refused, never overwritten.
"""

from __future__ import annotations

import fnmatch
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

from MechInterp.weights.checkpoint import (
    INDEX_NAME,
    LocalCheckpoint,
)
from MechInterp.weights.errors import WeightPatchError
from MechInterp.weights.patch import PatchPlan, tensor_to_bytes
from MechInterp.weights.safetensors_io import (
    fsync_path,
    hash_file_with_regions,
    read_header,
)
from MechInterp.weights.spec import WeightPatchSpec, canonical_sha256

MANIFEST_NAME = "mechinterp-weight-export.manifest.json"
_COPY_CHUNK = 8 * 1024 * 1024


def _copy_file(src: Path, dst: Path) -> None:
    with open(src, "rb") as fin, open(dst, "xb") as fout:
        for chunk in iter(lambda: fin.read(_COPY_CHUNK), b""):
            fout.write(chunk)
        fout.flush()
        os.fsync(fout.fileno())


def _sidecars(spec: WeightPatchSpec, ckpt: LocalCheckpoint, root_real: Path) -> list[Path]:
    cfg = spec.export
    chosen: list[Path] = []
    for entry in sorted(os.listdir(ckpt.root)):
        if entry.endswith(".safetensors") or entry == MANIFEST_NAME:
            continue
        if not any(fnmatch.fnmatchcase(entry, pat) for pat in cfg.sidecar_files):
            continue
        path = ckpt.root / entry
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if stat.S_ISLNK(st.st_mode):
            target = path.resolve(strict=False)
            if not spec.base.allow_symlinked_files or not target.is_file():
                raise WeightPatchError("path", f"sidecar {entry!r} is a disallowed or broken symlink")
            st = os.stat(target)
        if not stat.S_ISREG(st.st_mode):
            continue
        if st.st_size > cfg.max_sidecar_bytes:
            raise WeightPatchError("path", f"sidecar {entry!r} exceeds export.max_sidecar_bytes")
        chosen.append(path)
    return chosen


def _sha_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_COPY_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _rename_create_only(source: Path, destination: Path) -> None:
    """Publish atomically without replacing even an empty racing directory."""
    if sys.platform == "win32":
        # Windows rename refuses an existing destination, unlike POSIX rename.
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    else:
        raise WeightPatchError("path", "atomic create-only directory publication is unsupported on this platform")
    if result != 0:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise WeightPatchError("output_exists", "export destination appeared during export")
        raise OSError(code, os.strerror(code))


def export_checkpoint(
    spec: WeightPatchSpec,
    spec_sha256: str,
    plan: PatchPlan,
    ckpt: LocalCheckpoint,
    binding_report: dict,
) -> dict:
    """Write the patched checkpoint to ``spec.export.output_dir``; return the manifest."""
    if spec.export is None:
        raise WeightPatchError("schema", "the spec has no export block")
    final = Path(spec.export.output_dir)
    if os.path.lexists(final):
        raise WeightPatchError("output_exists", "export destination already exists; refusing to overwrite")
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.partial-", dir=parent))
    try:
        manifest = _build(spec, spec_sha256, plan, ckpt, binding_report, staging)
        if os.path.lexists(final):  # re-check immediately before the rename
            raise WeightPatchError("output_exists", "export destination appeared during export")
        fsync_path(staging)
        _rename_create_only(staging, final)
        try:
            fsync_path(parent)
        except OSError:  # some filesystems (e.g. drvfs/exFAT) cannot fsync a directory
            pass
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build(spec, spec_sha256, plan: PatchPlan, ckpt: LocalCheckpoint, binding_report, staging: Path) -> dict:
    root_real = ckpt.root.resolve(strict=True)
    modified_by_file: dict[str, list[str]] = {}
    for name in plan.modified:
        modified_by_file.setdefault(ckpt.tensor_to_file[name], []).append(name)

    out_files: dict[str, dict] = {}
    for fname, sf in sorted(ckpt.files.items()):
        src_whole, src_regions = ckpt.file_hashes(fname)
        dst = staging / fname
        _copy_file(sf.path, dst)
        names = modified_by_file.get(fname, [])
        if names:
            with open(dst, "r+b") as fh:
                for name in names:
                    entry = sf.entries[name]
                    start, _ = sf.absolute_range(entry)
                    fh.seek(start)
                    data = tensor_to_bytes(plan.modified[name].new_tensor)
                    if len(data) != entry.nbytes:
                        raise WeightPatchError("apply", f"{name}: patched byte length changed")
                    fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        # Readback verification against the base, whole shard.
        dst_sf = read_header(dst)
        with open(sf.path, "rb") as a, open(dst, "rb") as b:
            if a.read(sf.data_start) != b.read(sf.data_start):
                raise WeightPatchError("apply", f"{fname}: output header differs from the base header")
        if dst_sf.entries != sf.entries:
            raise WeightPatchError("apply", f"{fname}: output tensor layout differs from the base")
        dst_whole, dst_regions = hash_file_with_regions(dst_sf)
        for tname, region in dst_regions.items():
            if tname in plan.modified:
                if region != plan.modified[tname].after_sha256:
                    raise WeightPatchError("apply", f"{tname}: output bytes differ from the plan")
            elif region != src_regions[tname]:
                raise WeightPatchError("apply", f"{tname}: untouched tensor changed in the output")
        if not names and dst_whole != src_whole:
            raise WeightPatchError("apply", f"{fname}: unmodified shard is not byte-identical")
        out_files[fname] = {"sha256": dst_whole, "size": dst_sf.size, "modified_tensors": sorted(names)}

    copied: dict[str, dict] = {}
    if ckpt.index_path is not None:
        shutil.copyfile(ckpt.index_path, staging / INDEX_NAME)
        copied[INDEX_NAME] = {"sha256": _sha_of(staging / INDEX_NAME)}
    for path in _sidecars(spec, ckpt, root_real):
        if path.name == INDEX_NAME:
            continue
        shutil.copyfile(path, staging / path.name)
        copied[path.name] = {"sha256": _sha_of(staging / path.name)}

    try:
        import torch

        versions = {"python": sys.version.split()[0], "torch": torch.__version__}
    except Exception:  # pragma: no cover
        versions = {"python": sys.version.split()[0]}
    manifest = {
        "schema_version": "mechinterp-weight-export/v1",
        "patch_digest": plan.patch_digest(),
        "spec": {
            "sha256": spec_sha256,
            "canonical_sha256": canonical_sha256(spec),
            "description": spec.description,
        },
        "payload": {"sha256": plan.payload_sha256},
        "base": {
            "model_id": spec.base.model_id,
            "revision": spec.base.revision,
            "binding": binding_report,
            "files": {f: {"sha256": ckpt.file_hashes(f)[0]} for f in sorted(ckpt.files)},
        },
        "budget": spec.budget.model_dump(mode="json"),
        "totals": {
            "tensors_modified": len(plan.modified),
            "entries_changed": plan.total_entries_changed,
            "noop": plan.is_noop,
        },
        "tensors": [
            {**t.summary(), "file": ckpt.tensor_to_file[t.name]}
            for _, t in sorted(plan.targeted.items())
        ],
        "output": {"weight_files": out_files, "copied_files": copied},
        "engine": versions,
    }
    manifest_path = staging / MANIFEST_NAME
    with open(manifest_path, "x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    return manifest
