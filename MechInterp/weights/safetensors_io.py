"""Strict, pickle-free, torch-free safetensors header reader and byte hashing.

The weight-patch engine binds a patch to *exact bytes*. These helpers parse a
local ``.safetensors`` file's header with full structural validation and hash
raw tensor data regions straight from the file, so the digest covers the actual
stored dtype bytes (bfloat16 included) with no numeric round trip.

Nothing here loads a tensor into memory except ``read_region`` (one tensor), and
nothing here ever unpickles or touches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from MechInterp.weights.errors import WeightPatchError

MAX_HEADER_BYTES = 64 * 1024 * 1024
_CHUNK = 8 * 1024 * 1024

# safetensors dtype tag -> (itemsize, torch dtype name)
DTYPES: dict[str, tuple[int, str]] = {
    "F64": (8, "float64"),
    "F32": (4, "float32"),
    "F16": (2, "float16"),
    "BF16": (2, "bfloat16"),
    "I64": (8, "int64"),
    "I32": (4, "int32"),
    "I16": (2, "int16"),
    "I8": (1, "int8"),
    "U64": (8, "uint64"),
    "U32": (4, "uint32"),
    "U16": (2, "uint16"),
    "U8": (1, "uint8"),
    "BOOL": (1, "bool"),
    "F8_E4M3": (1, "float8_e4m3fn"),
    "F8_E5M2": (1, "float8_e5m2"),
}
FLOAT_TAGS = frozenset({"F64", "F32", "F16", "BF16"})
TORCH_NAME_TO_TAG = {name: tag for tag, (_, name) in DTYPES.items()}
FLOAT_TORCH_NAMES = frozenset(DTYPES[t][1] for t in FLOAT_TAGS)


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype_tag: str
    shape: tuple[int, ...]
    begin: int  # data-region-relative start
    end: int  # data-region-relative end (exclusive)

    @property
    def torch_dtype(self) -> str:
        return DTYPES[self.dtype_tag][1]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class SafetensorsFile:
    path: Path
    size: int
    header_len: int
    entries: dict[str, TensorEntry]
    metadata: dict[str, str]

    @property
    def data_start(self) -> int:
        return 8 + self.header_len

    def absolute_range(self, entry: TensorEntry) -> tuple[int, int]:
        return self.data_start + entry.begin, self.data_start + entry.end

    def ordered_entries(self) -> list[TensorEntry]:
        return sorted(self.entries.values(), key=lambda e: (e.begin, e.end, e.name))


def _reject_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise WeightPatchError("index", f"duplicate key {key!r} in JSON object")
        out[key] = value
    return out


def _reject_constant(token):
    raise WeightPatchError("index", f"non-finite JSON constant {token!r}")


def strict_json_loads(data: bytes | str):
    """JSON parse that refuses duplicate keys and NaN/Infinity constants."""
    try:
        return json.loads(
            data, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant
        )
    except WeightPatchError:
        raise
    except (ValueError, UnicodeDecodeError) as exc:
        raise WeightPatchError("index", f"malformed JSON: {exc}") from None


def _plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def read_header(path: str | Path) -> SafetensorsFile:
    """Parse and structurally validate a safetensors header.

    Enforced: bounded header, UTF-8 JSON object without duplicate keys, known
    dtype tags, non-negative integer shapes, ``end - begin == numel * itemsize``,
    tensors tiling the data region exactly (no gaps, no overlap, no trailing
    bytes), and a file size equal to ``8 + header + data``.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
        with open(p, "rb") as fh:
            prefix = fh.read(8)
            if len(prefix) != 8:
                raise WeightPatchError("index", f"{p.name}: truncated before header length")
            (header_len,) = struct.unpack("<Q", prefix)
            if header_len == 0 or header_len > MAX_HEADER_BYTES or 8 + header_len > size:
                raise WeightPatchError(
                    "index", f"{p.name}: invalid header length {header_len}"
                )
            raw = fh.read(header_len)
    except OSError as exc:
        raise WeightPatchError("path", f"cannot read {p.name}: {exc.strerror}") from None
    if len(raw) != header_len:
        raise WeightPatchError("index", f"{p.name}: truncated header")
    if not raw.lstrip(b" ").startswith(b"{"):
        raise WeightPatchError("index", f"{p.name}: header is not a JSON object")
    header = strict_json_loads(raw)
    if not isinstance(header, dict):
        raise WeightPatchError("index", f"{p.name}: header is not a JSON object")

    metadata: dict[str, str] = {}
    entries: dict[str, TensorEntry] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            if not isinstance(spec, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in spec.items()
            ):
                raise WeightPatchError("index", f"{p.name}: invalid __metadata__")
            metadata = dict(spec)
            continue
        if not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "data_offsets"}:
            raise WeightPatchError("index", f"{p.name}: malformed entry for {name!r}")
        tag = spec["dtype"]
        if tag not in DTYPES:
            raise WeightPatchError("index", f"{p.name}: unsupported dtype {tag!r} for {name!r}")
        shape = spec["shape"]
        if not isinstance(shape, list) or not all(_plain_int(d) and d >= 0 for d in shape):
            raise WeightPatchError("index", f"{p.name}: invalid shape for {name!r}")
        offsets = spec["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_plain_int(o) and o >= 0 for o in offsets)
            or offsets[0] > offsets[1]
        ):
            raise WeightPatchError("index", f"{p.name}: invalid data_offsets for {name!r}")
        numel = 1
        for d in shape:
            numel *= d
            if numel > 1 << 62:
                raise WeightPatchError("index", f"{p.name}: shape too large for {name!r}")
        if offsets[1] - offsets[0] != numel * DTYPES[tag][0]:
            raise WeightPatchError(
                "index", f"{p.name}: byte length does not match shape/dtype for {name!r}"
            )
        entries[name] = TensorEntry(name, tag, tuple(shape), offsets[0], offsets[1])

    data_len = size - 8 - header_len
    cursor = 0
    for entry in sorted(entries.values(), key=lambda e: (e.begin, e.end, e.name)):
        if entry.begin != cursor:
            raise WeightPatchError(
                "index", f"{p.name}: tensor data not contiguous at {entry.name!r}"
            )
        cursor = entry.end
    if cursor != data_len:
        raise WeightPatchError("index", f"{p.name}: data region size mismatch")
    return SafetensorsFile(p, size, header_len, entries, metadata)


def _stream(fh, count: int) -> Iterator[bytes]:
    remaining = count
    while remaining > 0:
        chunk = fh.read(min(_CHUNK, remaining))
        if not chunk:
            raise WeightPatchError("index", "file shorter than its header declares")
        remaining -= len(chunk)
        yield chunk


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                h.update(chunk)
    except OSError as exc:
        raise WeightPatchError("path", f"cannot read file: {exc.strerror}") from None
    return h.hexdigest()


def hash_file_with_regions(sf: SafetensorsFile) -> tuple[str, dict[str, str]]:
    """One sequential pass: (whole-file sha256, {tensor name: data-region sha256})."""
    whole = hashlib.sha256()
    regions: dict[str, str] = {}
    with open(sf.path, "rb") as fh:
        whole.update(fh.read(sf.data_start))
        for entry in sf.ordered_entries():
            region = hashlib.sha256()
            for chunk in _stream(fh, entry.nbytes):
                whole.update(chunk)
                region.update(chunk)
            regions[entry.name] = region.hexdigest()
        if fh.read(1):
            raise WeightPatchError("index", f"{sf.path.name}: trailing bytes after data")
    return whole.hexdigest(), regions


def read_region(sf: SafetensorsFile, entry: TensorEntry) -> bytearray:
    """Read exactly one tensor's raw bytes (a mutable buffer for torch.frombuffer)."""
    start, end = sf.absolute_range(entry)
    with open(sf.path, "rb") as fh:
        fh.seek(start)
        buf = bytearray(end - start)
        view = memoryview(buf)
        got = 0
        while got < len(buf):
            n = fh.readinto(view[got:])
            if not n:
                raise WeightPatchError("index", f"{sf.path.name}: short read for {entry.name!r}")
            got += n
    return buf


def fsync_path(path: str | Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
