"""Independent bounded MLX safetensors verification, with no ML imports.

An MLX adapter is not labeled PEFT. Final validation belongs to the explicitly
reloaded saved adapter; upstream in-loop validation is never final evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

from tuner.runtime.native_mlx_material import strict_json


_WIDTH = {"F16": 2, "BF16": 2, "F32": 4}


@dataclass(frozen=True)
class NativeTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


def adapter_tensors(raw: bytes) -> tuple[NativeTensor, ...]:
    if type(raw) is not bytes or not 10 <= len(raw) <= 512 * 1024 * 1024:
        raise ValueError("native adapter byte budget exceeded")
    header_size = struct.unpack_from("<Q", raw)[0]
    if not 2 <= header_size <= min(1024 * 1024, len(raw) - 8):
        raise ValueError("native adapter header is invalid")
    header = strict_json(raw[8:8 + header_size], maximum=1024 * 1024)
    if type(header) is not dict or not 1 <= len(header) <= 100000:
        raise ValueError("native adapter tensor inventory is invalid")
    payload_start = 8 + header_size
    tensors = []
    for name, item in header.items():
        if name == "__metadata__":
            if type(item) is not dict or any(type(k) is not str or type(v) is not str for k, v in item.items()):
                raise ValueError("native adapter metadata is invalid")
            continue
        if (type(name) is not str or not 1 <= len(name) <= 512
                or name.rsplit(".", 1)[-1] not in ("lora_a", "lora_b")
                or any(ord(c) < 32 for c in name)):
            raise ValueError("native adapter contains non-LoRA tensor")
        if type(item) is not dict or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("native tensor fields differ")
        dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
        if (type(dtype) is not str or dtype not in _WIDTH or type(shape) is not list
                or len(shape) != 2 or any(type(n) is not int or not 1 <= n <= 1048576 for n in shape)
                or type(offsets) is not list or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)):
            raise ValueError("native tensor layout is invalid")
        start, end = offsets
        if end - start != math.prod(shape) * _WIDTH[dtype] or payload_start + end > len(raw):
            raise ValueError("native tensor bounds differ")
        tensors.append(NativeTensor(name, dtype, tuple(shape), payload_start + start, payload_start + end))
    if not tensors:
        raise ValueError("native adapter has no tensors")
    cursor = payload_start
    for tensor in sorted(tensors, key=lambda t: t.start):
        if tensor.start != cursor:
            raise ValueError("native adapter has overlap or unclaimed bytes")
        cursor = tensor.end
    if cursor != len(raw):
        raise ValueError("native adapter has trailing bytes")
    return tuple(sorted(tensors, key=lambda t: t.name))


def _values(raw, tensor):
    view = memoryview(raw)[tensor.start:tensor.end]
    if tensor.dtype == "BF16":
        for (bits,) in struct.iter_unpack("<H", view):
            yield struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    else:
        for (value,) in struct.iter_unpack("<e" if tensor.dtype == "F16" else "<f", view):
            yield value


def finite_adapter_tensors(raw: bytes) -> tuple[NativeTensor, ...]:
    tensors = adapter_tensors(raw)
    if any(not math.isfinite(value) for tensor in tensors for value in _values(raw, tensor)):
        raise ValueError("native adapter has nonfinite parameters")
    return tensors


def verify_adapter_delta(initial: bytes, final: bytes, *, expected_shapes: tuple[tuple[str, tuple[int, ...]], ...]):
    before = adapter_tensors(initial)
    after = adapter_tensors(final)
    if (tuple((t.name, t.shape) for t in before) != expected_shapes
            or tuple((t.name, t.shape) for t in after) != expected_shapes
            or tuple(t.dtype for t in before) != tuple(t.dtype for t in after)):
        raise ValueError("native adapter parameter inventory differs")
    count, changed, norm = 0, 0, 0.0
    for old_tensor, new_tensor in zip(before, after):
        for old, new in zip(_values(initial, old_tensor), _values(final, new_tensor)):
            if not math.isfinite(old) or not math.isfinite(new):
                raise ValueError("native adapter has nonfinite parameters")
            delta = new - old
            count += 1
            changed += delta != 0
            norm += delta * delta
    if changed == 0 or not math.isfinite(norm) or norm <= 0:
        raise ValueError("native adapter has no finite parameter update")
    return {
        "initial_sha256": hashlib.sha256(initial).hexdigest(),
        "final_sha256": hashlib.sha256(final).hexdigest(),
        "parameter_count": count, "changed_parameters": changed,
        "delta_l2": math.sqrt(norm),
    }
