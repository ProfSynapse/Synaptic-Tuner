"""Tiny local checkpoints and patch specs for CPU weight-patch tests.

Hashes here are computed independently of the engine (raw dtype bytes via
``view(torch.uint8)``), so a bug in the engine's hashing cannot hide itself.
Nothing is downloaded; tensor names are arbitrary fixture names.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

REVISION = "a" * 40
OTHER_REVISION = "b" * 40

DOWN0 = "blk.0.ffn.down.weight"  # [out=8, in=16]
GATE0 = "blk.0.ffn.gate.weight"  # [out=16, in=8]
DOWN1 = "blk.1.ffn.down.weight"  # [8, 16]
EMBED = "tok.embed.weight"  # [32, 8]
NORM = "blk.0.norm.weight"  # float32 [8]


def raw_sha(t: torch.Tensor) -> str:
    flat = t.detach().contiguous().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_tensors(dtype=torch.bfloat16, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, dt=dtype):
        return torch.randn(*shape, generator=g).to(dt)

    return {
        DOWN0: rnd(8, 16),
        GATE0: rnd(16, 8),
        DOWN1: rnd(8, 16),
        EMBED: rnd(32, 8),
        NORM: rnd(8, dt=torch.float32),
    }


def write_checkpoint(root: Path, tensors=None, *, sharded: bool = True) -> dict[str, torch.Tensor]:
    """Write a checkpoint directory (sharded with an index, or one file)."""
    tensors = tensors or base_tensors()
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"fixture": True}))
    meta = {"format": "pt"}
    if sharded:
        shard_a = {k: tensors[k] for k in (DOWN0, GATE0, NORM)}
        shard_b = {k: tensors[k] for k in (DOWN1, EMBED)}
        save_file(shard_a, str(root / "model-00001-of-00002.safetensors"), metadata=meta)
        save_file(shard_b, str(root / "model-00002-of-00002.safetensors"), metadata=meta)
        weight_map = {k: "model-00001-of-00002.safetensors" for k in shard_a}
        weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard_b})
        total = sum(t.numel() * t.element_size() for t in tensors.values())
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
        )
    else:
        save_file(dict(tensors), str(root / "model.safetensors"), metadata=meta)
    return tensors


def pin(t: torch.Tensor) -> dict:
    return {
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": list(t.shape),
        "sha256": raw_sha(t),
    }


def make_spec(
    tmp: Path,
    tensors: dict[str, torch.Tensor],
    operations: list[dict],
    arrays: dict[str, torch.Tensor],
    *,
    checkpoint: Path | None = None,
    budget: dict | None = None,
    base_extra: dict | None = None,
    export_dir: Path | None = None,
    revision: str = REVISION,
    name: str = "patch",
) -> Path:
    """Write payload + spec YAML; return the spec path. Pins every targeted tensor."""
    tmp.mkdir(parents=True, exist_ok=True)
    payload_path = tmp / f"{name}.payload.safetensors"
    save_file({k: v.contiguous() for k, v in arrays.items()}, str(payload_path))
    targeted = []
    for op in operations:
        if op["tensor"] not in targeted:
            targeted.append(op["tensor"])
    base = {
        "model_id": "fixture/model",
        "revision": revision,
        "tensors": {n: pin(tensors[n]) for n in targeted},
    }
    if checkpoint is not None:
        base["checkpoint"] = str(checkpoint)
    base.update(base_extra or {})
    spec = {
        "schema_version": "mechinterp-weight-patch/v1",
        "base": base,
        "budget": {
            "allowed_tensors": ["blk.*"],
            "max_modified_tensors": 4,
            "max_modified_entries": 10_000,
            "max_relative_frobenius": 5.0,
            **(budget or {}),
        },
        "payload": {"path": payload_path.name, "sha256": file_sha(payload_path)},
        "operations": operations,
    }
    if export_dir is not None:
        spec["export"] = {"output_dir": str(export_dir)}
    spec_path = tmp / f"{name}.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


def rewrite_spec(spec_path: Path, mutate) -> None:
    data = yaml.safe_load(spec_path.read_text())
    mutate(data)
    spec_path.write_text(yaml.safe_dump(data, sort_keys=False))


def standard_ops_and_arrays(seed: int = 1):
    """rank-one on selected columns of DOWN0 + replace_rows on GATE0."""
    g = torch.Generator().manual_seed(seed)
    ops = [
        {
            "id": "r1",
            "tensor": DOWN0,
            "op": "add_rank_one",
            "scale": 0.05,
            "cols": [1, 4, 9],
            "arrays": {"u": "r1.u", "v": "r1.v"},
        },
        {
            "id": "rows",
            "tensor": GATE0,
            "op": "replace_rows",
            "rows": [2, 5],
            "arrays": {"values": "rows.values"},
        },
    ]
    arrays = {
        "r1.u": torch.randn(8, generator=g),
        "r1.v": torch.randn(3, generator=g),
        "rows.values": torch.randn(2, 8, generator=g),
    }
    return ops, arrays
