"""Check or explicitly refresh hashes in the fixed packaged-worker manifest.

The default is read-only. ``--write`` changes only the five already-declared
members' size/hash values and the closure digest. It cannot add members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tuner.runtime.packaged_worker_closure import (  # noqa: E402
    PACKAGED_WORKER_CLOSURE_SCHEMA,
    PACKAGED_WORKER_MEMBERS,
    MAX_RESOURCE_BYTES,
    stable_read,
)


_MANIFEST = Path("tuner/runtime/manifests/packaged-training-worker-v1.json")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest field")
        result[key] = value
    return result


def _refresh(root: Path, raw: bytes) -> bytes:
    document = json.loads(raw, object_pairs_hook=_unique)
    if (type(document) is not dict
            or set(document) != {"schema_version", "members", "closure_digest"}
            or document["schema_version"] != PACKAGED_WORKER_CLOSURE_SCHEMA
            or _canonical(document) + b"\n" != raw):
        raise ValueError("packaged worker manifest is invalid")
    members = document["members"]
    if (type(members) is not list or len(members) != len(PACKAGED_WORKER_MEMBERS)
            or tuple(member.get("path") if type(member) is dict else None for member in members)
            != PACKAGED_WORKER_MEMBERS):
        raise ValueError("packaged worker inventory differs")
    unsigned = {"schema_version": document["schema_version"], "members": members}
    if hashlib.sha256(_canonical(unsigned)).hexdigest() != document["closure_digest"]:
        raise ValueError("packaged worker manifest digest is invalid")
    refreshed: list[dict[str, object]] = []
    for name, member in zip(PACKAGED_WORKER_MEMBERS, members, strict=True):
        if type(member) is not dict or set(member) != {"path", "size_bytes", "sha256"}:
            raise ValueError("packaged worker member fields differ")
        if (type(member["size_bytes"]) is not int or not 1 <= member["size_bytes"] <= MAX_RESOURCE_BYTES
                or type(member["sha256"]) is not str or len(member["sha256"]) != 64):
            raise ValueError("packaged worker member metadata is invalid")
        content = stable_read(root / "tuner" / "runtime" / name, MAX_RESOURCE_BYTES)
        if not content:
            raise ValueError("packaged worker member is empty")
        refreshed.append({"path": name, "size_bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest()})
    updated = {"schema_version": PACKAGED_WORKER_CLOSURE_SCHEMA, "members": refreshed}
    updated["closure_digest"] = hashlib.sha256(_canonical(updated)).hexdigest()
    return _canonical(updated) + b"\n"


def _write_atomically(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".packaged-worker-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh only reviewed content hashes.")
    args = parser.parse_args(argv)
    selected = (_ROOT if root is None else root).resolve(strict=True)
    path = selected / _MANIFEST
    try:
        original = stable_read(path, MAX_RESOURCE_BYTES)
        updated = _refresh(selected, original)
        if updated == original:
            print("Packaged worker closure is CURRENT")
            return 0
        if not args.write:
            print("Packaged worker closure is STALE; review and rerun with --write.", file=sys.stderr)
            return 3
        _write_atomically(path, updated)
        if stable_read(path, MAX_RESOURCE_BYTES) != updated or _refresh(selected, updated) != updated:
            raise ValueError("written packaged worker closure is invalid")
        print("Packaged worker closure hashes refreshed")
        return 0
    except (OSError, ValueError, TypeError, UnicodeError):
        print("Packaged worker closure regeneration failed", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
