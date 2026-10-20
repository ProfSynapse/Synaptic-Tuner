"""Check fixed GRPO inventory hashes; initialization and refresh are explicit.

Run offline first, then outer (the outer pins the offline manifest bytes).
The default never writes. The sole reviewed additive migration admits the
constructor child and its three-library source commitment; no discovered
members are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile

_ROOT = Path(__file__).absolute().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tuner.runtime.offline_sft_worker import _canonical_json, _strict_document  # noqa: E402
from tuner.runtime.offline_grpo_worker import (  # noqa: E402
    captured_grpo_manifest_document, parse_offline_grpo_worker_manifest,
    GRPO_OFFLINE_MEMBERS, GRPO_CLOSURE_SCHEMA, GRPO_CLOSURE_REF,
    GRPO_ENTRYPOINT, GRPO_TRAINER_ENTRYPOINT, GRPO_BOOTSTRAP, GRPO_PACKAGED_SOURCE,
)
from tuner.runtime.offline_sft_worker import _parse_manifest, OfflineSFTWorkerError  # noqa: E402
from tuner.runtime.packaged_grpo_worker_closure import (  # noqa: E402
    captured_grpo_outer_document, parse_packaged_grpo_worker_closure,
    GRPO_OUTER_MEMBERS, GRPO_OUTER_SCHEMA,
)
from tuner.runtime.packaged_worker_closure import stable_read  # noqa: E402

_ADDED = frozenset({"tuner/runtime/packaged_grpo_constructor_child.py",
                    "tuner/runtime/manifests/grpo-constructor-library-sources-v1.json"})


def _reviewed_previous(raw, *, kind):
    """One-way policy migration for the single reviewed constructor child."""
    previous = _strict_document(raw)
    if _canonical_json(previous) + b"\n" != raw:
        raise ValueError
    current = GRPO_OFFLINE_MEMBERS if kind == "offline" else GRPO_OUTER_MEMBERS
    prior = tuple(path for path in current if path not in _ADDED)
    if tuple(item.get("path") for item in previous["members"]) != prior:
        raise ValueError
    if kind == "offline":
        _parse_manifest(previous, manifest_path=Path(GRPO_PACKAGED_SOURCE),
            expected_digest=previous["closure_digest"], method="grpo",
            _method_identity=(GRPO_CLOSURE_SCHEMA, GRPO_CLOSURE_REF,
                              GRPO_ENTRYPOINT, GRPO_TRAINER_ENTRYPOINT, GRPO_BOOTSTRAP),
            _method_members=prior)
    else:
        if (set(previous) != {"schema_version", "members", "closure_digest"}
                or previous["schema_version"] != GRPO_OUTER_SCHEMA
                or hashlib.sha256(_canonical_json({"schema_version": GRPO_OUTER_SCHEMA,
                    "members": previous["members"]})).hexdigest() != previous["closure_digest"]
                or any(set(item) != {"path", "size_bytes", "sha256"} for item in previous["members"])):
            raise ValueError
    return previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("offline", "outer"), required=True)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--write", action="store_true")
    flags.add_argument("--initialize", action="store_true")
    args = parser.parse_args(argv)
    try:
        if Path(__file__).is_symlink():
            raise ValueError
        name = "offline-grpo-worker-v1.json" if args.kind == "offline" else "packaged-grpo-worker-v1.json"
        path = _ROOT / "tuner/runtime/manifests" / name
        capture = captured_grpo_manifest_document if args.kind == "offline" else captured_grpo_outer_document
        updated = capture(_ROOT)
        payload = _canonical_json(updated) + b"\n"
        if args.kind == "offline":
            parse_offline_grpo_worker_manifest(payload)
        else:
            parse_packaged_grpo_worker_closure(payload, verify_sources=False)
        if args.initialize:
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            print("INITIALIZED")
            return 0
        previous_raw = stable_read(path, 1024 * 1024)
        previous = _strict_document(previous_raw)
        migration = False
        try:
            if args.kind == "offline":
                parse_offline_grpo_worker_manifest(previous_raw)
            else:
                parse_packaged_grpo_worker_closure(previous_raw, verify_sources=False)
        except (ValueError, OfflineSFTWorkerError):
            previous = _reviewed_previous(previous_raw, kind=args.kind)
            migration = True
        if previous_raw == payload:
            print("CURRENT")
            return 0
        if not args.write:
            print("STALE")
            return 3
        # Only content hashes and byte totals change; capture uses the same fixed
        # reviewed identities and path order. Refuse any metadata substitution.
        refreshed = {item["path"]: item for item in updated["members"]}
        additions = set(refreshed) - {item["path"] for item in previous["members"]}
        expected_additions = _ADDED if args.kind == "offline" else _ADDED - {
            "tuner/runtime/manifests/grpo-constructor-library-sources-v1.json"}
        if migration and additions != expected_additions:
            raise ValueError
        for old in previous["members"]:
            new = refreshed[old["path"]]
            if {key: value for key, value in old.items() if key not in {"size_bytes", "sha256"}} != {
                    key: value for key, value in new.items() if key not in {"size_bytes", "sha256"}}:
                raise ValueError
        descriptor, staging = tempfile.mkstemp(prefix=".grpo-manifest-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if stable_read(path, 1024 * 1024) != previous_raw:
                raise ValueError
            os.replace(staging, path)
        finally:
            if os.path.exists(staging):
                os.unlink(staging)
        print("UPDATED")
        return 0
    except (OSError, ValueError, TypeError, KeyError):
        print("GRPO_MANIFEST_REJECTED", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
