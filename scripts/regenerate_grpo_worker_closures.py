"""Check fixed GRPO inventory hashes; initialization and refresh are explicit.

Run offline first, then outer (the outer pins the offline manifest bytes).
The default never writes. Neither operation can discover or widen membership.
"""
from __future__ import annotations

import argparse
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
)
from tuner.runtime.packaged_grpo_worker_closure import (  # noqa: E402
    captured_grpo_outer_document, parse_packaged_grpo_worker_closure,
)
from tuner.runtime.packaged_worker_closure import stable_read  # noqa: E402


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
        if args.kind == "offline":
            parse_offline_grpo_worker_manifest(previous_raw)
        else:
            parse_packaged_grpo_worker_closure(previous_raw, verify_sources=False)
        if previous_raw == payload:
            print("CURRENT")
            return 0
        if not args.write:
            print("STALE")
            return 3
        # Only content hashes and byte totals change; capture uses the same fixed
        # reviewed identities and path order. Refuse any metadata substitution.
        for old, new in zip(previous["members"], updated["members"], strict=True):
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
