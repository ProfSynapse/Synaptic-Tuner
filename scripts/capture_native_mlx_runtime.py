#!/usr/bin/env python3
"""Read-only native MLX runtime capture/check; never launches training.

Run with the selected candidate Python and -B. Output is a new, exclusive 0600
file. Config contains `capture` (NativeCaptureConfig) and `expectation`
(MLXRuntimeExpectation); the lock is the original reviewed requirements.lock.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_material import (
    NativeCaptureConfig, capture_native_material, strict_json, verify_native_material,
)
from tuner.runtime.native_mlx_runtime import MLXRuntimeExpectation


def read(path, maximum):
    path = Path(path)
    with NativeTree(path.parent) as tree:
        return tree.read(path.name, maximum=maximum)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dependency-lock", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--check")
    parser.add_argument("--expected-digest")
    args = parser.parse_args(argv)
    try:
        config = strict_json(read(args.config, 1024 * 1024))
        if type(config) is not dict or set(config) != {"capture", "expectation"}:
            raise ValueError("configuration differs")
        lock = read(args.dependency_lock, 1024 * 1024)
        capture = NativeCaptureConfig.from_dict(config["capture"])
        expectation = MLXRuntimeExpectation.from_dict(config["expectation"])
        if args.check:
            if args.expected_digest is None:
                raise ValueError("check requires reviewed digest")
            raw = read(args.check, 16 * 1024 * 1024)
            # The supplied configuration remains binding in check mode.
            actual = capture_native_material(capture, expectation, dependency_lock=lock)
            if actual != raw:
                raise ValueError("configuration or observed runtime differs")
            verify_native_material(raw, expected_digest=args.expected_digest, dependency_lock=lock)
        else:
            if args.expected_digest is not None:
                raise ValueError("capture cannot self-approve a digest")
            raw = capture_native_material(capture, expectation, dependency_lock=lock)
            destination = Path(args.output)
            with NativeTree(destination.parent, private=True) as parent:
                fd = os.open(destination.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW, 0o600, dir_fd=parent.fd)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                parent.check()
        print("OBSERVED_NOT_ADMITTED sha256=" + hashlib.sha256(raw).hexdigest())
        return 0
    except Exception:
        print("NATIVE_CAPTURE_REJECTED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
