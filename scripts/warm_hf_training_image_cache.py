"""Warm the exact protected HF training-image cache without creating evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _authenticated_repo_root() -> Path:
    raw_script = Path(__file__)
    if raw_script.is_symlink():
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    script = raw_script.resolve(strict=True)
    if not script.is_file() or script.name != "warm_hf_training_image_cache.py":
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    root = script.parents[1]
    if script.parent.name != "scripts" or (root / "scripts" / script.name).resolve(strict=True) != script:
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    for relative in (
        "tuner/cloud/hf_training_image_lock.py",
        "tuner/cloud/hf_training_image_operation_lock.py",
        "tuner/cloud/hf_training_oci_registry.py",
    ):
        anchor = root / relative
        if anchor.is_symlink() or not anchor.is_file() or anchor.resolve(strict=True) != anchor.absolute():
            raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    return root


try:
    REPO_ROOT = _authenticated_repo_root()
except (OSError, RuntimeError):
    print("HF training image cache warm failed: SCRIPT_IDENTITY_INVALID", file=sys.stderr)
    raise SystemExit(125)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tuner.cloud.hf_training_image_lock import (
    TrainingImageLockError,
    subprocess_runner,
    warm_image_cache,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Exact docker.io/unsloth/unsloth@sha256 digest")
    parser.add_argument("--docker", required=True, type=Path, help="Explicit Docker executable")
    parser.add_argument("--docker-config", required=True, type=Path, help="Existing empty Docker config directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        warm_image_cache(
            image=args.image, docker=args.docker,
            docker_config=args.docker_config, runner=subprocess_runner,
        )
    except TrainingImageLockError as exc:
        print(f"HF training image cache warm failed: {exc.reason_code}", file=sys.stderr)
        return 125
    except Exception:
        print("HF training image cache warm failed: COMMAND_FAILED", file=sys.stderr)
        return 125
    print(json.dumps({"status": "CACHE_WARMED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
