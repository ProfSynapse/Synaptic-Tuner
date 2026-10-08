"""Capture or check a runtime-profile ``.inventory.json`` from a local image.

Runs a credential-free probe inside an image that is already present locally
(``docker run --pull never --network none``) and writes the canonical
``syntunia-python-distribution-inventory/v1`` document that
``tuner/runtime_profiles.py`` loads. It never pulls, pushes or contacts a
provider; pull the immutable base yourself first (``docker pull <ref>``).

Image mode probes one image as-is (a runtime profile whose image is used
unchanged, such as qwen35-sft-v1):

    python3 scripts/capture_runtime_profile_inventory.py \\
        --image unsloth/unsloth@sha256:<digest> \\
        --python /opt/unsloth-venv/bin/python3 \\
        --check Trainers/runtime_profiles/qwen35-sft-v1.inventory.json

Profile mode builds a local capture image from a packaged derived image
profile (its base plus every hash-pinned bootstrap wheel, installed with the
packaged build's offline command and ``pip check`` gate), probes the base and
the capture image, and keeps only the base's distributions at their installed
versions. Bootstrap wheels that replace a base distribution (an ML stack
change) are therefore recorded; the additive packaged-runtime closure is not,
matching an inventory captured from the base alone. Stage the bootstrap wheels
next to ``profile.yaml`` (or pass ``--wheel-dir``); each is hash-checked.

    python3 scripts/capture_runtime_profile_inventory.py \\
        --image unsloth/unsloth@sha256:<digest> \\
        --image-profile Trainers/image_profiles/<name>/profile.yaml \\
        --output Trainers/runtime_profiles/<name>.inventory.json

``--image`` is the immutable reference recorded in the inventory and must equal
the runtime profile's ``runtime.image``. ``--output`` refuses to overwrite;
``--check`` exits 1 and lists the differences when the file is stale.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tuner.runtime_profile_inventory import (  # noqa: E402
    PROBE_MAX_OUTPUT_BYTES,
    PROBE_TIMEOUT_SECONDS,
    InventoryCaptureError,
    build_inventory,
    capture_dockerfile,
    diff_inventories,
    overlay_probe,
    probe_command,
)

BUILD_TIMEOUT_SECONDS = 3600
CAPTURE_REPOSITORY = "synaptic-inventory-capture"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", required=True,
                        help="Immutable image reference recorded in the inventory")
    parser.add_argument("--python", default=None,
                        help="Absolute interpreter path inside the image "
                             "(profile mode: defaults to the profile's python_executable)")
    parser.add_argument("--docker", default="docker", help="Docker CLI to invoke")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-image", default=None,
                        help="Image mode: local image reference or ID to probe (default: --image)")
    source.add_argument("--image-profile", type=Path, default=None,
                        help="Profile mode: packaged derived image profile.yaml to build and capture")
    parser.add_argument("--wheel-dir", type=Path, default=None,
                        help="Profile mode: directory holding the bootstrap wheels "
                             "(default: the profile's directory)")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path, help="New inventory file to create")
    target.add_argument("--check", type=Path, help="Existing inventory file to compare")
    return parser


def _run(argv: list[str], *, timeout: int, label: str) -> bytes:
    completed = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-3000:]
        raise InventoryCaptureError(f"{label} exited {completed.returncode}: {detail.strip()}")
    return completed.stdout


def _probe(docker: str, image: str, python: str) -> bytes:
    output = _run(probe_command(docker=docker, image=image, python_executable=python),
                  timeout=PROBE_TIMEOUT_SECONDS, label="probe")
    if len(output) > PROBE_MAX_OUTPUT_BYTES:
        raise InventoryCaptureError("probe output is too large")
    return output


def _same_image(left: str, right: str) -> bool:
    return left.removeprefix("docker.io/") == right.removeprefix("docker.io/")


def _profile_probe(args: argparse.Namespace) -> tuple[bytes, str]:
    from tuner.cloud.derived_training_image import DerivedTrainingImageError, load_profile

    try:
        profile = load_profile(args.image_profile)
    except DerivedTrainingImageError as exc:
        raise InventoryCaptureError(f"image profile is invalid: {exc.code}") from exc
    if profile.packaged_runtime is None or profile.packages:
        raise InventoryCaptureError(
            "profile mode needs a packaged profile whose stack changes are hash-pinned "
            "bootstrap wheels (packages: [])"
        )
    if not _same_image(profile.base_image, args.image):
        raise InventoryCaptureError(
            f"--image {args.image} is not the profile base {profile.base_image}"
        )
    python = args.python or profile.python_executable
    if python != profile.python_executable:
        raise InventoryCaptureError("--python differs from the profile's python_executable")
    wheel_dir = args.wheel_dir or args.image_profile.parent
    bootstrap = profile.packaged_runtime["bootstrap"]
    scratch = REPO_ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    tag = f"{CAPTURE_REPOSITORY}/{profile.name}:{profile.canonical_sha256[7:19]}"
    with tempfile.TemporaryDirectory(prefix="inventory-capture-", dir=scratch) as raw:
        context = Path(raw)
        wheels = context / "wheels"
        wheels.mkdir()
        lines = []
        for item in bootstrap:
            source = wheel_dir / item["filename"]
            if not source.is_file():
                raise InventoryCaptureError(
                    f"bootstrap wheel {item['filename']} is not staged in {wheel_dir}"
                )
            if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
                raise InventoryCaptureError(f"bootstrap wheel {item['filename']} hash differs from the profile")
            shutil.copyfile(source, wheels / item["filename"])
            lines.append(f"/opt/synaptic-inventory-capture/{item['filename']} --hash=sha256:{item['sha256']}")
        (wheels / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
        (context / "Dockerfile").write_text(
            capture_dockerfile(base_image=profile.base_image, python_executable=python),
            encoding="utf-8",
        )
        started = time.monotonic()
        _run([args.docker, "build", "--pull=false", "--network", "none", "--platform", "linux/amd64",
              "--tag", tag, str(context)], timeout=BUILD_TIMEOUT_SECONDS, label="capture build")
        elapsed = time.monotonic() - started
    size = _run([args.docker, "image", "inspect", "--format", "{{.Size}}", tag],
                timeout=60, label="image inspect").decode().strip()
    print(f"built {tag} in {elapsed:.0f}s ({int(size) / 1e9:.2f} GB)")
    base_output = _probe(args.docker, args.image, python)
    derived_output = _probe(args.docker, tag, python)
    declared = {item["distribution"]: item["version"] for item in bootstrap}
    restricted, changed = overlay_probe(base_output, derived_output, declared=declared)
    for name, (before, after) in changed.items():
        print(f"  replaced {name}: {before} -> {after}")
    return restricted, python


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != payload:
        raise InventoryCaptureError(f"readback of {path} differs from the captured inventory")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise InventoryCaptureError(f"{args.output} already exists; refusing to overwrite")
        if args.image_profile is not None:
            probe_output, python = _profile_probe(args)
        else:
            if args.wheel_dir is not None:
                raise InventoryCaptureError("--wheel-dir applies only to --image-profile")
            if args.python is None:
                raise InventoryCaptureError("--python is required in image mode")
            python = args.python
            probe_output = _probe(args.docker, args.run_image or args.image, python)
        captured = build_inventory(probe_output, image=args.image, python_executable=python)
        if args.check is not None:
            expected = args.check.read_bytes()
            if expected == captured.payload:
                print(f"CURRENT {args.check} {captured.sha256} "
                      f"({captured.distribution_count} distributions)")
                return 0
            print(f"STALE {args.check}: captured {captured.sha256}", file=sys.stderr)
            for line in diff_inventories(expected, captured.payload) or ["byte encoding differs"]:
                print("  " + line, file=sys.stderr)
            return 1
        _write_new(args.output, captured.payload)
    except (InventoryCaptureError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"Inventory capture failed: {exc}", file=sys.stderr)
        return 2
    print(f"WROTE {args.output} ({captured.distribution_count} distributions)")
    print(f"inventory sha256: {captured.sha256}")
    for fact, value in captured.runtime.items():
        print(f"  {fact}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
