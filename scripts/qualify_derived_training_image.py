"""Plan, build, capture, and verify an immutable derived training image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _authenticated_repo_root() -> Path:
    raw = Path(__file__)
    if raw.is_symlink():
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    script = raw.resolve(strict=True)
    root = script.parents[1]
    if (
        script.name != "qualify_derived_training_image.py"
        or script.parent.name != "scripts"
        or (root / "scripts" / script.name).resolve(strict=True) != script
    ):
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    module = root / "tuner" / "cloud" / "derived_training_image.py"
    if module.is_symlink() or not module.is_file() or module.resolve(strict=True) != module.absolute():
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    return root


try:
    REPO_ROOT = _authenticated_repo_root()
except (OSError, RuntimeError):
    print("Derived training image workflow failed: SCRIPT_IDENTITY_INVALID", file=sys.stderr)
    raise SystemExit(125)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tuner.cloud.derived_training_image import (  # noqa: E402
    DerivedTrainingImageError,
    build_command,
    build_derived_image,
    capture_candidate,
    load_profile,
    plan_profile,
    render_dockerfile,
    verify_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Provider-free validation and deterministic plan")
    plan.add_argument("--config", required=True, type=Path)

    build = subparsers.add_parser("build", help="Build locally and retain an unverified receipt")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--docker", required=True, type=Path)
    build.add_argument("--docker-config", required=True, type=Path)
    build.add_argument("--tag", required=True)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--execute", action="store_true")

    capture = subparsers.add_parser("capture", help="Capture runtime and package provenance")
    capture.add_argument("--config", required=True, type=Path)
    capture.add_argument("--build-receipt", required=True, type=Path)
    capture.add_argument("--docker", required=True, type=Path)
    capture.add_argument("--docker-config", required=True, type=Path)
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("--execute", action="store_true")

    verify = subparsers.add_parser(
        "verify", help="Offline verification and diagnostic report"
    )
    verify.add_argument("--config", required=True, type=Path)
    verify.add_argument("--candidate", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    return parser


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            _json(plan_profile(args.config))
            return 0
        if args.command == "build":
            if not args.execute:
                profile = load_profile(args.config)
                dockerfile = render_dockerfile(profile)
                preview = build_command(
                    docker=args.docker,
                    docker_config=args.docker_config,
                    tag=args.tag,
                    dockerfile=Path("<temporary-context>") / "Dockerfile",
                    metadata_file=Path("<temporary-context>") / "build-metadata.json",
                )
                _json({
                    "schema_version": "synaptic-derived-training-image-build-plan/v1",
                    "status": "PLAN_ONLY",
                    "argv": list(preview.argv),
                    "dockerfile": dockerfile,
                })
                return 0
            result = build_derived_image(
                profile_path=args.config,
                docker=args.docker,
                docker_config=args.docker_config,
                tag=args.tag,
                output=args.output,
            )
            _json({"schema_version": result["schema_version"], "status": result["status"]})
            return 0
        if args.command == "capture":
            if not args.execute:
                load_profile(args.config)
                _json({
                    "schema_version": "synaptic-derived-training-image-capture-plan/v1",
                    "status": "PLAN_ONLY",
                    "build_receipt": str(args.build_receipt),
                    "output": str(args.output),
                })
                return 0
            result = capture_candidate(
                profile_path=args.config,
                build_receipt_path=args.build_receipt,
                docker=args.docker,
                docker_config=args.docker_config,
                output=args.output,
            )
            _json({"schema_version": result["schema_version"], "status": result["status"]})
            return 0
        result = verify_candidate(
            profile_path=args.config,
            candidate_path=args.candidate,
            output=args.output,
        )
        _json({"schema_version": result["schema_version"], "status": result["status"]})
        return 0
    except DerivedTrainingImageError as exc:
        print(f"Derived training image workflow failed: {exc.code}", file=sys.stderr)
        return 125
    except Exception:
        print("Derived training image workflow failed: COMMAND_FAILED", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
