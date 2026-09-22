"""Plan, build, capture, and verify an immutable derived training image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
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
    build_runtime_release,
    build_command,
    build_derived_image,
    capture_candidate,
    capture_final_runtime,
    load_profile,
    plan_profile,
    promote_runtime_release,
    render_dockerfile,
    verify_candidate,
    verify_runtime_release,
    qualify_local_runtime,
    verify_local_runtime,
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

    release = subparsers.add_parser(
        "release",
        help="Build, verify, or locally promote a provider-neutral runtime release",
    )
    release_actions = release.add_subparsers(dest="release_command", required=True)
    release_capture = release_actions.add_parser("capture", help="Measure the final installed runtime from verified image evidence")
    for action in (release_capture,):
        action.add_argument("--config", required=True, type=Path)
        action.add_argument("--build-receipt", required=True, type=Path)
        action.add_argument("--candidate", required=True, type=Path)
        action.add_argument("--image-verification", required=True, type=Path)
    release_capture.add_argument("--docker", required=True, type=Path)
    release_capture.add_argument("--docker-config", required=True, type=Path)
    release_capture.add_argument("--output", required=True, type=Path)
    release_capture.add_argument("--execute", action="store_true")
    release_build = release_actions.add_parser("build", help="Build canonical release bytes from final runtime capture")
    release_build.add_argument("--config", required=True, type=Path)
    release_build.add_argument("--build-receipt", required=True, type=Path)
    release_build.add_argument("--candidate", required=True, type=Path)
    release_build.add_argument("--image-verification", required=True, type=Path)
    release_build.add_argument("--final-runtime-capture", required=True, type=Path)
    release_build.add_argument("--compatibility", required=True, type=Path)
    release_build.add_argument("--output", required=True, type=Path)
    release_verify = release_actions.add_parser("verify", help="Verify a canonical release and packaged worker closure")
    release_verify.add_argument("--release", required=True, type=Path)
    for action in (release_verify,):
        action.add_argument("--final-runtime-capture", required=True, type=Path)
        action.add_argument("--config", required=True, type=Path)
        action.add_argument("--build-receipt", required=True, type=Path)
        action.add_argument("--candidate", required=True, type=Path)
        action.add_argument("--image-verification", required=True, type=Path)
    release_verify.add_argument("--output", required=True, type=Path)
    release_promote = release_actions.add_parser("promote", help="Record a local-only promotion from verified evidence")
    release_promote.add_argument("--release", required=True, type=Path)
    release_promote.add_argument("--verification", required=True, type=Path)
    release_promote.add_argument("--final-runtime-capture", required=True, type=Path)
    release_promote.add_argument("--config", required=True, type=Path)
    release_promote.add_argument("--build-receipt", required=True, type=Path)
    release_promote.add_argument("--candidate", required=True, type=Path)
    release_promote.add_argument("--image-verification", required=True, type=Path)
    release_promote.add_argument("--output", required=True, type=Path)
    closure = release_actions.add_parser("closure", help="Check the fixed packaged-worker closure; --write refreshes reviewed member hashes")
    closure.add_argument("--write", action="store_true")
    local = release_actions.add_parser("qualify-local", help="Plan or run offline installed-child CPU diagnostics; never training")
    local_verify = release_actions.add_parser("verify-local", help="Revalidate local CPU diagnostic evidence offline")
    for action in (local, local_verify):
        for flag in ("qualification-config", "release", "config", "build-receipt", "candidate", "image-verification", "final-runtime-capture", "output"):
            action.add_argument("--" + flag, required=True, type=Path)
    local.add_argument("--docker", required=True, type=Path)
    local.add_argument("--docker-config", required=True, type=Path)
    local.add_argument("--execute", action="store_true")
    local_verify.add_argument("--evidence", required=True, type=Path)
    return parser


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def _maintain_packaged_closure(*, write: bool) -> int:
    """Only refresh hashes of the independently fixed, dedicated inventory."""
    from tuner.runtime.packaged_worker_closure import (
        PACKAGED_WORKER_CLOSURE_SCHEMA, _canonical, load_packaged_worker_closure, stable_read,
    )
    root = REPO_ROOT / "tuner" / "runtime"
    target = root / "manifests" / "packaged-training-worker-v1.json"
    inventory = ("packaged_sft_child.py", "packaged_sft_execution.py", "packaged_training_worker.py", "packaged_worker_closure.py", "releases.py")
    raw = stable_read(target)
    current = json.loads(raw)
    if (set(current) != {"schema_version", "members", "closure_digest"}
            or current["schema_version"] != PACKAGED_WORKER_CLOSURE_SCHEMA
            or tuple(item["path"] for item in current["members"]) != inventory
            or any(set(item) != {"path", "size_bytes", "sha256"} for item in current["members"])
            or _canonical(current) + b"\n" != raw):
        raise DerivedTrainingImageError("CLOSURE_POLICY_INVALID")
    unsigned = {"schema_version": current["schema_version"], "members": []}
    for name in inventory:
        payload = stable_read(root / name)
        unsigned["members"].append({"path": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    refreshed = {**unsigned, "closure_digest": hashlib.sha256(_canonical(unsigned)).hexdigest()}
    expected = _canonical(refreshed) + b"\n"
    if raw != expected:
        if not write:
            print("PACKAGED_CLOSURE_DRIFT")
            return 3
        descriptor, temporary = tempfile.mkstemp(prefix=".packaged-closure-", suffix=".json", dir=target.parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        if stable_read(Path(temporary)) != expected or stable_read(target) != raw:
            raise DerivedTrainingImageError("CLOSURE_REFRESH_CONFLICT")
        os.replace(temporary, target)
    if load_packaged_worker_closure().digest != refreshed["closure_digest"]:
        raise DerivedTrainingImageError("CLOSURE_RECHECK_FAILED")
    print("PACKAGED_CLOSURE_CURRENT")
    return 0


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
                    runtime_inputs_digest=(hashlib.sha256((json.dumps(profile.packaged_runtime, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")).hexdigest() if profile.packaged_runtime is not None else None),
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
        if args.command == "release":
            if args.release_command in {"qualify-local", "verify-local"}:
                inputs = dict(qualification_config_path=args.qualification_config, release_path=args.release,
                    profile_path=args.config, build_receipt_path=args.build_receipt, candidate_path=args.candidate,
                    image_verification_path=args.image_verification, capture_path=args.final_runtime_capture, output=args.output)
                if args.release_command == "qualify-local":
                    result = qualify_local_runtime(**inputs, docker=args.docker, docker_config=args.docker_config, execute=args.execute)
                else:
                    result = verify_local_runtime(**inputs, evidence_path=args.evidence)
                _json(result)
                return 0
            if args.release_command == "closure":
                return _maintain_packaged_closure(write=args.write)
            if args.release_command == "build":
                release = build_runtime_release(capture_path=args.final_runtime_capture, compatibility_path=args.compatibility, output=args.output, profile_path=args.config, build_receipt_path=args.build_receipt, candidate_path=args.candidate, image_verification_path=args.image_verification)
                _json({
                    "schema_version": release.schema_version,
                    "status": "RELEASE_BUILT",
                    "runtime_release_digest": release.manifest_digest,
                })
                return 0
            if args.release_command == "verify":
                result = verify_runtime_release(release_path=args.release, capture_path=args.final_runtime_capture, profile_path=args.config, build_receipt_path=args.build_receipt, candidate_path=args.candidate, image_verification_path=args.image_verification, output=args.output)
                _json({"schema_version": result["schema_version"], "status": result["status"]})
                return 0
            if args.release_command == "capture":
                if not args.execute:
                    load_profile(args.config)
                    _json({"schema_version": "synaptic-packaged-runtime-final-capture-plan/v1", "status": "PLAN_ONLY", "image_verification": str(args.image_verification), "output": str(args.output)})
                    return 0
                result = capture_final_runtime(profile_path=args.config, build_receipt_path=args.build_receipt, candidate_path=args.candidate, image_verification_path=args.image_verification, docker=args.docker, docker_config=args.docker_config, output=args.output)
                _json({"schema_version": result["schema_version"], "status": result["status"]})
                return 0
            result = promote_runtime_release(
                release_path=args.release,
                verification_path=args.verification,
                final_runtime_capture_path=args.final_runtime_capture,
                profile_path=args.config, build_receipt_path=args.build_receipt,
                candidate_path=args.candidate, image_verification_path=args.image_verification,
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
    except BaseException:
        print("Derived training image workflow failed: COMMAND_FAILED", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
