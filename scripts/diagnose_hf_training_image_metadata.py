"""Run the single bounded metadata diagnostic for the protected cached image."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path


SUCCESS = {
    "schema_version": "synaptic-hf-training-image-metadata-diagnostic/v1",
    "status": "PASS",
}
_ATTRIBUTED_STAGES = frozenset({
    "preflight", "registry_initial", "operation_lock", "docker_authority_initial",
    "cache_identity_initial", "runtime_metadata", "cache_identity_final",
    "docker_authority_final", "registry_final", "final_integrity",
})
_ATTRIBUTED_CATEGORIES = frozenset({
    "timeout", "nonzero", "identity", "document", "runtime", "cleanup",
})
_RUNTIME_SUBSTAGES = frozenset({
    "child_unreported", "python_bootstrap", "python_runtime", "workspace_setup",
    "distribution_metadata", "torch_import", "safetensors_import",
    "transformers_import", "signature_introspection", "unsloth_spec",
    "unsloth_origin", "unsloth_package_root", "site_roots", "site_membership",
    "user_site_isolation", "origin_chain", "result_serialization",
})
_PYTHON_IMPLEMENTATIONS = frozenset({"CPython", "PyPy", "GraalPy", "Jython", "IronPython"})
_PYTHON_VERSION = re.compile(
    r"^[1-9][0-9]{0,2}\.(?:0|[1-9][0-9]{0,2})\.(?:0|[1-9][0-9]{0,2})$",
)


def _line(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"


def _failure(reason_code: str) -> int:
    sys.stderr.write(_line({
        "reason_code": reason_code,
        "schema_version": "synaptic-hf-training-image-metadata-diagnostic-error/v1",
        "status": "FAILED",
    }))
    return 125


def _authenticated_repo_root() -> Path:
    raw_script = Path(__file__)
    if raw_script.is_symlink():
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    script = raw_script.resolve(strict=True)
    if not script.is_file() or script.name != "diagnose_hf_training_image_metadata.py":
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
    raise SystemExit(_failure("SCRIPT_IDENTITY_INVALID"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tuner.cloud.hf_training_image_lock import (
    MetadataDiagnosticStageError,
    PythonRuntimeIdentityDiagnosticError,
    RuntimeSubstageDiagnosticError,
    diagnose_runtime_metadata,
    diagnose_runtime_metadata_attributed,
    diagnose_runtime_substage_attributed,
    observe_python_runtime_identity,
    subprocess_runner,
)


class _ArgumentInvalid(RuntimeError):
    pass


class _ClosedParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentInvalid


def build_parser() -> argparse.ArgumentParser:
    parser = _ClosedParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--docker", required=True, type=Path)
    parser.add_argument("--docker-config", required=True, type=Path)
    attribution = parser.add_mutually_exclusive_group()
    attribution.add_argument("--stage-attribution", action="store_true")
    attribution.add_argument("--runtime-substage-attribution", action="store_true")
    attribution.add_argument("--python-runtime-identity", action="store_true")
    return parser


def _stage_failure(*, failed_stage: str, category: str) -> int:
    sys.stderr.write(_line({
        "category": category,
        "failed_stage": failed_stage,
        "reason_code": "DIAGNOSTIC_STAGE_REJECTED",
        "schema_version": "synaptic-hf-training-image-metadata-diagnostic-stage-error/v1",
        "status": "FAILED",
    }))
    return 125


def _runtime_substage_failure(*, runtime_substage: str) -> int:
    sys.stderr.write(_line({
        "reason_code": "RUNTIME_SUBSTAGE_REJECTED",
        "runtime_substage": runtime_substage,
        "schema_version": "synaptic-hf-training-runtime-substage-error/v1",
        "status": "FAILED",
    }))
    return 125


def _python_identity_failure() -> int:
    sys.stderr.write(_line({
        "reason_code": "PYTHON_RUNTIME_IDENTITY_REJECTED",
        "schema_version": "synaptic-hf-training-python-runtime-identity-error/v1",
        "status": "FAILED",
    }))
    return 125


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except _ArgumentInvalid:
        return _failure("ARGUMENT_INVALID")
    if platform.python_implementation() != "CPython" or platform.python_version() != "3.12.7":
        return _failure("INTERPRETER_INVALID")
    try:
        if args.python_runtime_identity:
            diagnostic = observe_python_runtime_identity
        elif args.runtime_substage_attribution:
            diagnostic = diagnose_runtime_substage_attributed
        elif args.stage_attribution:
            diagnostic = diagnose_runtime_metadata_attributed
        else:
            diagnostic = diagnose_runtime_metadata
        result = diagnostic(
            image=args.image, docker=args.docker, docker_config=args.docker_config,
            runner=subprocess_runner,
        )
    except MetadataDiagnosticStageError as exc:
        if (
            not (
                args.stage_attribution or args.runtime_substage_attribution
                or args.python_runtime_identity
            )
            or type(exc) is not MetadataDiagnosticStageError
        ):
            return _failure("DIAGNOSTIC_REJECTED")
        try:
            failed_stage = exc.failed_stage
            category = exc.category
            valid = (
                type(failed_stage) is str and failed_stage in _ATTRIBUTED_STAGES
                and type(category) is str and category in _ATTRIBUTED_CATEGORIES
            )
        except BaseException:
            return _failure("DIAGNOSTIC_REJECTED")
        if not valid:
            return _failure("DIAGNOSTIC_REJECTED")
        return _stage_failure(failed_stage=failed_stage, category=category)
    except RuntimeSubstageDiagnosticError as exc:
        if (
            not args.runtime_substage_attribution
            or type(exc) is not RuntimeSubstageDiagnosticError
        ):
            return _failure("DIAGNOSTIC_REJECTED")
        try:
            runtime_substage = exc.runtime_substage
            valid = type(runtime_substage) is str and runtime_substage in _RUNTIME_SUBSTAGES
        except BaseException:
            return _failure("DIAGNOSTIC_REJECTED")
        if not valid:
            return _failure("DIAGNOSTIC_REJECTED")
        return _runtime_substage_failure(runtime_substage=runtime_substage)
    except PythonRuntimeIdentityDiagnosticError as exc:
        if (
            not args.python_runtime_identity
            or type(exc) is not PythonRuntimeIdentityDiagnosticError
        ):
            return _failure("DIAGNOSTIC_REJECTED")
        return _python_identity_failure()
    except Exception:
        return _failure("DIAGNOSTIC_REJECTED")
    if args.python_runtime_identity:
        try:
            valid = (
                type(result) is dict
                and set(result) == {"implementation", "schema_version", "status", "version"}
                and result.get("schema_version") == "synaptic-hf-training-python-runtime-identity/v1"
                and result.get("status") == "OBSERVED"
                and type(result.get("implementation")) is str
                and result["implementation"] in _PYTHON_IMPLEMENTATIONS
                and type(result.get("version")) is str
                and _PYTHON_VERSION.fullmatch(result["version"]) is not None
            )
        except BaseException:
            valid = False
        if not valid:
            return _failure("DIAGNOSTIC_REJECTED")
        sys.stdout.write(json.dumps(
            result, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ))
        return 0
    if result != SUCCESS:
        return _failure("DIAGNOSTIC_REJECTED")
    sys.stdout.write(_line(SUCCESS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
