"""Candidate evidence for an engine-controlled Modal training-image build.

Image construction and inspection precede issuance of the final release.  The
candidate is deliberately not a runtime release or training authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import os
import queue
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Callable
from typing import Mapping

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV2
from tuner.cloud.derived_training_image import load_profile
from tuner.runtime.packaged_worker_closure import stable_read
from tuner.execution.providers.modal.modal_wheel_builder import (
    builder_lock_bytes, create_offline_wheel_builder,
)


_IMAGE_ID = re.compile(r"^im-[A-Za-z0-9]{1,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CAPTURE_BODY_BYTES = 128 * 1024
_SOURCE_STATUS_TIMEOUT_SECONDS = 120
_BUILDER_ATTESTATION = object()
_WHEEL_SOURCE_PATHS = (
    "pyproject.toml", "README.md", "LICENSE",
    "tuner", "synaptic_tuner", "shared", "SynthChat", "Evaluator",
    "MechInterp", "Trainers",
)
_SOURCE_STATE_REASONS = (
    "INPUT_INVALID", "HEAD_BEFORE_UNAVAILABLE", "HEAD_BEFORE_MISMATCH",
    "STATUS_BEFORE_TIMEOUT", "STATUS_BEFORE_UNAVAILABLE", "STATUS_BEFORE_DIRTY",
    "HEAD_AFTER_UNAVAILABLE", "HEAD_AFTER_MISMATCH",
    "STATUS_AFTER_TIMEOUT", "STATUS_AFTER_UNAVAILABLE", "STATUS_AFTER_DIRTY",
)


class SourceArchiveInvalid(ValueError):
    """The bound engine commit could not supply a bounded wheel source archive."""


class SourceWheelFailure(ValueError):
    """Closed local source-wheel failure without subprocess or filesystem detail."""

    __slots__ = ("reason",)
    _REASONS = frozenset({
        "SOURCE_STATE_INVALID", *_SOURCE_STATE_REASONS,
        "BUILDER_SETUP_FAILED", "OFFLINE_WHEEL_TIMEOUT",
        "OFFLINE_WHEEL_FAILED", "WHEEL_INVENTORY_INVALID",
    })

    def __init__(self, reason: str) -> None:
        if reason not in self._REASONS:
            raise ValueError("source wheel reason is invalid")
        super().__init__("source_wheel_unavailable")
        self.reason = reason


class ModalBoundedOperationFailure(RuntimeError):
    """A bounded call failed; its provider exception is never retained."""

    __slots__ = ("reason",)

    def __init__(self, code: str, reason: str) -> None:
        if reason not in {"TIMEOUT", "OPERATION_FAILED", "INSPECTOR_REJECTED", "OUTPUT_INVALID"}:
            raise ValueError("bounded failure reason is invalid")
        super().__init__(code)
        self.reason = reason


class ModalBuildStageFailure(RuntimeError):
    """Closed build/capture stage for host projection, without provider text."""

    __slots__ = ("stage", "reason")
    _REASONS = {
        "SOURCE_WHEEL": frozenset({
            "LOCAL_BUILD_FAILED", "SOURCE_STATE_INVALID", *_SOURCE_STATE_REASONS,
            "BUILDER_SETUP_FAILED",
            "OFFLINE_WHEEL_TIMEOUT", "OFFLINE_WHEEL_FAILED", "WHEEL_INVENTORY_INVALID",
        }),
        "BUILD_INPUTS": frozenset({"INVALID"}),
        "IMAGE_BUILD": frozenset({"TIMEOUT", "OPERATION_FAILED", "IDENTITY_MISSING"}),
        "CAPTURE_CREATE": frozenset({"TIMEOUT", "OPERATION_FAILED"}),
        "CAPTURE_OUTPUT": frozenset({"TIMEOUT", "OPERATION_FAILED", "INSPECTOR_REJECTED", "OUTPUT_INVALID"}),
        "CAPTURE_CLEANUP": frozenset({"TIMEOUT", "OPERATION_FAILED"}),
        "CAPTURE_VALIDATE": frozenset({"INVALID"}),
    }

    def __init__(self, stage: str, reason: str) -> None:
        if reason not in self._REASONS.get(stage, ()):
            raise ValueError("Modal build stage diagnosis is invalid")
        super().__init__("modal_build_stage_unavailable")
        self.stage, self.reason = stage, reason


class _CaptureOutputFailure(ValueError):
    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        if reason not in {"INSPECTOR_REJECTED", "OUTPUT_INVALID"}:
            raise ValueError("Modal capture reason is invalid")
        super().__init__("Modal capture output is invalid")
        self.reason = reason


def _canonical_capture_bytes(document: dict[str, object]) -> bytes:
    """Encode an inspector report with its own bounded output contract."""
    try:
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("Modal build capture is invalid") from None
    if len(encoded) > _MAX_CAPTURE_BODY_BYTES:
        raise ValueError("Modal build capture is too large")
    return encoded


def plan_modal_build_material(profile_path: Path) -> dict[str, object]:
    """Read-only intent; the final material digest follows the wheel build."""
    profile = load_profile(profile_path)
    if profile.packaged_runtime is None or profile.packages:
        raise ValueError("only the pinned packaged Modal profile is supported")
    root = Path(__file__).resolve().parents[4]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, timeout=30, check=False,
    )
    source_commit = commit.stdout.decode("ascii", "strict").strip() if commit.returncode == 0 else ""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("accepted engine source identity is unavailable")
    inspector = root / "tuner" / "runtime" / "modal_build_inspector.py"
    builder_lock_digest = hashlib.sha256(builder_lock_bytes()).hexdigest()
    intent = {
        "schema_version": "synaptic-modal-build-material-intent/v1",
        "profile_name": profile.name,
        "profile_digest": profile.canonical_sha256,
        "base_image": profile.base_image,
        "engine_source_commit": source_commit,
        "inspector_sha256": hashlib.sha256(stable_read(inspector, 128 * 1024)).hexdigest(),
        "builder_policy": "synaptic-modal-wheel-build/v2",
        "builder_lock_sha256": builder_lock_digest,
    }
    return {**intent, "intent_digest": hashlib.sha256(canonical_bytes(intent)).hexdigest()}


def prepare_current_source_wheel(source_root: Path, output_dir: Path, *,
                                 expected_source_commit: str,
                                 builder_cache_root: Path | None = None) -> tuple[Path, str]:
    """Build the accepted clean engine commit offline, without a customer repo."""
    try:
        root = source_root.resolve(strict=True)
        if (not (root / "pyproject.toml").is_file()
                or type(expected_source_commit) is not str
                or re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is None):
            raise ValueError
    except Exception:
        raise SourceWheelFailure("INPUT_INVALID") from None
    try:
        if not output_dir.is_dir():
            raise ValueError
    except Exception:
        raise SourceWheelFailure("BUILDER_SETUP_FAILED") from None

    def bound_head(phase: str) -> None:
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, timeout=30, check=False,
            )
            if head.returncode != 0 or type(head.stdout) is not bytes:
                raise ValueError
            observed = head.stdout.decode("ascii", "strict").strip()
            if re.fullmatch(r"[0-9a-f]{40}", observed) is None:
                raise ValueError
        except Exception:
            raise SourceWheelFailure("HEAD_" + phase + "_UNAVAILABLE") from None
        if observed != expected_source_commit:
            raise SourceWheelFailure("HEAD_" + phase + "_MISMATCH")

    def clean_status(phase: str) -> None:
        try:
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, timeout=_SOURCE_STATUS_TIMEOUT_SECONDS, check=False,
            )
        except subprocess.TimeoutExpired:
            raise SourceWheelFailure("STATUS_" + phase + "_TIMEOUT") from None
        except Exception:
            raise SourceWheelFailure("STATUS_" + phase + "_UNAVAILABLE") from None
        if (status.returncode != 0 or type(status.stdout) is not bytes
                or type(status.stderr) is not bytes or status.stderr):
            raise SourceWheelFailure("STATUS_" + phase + "_UNAVAILABLE")
        if status.stdout:
            raise SourceWheelFailure("STATUS_" + phase + "_DIRTY")

    bound_head("BEFORE")
    clean_status("BEFORE")
    try:
        archive = subprocess.run(
            ["git", "-C", str(root), "archive", "--format=tar",
             expected_source_commit, *_WHEEL_SOURCE_PATHS],
            capture_output=True, timeout=120, check=False,
        )
    except Exception:
        raise SourceArchiveInvalid("accepted engine source archive failed") from None
    bound_head("AFTER")
    clean_status("AFTER")
    if archive.returncode != 0 or not archive.stdout or len(archive.stdout) > 512 * 1024 * 1024:
        raise SourceArchiveInvalid("accepted engine source archive failed")
    try:
        private_scratch = tempfile.TemporaryDirectory(prefix="synaptic-wheel-source-")
    except Exception:
        raise SourceWheelFailure("BUILDER_SETUP_FAILED") from None
    with private_scratch as scratch:
        try:
            source = Path(scratch) / "source"
            source.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as members:
                for member in members:
                    if not member.isfile() and not member.isdir():
                        raise ValueError("engine source archive contains an unsupported member")
                    destination = source.joinpath(*Path(member.name).parts)
                    if not destination.resolve().is_relative_to(source.resolve()):
                        raise ValueError("engine source archive path escapes scratch")
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        if member.size > 64 * 1024 * 1024:
                            raise ValueError("engine source member is too large")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        stream = members.extractfile(member)
                        if stream is None:
                            raise ValueError("engine source member is unreadable")
                        destination.write_bytes(stream.read())
        except OSError:
            raise SourceWheelFailure("BUILDER_SETUP_FAILED") from None
        except Exception:
            raise SourceArchiveInvalid("accepted engine source archive failed") from None
        cache = builder_cache_root if builder_cache_root is not None else Path(scratch) / "builder-cache"
        try:
            builder_python = create_offline_wheel_builder(Path(scratch), cache)
        except Exception:
            raise SourceWheelFailure("BUILDER_SETUP_FAILED") from None
        command = [str(builder_python), "-I", "-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation", "--no-cache-dir", "--wheel-dir", str(output_dir), str(source)]
        build_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONNOUSERSITE": "1",
        }
        try:
            result = subprocess.run(command, capture_output=True, timeout=300, check=False, env=build_env)
        except subprocess.TimeoutExpired:
            raise SourceWheelFailure("OFFLINE_WHEEL_TIMEOUT") from None
        except Exception:
            raise SourceWheelFailure("OFFLINE_WHEEL_FAILED") from None
        if result.returncode != 0:
            raise SourceWheelFailure("OFFLINE_WHEEL_FAILED")
    try:
        wheels = list(output_dir.glob("synaptic_tuner-*-py3-none-any.whl"))
        if len(wheels) != 1:
            raise ValueError
        wheel = wheels[0]
        raw = wheel.read_bytes()
        if not raw or len(raw) > 256 * 1024 * 1024:
            raise ValueError
    except Exception:
        raise SourceWheelFailure("WHEEL_INVENTORY_INVALID") from None
    return wheel, hashlib.sha256(raw).hexdigest()


def _bounded(operation: Callable[[], object], *, deadline: float, code: str,
             late_cleanup: Callable[[object], None] | None = None) -> object:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ModalBoundedOperationFailure(code, "TIMEOUT")
    result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)
    lock = threading.Lock()
    abandoned = False
    completed = False
    completed_value: object | None = None

    def invoke() -> None:
        nonlocal completed, completed_value
        try:
            value = operation()
            with lock:
                completed = True
                completed_value = value
                cleanup = abandoned
            result.put((True, value))
            if cleanup and late_cleanup is not None:
                try:
                    late_cleanup(value)
                except BaseException:
                    pass
        except BaseException as error:
            result.put((False, error))

    threading.Thread(target=invoke, daemon=True, name="modal-training-build").start()
    try:
        ok, value = result.get(timeout=remaining)
    except queue.Empty:
        with lock:
            abandoned = True
            cleanup_value = completed_value if completed else None
        if cleanup_value is not None and late_cleanup is not None:
            try:
                late_cleanup(cleanup_value)
            except BaseException:
                pass
        raise ModalBoundedOperationFailure(code, "TIMEOUT") from None
    if not ok:
        reason = value.reason if type(value) is _CaptureOutputFailure else "OPERATION_FAILED"
        raise ModalBoundedOperationFailure(code, reason) from None
    return value


def _capture_output(sandbox: object) -> bytes:
    output = bytearray()
    for chunk in sandbox.stdout:
        if type(chunk) is str:
            chunk = chunk.encode("utf-8")
        if type(chunk) is not bytes or len(chunk) > _MAX_CAPTURE_BODY_BYTES + 1 - len(output):
            raise _CaptureOutputFailure("OUTPUT_INVALID")
        output.extend(chunk)
    if len(output) == _MAX_CAPTURE_BODY_BYTES + 1 and output[-1:] != b"\n":
        raise _CaptureOutputFailure("OUTPUT_INVALID")
    sandbox.wait(raise_on_termination=False)
    if type(sandbox.returncode) is not int or sandbox.returncode != 0:
        raise _CaptureOutputFailure("INSPECTOR_REJECTED")
    return bytes(output)


def _terminate_sandbox(sandbox: object) -> None:
    try:
        _cleanup_sandbox(sandbox)
    except BaseException:
        pass


def _cleanup_sandbox(sandbox: object) -> None:
    deadline = time.monotonic() + 30
    _bounded(lambda: sandbox.terminate(wait=False), deadline=deadline,
             code="modal_training_capture_cleanup_unresolved")
    while time.monotonic() < deadline:
        state = _bounded(sandbox.poll, deadline=deadline,
                         code="modal_training_capture_cleanup_unresolved")
        if state is not None:
            return
        time.sleep(0.05)
    raise ModalBoundedOperationFailure("modal_training_capture_cleanup_unresolved", "TIMEOUT")


def capture_modal_build_candidate(
    *, sdk: object, client: object, profile_path: Path, build_claim: Callable[[], None],
    app_name: str, environment_name: str, expected_intent_digest: str,
    build_app: object | None = None,
    builder_cache_root: Path | None = None,
) -> ModalBuildCandidateV1:
    """Build the fixed pinned image, then measure it before release construction.

    ``build_claim`` durably consumes one-shot authority before any provider
    operation.  The host resolves ``profile_path`` from its allowlisted profile;
    it is never a customer-provided wheel or source path.
    """
    if getattr(sdk, "__version__", None) != "1.5.4" or client is None or not callable(build_claim):
        raise ValueError("Modal build client or authorization is invalid")
    intent = plan_modal_build_material(profile_path)
    if _SHA256.fullmatch(expected_intent_digest) is None or intent["intent_digest"] != expected_intent_digest:
        raise ValueError("Modal build intent changed")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,126}", app_name) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,126}", environment_name):
        raise ValueError("Modal build scope is invalid")
    profile = load_profile(profile_path)
    if profile.packaged_runtime is None or profile.packages:
        raise ValueError("only the pinned packaged Modal profile is supported")
    root = Path(__file__).resolve().parents[4]
    inspector = root / "tuner" / "runtime" / "modal_build_inspector.py"
    inspector_raw = stable_read(inspector, 128 * 1024)
    inspector_digest = hashlib.sha256(inspector_raw).hexdigest()
    deadline = time.monotonic() + 3600
    with tempfile.TemporaryDirectory(prefix="synaptic-modal-build-") as scratch:
        staged = Path(scratch)
        try:
            wheel, digest = prepare_current_source_wheel(
                root, staged, expected_source_commit=str(intent["engine_source_commit"]),
                builder_cache_root=builder_cache_root,
            )
        except SourceArchiveInvalid:
            raise
        except SourceWheelFailure as error:
            raise ModalBuildStageFailure("SOURCE_WHEEL", error.reason) from None
        except Exception:
            raise ModalBuildStageFailure("SOURCE_WHEEL", "LOCAL_BUILD_FAILED") from None
        try:
            packaged = json.loads(json.dumps(profile.packaged_runtime))
            packaged["wheel"]["filename"] = wheel.name
            packaged["wheel"]["sha256"] = digest
            inputs_raw = (json.dumps(packaged, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
            staged_inputs = staged / "build-inputs.json"
            staged_inputs.write_bytes(inputs_raw)
            requirements = []
            all_wheels = [packaged["wheel"], *packaged["bootstrap"]]
            for item in all_wheels:
                path = wheel if item is packaged["wheel"] else profile.artifact_root / item["filename"]
                raw = stable_read(path, 256 * 1024 * 1024)
                if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                    raise ValueError("pinned Modal wheel differs")
                if path != wheel:
                    target = staged / item["filename"]
                    target.write_bytes(raw)
                requirements.append(f"/opt/synaptic-runtime/{item['filename']} --hash=sha256:{item['sha256']}")
            staged_requirements = staged / "requirements.txt"
            staged_requirements.write_text("\n".join(requirements) + "\n", encoding="ascii")
            python = profile.python_executable
            command = (
                "set -eu; before=\"$(" + python + " -I -m pip check 2>&1)\" || test \"$?\" -eq 1; "
                + python + " -I -m pip install --no-index --no-deps --require-hashes --no-cache-dir -r /opt/synaptic-runtime/requirements.txt; "
                + "after=\"$(" + python + " -I -m pip check 2>&1)\" || test \"$?\" -eq 1; "
                + "test \"$before\" = \"$after\""
            )
            recipe = {
                "sdk_version": "1.5.4", "python_executable": python,
                "commands": [command],
                "installer_flags": ["--no-index", "--no-deps", "--require-hashes", "--no-cache-dir"],
                "wheels": [{"filename": item["filename"], "sha256": item["sha256"]} for item in all_wheels],
                "builder_policy": "synaptic-modal-wheel-build/v2",
                "builder_lock_sha256": hashlib.sha256(builder_lock_bytes()).hexdigest(),
            }
            material = {
                "kind": "modal_build", "base_image": {
                    "reference": profile.base_image, "digest": profile.base_image.split("@sha256:", 1)[1],
                },
                "build_inputs": recipe,
                "build_inputs_digest": hashlib.sha256(canonical_bytes(recipe)).hexdigest(),
            }
            if plan_modal_build_material(profile_path)["intent_digest"] != expected_intent_digest:
                raise ValueError("Modal build intent changed before claim")
        except Exception:
            raise ModalBuildStageFailure("BUILD_INPUTS", "INVALID") from None
        build_claim()
        app = (build_app if build_app is not None else _bounded(
            lambda: sdk.App.lookup(app_name, create_if_missing=False, client=client,
                                   environment_name=environment_name),
            deadline=deadline, code="modal_build_app_lookup_failed"))
        try:
            image = sdk.Image.from_registry(profile.base_image).entrypoint([])
            image = image.add_local_file(staged_inputs, "/opt/synaptic-runtime/build-inputs.json", copy=True)
            image = image.add_local_file(staged_requirements, "/opt/synaptic-runtime/requirements.txt", copy=True)
            for item in all_wheels:
                image = image.add_local_file(staged / item["filename"], "/opt/synaptic-runtime/" + item["filename"], copy=True)
            image = image.run_commands(command)
            image = _bounded(lambda: image.build(app), deadline=deadline, code="modal_training_image_build_failed")
        except ModalBoundedOperationFailure as error:
            raise ModalBuildStageFailure("IMAGE_BUILD", error.reason) from None
        except Exception:
            raise ModalBuildStageFailure("IMAGE_BUILD", "OPERATION_FAILED") from None
        image_id = getattr(image, "object_id", None)
        if type(image_id) is not str or _IMAGE_ID.fullmatch(image_id) is None:
            raise ModalBuildStageFailure("IMAGE_BUILD", "IDENTITY_MISSING")
        try:
            sandbox = _bounded(
                lambda: sdk.Sandbox.create(
                    python, "-I", "-m", "tuner.runtime.modal_build_inspector",
                    app=app, image=image, cpu=1.0, memory=2048, timeout=300, idle_timeout=300,
                    block_network=True, client=client,
                ), deadline=deadline, code="modal_training_capture_create_ambiguous", late_cleanup=_terminate_sandbox,
            )
        except ModalBoundedOperationFailure as error:
            raise ModalBuildStageFailure("CAPTURE_CREATE", error.reason) from None
        capture_failure: BaseException | None = None
        try:
            try:
                raw = _bounded(lambda: _capture_output(sandbox), deadline=deadline,
                               code="modal_training_capture_failed")
            except ModalBoundedOperationFailure as error:
                raise ModalBuildStageFailure("CAPTURE_OUTPUT", error.reason) from None
        except BaseException as error:
            capture_failure = error
            raise
        finally:
            try:
                _cleanup_sandbox(sandbox)
            except ModalBoundedOperationFailure as error:
                if capture_failure is None:
                    raise ModalBuildStageFailure("CAPTURE_CLEANUP", error.reason) from None
            except Exception:
                if capture_failure is None:
                    raise ModalBuildStageFailure("CAPTURE_CLEANUP", "OPERATION_FAILED") from None
        try:
            if hashlib.sha256(stable_read(inspector, 128 * 1024)).hexdigest() != inspector_digest:
                raise ValueError("Modal training inspector changed")
            candidate = ModalBuildCandidateV1.from_capture(
                material=material, raw=raw, expected_image_id=image_id,
                expected_inspector_sha256=inspector_digest,
                expected_build_inputs_digest=hashlib.sha256(inputs_raw).hexdigest(),
            )
            candidate._attest_build(_BUILDER_ATTESTATION)
        except Exception:
            raise ModalBuildStageFailure("CAPTURE_VALIDATE", "INVALID") from None
        return candidate


@dataclass(frozen=True, slots=True)
class ModalBuildCandidateV1:
    """Bounded capture of one actual Modal image and its installed inventory."""

    image_id: str
    _material_raw: bytes
    _capture_raw: bytes
    inspector_sha256: str
    capture_digest: str
    _builder_attested: bool = field(init=False, default=False, repr=False)

    @property
    def material(self) -> dict[str, object]:
        return json.loads(self._material_raw)

    @property
    def measured(self) -> dict[str, object]:
        return json.loads(self._capture_raw)["measured"]

    def _validate_capture(self) -> None:
        if (type(self._capture_raw) is not bytes or not self._capture_raw
                or len(self._capture_raw) > _MAX_CAPTURE_BODY_BYTES + 1
                or self._capture_raw[-1:] != b"\n"
                or hashlib.sha256(self._capture_raw).hexdigest() != self.capture_digest
                or _IMAGE_ID.fullmatch(self.image_id) is None
                or _SHA256.fullmatch(self.inspector_sha256) is None):
            raise ValueError("Modal candidate capture changed")
        document = json.loads(self._capture_raw)
        if (type(document) is not dict
                or document.get("schema_version") != "synaptic-modal-build-capture/v1"
                or document.get("image_id") != self.image_id
                or document.get("inspector_sha256") != self.inspector_sha256
                or _canonical_capture_bytes(document) + b"\n" != self._capture_raw):
            raise ValueError("Modal candidate capture differs")
        material = self.material
        if material.get("kind") != "modal_build" or canonical_bytes(material) != self._material_raw:
            raise ValueError("Modal candidate material changed")

    def _attest_build(self, token: object) -> None:
        if token is not _BUILDER_ATTESTATION:
            raise ValueError("Modal build attestation is private")
        self._validate_capture()
        object.__setattr__(self, "_builder_attested", True)

    @classmethod
    def from_capture(
        cls, *, material: Mapping[str, object], raw: bytes,
        expected_image_id: str, expected_inspector_sha256: str,
        expected_build_inputs_digest: str,
    ) -> "ModalBuildCandidateV1":
        if type(raw) is not bytes or not raw or len(raw) > _MAX_CAPTURE_BODY_BYTES + 1 or raw[-1:] != b"\n":
            raise ValueError("Modal build capture is invalid")
        if _IMAGE_ID.fullmatch(expected_image_id) is None or _SHA256.fullmatch(expected_inspector_sha256) is None:
            raise ValueError("Modal build identity is invalid")
        try:
            document = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ValueError("Modal build capture is invalid") from None
        if type(document) is not dict or set(document) != {"schema_version", "image_id", "inspector_sha256", "measured"}:
            raise ValueError("Modal build capture fields differ")
        if (document["schema_version"] != "synaptic-modal-build-capture/v1"
                or document["image_id"] != expected_image_id
                or document["inspector_sha256"] != expected_inspector_sha256
                or type(document["measured"]) is not dict
                or _canonical_capture_bytes(document) + b"\n" != raw):
            raise ValueError("Modal build capture identity differs")
        selected = dict(material)
        if selected.get("kind") != "modal_build":
            raise ValueError("Modal build material is invalid")
        measured = document["measured"]
        inputs = selected.get("build_inputs")
        if (type(inputs) is not dict or _SHA256.fullmatch(expected_build_inputs_digest) is None
                or measured.get("build_inputs_digest") != expected_build_inputs_digest):
            raise ValueError("Modal build inputs are invalid")
        wheels = inputs.get("wheels")
        if type(wheels) is not list or not wheels:
            raise ValueError("Modal build wheels are invalid")
        wheel = measured.get("package")
        if type(wheel) is not dict or type(wheels[0]) is not dict or wheels[0].get("sha256") != wheel.get("digest"):
            raise ValueError("Modal engine wheel differs from capture")
        result = cls(expected_image_id, canonical_bytes(selected), raw, expected_inspector_sha256, hashlib.sha256(raw).hexdigest())
        result._validate_capture()
        return result

    def validate_release(self, release: PackagedTrainingRuntimeReleaseV2, *, require_builder: bool = True) -> None:
        self._validate_capture()
        if require_builder and self._builder_attested is not True:
            raise ValueError("Modal candidate lacks provider-build attestation")
        if type(release) is not PackagedTrainingRuntimeReleaseV2 or release.material != self.material:
            raise ValueError("Modal candidate material differs from release")
        measured = self.measured
        checks = {
            "package_name": ("package", "name"),
            "package_version": ("package", "version"),
            "package_digest": ("package", "digest"),
            "source_provenance_digest": ("package", "source_provenance_digest"),
            "worker_entrypoint": ("worker", "entrypoint"),
            "worker_closure_digest": ("worker", "closure_digest"),
            "python_implementation": ("python", "implementation"),
            "python_version": ("python", "version"),
            "python_executable": ("python", "executable"),
            "python_executable_digest": ("python", "executable_digest"),
            "installed_distributions_digest": ("installed_distributions", "digest"),
            "installed_distribution_count": ("installed_distributions", "count"),
            "platform_system": ("platform", "system"),
            "platform_machine": ("platform", "machine"),
            "cuda_version": ("platform", "cuda_version"),
        }
        for field, path in checks.items():
            value: object = measured
            for part in path:
                if type(value) is not dict:
                    raise ValueError("Modal candidate measurement is invalid")
                value = value[part]
            if getattr(release, field) != value:
                raise ValueError("Modal candidate measurement differs from release")
        if (release.to_dict()["platform"]["runtime_facts"] != measured["platform"]["runtime_facts"]
                or measured["contracts"] != release.to_dict()["contracts"]
                or measured["capabilities"]["compatibility"] != release.to_dict()["compatibility"]):
            raise ValueError("Modal candidate capabilities differ from release")


def build_modal_runtime_release_v2(candidate: ModalBuildCandidateV1, *, release_ref: str) -> PackagedTrainingRuntimeReleaseV2:
    """Issue the final release only from a captured, measured image."""
    if type(candidate) is not ModalBuildCandidateV1:
        raise TypeError("exact Modal build candidate required")
    measured = candidate.measured
    compatibility = measured["capabilities"]["compatibility"]
    contracts = measured["contracts"]
    release = PackagedTrainingRuntimeReleaseV2.build(
        release_ref=release_ref, material=candidate.material,
        package_name=measured["package"]["name"],
        package_version=measured["package"]["version"],
        package_digest=measured["package"]["digest"],
        source_provenance_digest=measured["package"]["source_provenance_digest"],
        worker_entrypoint=measured["worker"]["entrypoint"],
        worker_closure_digest=measured["worker"]["closure_digest"],
        python_implementation=measured["python"]["implementation"],
        python_version=measured["python"]["version"],
        python_executable=measured["python"]["executable"],
        python_executable_digest=measured["python"]["executable_digest"],
        installed_distributions_digest=measured["installed_distributions"]["digest"],
        installed_distribution_count=measured["installed_distributions"]["count"],
        platform_system=measured["platform"]["system"],
        platform_machine=measured["platform"]["machine"],
        cuda_version=measured["platform"]["cuda_version"],
        runtime_facts=measured["platform"]["runtime_facts"],
        compatible_methods=tuple(compatibility["methods"]),
        compatible_models=tuple((item["ref"], item["revision"]) for item in compatibility["models"]),
        compatible_dataset_formats=tuple(compatibility["dataset_formats"]),
        workload_schema=contracts["workload_schema"],
        prepared_input_schema=contracts["prepared_input_schema"],
        artifact_contract_schema=contracts["artifact_contract_schema"],
    )
    candidate.validate_release(release, require_builder=False)
    return release
