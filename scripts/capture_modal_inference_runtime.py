"""Capture candidate runtime metadata in an existing Modal app.

This launcher is intentionally not a runtime-lock issuer.  It runs the
credential-free inspector in a digest-selected public image and returns the
inspector's candidate report together with local capture provenance.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import tempfile
from typing import Any, Callable
import stat

_IMAGE = re.compile(r"docker\.io/vllm/vllm-openai@sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SDK_VERSION = "1.5.4"
_REMOTE_SCRIPT = "/opt/synaptic/inspect_modal_inference_runtime.py"
_MAX_CAPTURE_BYTES = 1024 * 1024
_HOST_TIMEOUT_SECONDS = 600.0
_SANDBOX_TIMEOUT_SECONDS = 300
_CLEANUP_TIMEOUT_SECONDS = 10.0


class ModalInferenceRuntimeCaptureError(RuntimeError):
    """Closed failure retaining an exact returned Sandbox when one is known."""

    def __init__(
        self,
        reason_code: str,
        *,
        cleanup_lease: object | None = None,
        sandbox_id: str | None = None,
        cleanup_requested: bool = False,
        cleanup_confirmed: bool = False,
    ) -> None:
        self.reason_code = reason_code
        self.cleanup_lease = cleanup_lease
        self.sandbox_id = sandbox_id
        self.cleanup_requested = cleanup_requested
        self.cleanup_confirmed = cleanup_confirmed
        super().__init__(reason_code)


class ModalInferenceRuntimeCapture:
    """Immutable capture output; neither image nor source attestation."""

    __slots__ = ("_raw", "_sandbox_id", "_cleanup_requested", "_token")

    def __init__(self, *args: object, **kwargs: object) -> None:
        if kwargs.pop("_token", None) is not _CAPTURE_TOKEN or args or kwargs:
            raise TypeError("modal_inference_runtime_capture_invalid")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("modal_inference_runtime_capture_immutable")

    @property
    def canonical_bytes(self) -> bytes:
        return bytes(self._raw)

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    @property
    def cleanup_requested(self) -> bool:
        return self._cleanup_requested


_CAPTURE_TOKEN = object()


class _ClosedParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ModalInferenceRuntimeCaptureError("capture_input_invalid")


def _exact_text(value: object, pattern: re.Pattern[str] | None = None) -> str:
    try:
        encoded_size = len(value.encode("utf-8")) if type(value) is str else 0
    except UnicodeError:
        raise ModalInferenceRuntimeCaptureError("capture_input_invalid") from None
    if (
        type(value) is not str
        or not value
        or encoded_size > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or (pattern is not None and pattern.fullmatch(value) is None)
    ):
        raise ModalInferenceRuntimeCaptureError("capture_input_invalid")
    return value


def _script_path() -> Path:
    return Path(__file__).with_name("inspect_modal_inference_runtime.py")


def _script_source(path: Path) -> tuple[bytes, str]:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ModalInferenceRuntimeCaptureError("inspection_source_invalid")
        digest = hashlib.sha256()
        content = bytearray()
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_CAPTURE_BYTES + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_CAPTURE_BYTES:
                raise ModalInferenceRuntimeCaptureError("inspection_source_invalid")
            digest.update(chunk)
            content.extend(chunk)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_nlink,
        )
        if size == 0 or size != before.st_size or identity(after) != identity(before):
            raise ModalInferenceRuntimeCaptureError("inspection_source_invalid")
        return bytes(content), digest.hexdigest()
    except ModalInferenceRuntimeCaptureError:
        raise
    except OSError:
        raise ModalInferenceRuntimeCaptureError("inspection_source_invalid") from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                if sys.exc_info()[0] is None:
                    raise ModalInferenceRuntimeCaptureError(
                        "inspection_source_invalid"
                    ) from None


class _PendingOperation:
    def __init__(
        self,
        operation: Callable[[], Any],
        late_result: Callable[[object], object] | None = None,
        ownership: "_OperationOwnership | None" = None,
    ):
        self.result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._abandoned = False
        self._completed = False
        self._cleanup_started = False
        self._successful = False
        self._value: object | None = None
        self._late_result = late_result
        self.late_value: object | None = None
        self.late_cleanup: object | None = None
        if ownership is not None:
            ownership.attach(self)

        def claim_cleanup() -> object | None:
            with self._lock:
                if (
                    not self._abandoned
                    or not self._completed
                    or not self._successful
                    or self._cleanup_started
                ):
                    return None
                self._cleanup_started = True
                self.late_value = self._value
                return self._value

        def invoke() -> None:
            try:
                value = operation()
            except BaseException as error:
                with self._lock:
                    self._completed = True
                self.result.put((False, error))
                return
            with self._lock:
                self._completed = True
                self._successful = True
                self._value = value
            if ownership is not None:
                ownership.retain(value)
            self.result.put((True, value))
            cleanup_value = claim_cleanup()
            if late_result is not None and cleanup_value is not None:
                try:
                    self.late_cleanup = late_result(cleanup_value)
                except BaseException:
                    self.late_cleanup = (False, False)

        self.worker = threading.Thread(
            target=invoke, daemon=True, name="modal-runtime-capture"
        )
        self.worker.start()

    def abandon(self) -> None:
        cleanup_value = None
        with self._lock:
            self._abandoned = True
            if self._completed and self._successful and not self._cleanup_started:
                self._cleanup_started = True
                self.late_value = self._value
                cleanup_value = self._value
        if cleanup_value is not None and self._late_result is not None:
            try:
                self.late_cleanup = self._late_result(cleanup_value)
            except BaseException:
                self.late_cleanup = (False, False)

    def recovery_lease(self) -> "_PendingOperation | None":
        with self._lock:
            if self._completed and not self._successful:
                return None
            return self


class _OperationOwnership:
    """Caller-visible ownership installed before an asynchronous create starts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: _PendingOperation | None = None
        self._value: object | None = None

    def attach(self, pending: _PendingOperation) -> None:
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("operation ownership already attached")
            self._pending = pending

    def retain(self, value: object) -> None:
        with self._lock:
            self._value = value

    def transfer(self, value: object) -> None:
        with self._lock:
            if self._value is not value:
                raise RuntimeError("operation ownership transfer differs")
            self._value = None

    def retained_value(self) -> object | None:
        with self._lock:
            return self._value

    def abandon_pending(self) -> None:
        with self._lock:
            pending = self._pending
        if pending is not None:
            pending.abandon()

    def cleanup_lease(self) -> object | None:
        with self._lock:
            value = self._value
            pending = self._pending
        if value is not None:
            return value
        return None if pending is None else pending.recovery_lease()


def _bounded_call(
    operation: Callable[[], Any],
    *,
    deadline: float,
    timeout_code: str,
    failure_code: str = "modal_operation_failed",
    late_result: Callable[[object], object] | None = None,
    ownership: _OperationOwnership | None = None,
) -> Any:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ModalInferenceRuntimeCaptureError(timeout_code)
    pending = _PendingOperation(operation, late_result, ownership)
    try:
        successful, value = pending.result.get(timeout=remaining)
    except queue.Empty:
        pending.abandon()
        raise ModalInferenceRuntimeCaptureError(
            timeout_code, cleanup_lease=pending
        ) from None
    except BaseException as error:
        pending.abandon()
        setattr(error, "cleanup_lease", pending)
        raise
    if successful:
        return value
    if isinstance(value, (KeyboardInterrupt, SystemExit)):
        raise value
    raise ModalInferenceRuntimeCaptureError(failure_code) from None


def _read_output(sandbox: object) -> bytes:
    output = bytearray()
    stream = getattr(sandbox, "stdout")
    for chunk in stream:
        if type(chunk) is str:
            try:
                chunk = chunk.encode("utf-8")
            except UnicodeError:
                raise ModalInferenceRuntimeCaptureError(
                    "capture_output_invalid"
                ) from None
        if type(chunk) is not bytes:
            raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
        if len(chunk) > _MAX_CAPTURE_BYTES - len(output):
            raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
        output.extend(chunk)
    sandbox.wait(raise_on_termination=False)
    exit_code = getattr(sandbox, "returncode")
    if type(exit_code) is not int or exit_code != 0:
        raise ModalInferenceRuntimeCaptureError("inspection_failed")
    return bytes(output)


def _cleanup(sandbox: object) -> tuple[bool, bool]:
    deadline = time.monotonic() + _CLEANUP_TIMEOUT_SECONDS
    requested = False
    confirmed = False
    try:
        _bounded_call(
            lambda: sandbox.terminate(wait=False),
            deadline=deadline,
            timeout_code="cleanup_unresolved",
        )
        requested = True
        poll_deadline = deadline
        while time.monotonic() < poll_deadline:
            result = _bounded_call(
                sandbox.poll,
                deadline=poll_deadline,
                timeout_code="cleanup_unresolved",
            )
            if result is not None:
                confirmed = True
                break
            time.sleep(0.05)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        pass
    return requested, confirmed


def _sandbox_id(sandbox: object) -> str:
    return _exact_text(getattr(sandbox, "object_id"), _NAME)


def _late_cleanup(sandbox: object) -> tuple[bool, bool]:
    return _cleanup(sandbox)


def _parse_candidate(
    raw: bytes, *, image: str, source_commit: str
) -> dict[str, object]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
    document = raw[:-1]

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if type(key) is not str or key in result:
                raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
            result[key] = value
        return result

    try:
        candidate = json.loads(document, object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError):
        raise ModalInferenceRuntimeCaptureError("capture_output_invalid") from None
    if type(candidate) is not dict or set(candidate) != {
        "distributions",
        "operator_selection",
        "python",
        "requirements",
        "schema_version",
        "status",
    }:
        raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
    requirements = candidate["requirements"]
    if type(requirements) is not dict or set(requirements) != {"modal", "vllm"}:
        raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
    for name in ("modal", "vllm"):
        item = requirements[name]
        if (
            type(item) is not dict
            or set(item) != {"present", "version"}
            or type(item["present"]) is not bool
            or (item["present"] and type(item["version"]) is not str)
            or (not item["present"] and item["version"] is not None)
        ):
            raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
    distributions = candidate["distributions"]
    python = candidate["python"]
    if (
        type(distributions) is not dict
        or len(distributions) > 512
        or any(
            type(key) is not str or type(value) is not str
            for key, value in distributions.items()
        )
        or type(python) is not dict
        or set(python)
        != {"executable", "executable_sha256", "implementation", "version"}
        or python["implementation"] != "cpython"
        or type(python["executable"]) is not str
        or type(python["version"]) is not str
        or type(python["executable_sha256"]) is not str
        or _SHA256.fullmatch(python["executable_sha256"]) is None
    ):
        raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
    for name in ("modal", "vllm"):
        requirement = requirements[name]
        if requirement["present"]:
            if distributions.get(name) != requirement["version"]:
                raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
        elif name in distributions:
            raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
    if (
        candidate["schema_version"]
        != "synaptic-modal-inference-runtime-inspection-candidate/v1"
        or candidate["status"] != "CANDIDATE_ONLY"
        or candidate["operator_selection"]
        != {"image": image, "source_commit": source_commit}
        or json.dumps(
            candidate,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        != document
    ):
        raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
    return candidate


def capture_modal_inference_runtime(
    *,
    sdk: object,
    client: object,
    app_name: str,
    environment_name: str,
    image: str,
    source_commit: str,
) -> ModalInferenceRuntimeCapture:
    """Run the inspector once, with a finite local and remote lifetime."""

    app_name = _exact_text(app_name, _NAME)
    environment_name = _exact_text(environment_name, _NAME)
    image = _exact_text(image, _IMAGE)
    source_commit = _exact_text(source_commit, _COMMIT)
    if client is None or getattr(sdk, "__version__", None) != _SDK_VERSION:
        raise ModalInferenceRuntimeCaptureError("modal_sdk_invalid")
    script = _script_path()
    script_bytes, script_digest = _script_source(script)
    deadline = time.monotonic() + _HOST_TIMEOUT_SECONDS
    sandbox = None
    sandbox_id = None
    active_error: BaseException | None = None
    requested = False
    confirmed = False
    create_ownership = _OperationOwnership()
    try:
        app = _bounded_call(
            lambda: sdk.App.lookup(
                app_name,
                create_if_missing=False,
                client=client,
                environment_name=environment_name,
            ),
            deadline=deadline,
            timeout_code="app_lookup_timeout",
        )
        temporary = tempfile.TemporaryDirectory(prefix="synaptic-modal-inspection-")
        staged = Path(temporary.name) / "inspect_modal_inference_runtime.py"
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            offset = 0
            while offset < len(script_bytes):
                written = os.write(descriptor, script_bytes[offset:])
                if type(written) is not int or written <= 0:
                    raise ModalInferenceRuntimeCaptureError("inspection_source_invalid")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        image_value = (
            sdk.Image.from_registry(image)
            .entrypoint([])
            .add_local_file(staged, _REMOTE_SCRIPT, copy=True)
        )
        if (
            _script_source(staged)[1] != script_digest
            or _script_source(script)[1] != script_digest
        ):
            raise ModalInferenceRuntimeCaptureError("inspection_source_changed")

        def create_sandbox() -> object:
            try:
                return sdk.Sandbox.create(
                    "python3",
                    _REMOTE_SCRIPT,
                    "--image",
                    image,
                    "--source-commit",
                    source_commit,
                    app=app,
                    image=image_value,
                    cpu=1.0,
                    memory=2048,
                    timeout=_SANDBOX_TIMEOUT_SECONDS,
                    idle_timeout=_SANDBOX_TIMEOUT_SECONDS,
                    block_network=True,
                    client=client,
                )
            finally:
                temporary.cleanup()

        sandbox = _bounded_call(
            create_sandbox,
            deadline=deadline,
            timeout_code="sandbox_create_ambiguous",
            failure_code="sandbox_create_ambiguous",
            late_result=_late_cleanup,
            ownership=create_ownership,
        )
        create_ownership.transfer(sandbox)
        sandbox_id = _sandbox_id(sandbox)
        raw = _bounded_call(
            lambda: _read_output(sandbox),
            deadline=deadline,
            timeout_code="capture_timeout",
        )
        if _script_source(script)[1] != script_digest:
            raise ModalInferenceRuntimeCaptureError("inspection_source_changed")
        candidate = _parse_candidate(raw, image=image, source_commit=source_commit)
        envelope = {
            "candidate": candidate,
            "inspection_script_sha256": script_digest,
            "operator_selection_only": True,
            "sandbox_id": sandbox_id,
            "schema_version": "synaptic-modal-inference-runtime-capture/v1",
        }
        encoded = json.dumps(
            envelope,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        if len(encoded) > _MAX_CAPTURE_BYTES:
            raise ModalInferenceRuntimeCaptureError("capture_output_invalid")
    except BaseException as error:
        active_error = error
    finally:
        if sandbox is None:
            sandbox = create_ownership.retained_value()
            if sandbox is None:
                create_ownership.abandon_pending()
        if sandbox is not None:
            try:
                requested, confirmed = _cleanup(sandbox)
            except BaseException as cleanup_error:
                if active_error is None:
                    active_error = cleanup_error
    if active_error is not None:
        if isinstance(active_error, (KeyboardInterrupt, SystemExit)):
            if getattr(active_error, "cleanup_lease", None) is None:
                setattr(
                    active_error,
                    "cleanup_lease",
                    (
                        sandbox
                        if sandbox is not None
                        else create_ownership.cleanup_lease()
                    ),
                )
            setattr(active_error, "sandbox_id", sandbox_id)
            setattr(active_error, "cleanup_requested", requested)
            setattr(active_error, "cleanup_confirmed", confirmed)
            raise active_error
        reason_code = (
            active_error.reason_code
            if isinstance(active_error, ModalInferenceRuntimeCaptureError)
            else "modal_inference_runtime_capture_failed"
        )
        raise ModalInferenceRuntimeCaptureError(
            reason_code,
            cleanup_lease=(
                active_error.cleanup_lease
                if sandbox is None
                and isinstance(active_error, ModalInferenceRuntimeCaptureError)
                and active_error.cleanup_lease is not None
                else (
                    sandbox if sandbox is not None else create_ownership.cleanup_lease()
                )
            ),
            sandbox_id=sandbox_id,
            cleanup_requested=requested,
            cleanup_confirmed=confirmed,
        ) from None
    if not requested or not confirmed:
        raise ModalInferenceRuntimeCaptureError(
            "cleanup_unresolved",
            cleanup_lease=sandbox,
            sandbox_id=sandbox_id,
            cleanup_requested=requested,
            cleanup_confirmed=confirmed,
        ) from None
    result = ModalInferenceRuntimeCapture(_token=_CAPTURE_TOKEN)
    object.__setattr__(result, "_raw", encoded)
    object.__setattr__(result, "_sandbox_id", sandbox_id)
    object.__setattr__(result, "_cleanup_requested", True)
    object.__setattr__(result, "_token", _CAPTURE_TOKEN)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = _ClosedParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--modal-profile",
        type=lambda value: _exact_text(value, _NAME),
        help="exact named Modal profile; never falls back to environment auth",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        token_id = None
        token_secret = None
        if arguments.modal_profile is None:
            token_id = os.environ.get("MODAL_TOKEN_ID")
            token_secret = os.environ.get("MODAL_TOKEN_SECRET")
            if (
                not token_id
                or not token_id.strip()
                or not token_secret
                or not token_secret.strip()
            ):
                raise ModalInferenceRuntimeCaptureError("modal_credentials_missing")
        spec = importlib.util.find_spec("modal")
        if spec is None:
            raise ModalInferenceRuntimeCaptureError("modal_sdk_invalid")

        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                import modal

                if arguments.modal_profile is not None:
                    from modal.config import config

                    token_id = config.get(
                        "token_id", profile=arguments.modal_profile, use_env=False
                    )
                    token_secret = config.get(
                        "token_secret", profile=arguments.modal_profile, use_env=False
                    )
                    if (
                        type(token_id) is not str
                        or not token_id.strip()
                        or type(token_secret) is not str
                        or not token_secret.strip()
                    ):
                        raise ModalInferenceRuntimeCaptureError(
                            "modal_credentials_missing"
                        )
                client = modal.Client.from_credentials(token_id, token_secret)
                capture = capture_modal_inference_runtime(
                    sdk=modal,
                    client=client,
                    app_name=arguments.app,
                    environment_name=arguments.environment,
                    image=arguments.image,
                    source_commit=arguments.source_commit,
                )
        sys.stdout.buffer.write(capture.canonical_bytes + b"\n")
        return 0
    except (KeyboardInterrupt, SystemExit) as error:
        if isinstance(error, SystemExit) and error.code == 0:
            return 0
        sys.stderr.write(
            json.dumps(
                {
                    "cleanup_confirmed": bool(
                        getattr(error, "cleanup_confirmed", False)
                    ),
                    "cleanup_requested": bool(
                        getattr(error, "cleanup_requested", False)
                    ),
                    "ownership_known": getattr(error, "cleanup_lease", None)
                    is not None,
                    "sandbox_id": getattr(error, "sandbox_id", None),
                    "status": "INTERRUPTED",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 125
    except Exception as error:
        payload = {"reason_code": "capture_failed", "status": "FAILED"}
        if isinstance(error, ModalInferenceRuntimeCaptureError):
            payload.update(
                {
                    "cleanup_confirmed": error.cleanup_confirmed,
                    "cleanup_requested": error.cleanup_requested,
                    "ownership_known": error.cleanup_lease is not None,
                    "sandbox_id": error.sandbox_id,
                }
            )
            if error.reason_code == "sandbox_create_ambiguous":
                payload["create_ambiguous"] = True
        sys.stderr.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
