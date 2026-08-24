"""Closed provider operations for the protected HF A10G training smoke."""

from __future__ import annotations

import inspect
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_UP
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Sequence

from tuner.cloud.hf_training_smoke_contract import (
    APPROVAL_SCHEMA,
    ARTIFACT_SLOT_INPUT_SCHEMA,
    HARDWARE_MAX_HOURLY_COST_MICRO_USD,
    HARDWARE_MAX_TIMEOUT_COST_MICRO_USD,
    HARDWARE_QUOTE_ENDPOINT,
    HARDWARE_QUOTE_MAX_UNIT_COST_MICRO_USD,
    HARDWARE_QUOTE_UNIT_LABEL,
    PREFLIGHT_SCHEMA,
    SUBMISSION_SCHEMA,
    derive_hf_training_artifact_prefix,
    derive_hf_training_artifact_slot,
    document_sha256,
    seal_training_document,
)
from tuner.cloud.hf_training_smoke_workload import ProtectedWorkload
from tuner.core.exceptions import CloudProviderError


HF_HUB_VERSION = "1.27.0"
HF_ENDPOINT = HARDWARE_QUOTE_ENDPOINT
HARDWARE_FLAVOR = "a10g-small"
PROVIDER_TIMEOUT_SECONDS = 1800
CANCEL_AFTER_SECONDS = 1500
OBSERVE_UNTIL_SECONDS = 2100
HTTP_TIMEOUT_SECONDS = (10.0, 30.0, 30.0, 10.0)
MAX_REDIRECTS = 5
FIXED_NONSECRET_ENV = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_DISABLED": "true",
}
_FORBIDDEN_ENV = frozenset(
    {
        "HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN", "HF_ENDPOINT",
        "HF_INFERENCE_ENDPOINT", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "NO_PROXY", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
)
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_PROVIDER_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_PROVIDER_LABEL_DIGEST_PREFIX = 48
_DOWNLOAD_PASS = b'{"schema_version":"synaptic-hf-training-download-child/v1","status":"PASS"}\n'
_PROVIDER_STAGES = frozenset({"SCHEDULING", "RUNNING", "COMPLETED", "ERROR", "DELETED", "CANCELED"})
_PROVIDER_FAILURE_REASONS = {
    400: "PROVIDER_REQUEST_REJECTED",
    401: "PROVIDER_AUTH_REJECTED",
    402: "PROVIDER_PAYMENT_REJECTED",
    403: "PROVIDER_AUTH_REJECTED",
    404: "PROVIDER_REQUEST_REJECTED",
    409: "PROVIDER_REQUEST_REJECTED",
    413: "PROVIDER_REQUEST_REJECTED",
    422: "PROVIDER_REQUEST_REJECTED",
    429: "PROVIDER_RATE_LIMITED",
}
_SAFE_PROVIDER_FAILURE_REASONS = frozenset({
    *_PROVIDER_FAILURE_REASONS.values(),
    "PROVIDER_SERVICE_ERROR",
    "PROVIDER_TRANSPORT_ERROR",
})


class _ProviderSubmissionFailure(Exception):
    """Carry only a closed nonsecret failure class across the submit boundary."""

    __slots__ = ("reason_code",)

    def __init__(self, reason_code: str) -> None:
        if reason_code not in _SAFE_PROVIDER_FAILURE_REASONS:
            reason_code = "PROVIDER_TRANSPORT_ERROR"
        self.reason_code = reason_code
        super().__init__("HF provider submission failed")


def _provider_failure_reason(error: Exception) -> str:
    """Return one bounded nonsecret class for a post-boundary provider failure."""

    if type(error) is _ProviderSubmissionFailure:
        return error.reason_code
    try:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if type(status) is not int:
            return "PROVIDER_OUTCOME_AMBIGUOUS"
        if status in _PROVIDER_FAILURE_REASONS:
            return _PROVIDER_FAILURE_REASONS[status]
        if 500 <= status <= 599:
            return "PROVIDER_SERVICE_ERROR"
    except BaseException:
        return "PROVIDER_OUTCOME_AMBIGUOUS"
    return "PROVIDER_OUTCOME_AMBIGUOUS"


def _close_provider_preserving_pending(provider: object) -> None:
    pending = sys.exc_info()[0] is not None
    try:
        provider.close()  # type: ignore[attr-defined]
    except BaseException:
        if not pending:
            raise


def _advance_status_intervals(
    intervals: list[dict[str, str]], current_stage: str | None,
    current_start: str, observed_stage: str, observed_at: str,
) -> tuple[str, str]:
    if observed_stage not in _PROVIDER_STAGES:
        raise CloudProviderError("HF provider returned an unknown job stage")
    if current_stage is None:
        return observed_stage, current_start
    if observed_stage != current_stage:
        intervals.append({
            "status": current_stage, "started_at": current_start, "ended_at": observed_at,
        })
        return observed_stage, observed_at
    return current_stage, current_start


def _terminate_download_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt" and process.pid:
            completed = subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10, check=False, env={"PATH": os.defpath},
            )
            if completed.returncode:
                process.kill()
        elif process.pid:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _run_download_child(command: tuple[str, ...], secret: bytearray) -> tuple[int, bytes, bytes]:
    process: subprocess.Popen[bytes] | None = None
    streams: list[object] = []
    threads: list[threading.Thread] = []
    cleanup_done = False

    def cleanup() -> bool:
        if process is None:
            return False
        _terminate_download_tree(process)
        try:
            process.wait(timeout=10)
            gone = True
        except BaseException:
            _terminate_download_tree(process)
            try:
                gone = process.poll() is not None
            except BaseException:
                gone = False
        closed = True
        for stream in streams:
            try:
                stream.close()  # type: ignore[attr-defined]
            except BaseException:
                closed = False
        joined = True
        for thread in threads:
            try:
                if thread.ident is not None:
                    thread.join(timeout=5)
                    joined = joined and not thread.is_alive()
            except BaseException:
                joined = False
        return gone and closed and joined

    try:
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True})
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False, env={}, close_fds=True, **group,
        )
        streams = [process.stdin, process.stdout, process.stderr]
        if any(stream is None for stream in streams):
            raise RuntimeError("missing owned process pipe")
        results = [bytearray(), bytearray()]
        overflow = threading.Event()

        def consume(index: int, stream: object) -> None:
            try:
                while True:
                    chunk = stream.read(4096)  # type: ignore[attr-defined]
                    if not chunk:
                        return
                    remaining = 4097 - len(results[index])
                    results[index].extend(chunk[:remaining])
                    if len(results[index]) > 4096:
                        overflow.set()
                        _terminate_download_tree(process)  # type: ignore[arg-type]
                        return
            except (OSError, ValueError):
                overflow.set()
                _terminate_download_tree(process)  # type: ignore[arg-type]

        for index, stream in enumerate((process.stdout, process.stderr)):
            thread = threading.Thread(target=consume, args=(index, stream), daemon=True)
            threads.append(thread)
            thread.start()
        process.stdin.write(secret)
        process.stdin.close()
        try:
            code = process.wait(timeout=900)
        except subprocess.TimeoutExpired as exc:
            cleanup_done = True
            if not cleanup():
                raise CloudProviderError("Protected artifact downloader cleanup is unproven") from None
            raise CloudProviderError("Protected artifact download timed out") from exc
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads) or overflow.is_set():
            raise RuntimeError("bounded child output failure")
        for stream in streams:
            stream.close()  # type: ignore[attr-defined]
        cleanup_done = True
        return code, bytes(results[0]), bytes(results[1])
    except BaseException as exc:
        if process is None:
            if isinstance(exc, OSError):
                raise CloudProviderError("Protected artifact download could not be executed") from exc
            raise
        proven = True if cleanup_done else cleanup()
        if not proven:
            raise CloudProviderError("Protected artifact downloader cleanup is unproven") from None
        if isinstance(exc, Exception):
            if isinstance(exc, CloudProviderError):
                raise
            raise CloudProviderError("Protected artifact download was rejected") from exc
        raise


def normalize_artifact_inventory(
    values: Sequence[object], *, prefix: str,
) -> tuple[dict[str, object], ...]:
    """Reduce a provider listing to the exact closed verifier inventory."""

    from tuner.cloud.hf_training_smoke_artifacts import (
        EXPECTED_PATHS, MAX_FILE_BYTES, MAX_TOTAL_BYTES,
    )

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise CloudProviderError("Artifact provider inventory is invalid")
    root = prefix + "/"
    normalized: list[dict[str, object]] = []
    folders: set[str] = set()
    for value in values:
        kind = _field(value, "type")
        path = _field(value, "path")
        if kind == "directory":
            if not isinstance(path, str):
                raise CloudProviderError("Artifact provider inventory is invalid")
            folders.add(path)
            continue
        if kind != "file":
            raise CloudProviderError("Artifact provider inventory is invalid")
        size = _field(value, "size")
        xet_hash = _field(value, "xet_hash")
        if (
            not isinstance(path, str) or not path.startswith(root)
            or type(size) is not int or size < 0
            or (
                xet_hash is not None
                and (not isinstance(xet_hash, str) or re.fullmatch(r"[0-9a-f]{64}", xet_hash) is None)
            )
        ):
            raise CloudProviderError("Artifact provider inventory is invalid")
        normalized.append({
            "path": path[len(root):], "bytes": size, "provider_xet_hash": xet_hash,
        })
    normalized.sort(key=lambda item: str(item["path"]))
    if folders != {f"{prefix}/checkpoint-1", f"{prefix}/final_model"}:
        raise CloudProviderError("Artifact provider folder inventory is not exact")
    if tuple(item["path"] for item in normalized) != tuple(sorted(EXPECTED_PATHS)):
        raise CloudProviderError("Artifact provider inventory is not exact")
    sizes = [int(item["bytes"]) for item in normalized]
    if any(size > MAX_FILE_BYTES for size in sizes) or sum(sizes) > MAX_TOTAL_BYTES:
        raise CloudProviderError("Artifact provider inventory exceeds verifier bounds")
    return tuple(normalized)


@dataclass(frozen=True)
class HardwareQuote:
    endpoint: str
    flavor: str
    unit_cost_micro_usd: int
    unit_label: str
    hourly_cost_micro_usd: int
    timeout_cost_micro_usd: int
    fetched_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "flavor": self.flavor,
            "unit_cost_micro_usd": self.unit_cost_micro_usd,
            "unit_label": self.unit_label,
            "hourly_cost_micro_usd": self.hourly_cost_micro_usd,
            "timeout_cost_micro_usd": self.timeout_cost_micro_usd,
            "fetched_at": self.fetched_at,
        }


@dataclass(frozen=True)
class ProviderJob:
    namespace: str
    job_id: str
    created_at: str
    status: str | None = None

    def identity(self) -> dict[str, str]:
        return {
            "namespace": self.namespace,
            "job_id": self.job_id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ProviderJobExpectation:
    image: str
    command: tuple[str, ...]
    name: str
    labels: tuple[tuple[str, str], ...]
    volumes: tuple[object, object]
    namespace: str


def provider_job_identity(approval: Mapping[str, object]) -> tuple[str, dict[str, str]]:
    authorization = approval.get("authorization_id")
    bindings = approval.get("bindings")
    if not isinstance(authorization, str) or re.fullmatch(r"[0-9a-f]{64}", authorization) is None:
        raise CloudProviderError("HF training authorization identity is invalid")
    if not isinstance(bindings, Mapping):
        raise CloudProviderError("HF training approval bindings are invalid")
    for key in ("workload_digest", "artifact_slot_id"):
        value = bindings.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise CloudProviderError("HF training approval binding identity is invalid")
    labels = {
        "synaptic-kind": "hf-training-smoke",
        "synaptic-auth": authorization[:_PROVIDER_LABEL_DIGEST_PREFIX],
    }
    if any(
        _PROVIDER_LABEL.fullmatch(key) is None or _PROVIDER_LABEL.fullmatch(value) is None
        for key, value in labels.items()
    ):
        raise CloudProviderError("HF training recovery labels are invalid")
    return f"synaptic-hf-training-smoke-{authorization[:12]}", labels


def build_preflight_document(
    *, experiment_id: str, run_id: str, tracking_root_id: str, occurred_at: str,
    source: Mapping[str, object], runtime_lock: Mapping[str, object],
    runtime_lock_uri: str, runtime_lock_sha256: str, workload: ProtectedWorkload,
    dataset_bytes: int, dataset_row_sha256: str, source_bucket_id: str,
    source_prefix: str, artifact_bucket_id: str, artifact_base_prefix: str,
    expected_namespace: str, hardware_quote: HardwareQuote,
) -> dict[str, object]:
    """Seal the exact preflight after all read-only provider checks complete."""

    from tuner.cloud.hf_training_smoke_workload import (
        DATASET, DATASET_GIT_BLOB, DATASET_SHA256, MODEL, MODEL_REVISION,
    )

    slot_input = {
        "schema_version": ARTIFACT_SLOT_INPUT_SCHEMA,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "tracking_root_id": tracking_root_id,
        "source_lock_sha256": source["source_lock"]["sha256"],
        "workload_digest": workload.workload_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "artifact_bucket_id": artifact_bucket_id,
        "artifact_base_prefix": artifact_base_prefix,
    }
    slot_id = derive_hf_training_artifact_slot(slot_input)
    try:
        slot_index = workload.remote_argv.index("--artifact-slot")
    except ValueError as exc:
        raise CloudProviderError("Protected workload lacks an artifact slot") from exc
    if slot_index + 1 >= len(workload.remote_argv) or workload.remote_argv[slot_index + 1] != slot_id:
        raise CloudProviderError("Protected workload does not bind the derived artifact slot")
    document = {
        "schema_version": PREFLIGHT_SCHEMA,
        "experiment_id": experiment_id, "run_id": run_id,
        "tracking_root_id": tracking_root_id, "occurred_at": occurred_at,
        "status": "PASS", "source": dict(source),
        "runtime_lock": {"uri": runtime_lock_uri, "sha256": runtime_lock_sha256},
        "workload_digest": workload.workload_sha256,
        "model": {"repository": MODEL, "revision": MODEL_REVISION},
        "dataset": {
            "path": DATASET, "sha256": DATASET_SHA256, "git_blob": DATASET_GIT_BLOB,
            "bytes": dataset_bytes, "row_count": 1, "row_sha256": dataset_row_sha256,
        },
        "image": dict(runtime_lock["image"]), "hardware": hardware_quote.as_dict(),
        "artifact_slot_input": slot_input, "artifact_slot_id": slot_id,
        "volumes": [
            {"bucket_id": source_bucket_id, "prefix": source_prefix,
             "mount_path": "/workspace/synaptic-bootstrap-input", "read_only": True},
            {"bucket_id": artifact_bucket_id,
             "prefix": derive_hf_training_artifact_prefix(artifact_base_prefix, slot_id),
             "mount_path": "/workspace/artifacts", "read_only": False},
        ],
        "command": {
            "remote_argv_sha256": workload.remote_argv_sha256,
            "provider_command_sha256": workload.provider_command_sha256,
        },
        "launcher_auth": {"mode": "explicit_file", "expected_namespace": expected_namespace},
        "job_secrets": [],
    }
    return seal_training_document(document)


def build_approval_document(
    *, preflight: Mapping[str, object], preflight_uri: str,
    user_authorization_reference: str, issued_at: str, expires_at: str,
) -> dict[str, object]:
    hardware = preflight["hardware"]
    source_volume, artifact_volume = preflight["volumes"]
    source = preflight["source"]
    document = {
        "schema_version": APPROVAL_SCHEMA, "kind": "hf.training-smoke",
        "experiment_id": preflight["experiment_id"], "run_id": preflight["run_id"],
        "tracking_root_id": preflight["tracking_root_id"],
        "preflight": {"uri": preflight_uri, "sha256": document_sha256(preflight)},
        "user_authorization_reference": user_authorization_reference,
        "issued_at": issued_at, "expires_at": expires_at, "hardware": HARDWARE_FLAVOR,
        "hardware_quote": {
            "preflight_sha256": document_sha256(preflight),
            "unit_cost_micro_usd": hardware["unit_cost_micro_usd"],
            "hourly_cost_micro_usd": hardware["hourly_cost_micro_usd"],
            "timeout_cost_micro_usd": hardware["timeout_cost_micro_usd"],
            "fetched_at": hardware["fetched_at"],
        },
        "provider_timeout_seconds": PROVIDER_TIMEOUT_SECONDS,
        "cancel_after_seconds": CANCEL_AFTER_SECONDS,
        "observe_until_seconds": OBSERVE_UNTIL_SECONDS,
        "maximum_submissions": 1, "maximum_retries": 0,
        "publication": False, "ssh": False, "ports": False, "wandb": False,
        "launcher_auth": dict(preflight["launcher_auth"]), "job_secrets": [],
        "bindings": {
            "source_lock_sha256": source["source_lock"]["sha256"],
            "workload_digest": preflight["workload_digest"],
            "runtime_lock_sha256": preflight["runtime_lock"]["sha256"],
            "model_revision": preflight["model"]["revision"],
            "dataset_sha256": preflight["dataset"]["sha256"],
            "image_child_digest": preflight["image"]["child_digest"],
            "remote_argv_sha256": preflight["command"]["remote_argv_sha256"],
            "provider_command_sha256": preflight["command"]["provider_command_sha256"],
            "source_bucket_id": source_volume["bucket_id"], "source_prefix": source_volume["prefix"],
            "artifact_bucket_id": artifact_volume["bucket_id"],
            "artifact_base_prefix": preflight["artifact_slot_input"]["artifact_base_prefix"],
            "artifact_slot_id": preflight["artifact_slot_id"],
            "artifact_prefix": artifact_volume["prefix"],
        },
    }
    return seal_training_document(document)


def build_submission_event(
    *, approval: Mapping[str, object], approval_uri: str, state: str, sequence: int,
    occurred_at: str, previous_event: Mapping[str, object] | None = None,
    previous_event_uri: str | None = None, provider_job: ProviderJob | None = None,
    reason_code: str | None = None,
) -> dict[str, object]:
    previous_ref = None
    if previous_event is not None:
        if previous_event_uri is None:
            raise CloudProviderError("Submission predecessor URI is unavailable")
        previous_ref = {"uri": previous_event_uri, "sha256": document_sha256(previous_event)}
    document = {
        "schema_version": SUBMISSION_SCHEMA, "authorization_id": approval["authorization_id"],
        "approval": {"uri": approval_uri, "sha256": document_sha256(approval)},
        "experiment_id": approval["experiment_id"], "run_id": approval["run_id"],
        "tracking_root_id": approval["tracking_root_id"], "state": state,
        "sequence": sequence, "occurred_at": occurred_at, "previous_event": previous_ref,
        "provider_job": provider_job.identity() if provider_job is not None else None,
        "reason_code": reason_code,
        "provider_effect_possible": state != "NOT_SUBMITTED",
    }
    return seal_training_document(document)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def download_exact_artifacts(
    *, token: str, bucket_id: str, prefix: str,
    inventory: Sequence[Mapping[str, object]], destination: os.PathLike[str] | str,
    python_executable: os.PathLike[str] | str | None = None,
) -> None:
    """Invoke the owned bounded downloader with the credential only on stdin."""

    if platform.python_implementation() != "CPython" or sys.version_info[:3] != (3, 12, 7):
        raise CloudProviderError("Protected HF launcher requires exact CPython 3.12.7")
    if not isinstance(token, str) or not token or len(token) > 4096 or any(c in token for c in "\r\n"):
        raise CloudProviderError("HF provider credential is invalid")
    target = os.path.realpath(os.fspath(destination))
    if not os.path.isdir(target) or os.listdir(target):
        raise CloudProviderError("Artifact download destination is not an existing empty directory")
    from tuner.cloud.hf_training_smoke_artifacts import EXPECTED_PATHS, MAX_FILE_BYTES, MAX_TOTAL_BYTES
    normalized = list(inventory)
    if (
        len(normalized) != len(EXPECTED_PATHS)
        or [item.get("path") if isinstance(item, Mapping) else None for item in normalized]
        != sorted(EXPECTED_PATHS)
        or any(
            set(item) != {"path", "bytes", "provider_xet_hash"}
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
            or (
                item["provider_xet_hash"] is not None
                and (
                    not isinstance(item["provider_xet_hash"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", item["provider_xet_hash"]) is None
                )
            )
            for item in normalized
            if isinstance(item, Mapping)
        )
        or any(not isinstance(item, Mapping) for item in normalized)
    ):
        raise CloudProviderError("Artifact provider inventory is not exact")
    sizes = [int(item["bytes"]) for item in normalized]
    if any(size > MAX_FILE_BYTES for size in sizes) or sum(sizes) > MAX_TOTAL_BYTES:
        raise CloudProviderError("Artifact provider inventory exceeds verifier bounds")
    try:
        inventory_raw = json.dumps(
            normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CloudProviderError("Artifact provider inventory is invalid") from exc
    if not inventory_raw or len(inventory_raw) > 64 * 1024:
        raise CloudProviderError("Artifact provider inventory exceeds its bound")
    executable = os.path.realpath(os.fspath(python_executable or sys.executable))
    child = os.path.realpath(os.path.join(os.path.dirname(__file__), "hf_training_smoke_download_child.py"))
    command = (
        executable, "-I", "-B", child, "--bucket-id", bucket_id, "--prefix", prefix,
        "--inventory", "INVENTORY", "--destination", target,
    )
    secret = bytearray(token.encode("ascii"))
    try:
        with tempfile.TemporaryDirectory(prefix="synaptic-hf-training-download-") as temporary:
            inventory_path = os.path.join(temporary, "inventory.json")
            with open(inventory_path, "xb") as handle:
                handle.write(inventory_raw)
            actual = tuple(inventory_path if value == "INVENTORY" else value for value in command)
            returncode, stdout, stderr = _run_download_child(actual, secret)
            if stderr or returncode != 0 or stdout != _DOWNLOAD_PASS:
                raise CloudProviderError("Protected artifact download was rejected")
    finally:
        for index in range(len(secret)):
            secret[index] = 0


def _require_clean_environment(environment: Mapping[str, str]) -> None:
    present = sorted(
        str(name) for name, value in environment.items()
        if str(name).upper() in _FORBIDDEN_ENV and isinstance(value, str) and value.strip()
    )
    if present:
        raise CloudProviderError("Protected HF provider environment is not isolated")


def _explicit_parameter(callable_value: object, name: str) -> None:
    if not callable(callable_value):
        raise CloudProviderError("Installed HF provider client is incomplete")
    try:
        parameter = inspect.signature(callable_value).parameters.get(name)
    except (TypeError, ValueError) as exc:
        raise CloudProviderError("Installed HF provider client cannot be authenticated") from exc
    if parameter is None or parameter.kind is inspect.Parameter.VAR_KEYWORD:
        raise CloudProviderError("Installed HF provider client has an unreviewed signature")


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _timestamp(value: object) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise CloudProviderError("HF provider timestamp is not timezone-aware")
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if not isinstance(value, str):
        raise CloudProviderError("HF provider timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CloudProviderError("HF provider timestamp is invalid") from exc
    rendered = parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if rendered != value:
        raise CloudProviderError("HF provider timestamp is not canonical")
    return rendered


class HFTrainingSmokeProvider:
    """Pinned HF 1.27 adapter; every authenticated call receives ``token`` explicitly."""

    def __init__(
        self, *, token: str, hub: ModuleType, api: object, clients: list[object],
        request_error_type: type[Exception],
    ) -> None:
        if not isinstance(token, str) or not token or len(token) > 4096 or any(c in token for c in "\r\n"):
            raise CloudProviderError("HF provider credential is invalid")
        self._token = token
        self._hub = hub
        self._api = api
        self._clients = clients
        self._request_error_type = request_error_type
        for method, parameters in (
            (getattr(api, "whoami", None), ("token",)),
            (getattr(api, "list_jobs_hardware", None), ("token",)),
            (getattr(api, "run_job", None), ("token", "secrets", "volumes", "namespace")),
            (getattr(api, "inspect_job", None), ("token", "namespace")),
            (getattr(api, "cancel_job", None), ("token", "namespace")),
            (getattr(api, "list_jobs", None), ("token", "namespace", "labels", "timeout")),
            (getattr(api, "list_bucket_tree", None), ("token", "bucket_id", "prefix", "recursive")),
        ):
            for parameter in parameters:
                _explicit_parameter(method, parameter)

    @property
    def sdk(self) -> object:
        return self._hub

    def close(self) -> None:
        failed = False
        for client in self._clients:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    failed = True
        self._token = ""
        if failed:
            raise CloudProviderError("HF provider client cleanup failed")

    def authenticate_namespace(self, expected_namespace: str) -> str:
        if _PROVIDER_ID.fullmatch(expected_namespace) is None:
            raise CloudProviderError("Expected HF namespace is invalid")
        _explicit_parameter(self._api.whoami, "token")
        identity = self._api.whoami(token=self._token)
        namespace = _field(identity, "name")
        if namespace != expected_namespace:
            raise CloudProviderError("HF credential owner does not match approval namespace")
        return expected_namespace

    def quote_a10g(self, *, now: Callable[[], str] = _utc_now) -> HardwareQuote:
        method = self._api.list_jobs_hardware
        _explicit_parameter(method, "token")
        values = method(token=self._token)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise CloudProviderError("HF hardware quote document is invalid")
        matches = [item for item in values if _field(item, "name") == HARDWARE_FLAVOR]
        if len(matches) != 1:
            raise CloudProviderError("HF hardware quote is not unique")
        item = matches[0]
        unit = _field(item, "unit_cost_micro_usd")
        label = _field(item, "unit_label")
        if type(unit) is not int or not 1 <= unit <= HARDWARE_QUOTE_MAX_UNIT_COST_MICRO_USD:
            raise CloudProviderError("HF hardware quote cost is invalid")
        if label != HARDWARE_QUOTE_UNIT_LABEL:
            raise CloudProviderError("HF hardware quote unit is invalid")
        hourly = unit * 60
        timeout = unit * 30
        if hourly > HARDWARE_MAX_HOURLY_COST_MICRO_USD or timeout > HARDWARE_MAX_TIMEOUT_COST_MICRO_USD:
            raise CloudProviderError("HF hardware quote exceeds the approved cost envelope")
        return HardwareQuote(HF_ENDPOINT, HARDWARE_FLAVOR, unit, label, hourly, timeout, now())

    def submit(
        self, *, image: str, command: tuple[str, ...], name: str,
        labels: Mapping[str, str], volumes: tuple[object, object], namespace: str,
    ) -> ProviderJob:
        authorization = labels.get("synaptic-auth") if isinstance(labels, Mapping) else None
        expected_labels = {
            "synaptic-kind": "hf-training-smoke",
            "synaptic-auth": authorization,
        }
        if (
            dict(labels) != expected_labels
            or not isinstance(authorization, str)
            or re.fullmatch(r"[0-9a-f]{48}", authorization) is None
            or name != f"synaptic-hf-training-smoke-{authorization[:12]}"
        ):
            raise CloudProviderError("HF provider job identity is not exact")
        method = self._api.run_job
        _explicit_parameter(method, "token")
        try:
            job = method(
                image=image, command=list(command), env=dict(FIXED_NONSECRET_ENV), secrets={},
                flavor=HARDWARE_FLAVOR, timeout=PROVIDER_TIMEOUT_SECONDS, name=name,
                labels=dict(labels), volumes=list(volumes), expose=[], ssh=False,
                namespace=namespace, token=self._token,
            )
        except self._request_error_type:
            raise _ProviderSubmissionFailure("PROVIDER_TRANSPORT_ERROR") from None
        submitted = self._normalize_job(job, namespace=namespace)
        expectation = ProviderJobExpectation(
            image=image, command=command, name=name,
            labels=tuple(sorted((str(key), str(value)) for key, value in labels.items())),
            volumes=volumes, namespace=namespace,
        )
        inspected = self.inspect(submitted.job_id, namespace=namespace, expected=expectation)
        if inspected.identity() != submitted.identity():
            raise CloudProviderError("HF provider submission identity changed during inspection")
        return inspected

    def inspect(
        self, job_id: str, *, namespace: str,
        expected: ProviderJobExpectation | None = None,
    ) -> ProviderJob:
        if _JOB_ID.fullmatch(job_id) is None:
            raise CloudProviderError("HF provider job identity is invalid")
        method = self._api.inspect_job
        _explicit_parameter(method, "token")
        raw = method(job_id=job_id, namespace=namespace, token=self._token)
        if expected is not None:
            self._authenticate_job_spec(raw, expected)
        normalized = self._normalize_job(raw, namespace=namespace)
        if normalized.job_id != job_id:
            raise CloudProviderError("HF provider returned another job identity")
        return normalized

    def cancel(self, job_id: str, *, namespace: str) -> None:
        method = self._api.cancel_job
        _explicit_parameter(method, "token")
        method(job_id=job_id, namespace=namespace, token=self._token)

    def list_jobs(self, *, namespace: str, labels: Mapping[str, str]) -> tuple[object, ...]:
        if (
            _PROVIDER_ID.fullmatch(namespace) is None
            or not isinstance(labels, Mapping)
            or not labels
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or _PROVIDER_LABEL.fullmatch(key) is None
                or _PROVIDER_LABEL.fullmatch(value) is None
                for key, value in labels.items()
            )
        ):
            raise CloudProviderError("HF provider recovery query is invalid")
        method = self._api.list_jobs
        result = method(labels=dict(labels), timeout=30, namespace=namespace, token=self._token)
        if isinstance(result, (str, bytes, Mapping)):
            raise CloudProviderError("HF provider job listing is invalid")
        try:
            iterator = iter(result)
            bounded: list[object] = []
            for item in iterator:
                bounded.append(item)
                if len(bounded) == 2:
                    break
        except Exception as exc:
            raise CloudProviderError("HF provider job listing is invalid") from exc
        return tuple(bounded)

    def list_bucket_tree(self, *, bucket_id: str, prefix: str) -> tuple[object, ...]:
        method = self._api.list_bucket_tree
        result = method(
            bucket_id=bucket_id, prefix=prefix, recursive=True, token=self._token,
        )
        if isinstance(result, (str, bytes, Mapping)):
            raise CloudProviderError("HF Bucket listing is invalid")
        try:
            bounded: list[object] = []
            for value in result:
                bounded.append(value)
                if len(bounded) == 65:
                    break
        except Exception as exc:
            raise CloudProviderError("HF Bucket listing is invalid") from exc
        if len(bounded) > 64:
            raise CloudProviderError("HF Bucket listing exceeds its bound")
        return tuple(bounded)

    @staticmethod
    def _normalize_job(job: object, *, namespace: str) -> ProviderJob:
        job_id = _field(job, "id") or _field(job, "job_id")
        owner = _field(_field(job, "owner"), "name")
        created = _field(job, "created_at")
        status_object = _field(job, "status")
        status = _field(status_object, "stage") if status_object is not None else None
        status = getattr(status, "value", status)
        if owner != namespace or not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
            raise CloudProviderError("HF provider returned an invalid job identity")
        if status is not None and not isinstance(status, str):
            raise CloudProviderError("HF provider returned an invalid job status")
        return ProviderJob(owner, job_id, _timestamp(created), status)

    @staticmethod
    def _authenticate_job_spec(job: object, expected: ProviderJobExpectation) -> None:
        expected_volumes: list[object] = []
        for volume in expected.volumes:
            to_dict = getattr(volume, "to_dict", None)
            if not callable(to_dict):
                raise CloudProviderError("HF provider expected volume is not inspectable")
            expected_volumes.append(dict(to_dict()))
        actual_volumes = _field(job, "volumes")
        if isinstance(actual_volumes, Sequence) and not isinstance(actual_volumes, (str, bytes)):
            rendered = []
            for volume in actual_volumes:
                to_dict = getattr(volume, "to_dict", None)
                rendered.append(dict(to_dict()) if callable(to_dict) else dict(volume) if isinstance(volume, Mapping) else None)
        else:
            rendered = None
        returned_labels = dict(expected.labels)
        returned_labels["name"] = expected.name
        exact = {
            "docker_image": expected.image,
            "space_id": None,
            "command": list(expected.command),
            "arguments": [],
            "environment": dict(FIXED_NONSECRET_ENV),
            "flavor": HARDWARE_FLAVOR,
            "labels": returned_labels,
            "volumes": expected_volumes,
            "endpoint": HF_ENDPOINT,
        }
        actual = {key: _field(job, key) for key in exact}
        actual["volumes"] = rendered
        status = _field(job, "status")
        returned_secrets = _field(job, "secrets")
        secrets_are_empty = (
            returned_secrets is None
            or isinstance(returned_secrets, Mapping) and not returned_secrets
            or isinstance(returned_secrets, Sequence)
            and not isinstance(returned_secrets, (str, bytes))
            and not returned_secrets
        )
        if (
            actual != exact
            or _field(_field(job, "owner"), "name") != expected.namespace
            or not secrets_are_empty
            or _field(status, "expose_urls") not in (None, [])
            or _field(status, "ssh_url") is not None
        ):
            raise CloudProviderError("HF provider job specification is not approval-authenticated")


def create_provider(
    token: str, *, environment: Mapping[str, str] | None = None,
    huggingface_hub: ModuleType | None = None, httpx: ModuleType | None = None,
) -> HFTrainingSmokeProvider:
    _require_clean_environment(os.environ if environment is None else environment)
    if huggingface_hub is None:
        import huggingface_hub as hub_module
        huggingface_hub = hub_module
    if httpx is None:
        import httpx as httpx_module
        httpx = httpx_module
    probe_provider_contract(huggingface_hub)
    clients: list[object] = []

    request_error_type = getattr(httpx, "RequestError", None)
    if (
        not isinstance(request_error_type, type)
        or not issubclass(request_error_type, Exception)
    ):
        raise CloudProviderError("Installed HTTP provider client is incomplete")

    def factory():
        client = httpx.Client(
            base_url=HF_ENDPOINT,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
            follow_redirects=True, max_redirects=MAX_REDIRECTS, trust_env=False,
        )
        clients.append(client)
        return client

    try:
        huggingface_hub.set_client_factory(factory)
        api = huggingface_hub.HfApi(endpoint=HF_ENDPOINT, token=False)
        return HFTrainingSmokeProvider(
            token=token, hub=huggingface_hub, api=api, clients=clients,
            request_error_type=request_error_type,
        )
    except BaseException:
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        raise


def probe_provider_contract(huggingface_hub: ModuleType) -> None:
    """Authenticate the installed provider surface before any durable effect claim."""

    if getattr(huggingface_hub, "__version__", None) != HF_HUB_VERSION:
        raise CloudProviderError("Installed HF provider client has the wrong version")
    api_type = getattr(huggingface_hub, "HfApi", None)
    _explicit_parameter(api_type, "endpoint")
    _explicit_parameter(api_type, "token")
    for name, parameters in (
        ("whoami", ("token",)),
        ("list_jobs_hardware", ("token",)),
        ("run_job", ("token", "secrets", "volumes", "namespace")),
        ("inspect_job", ("token", "namespace")),
        ("cancel_job", ("token", "namespace")),
        ("list_jobs", ("token", "namespace", "labels", "timeout")),
        ("list_bucket_tree", ("token", "bucket_id", "prefix", "recursive")),
    ):
        method = getattr(api_type, name, None)
        for parameter in parameters:
            _explicit_parameter(method, parameter)


def _cli_state(args: object, context: object):
    """Load the already-provisioned source state without reading a credential."""

    from shared.experiment_tracking.experiment import load_experiment
    from shared.experiment_tracking.service import TrackingService
    from tuner.cloud.hf_provisioning import consume_hf_source_transport, load_canonical_json
    from tuner.handlers.hf_source_handler import _external_runtime_layout, _require_external_base_dir
    from tuner.handlers.stages._util import hf_source_preparation_from_consumable

    experiment_id = str(getattr(args, "experiment_id", "") or "").strip()
    if not experiment_id:
        raise CloudProviderError("Protected HF training experiment identity is required")
    base_dir = _require_external_base_dir(getattr(args, "base_dir", None), context=context)
    tracking = TrackingService(base_dir=base_dir)
    experiment = load_experiment(experiment_id, tracking.base_dir)
    tracking.require_consumable_hf_transport(experiment)
    required = (
        experiment.source_lock_uri, experiment.source_lock_sha256,
        experiment.source_transport_uri, experiment.source_transport_sha256,
        experiment.provisioning_evidence_uri, experiment.provisioning_evidence_sha256,
    )
    if any(not isinstance(value, str) or not value for value in required):
        raise CloudProviderError("Protected HF source tracking bindings are incomplete")
    descriptor_path = tracking.resolve_uri(experiment.source_transport_uri)
    evidence = load_canonical_json(
        tracking.resolve_uri(experiment.provisioning_evidence_uri), maximum_bytes=64 * 1024,
    )
    consumed = consume_hf_source_transport(
        context, transport_root=descriptor_path.parent,
        descriptor_uri=experiment.source_transport_uri,
        source_lock_uri=experiment.source_lock_uri, evidence=evidence,
    )
    preparation = hf_source_preparation_from_consumable(
        consumed, context=context,
        runtime_layout=_external_runtime_layout(context, base_dir),
        provisioning_evidence_uri=experiment.provisioning_evidence_uri,
    )
    if (
        preparation.descriptor_sha256 != experiment.source_transport_sha256
        or preparation.provisioning_evidence_sha256 != experiment.provisioning_evidence_sha256
    ):
        raise CloudProviderError("Protected HF source tracking identity changed")
    return tracking, experiment, preparation


def _approve_action(args: object, context: object) -> dict[str, object]:
    from tuner.cloud.hf_provisioning import load_canonical_json

    tracking, experiment, _preparation = _cli_state(args, context)
    if experiment.hf_training_preflight_state != "PASS":
        raise CloudProviderError("Protected HF training preflight is unavailable")
    preflight_uri = str(experiment.hf_training_preflight_uri or "")
    preflight = load_canonical_json(tracking.resolve_uri(preflight_uri), maximum_bytes=256 * 1024)
    values = {}
    for name in ("authorization_reference", "issued_at", "expires_at"):
        value = str(getattr(args, name, "") or "").strip()
        if not value:
            raise CloudProviderError("Protected HF training approval argument is unavailable")
        values[name] = value
    approval = build_approval_document(
        preflight=preflight, preflight_uri=preflight_uri,
        user_authorization_reference=values["authorization_reference"],
        issued_at=values["issued_at"], expires_at=values["expires_at"],
    )
    tracking.record_hf_training_approval(experiment, approval)
    return {
        "status": "APPROVED", "authorization_id": approval["authorization_id"],
        "submitted": False,
    }


def _required_cli(args: object, name: str) -> str:
    value = str(getattr(args, name, "") or "").strip()
    if not value:
        raise CloudProviderError("Protected HF training argument is unavailable")
    return value


def _preflight_action(args: object, context: object) -> dict[str, object]:
    import hashlib

    from shared.experiment_tracking.root_identity import ensure_tracking_root_identity
    from tuner.cloud.hf_training_smoke_workload import (
        DATASET, DATASET_SHA256, RUNTIME_LOCK_PATH,
        build_workload, validate_runtime_lock,
    )
    from tuner.cloud.hf_volume_transport import (
        HFArtifactVolumeSpec, prove_writable_artifact_volume,
        validate_disjoint_volume_prefixes,
    )
    from tuner.handlers._hf_secret_file import preflight_hf_secret_file, read_claimed_hf_token

    tracking, experiment, preparation = _cli_state(args, context)
    preparation.require_consumable()
    source_spec = preparation.volume_spec
    assert source_spec is not None
    source_bucket = _required_cli(args, "source_bucket_id")
    source_prefix = _required_cli(args, "source_prefix")
    artifact_bucket = _required_cli(args, "artifact_bucket_id")
    artifact_base = _required_cli(args, "artifact_prefix")
    namespace = _required_cli(args, "expected_namespace")
    if source_spec.source != source_bucket or source_spec.path != source_prefix:
        raise CloudProviderError("Protected HF source Volume does not match provisioned identity")
    repository = Path(getattr(context, "project_root", "")).resolve(strict=True)
    runtime_lock, _runtime_raw = validate_runtime_lock(repository / RUNTIME_LOCK_PATH)
    runtime_ref = tracking.snapshot_hf_training_runtime_lock(experiment, runtime_lock)
    root_id = str(ensure_tracking_root_identity(tracking.base_dir)["root_id"])
    preliminary = build_workload(
        repository, source_lock_sha256=preparation.source_lock_sha256,
        artifact_slot="0" * 64,
    )
    slot_input = {
        "schema_version": ARTIFACT_SLOT_INPUT_SCHEMA,
        "experiment_id": experiment.experiment_id,
        "run_id": preparation.source_lock.run_id,
        "tracking_root_id": root_id,
        "source_lock_sha256": preparation.source_lock_sha256,
        "workload_digest": preliminary.workload_sha256,
        "runtime_lock_sha256": runtime_ref["sha256"],
        "artifact_bucket_id": artifact_bucket,
        "artifact_base_prefix": artifact_base,
    }
    slot_id = derive_hf_training_artifact_slot(slot_input)
    workload = build_workload(
        repository, source_lock_sha256=preparation.source_lock_sha256,
        artifact_slot=slot_id, source_volume_spec=source_spec,
        expected_project_root=preparation.physical_project_root,
        expected_engine_root=preparation.physical_engine_root,
        expected_project_commit=preparation.source_lock.project_source.commit,
        expected_engine_commit=preparation.source_lock.engine_source.commit,
        expected_mode=preparation.source_lock.mode,
    )
    artifact_prefix = derive_hf_training_artifact_prefix(artifact_base, slot_id)
    artifact_spec = HFArtifactVolumeSpec(source=artifact_bucket, path=artifact_prefix)
    validate_disjoint_volume_prefixes(source_spec, artifact_spec)
    secret_file = preflight_hf_secret_file(getattr(args, "env_file", None), context=context)
    token = read_claimed_hf_token(secret_file)
    provider = create_provider(token)
    try:
        provider.authenticate_namespace(namespace)
        quote = provider.quote_a10g()
        if provider.list_bucket_tree(bucket_id=artifact_bucket, prefix=artifact_prefix):
            raise CloudProviderError("Protected artifact slot is not empty")
        preparation.prove_volume(provider.sdk)
        prove_writable_artifact_volume(provider.sdk, artifact_spec)
    finally:
        _close_provider_preserving_pending(provider)
        token = ""
    descriptor = preparation.consumable_transport.prepared.descriptor
    source = {
        "descriptor": {"uri": preparation.descriptor_uri, "sha256": preparation.descriptor_sha256},
        "source_lock": {"uri": preparation.source_lock_uri, "sha256": preparation.source_lock_sha256},
        "provisioning_evidence": {
            "uri": preparation.provisioning_evidence_uri,
            "sha256": preparation.provisioning_evidence_sha256,
        },
        "bundle_sha256": descriptor["bundle"]["content_sha256"],
        "capsule_manifest_sha256": descriptor["capsule"]["manifest"]["sha256"],
        "checkout_policy_sha256": descriptor["checkout_policy"]["sha256"],
        "project_commit": preparation.source_lock.project_source.commit,
        "engine_commit": preparation.source_lock.engine_source.commit,
    }
    dataset_raw = (repository / DATASET).read_bytes()
    if (
        not dataset_raw or len(dataset_raw) > 10 * 1024 * 1024
        or hashlib.sha256(dataset_raw).hexdigest() != DATASET_SHA256
    ):
        raise CloudProviderError("Protected dataset identity changed")
    rows = dataset_raw.splitlines()
    if len(rows) != 1 or not rows[0]:
        raise CloudProviderError("Protected dataset is not exactly one row")
    preflight = build_preflight_document(
        experiment_id=experiment.experiment_id, run_id=preparation.source_lock.run_id,
        tracking_root_id=root_id, occurred_at=_utc_now(), source=source,
        runtime_lock=runtime_lock, runtime_lock_uri=runtime_ref["uri"],
        runtime_lock_sha256=runtime_ref["sha256"], workload=workload,
        dataset_bytes=len(dataset_raw), dataset_row_sha256=hashlib.sha256(rows[0]).hexdigest(),
        source_bucket_id=source_bucket, source_prefix=source_prefix,
        artifact_bucket_id=artifact_bucket, artifact_base_prefix=artifact_base,
        expected_namespace=namespace, hardware_quote=quote,
    )
    tracking.record_hf_training_preflight(experiment, preflight)
    return {
        "status": "PASS", "preflight_id": preflight["preflight_id"],
        "artifact_slot_id": slot_id, "submitted": False,
    }


def _load_training_document(tracking: object, uri: str) -> dict[str, object]:
    from tuner.cloud.hf_provisioning import load_canonical_json
    return load_canonical_json(tracking.resolve_uri(uri), maximum_bytes=256 * 1024)


def _execute_action(args: object, context: object) -> dict[str, object]:
    from tuner.cloud.hf_training_smoke_workload import build_workload
    from tuner.cloud.hf_volume_transport import HFArtifactVolumeSpec, prove_writable_artifact_volume
    from tuner.handlers._hf_secret_file import preflight_hf_secret_file, read_claimed_hf_token

    tracking, experiment, preparation = _cli_state(args, context)
    if experiment.hf_training_submission_state != "APPROVED":
        raise CloudProviderError("Protected HF training approval is unavailable")
    approval_uri = str(experiment.hf_training_approval_uri or "")
    approval = _load_training_document(tracking, approval_uri)
    now = _utc_now()
    issued = datetime.fromisoformat(str(approval["issued_at"]).replace("Z", "+00:00"))
    expires = datetime.fromisoformat(str(approval["expires_at"]).replace("Z", "+00:00"))
    quote_at = datetime.fromisoformat(str(approval["hardware_quote"]["fetched_at"]).replace("Z", "+00:00"))
    current = datetime.fromisoformat(now.replace("Z", "+00:00"))
    if current < issued or current >= expires or (current - quote_at).total_seconds() > 900:
        raise CloudProviderError("Protected HF training approval or quote is stale")
    import huggingface_hub

    probe_provider_contract(huggingface_hub)
    preparation.require_consumable()
    source_spec = preparation.volume_spec
    assert source_spec is not None
    repository = Path(getattr(context, "project_root", "")).resolve(strict=True)
    bindings = approval["bindings"]
    workload = build_workload(
        repository, source_lock_sha256=str(bindings["source_lock_sha256"]),
        artifact_slot=str(bindings["artifact_slot_id"]), source_volume_spec=source_spec,
        expected_project_root=preparation.physical_project_root,
        expected_engine_root=preparation.physical_engine_root,
        expected_project_commit=preparation.source_lock.project_source.commit,
        expected_engine_commit=preparation.source_lock.engine_source.commit,
        expected_mode=preparation.source_lock.mode,
    )
    if (
        workload.workload_sha256 != bindings["workload_digest"]
        or workload.remote_argv_sha256 != bindings["remote_argv_sha256"]
        or workload.provider_command_sha256 != bindings["provider_command_sha256"]
        or workload.image.split("@", 1)[1] != bindings["image_child_digest"]
    ):
        raise CloudProviderError("Protected HF training workload changed after approval")
    source_volume = preparation.prove_volume(huggingface_hub).provider_volume
    artifact_volume = prove_writable_artifact_volume(
        huggingface_hub,
        HFArtifactVolumeSpec(source=str(bindings["artifact_bucket_id"]), path=str(bindings["artifact_prefix"])),
    ).provider_volume
    secret_file = preflight_hf_secret_file(getattr(args, "env_file", None), context=context)
    claim_document = build_submission_event(
        approval=approval, approval_uri=approval_uri, state="SUBMITTING", sequence=1,
        occurred_at=now,
    )
    claim = tracking.claim_hf_training_submission(experiment, claim_document)
    if not claim.provider_attempt_authorized:
        return {"status": claim.state, "submitted": False}
    name, labels = provider_job_identity(approval)

    def terminal(state: str, *, job: ProviderJob | None, reason: str | None) -> None:
        document = build_submission_event(
            approval=approval, approval_uri=approval_uri, state=state, sequence=2,
            occurred_at=_utc_now(), previous_event=claim.document,
            previous_event_uri=claim.uri, provider_job=job, reason_code=reason,
        )
        tracking.record_hf_training_submission_terminal(experiment, document)

    provider = None
    token = ""
    provider_call_started = False
    try:
        token = read_claimed_hf_token(secret_file)
        provider = create_provider(token)
        namespace = str(approval["launcher_auth"]["expected_namespace"])
        provider.authenticate_namespace(namespace)
        if provider.list_bucket_tree(
            bucket_id=str(bindings["artifact_bucket_id"]),
            prefix=str(bindings["artifact_prefix"]),
        ):
            terminal("NOT_SUBMITTED", job=None, reason="PREFIX_NOT_EMPTY")
            return {
                "status": "NOT_SUBMITTED", "submitted": False,
                "retry_allowed": False, "reason_code": "PREFIX_NOT_EMPTY",
            }
        provider_call_started = True
        job = provider.submit(
            image=workload.image, command=workload.provider_command, name=name,
            labels=labels, volumes=(source_volume, artifact_volume), namespace=namespace,
        )
        terminal("SUBMITTED", job=job, reason=None)
        return {"status": "SUBMITTED", **job.identity(), "retry_allowed": False}
    except BaseException as exc:
        if isinstance(exc, Exception):
            if provider_call_started:
                state = "AMBIGUOUS"
                reason = _provider_failure_reason(exc)
            else:
                state = "NOT_SUBMITTED"
                reason = "LOCAL_PRECALL_FAILURE"
            try:
                terminal(state, job=None, reason=reason)
            except BaseException:
                raise CloudProviderError("Protected HF training terminal state could not be recorded") from None
            raise CloudProviderError("Protected HF training submission was rejected") from None
        try:
            terminal("AMBIGUOUS", job=None, reason="INTERRUPTED_AFTER_CLAIM")
        except Exception:
            pass
        raise
    finally:
        if provider is not None:
            _close_provider_preserving_pending(provider)
        token = ""


def _recover_action(args: object, context: object) -> dict[str, object]:
    from tuner.cloud.hf_training_smoke_workload import build_workload
    from tuner.cloud.hf_volume_transport import HFArtifactVolumeSpec, prove_writable_artifact_volume
    from tuner.handlers._hf_secret_file import preflight_hf_secret_file, read_claimed_hf_token

    tracking, experiment, preparation = _cli_state(args, context)
    if experiment.hf_training_submission_state != "AMBIGUOUS":
        raise CloudProviderError("Protected HF training recovery requires AMBIGUOUS state")
    approval_uri = str(experiment.hf_training_approval_uri or "")
    approval = _load_training_document(tracking, approval_uri)
    previous_uri = str(experiment.hf_training_submission_event_uri or "")
    previous = _load_training_document(tracking, previous_uri)
    bindings = approval["bindings"]
    preparation.require_consumable()
    source_spec = preparation.volume_spec
    assert source_spec is not None
    repository = Path(getattr(context, "project_root", "")).resolve(strict=True)
    workload = build_workload(
        repository, source_lock_sha256=str(bindings["source_lock_sha256"]),
        artifact_slot=str(bindings["artifact_slot_id"]), source_volume_spec=source_spec,
        expected_project_root=preparation.physical_project_root,
        expected_engine_root=preparation.physical_engine_root,
        expected_project_commit=preparation.source_lock.project_source.commit,
        expected_engine_commit=preparation.source_lock.engine_source.commit,
        expected_mode=preparation.source_lock.mode,
    )
    if (
        workload.workload_sha256 != bindings["workload_digest"]
        or workload.remote_argv_sha256 != bindings["remote_argv_sha256"]
        or workload.provider_command_sha256 != bindings["provider_command_sha256"]
        or workload.image.split("@", 1)[1] != bindings["image_child_digest"]
    ):
        raise CloudProviderError("Protected HF training workload changed after approval")
    import huggingface_hub

    source_volume = preparation.prove_volume(huggingface_hub).provider_volume
    artifact_volume = prove_writable_artifact_volume(
        huggingface_hub,
        HFArtifactVolumeSpec(source=str(bindings["artifact_bucket_id"]), path=str(bindings["artifact_prefix"])),
    ).provider_volume
    name, labels = provider_job_identity(approval)
    namespace = str(approval["launcher_auth"]["expected_namespace"])
    expected = ProviderJobExpectation(
        workload.image, workload.provider_command, name, tuple(sorted(labels.items())),
        (source_volume, artifact_volume), namespace,
    )
    secret_file = preflight_hf_secret_file(getattr(args, "env_file", None), context=context)
    token = read_claimed_hf_token(secret_file)
    provider = create_provider(token)
    try:
        provider.authenticate_namespace(namespace)
        candidates = provider.list_jobs(namespace=namespace, labels=labels)
        if len(candidates) != 1:
            return {"status": "AMBIGUOUS", "recovered": False, "retry_allowed": False}
        raw_id = _field(candidates[0], "id") or _field(candidates[0], "job_id")
        if not isinstance(raw_id, str):
            return {"status": "AMBIGUOUS", "recovered": False, "retry_allowed": False}
        job = provider.inspect(raw_id, namespace=namespace, expected=expected)
    except Exception:
        return {"status": "AMBIGUOUS", "recovered": False, "retry_allowed": False}
    finally:
        _close_provider_preserving_pending(provider)
        token = ""
    recovered = build_submission_event(
        approval=approval, approval_uri=approval_uri, state="SUBMITTED", sequence=3,
        occurred_at=_utc_now(), previous_event=previous, previous_event_uri=previous_uri,
        provider_job=job, reason_code="RECOVERY_CONFIRMED_SUBMITTED",
    )
    tracking.recover_hf_training_submission(experiment, recovered)
    return {"status": "SUBMITTED", **job.identity(), "recovered": True, "retry_allowed": False}


def _downstream_base(
    approval: Mapping[str, object], approval_uri: str,
    submission: Mapping[str, object], submission_uri: str,
) -> dict[str, object]:
    return {
        "authorization_id": approval["authorization_id"],
        "approval": {"uri": approval_uri, "sha256": document_sha256(approval)},
        "submission": {"uri": submission_uri, "sha256": document_sha256(submission)},
        "provider_job": dict(submission["provider_job"]),
        "experiment_id": approval["experiment_id"], "run_id": approval["run_id"],
        "tracking_root_id": approval["tracking_root_id"],
    }


def _cancellation_document(
    *, approval: Mapping[str, object], approval_uri: str,
    submission: Mapping[str, object], submission_uri: str, state: str, sequence: int,
    previous: Mapping[str, object] | None = None, previous_uri: str | None = None,
    reason_code: str | None = None,
) -> dict[str, object]:
    from tuner.cloud.hf_training_smoke_contract import CANCELLATION_SCHEMA
    document = {
        "schema_version": CANCELLATION_SCHEMA,
        **_downstream_base(approval, approval_uri, submission, submission_uri),
        "state": state, "sequence": sequence, "occurred_at": _utc_now(),
        "previous_event": (
            {"uri": previous_uri, "sha256": document_sha256(previous)}
            if previous is not None and previous_uri is not None else None
        ),
        "reason_code": reason_code,
        "provider_effect_possible": state != "NOT_REQUIRED",
    }
    return seal_training_document(document)


def _observation_document(
    *, approval: Mapping[str, object], approval_uri: str,
    submission: Mapping[str, object], submission_uri: str, state: str,
    started_at: str, ended_at: str, status_intervals: Sequence[Mapping[str, str]],
    previous: Mapping[str, object] | None = None, previous_uri: str | None = None,
) -> dict[str, object]:
    from tuner.cloud.hf_training_smoke_contract import OBSERVATION_SCHEMA
    if (previous is None) is not (previous_uri is None):
        raise CloudProviderError("Observation predecessor is incomplete")
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    elapsed = max(Decimal(0), Decimal(str((end - start).total_seconds())))
    unit = Decimal(int(approval["hardware_quote"]["unit_cost_micro_usd"])) / Decimal(1_000_000)
    hourly = unit * Decimal(60)
    estimated = (elapsed * unit / Decimal(60)).quantize(Decimal("0.000001"), rounding=ROUND_UP)
    hourly_text = format(hourly.quantize(Decimal("0.000001")), "f").rstrip("0").rstrip(".")
    estimated_text = format(estimated, "f").rstrip("0").rstrip(".") or "0"
    terminal = state in {"COMPLETED", "ERROR", "CANCELLED"}
    document = {
        "schema_version": OBSERVATION_SCHEMA,
        **_downstream_base(approval, approval_uri, submission, submission_uri),
        "state": state, "terminal": terminal, "occurred_at": ended_at,
        "previous_event": (
            {"uri": previous_uri, "sha256": document_sha256(previous)}
            if previous is not None and previous_uri is not None else None
        ),
        "status_intervals": [dict(interval) for interval in status_intervals],
        "hourly_price_usd": hourly_text, "estimated_cost_usd": estimated_text,
        "cost_bounded_completion": terminal and elapsed <= Decimal(PROVIDER_TIMEOUT_SECONDS),
    }
    return seal_training_document(document)


def _observe_action(args: object, context: object) -> dict[str, object]:
    import time
    from tuner.cloud.hf_training_smoke_workload import build_workload
    from tuner.cloud.hf_volume_transport import HFArtifactVolumeSpec, prove_writable_artifact_volume

    tracking, experiment, preparation = _cli_state(args, context)
    if experiment.hf_training_submission_state != "SUBMITTED":
        raise CloudProviderError("Protected HF training observation requires SUBMITTED state")
    approval_uri = str(experiment.hf_training_approval_uri or "")
    submission_uri = str(experiment.hf_training_submission_event_uri or "")
    approval = _load_training_document(tracking, approval_uri)
    submission = _load_training_document(tracking, submission_uri)
    submission_previous = submission.get("previous_event")
    if not isinstance(submission_previous, Mapping):
        raise CloudProviderError("HF submission claim evidence is unavailable")
    submission_claim_uri = submission_previous.get("uri")
    submission_claim_sha256 = submission_previous.get("sha256")
    if type(submission_claim_uri) is not str or type(submission_claim_sha256) is not str:
        raise CloudProviderError("HF submission claim evidence is invalid")
    submission_claim = _load_training_document(tracking, submission_claim_uri)
    if document_sha256(submission_claim) != submission_claim_sha256:
        raise CloudProviderError("HF submission claim evidence changed")
    from tuner.cloud.hf_training_smoke_contract import validate_observation_event, validate_submission_event
    validate_submission_event(submission, approval=approval, previous_event=submission_claim)

    previous_observation: Mapping[str, object] | None = None
    previous_observation_uri: str | None = None
    observation_state = getattr(experiment, "hf_training_observation_state", None)
    if observation_state is not None:
        if observation_state != "STOPPED":
            raise CloudProviderError("Terminal HF training observation cannot be replaced")
        previous_observation_uri = str(
            getattr(experiment, "hf_training_observation_event_uri", "") or ""
        )
        previous_observation_sha256 = str(
            getattr(experiment, "hf_training_observation_event_sha256", "") or ""
        )
        if not previous_observation_uri or not previous_observation_sha256:
            raise CloudProviderError("Stopped HF training observation evidence is unavailable")
        previous_observation = _load_training_document(tracking, previous_observation_uri)
        if document_sha256(previous_observation) != previous_observation_sha256:
            raise CloudProviderError("Stopped HF training observation evidence changed")
        validate_observation_event(previous_observation)
        if previous_observation["state"] != "STOPPED" or previous_observation["terminal"] is not False:
            raise CloudProviderError("Stopped HF training observation evidence is invalid")
    provider_job = submission["provider_job"]
    namespace = str(provider_job["namespace"])
    job_id = str(provider_job["job_id"])
    bindings = approval["bindings"]
    preparation.require_consumable()
    source_spec = preparation.volume_spec
    assert source_spec is not None
    repository = Path(getattr(context, "project_root", "")).resolve(strict=True)
    workload = build_workload(
        repository, source_lock_sha256=str(bindings["source_lock_sha256"]),
        artifact_slot=str(bindings["artifact_slot_id"]), source_volume_spec=source_spec,
        expected_project_root=preparation.physical_project_root,
        expected_engine_root=preparation.physical_engine_root,
        expected_project_commit=preparation.source_lock.project_source.commit,
        expected_engine_commit=preparation.source_lock.engine_source.commit,
        expected_mode=preparation.source_lock.mode,
    )
    if (
        workload.workload_sha256 != bindings["workload_digest"]
        or workload.remote_argv_sha256 != bindings["remote_argv_sha256"]
        or workload.provider_command_sha256 != bindings["provider_command_sha256"]
        or workload.image.split("@", 1)[1] != bindings["image_child_digest"]
    ):
        raise CloudProviderError("Protected HF training workload changed after approval")
    import huggingface_hub

    source_volume = preparation.prove_volume(huggingface_hub).provider_volume
    artifact_volume = prove_writable_artifact_volume(
        huggingface_hub,
        HFArtifactVolumeSpec(source=str(bindings["artifact_bucket_id"]), path=str(bindings["artifact_prefix"])),
    ).provider_volume
    job_name, labels = provider_job_identity(approval)
    expected = ProviderJobExpectation(
        workload.image, workload.provider_command, job_name, tuple(sorted(labels.items())),
        (source_volume, artifact_volume), namespace,
    )
    claim_time = datetime.fromisoformat(str(submission_claim["occurred_at"]).replace("Z", "+00:00"))
    provider_time = datetime.fromisoformat(str(provider_job["created_at"]).replace("Z", "+00:00"))
    if abs((claim_time - provider_time).total_seconds()) > 120:
        raise CloudProviderError("HF provider and durable submission clocks exceed skew bound")
    started = min(claim_time, provider_time)
    cancel_at = started.timestamp() + CANCEL_AFTER_SECONDS
    stop_at = started.timestamp() + OBSERVE_UNTIL_SECONDS
    from tuner.handlers._hf_secret_file import preflight_hf_secret_file, read_claimed_hf_token
    secret_file = preflight_hf_secret_file(getattr(args, "env_file", None), context=context)
    token = read_claimed_hf_token(secret_file)
    provider = create_provider(token)
    cancellation_claimed = False
    cancellation_attempted = False
    interval_start = (
        str(previous_observation["occurred_at"])
        if previous_observation is not None
        else started.isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    interval_stage: str | None = None
    intervals: list[dict[str, str]] = []

    def record_stage(stage: str, observed_at: str) -> None:
        nonlocal interval_stage, interval_start
        interval_stage, interval_start = _advance_status_intervals(
            intervals, interval_stage, interval_start, stage, observed_at,
        )

    try:
        provider.authenticate_namespace(namespace)
        while True:
            observed = provider.inspect(job_id, namespace=namespace, expected=expected)
            if observed.identity() != dict(provider_job):
                raise CloudProviderError("HF provider observation identity changed")
            status = (observed.status or "").upper()
            terminal_map = {
                "COMPLETED": "COMPLETED", "ERROR": "ERROR", "DELETED": "ERROR",
                "CANCELED": "CANCELLED",
            }
            record_stage(status, _utc_now())
            current = time.time()
            if status in terminal_map:
                state = terminal_map[status]
                break
            if current >= stop_at:
                if previous_observation is not None:
                    raise CloudProviderError("Stopped HF training observation remains nonterminal")
                state = "STOPPED"
                break
            if current >= cancel_at and not cancellation_claimed:
                cancellation_claimed = True
                claim_doc = _cancellation_document(
                    approval=approval, approval_uri=approval_uri,
                    submission=submission, submission_uri=submission_uri,
                    state="CLAIMED", sequence=1,
                )
                claim = tracking.claim_hf_training_cancellation(experiment, claim_doc)
                if claim.provider_attempt_authorized:
                    rechecked = provider.inspect(job_id, namespace=namespace, expected=expected)
                    if rechecked.identity() != dict(provider_job):
                        raise CloudProviderError("HF provider observation identity changed")
                    re_status = (rechecked.status or "").upper()
                    record_stage(re_status, _utc_now())
                    if re_status in terminal_map:
                        terminal_cancel = _cancellation_document(
                            approval=approval, approval_uri=approval_uri,
                            submission=submission, submission_uri=submission_uri,
                            state="NOT_REQUIRED", sequence=2, previous=claim.document,
                            previous_uri=claim.uri, reason_code="TERMINAL_ON_REINSPECTION",
                        )
                    elif time.time() >= stop_at:
                        terminal_cancel = _cancellation_document(
                            approval=approval, approval_uri=approval_uri,
                            submission=submission, submission_uri=submission_uri,
                            state="AMBIGUOUS", sequence=2, previous=claim.document,
                            previous_uri=claim.uri, reason_code="INTERRUPTED_AFTER_CLAIM",
                        )
                    else:
                        try:
                            cancellation_attempted = True
                            provider.cancel(job_id, namespace=namespace)
                            terminal_cancel = _cancellation_document(
                                approval=approval, approval_uri=approval_uri,
                                submission=submission, submission_uri=submission_uri,
                                state="REQUESTED", sequence=2, previous=claim.document,
                                previous_uri=claim.uri,
                            )
                        except BaseException as exc:
                            reason = "CANCEL_OUTCOME_AMBIGUOUS" if isinstance(exc, Exception) else "INTERRUPTED_AFTER_CLAIM"
                            terminal_cancel = _cancellation_document(
                                approval=approval, approval_uri=approval_uri,
                                submission=submission, submission_uri=submission_uri,
                                state="AMBIGUOUS", sequence=2, previous=claim.document,
                                previous_uri=claim.uri, reason_code=reason,
                            )
                            tracking.record_hf_training_cancellation_terminal(experiment, terminal_cancel)
                            raise
                    tracking.record_hf_training_cancellation_terminal(experiment, terminal_cancel)
                    if time.time() >= stop_at and re_status not in terminal_map:
                        if previous_observation is not None:
                            raise CloudProviderError("Stopped HF training observation remains nonterminal")
                        state = "STOPPED"
                        break
            time.sleep(min(30.0, max(0.0, min(cancel_at if not cancellation_claimed else stop_at, stop_at) - current)))
        ended = _utc_now()
        if interval_stage is None:
            raise CloudProviderError("HF provider returned no observable job stage")
        intervals.append({"status": interval_stage, "started_at": interval_start, "ended_at": ended})
        observation = _observation_document(
            approval=approval, approval_uri=approval_uri,
            submission=submission, submission_uri=submission_uri,
            state=state, started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
            ended_at=ended, status_intervals=intervals,
            previous=previous_observation, previous_uri=previous_observation_uri,
        )
        tracking.record_hf_training_observation(experiment, observation)
        return {"status": state, "terminal": state != "STOPPED", "cancel_attempted": cancellation_attempted}
    finally:
        _close_provider_preserving_pending(provider)
        token = ""


def _result_document(
    *, approval: Mapping[str, object], approval_uri: str,
    submission: Mapping[str, object], submission_uri: str,
    observation: Mapping[str, object], observation_uri: str, state: str,
    previous: Mapping[str, object] | None = None, previous_uri: str | None = None,
    inventory: Sequence[Mapping[str, object]] = (), optimizer_proof: Mapping[str, object] | None = None,
    pre_digest: str | None = None, post_digest: str | None = None,
    verified_digest: str | None = None, reason_code: str | None = None,
) -> dict[str, object]:
    from tuner.cloud.hf_training_smoke_contract import RESULT_SCHEMA
    bindings = approval["bindings"]
    document = {
        "schema_version": RESULT_SCHEMA, "authorization_id": approval["authorization_id"],
        "approval": {"uri": approval_uri, "sha256": document_sha256(approval)},
        "submission": {"uri": submission_uri, "sha256": document_sha256(submission)},
        "observation": {"uri": observation_uri, "sha256": document_sha256(observation)},
        "provider_job": dict(submission["provider_job"]),
        "experiment_id": approval["experiment_id"], "run_id": approval["run_id"],
        "tracking_root_id": approval["tracking_root_id"], "state": state,
        "occurred_at": _utc_now(),
        "previous_result": (
            {"uri": previous_uri, "sha256": document_sha256(previous)}
            if previous is not None and previous_uri is not None else None
        ),
        "artifact_prefix": {
            "bucket_id": bindings["artifact_bucket_id"],
            "base_prefix": bindings["artifact_base_prefix"],
            "slot_id": bindings["artifact_slot_id"], "prefix": bindings["artifact_prefix"],
            "pre_download_inventory_sha256": pre_digest,
            "post_download_inventory_sha256": post_digest,
            "verified_inventory_sha256": verified_digest,
        },
        "inventory": list(inventory), "optimizer_proof": dict(optimizer_proof) if optimizer_proof else None,
        "publication": False, "ssh": False, "ports": False, "wandb": False,
        "job_secrets": [], "reason_code": reason_code,
    }
    return seal_training_document(document)


def _verify_claimed(
    *, args: object, context: object, tracking: object, experiment: object,
    approval: Mapping[str, object], approval_uri: str,
    submission: Mapping[str, object], submission_uri: str,
    observation: Mapping[str, object], observation_uri: str,
    claim: object,
) -> dict[str, object]:
    from tuner.cloud.hf_training_smoke_artifacts import ArtifactExpectation, build_inventory, verify_artifact_tree
    from tuner.handlers._hf_secret_file import preflight_hf_secret_file, read_claimed_hf_token

    provider = None
    temporary = None
    token = ""
    try:
        preflight = _load_training_document(tracking, str(approval["preflight"]["uri"]))
        runtime_lock = _load_training_document(tracking, str(preflight["runtime_lock"]["uri"]))
        runtime = runtime_lock["runtime"]
        expectation = ArtifactExpectation(
            source_lock_sha256=str(approval["bindings"]["source_lock_sha256"]),
            workload_sha256=str(approval["bindings"]["workload_digest"]),
            model_revision=str(approval["bindings"]["model_revision"]),
            dataset_sha256=str(approval["bindings"]["dataset_sha256"]),
            artifact_slot=str(approval["bindings"]["artifact_slot_id"]),
            runtime_lock_id=str(runtime_lock["lock_id"]),
            runtime_python_implementation=str(runtime["python_implementation"]),
            runtime_python=str(runtime["python"]),
            runtime_packages=tuple(sorted((str(k), str(v)) for k, v in runtime["packages"].items())),
            runtime_signatures=tuple(sorted((str(k), str(v)) for k, v in runtime["signatures"].items())),
        )
        bindings = approval["bindings"]
        namespace = str(approval["launcher_auth"]["expected_namespace"])
        secret_file = preflight_hf_secret_file(getattr(args, "env_file", None), context=context)
        token = read_claimed_hf_token(secret_file)
        provider = create_provider(token)
        provider.authenticate_namespace(namespace)
        pre = normalize_artifact_inventory(
            provider.list_bucket_tree(
                bucket_id=str(bindings["artifact_bucket_id"]), prefix=str(bindings["artifact_prefix"]),
            ), prefix=str(bindings["artifact_prefix"]),
        )
        pre_digest = document_sha256(list(pre))
        temporary = tempfile.TemporaryDirectory(prefix="synaptic-hf-training-verify-")
        root = Path(temporary.name)
        download_error: Exception | None = None
        try:
            download_exact_artifacts(
                token=token, bucket_id=str(bindings["artifact_bucket_id"]),
                prefix=str(bindings["artifact_prefix"]), inventory=pre, destination=root,
            )
        except Exception as exc:
            download_error = exc
        post = normalize_artifact_inventory(
            provider.list_bucket_tree(
                bucket_id=str(bindings["artifact_bucket_id"]), prefix=str(bindings["artifact_prefix"]),
            ), prefix=str(bindings["artifact_prefix"]),
        )
        post_digest = document_sha256(list(post))
        if post != pre or post_digest != pre_digest:
            raise CloudProviderError("Artifact provider inventory changed during download")
        if download_error is not None:
            raise CloudProviderError("Protected artifact download was rejected") from None
        verified = verify_artifact_tree(root, expectation)
        local = build_inventory(root)["files"]
        result_inventory = [
            {"path": item["path"], "bytes": item["size"], "sha256": item["sha256"]}
            for item in local
        ]
        verified_digest = document_sha256(result_inventory)
        proof = verified["optimizer_proof"]
        terminal = _result_document(
            approval=approval, approval_uri=approval_uri,
            submission=submission, submission_uri=submission_uri,
            observation=observation, observation_uri=observation_uri, state="VERIFIED",
            previous=claim.document, previous_uri=claim.uri,
            inventory=result_inventory, optimizer_proof=proof,
            pre_digest=pre_digest, post_digest=post_digest, verified_digest=verified_digest,
        )
        tracking.record_hf_training_result(experiment, terminal)
        return {"status": "VERIFIED", "result_id": terminal["result_id"], "verified": True}
    finally:
        if temporary is not None:
            temporary.cleanup()
        if provider is not None:
            _close_provider_preserving_pending(provider)
        token = ""


def _verify_action(args: object, context: object) -> dict[str, object]:
    from tuner.cloud.hf_training_smoke_artifacts import TrainingSmokeArtifactError

    tracking, experiment, _preparation = _cli_state(args, context)
    if experiment.hf_training_observation_state != "COMPLETED":
        raise CloudProviderError("Protected HF training completion is unavailable")
    approval_uri = str(experiment.hf_training_approval_uri or "")
    submission_uri = str(experiment.hf_training_submission_event_uri or "")
    observation_uri = str(experiment.hf_training_observation_event_uri or "")
    approval = _load_training_document(tracking, approval_uri)
    submission = _load_training_document(tracking, submission_uri)
    observation = _load_training_document(tracking, observation_uri)
    if observation.get("state") != "COMPLETED" or observation.get("terminal") is not True:
        raise CloudProviderError("Protected HF training completion is invalid")
    previous_result: Mapping[str, object] | None = None
    previous_result_uri: str | None = None
    result_state = getattr(experiment, "hf_training_result_state", None)
    if result_state == "INCONCLUSIVE":
        previous_result_uri = str(getattr(experiment, "hf_training_result_uri", "") or "")
        previous_result_sha256 = str(
            getattr(experiment, "hf_training_result_sha256", "") or ""
        )
        if not previous_result_uri or not previous_result_sha256:
            raise CloudProviderError("Inconclusive HF artifact result evidence is unavailable")
        previous_result = _load_training_document(tracking, previous_result_uri)
        if document_sha256(previous_result) != previous_result_sha256:
            raise CloudProviderError("Inconclusive HF artifact result evidence changed")
    elif result_state not in {None, "VERIFYING"}:
        raise CloudProviderError("Protected HF artifact result is terminal")
    claim_doc = _result_document(
        approval=approval, approval_uri=approval_uri,
        submission=submission, submission_uri=submission_uri,
        observation=observation, observation_uri=observation_uri, state="VERIFYING",
        previous=previous_result, previous_uri=previous_result_uri,
    )
    claim = tracking.claim_hf_training_verification(experiment, claim_doc)
    if not claim.provider_attempt_authorized:
        return {"status": claim.state, "verified": False}
    try:
        return _verify_claimed(
            args=args, context=context, tracking=tracking, experiment=experiment,
            approval=approval, approval_uri=approval_uri,
            submission=submission, submission_uri=submission_uri,
            observation=observation, observation_uri=observation_uri, claim=claim,
        )
    except Exception as exc:
        terminal_state = "INVALID" if isinstance(exc, TrainingSmokeArtifactError) else "INCONCLUSIVE"
        terminal = _result_document(
            approval=approval, approval_uri=approval_uri,
            submission=submission, submission_uri=submission_uri,
            observation=observation, observation_uri=observation_uri, state=terminal_state,
            previous=claim.document, previous_uri=claim.uri,
            reason_code="ARTIFACT_INVALID" if terminal_state == "INVALID" else "VERIFICATION_INCONCLUSIVE",
        )
        tracking.record_hf_training_result(experiment, terminal)
        raise CloudProviderError("Protected HF artifact verification was rejected") from None
    except BaseException:
        try:
            terminal = _result_document(
                approval=approval, approval_uri=approval_uri,
                submission=submission, submission_uri=submission_uri,
                observation=observation, observation_uri=observation_uri, state="INCONCLUSIVE",
                previous=claim.document, previous_uri=claim.uri,
                reason_code="VERIFICATION_INTERRUPTED",
            )
            tracking.record_hf_training_result(experiment, terminal)
        except Exception:
            pass
        raise


def run_training_smoke_action(
    action: str, *, args: object, context: object,
) -> dict[str, object]:
    """Execute one exact CLI transition; the handler owns sanitized presentation."""

    if platform.python_implementation() != "CPython" or sys.version_info[:3] != (3, 12, 7):
        raise CloudProviderError("Protected HF launcher requires exact CPython 3.12.7")
    if action == "approve":
        return _approve_action(args, context)
    if action == "preflight":
        return _preflight_action(args, context)
    if action == "execute":
        return _execute_action(args, context)
    if action == "recover":
        return _recover_action(args, context)
    if action == "observe":
        return _observe_action(args, context)
    if action == "verify":
        return _verify_action(args, context)
    raise CloudProviderError("Protected HF training action is invalid")


__all__ = [
    "CANCEL_AFTER_SECONDS", "FIXED_NONSECRET_ENV", "HARDWARE_FLAVOR",
    "HFTrainingSmokeProvider", "HF_ENDPOINT", "HF_HUB_VERSION", "HardwareQuote",
    "OBSERVE_UNTIL_SECONDS", "PROVIDER_TIMEOUT_SECONDS", "ProviderJob", "create_provider",
    "probe_provider_contract", "build_approval_document", "build_preflight_document",
    "build_submission_event", "download_exact_artifacts",
    "normalize_artifact_inventory",
    "ProviderJobExpectation", "provider_job_identity",
    "run_training_smoke_action",
]
