"""Run scripts/chat_model.py once inside a finite Modal GPU Sandbox.

Location: examples/model_chat/modal_launch.py. Standalone: no training run,
coordinator, Foundation or signed admission. The chat configuration is the
exact six-key file scripts/chat_model.py reads; it is validated here by that
script's own loader and shipped into the Sandbox unchanged. A finished Sandbox
exposes only its log streams, so the container prints the retained
chat-result.jsonl to stdout; the raw reply therefore appears in Modal's log
stream. Credentials come only from the named Modal profile, never from the
environment. Exact-instance shutdown proof is terminate + from_id().poll().
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import sys

ENGINE = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ENGINE))

from scripts.chat_model import load_configuration

PROVIDER_SCHEMA = "synaptic-model-chat-modal-provider/v1"
PROVIDER_KEYS = (
    "schema_version",
    "environment_name",
    "app_name",
    "gpu",
    "cpu_millicores",
    "memory_mb",
    "startup_margin_seconds",
)
STARTUP_RUNTIME_KEYS = frozenset({"provider_image_id", "python_executable"})
SDK_VERSION = "1.5.4"
LOCK_PATH = ENGINE / "tuner/execution/providers/modal/inference-runtime.lock.json"
RESULT_MARKER = "--- chat-result.jsonl"
RESULT_STATUSES = ("CLAIMED", "REPLY_SAVED", "CHAT_CONTEXT_CLOSED")
MOUNT_IGNORE = (
    "**/__pycache__",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/tests",
    "**/tests/**",
    "**/Datasets",
    "**/Datasets/**",
)
MOUNTED_DIRECTORIES = ("Evaluator", "tuner", "synaptic_tuner")
# Exact container entrypoint. The result file is printed because nothing can
# read a file out of a Sandbox whose entrypoint has exited.
CONTAINER_SCRIPT = (
    "set -euo pipefail; mkdir -m 700 /root/chat-attempt; "
    "python3 -B /engine/scripts/chat_model.py --configuration /engine/chat.json --check; "
    "status=0; python3 -B /engine/scripts/chat_model.py --configuration /engine/chat.json "
    "--output-directory /root/chat-attempt || status=$?; "
    'echo "--- chat-result.jsonl"; cat /root/chat-attempt/chat-result.jsonl 2>/dev/null '
    '|| echo "--- chat-result.jsonl was not written" >&2; exit "$status"'
)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_ATTEMPT = re.compile(r"[A-Za-z0-9._-]{1,64}")
_GPU = re.compile(r"[A-Za-z0-9][A-Za-z0-9:.-]{0,31}")
_IMAGE = re.compile(r"[a-z0-9][a-z0-9./_-]{0,255}@sha256:[0-9a-f]{64}")
_REASON = re.compile(r"[a-z][a-z0-9_]{0,63}")
_LOG_LIMIT = 1024 * 1024
PROBE_SCRIPT = ENGINE / "scripts/probe_vllm_startup.py"
PROBE_SOURCE_MEMBERS = (
    "scripts/probe_vllm_startup.py",
    "tuner/inference/vllm_runtime.py",
    "tuner/inference/owned_process.py",
    "tuner/inference/serving_target.py",
    "tuner/inference/retrieved_model.py",
    "tuner/execution/providers/modal/model_snapshot.py",
    "tuner/execution/providers/modal/mounted_io.py",
)
_SOURCE_FILE_LIMIT = 512 * 1024 * 1024
_SOURCE_TOTAL_LIMIT = 768 * 1024 * 1024
_SOURCE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


class ModalChatLaunchError(RuntimeError):
    """Closed launcher failure; the message is a fixed reason code."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ModalChatLaunchError("modal_chat_arguments_invalid")


@dataclass(frozen=True)
class ModalChatProvider:
    environment_name: str
    app_name: str
    gpu: str
    cpu_millicores: int
    memory_mb: int
    startup_margin_seconds: int
    startup_runtime: dict[str, str] | None = None


def _invalid(*args):
    raise ModalChatLaunchError("modal_chat_provider_invalid")


def _unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            _invalid()
        result[name] = value
    return result


def load_provider(path: Path) -> ModalChatProvider:
    """Strict read: regular file <= 64 KiB, unique keys, exact key set, bounds."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as selected:
        info = os.fstat(selected.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
            _invalid()
        raw = selected.read(65537)
    if len(raw) != info.st_size:
        raise ModalChatLaunchError("modal_chat_provider_changed")
    try:
        document = json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid)
    except ValueError:
        _invalid()
    if (
        type(document) is not dict
        or set(document) not in (set(PROVIDER_KEYS), set(PROVIDER_KEYS) | {"startup_runtime"})
        or document["schema_version"] != PROVIDER_SCHEMA
    ):
        _invalid()
    for name, pattern in (("environment_name", _NAME), ("app_name", _NAME), ("gpu", _GPU)):
        if type(document[name]) is not str or pattern.fullmatch(document[name]) is None:
            _invalid()
    for name, low, high in (
        ("cpu_millicores", 125, 256000),
        ("memory_mb", 128, 1048576),
        ("startup_margin_seconds", 1, 600),
    ):
        if type(document[name]) is not int or not low <= document[name] <= high:
            _invalid()
    runtime = document.get("startup_runtime")
    if "startup_runtime" in document and runtime is None:
        _invalid()
    if runtime is not None:
        if type(runtime) is not dict or set(runtime) != STARTUP_RUNTIME_KEYS:
            _invalid()
        image_id, python = runtime["provider_image_id"], runtime["python_executable"]
        if (type(image_id) is not str or re.fullmatch(r"im-[A-Za-z0-9]{1,64}", image_id) is None
                or type(python) is not str or not python.startswith("/")
                or PurePosixPath(python).as_posix() != python
                or any(part in {".", ".."} for part in python.split("/"))
                or re.fullmatch(r"/[A-Za-z0-9._/-]+", python) is None):
            _invalid()
        runtime = {"provider_image_id": image_id, "python_executable": python}
    return ModalChatProvider(*(document[name] for name in PROVIDER_KEYS[1:]), runtime)


def image_reference() -> str:
    with open(LOCK_PATH, "rb") as handle:
        value = json.load(handle).get("base_registry_reference")
    if type(value) is not str or "@sha256:" not in value or _IMAGE.fullmatch(value) is None:
        raise ModalChatLaunchError("modal_chat_image_reference_invalid")
    return value


def _attempt(name) -> str:
    if type(name) is not str or _ATTEMPT.fullmatch(name) is None:
        raise ModalChatLaunchError("modal_chat_attempt_invalid")
    return name


def _output_directory(output) -> int:
    """Apply the chat command's private-directory checks; return a retained dir fd."""
    if (
        not isinstance(output, Path)
        or not output.is_absolute()
        or not output.is_dir()
        or output.resolve(strict=True) != output
        or output.stat().st_uid != os.geteuid()
        or output.stat().st_mode & 0o077
    ):
        raise ModalChatLaunchError("modal_chat_output_invalid")
    directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    retained = os.fstat(directory)
    if retained.st_uid != os.geteuid() or retained.st_mode & 0o077:
        os.close(directory)
        raise ModalChatLaunchError("modal_chat_output_invalid")
    return directory


def _write_private(directory: int, name: str, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(name, flags, 0o600, dir_fd=directory), "w", encoding="utf-8") as saved:
        saved.write(text)
        saved.flush()
        os.fsync(saved.fileno())


def _save_private_probe_stderr(directory: int, value: object) -> int:
    """Retain only a bounded private startup log tail; never emit it."""
    if type(value) is str:
        raw = value.encode("utf-8")
    elif type(value) is bytes:
        raw = value
    else:
        raise ModalChatLaunchError("modal_chat_startup_log_invalid")
    if len(raw) > 65536:
        raise ModalChatLaunchError("modal_chat_startup_log_invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open("startup-vllm-stderr.log", flags, 0o600, dir_fd=directory), "wb") as saved:
        saved.write(raw)
        saved.flush()
        os.fsync(saved.fileno())
    return len(raw)


def _probe_source_hashes() -> dict[str, str]:
    return {
        member: hashlib.sha256((ENGINE / member).read_bytes()).hexdigest()
        for member in PROBE_SOURCE_MEMBERS
    }


def _local_source_files(selected: dict) -> tuple[tuple[Path, str, tuple[int, ...]], ...]:
    """Admit only exact named regular files before any provider effect."""
    source = selected.get("local_source")
    if source is None:
        return ()
    if type(source) is not dict or set(source) != {"adapter", "tokenizer"}:
        raise ModalChatLaunchError("modal_chat_startup_source_invalid")
    files = []
    total = 0
    for group in ("adapter", "tokenizer"):
        entries = source[group]
        if type(entries) is not list or not 1 <= len(entries) <= 16:
            raise ModalChatLaunchError("modal_chat_startup_source_invalid")
        names = set()
        for entry in entries:
            if type(entry) is not dict or set(entry) != {"path", "name", "sha256", "size_bytes"}:
                raise ModalChatLaunchError("modal_chat_startup_source_invalid")
            name, raw_path = entry["name"], entry["path"]
            size, digest = entry["size_bytes"], entry["sha256"]
            if (type(name) is not str or name in {".", ".."} or _SOURCE_NAME.fullmatch(name) is None
                    or name in names or type(raw_path) is not str or not raw_path
                    or type(size) is not int or not 0 <= size <= _SOURCE_FILE_LIMIT
                    or type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                raise ModalChatLaunchError("modal_chat_startup_source_invalid")
            names.add(name)
            try:
                path = Path(raw_path)
                if not path.is_absolute() or path.resolve(strict=True) != path:
                    raise ModalChatLaunchError("modal_chat_startup_source_invalid")
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                try:
                    before = os.fstat(descriptor)
                    if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                        raise ModalChatLaunchError("modal_chat_startup_source_invalid")
                    actual = hashlib.sha256()
                    read_size = 0
                    while True:
                        chunk = os.read(descriptor, min(1024 * 1024, size + 1 - read_size))
                        if not chunk:
                            break
                        read_size += len(chunk)
                        if read_size > size:
                            raise ModalChatLaunchError("modal_chat_startup_source_invalid")
                        actual.update(chunk)
                    after = os.fstat(descriptor)
                    identity = tuple(getattr(before, field) for field in
                                     ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
                    if (identity != tuple(getattr(after, field) for field in
                                          ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
                            or read_size != size or actual.hexdigest() != digest):
                        raise ModalChatLaunchError("modal_chat_startup_source_invalid")
                finally:
                    os.close(descriptor)
            except OSError:
                raise ModalChatLaunchError("modal_chat_startup_source_invalid") from None
            total += size
            if total > _SOURCE_TOTAL_LIMIT:
                raise ModalChatLaunchError("modal_chat_startup_source_invalid")
            files.append((path, f"/engine/startup-inputs/{group}/{name}", identity))
    return tuple(files)


def _stage_local_source_files(selected: dict, output: Path, directory: int,
                              admitted: tuple) -> tuple:
    """Copy verified bytes into exclusive private files used for image upload."""
    if not admitted:
        return ()
    os.mkdir("startup-inputs", 0o700, dir_fd=directory)
    stage_root = os.open("startup-inputs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                         dir_fd=directory)
    try:
        staged = {"adapter": [], "tokenizer": []}
        index = 0
        for group in ("adapter", "tokenizer"):
            os.mkdir(group, 0o700, dir_fd=stage_root)
            group_fd = os.open(group, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=stage_root)
            try:
                for entry in selected["local_source"][group]:
                    original, remote, identity = admitted[index]
                    index += 1
                    source_fd = os.open(original, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    try:
                        before = os.fstat(source_fd)
                        current = tuple(getattr(before, field) for field in
                                        ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
                        if current != identity or not stat.S_ISREG(before.st_mode):
                            raise ModalChatLaunchError("modal_chat_startup_source_changed")
                        dest_fd = os.open(entry["name"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                          0o600, dir_fd=group_fd)
                        digest = hashlib.sha256()
                        count = 0
                        with os.fdopen(dest_fd, "wb") as destination:
                            while True:
                                block = os.read(source_fd, min(1024 * 1024, entry["size_bytes"] - count + 1))
                                if not block:
                                    break
                                count += len(block)
                                if count > entry["size_bytes"]:
                                    raise ModalChatLaunchError("modal_chat_startup_source_changed")
                                digest.update(block)
                                destination.write(block)
                            destination.flush()
                            os.fsync(destination.fileno())
                        after = os.fstat(source_fd)
                        if (current != tuple(getattr(after, field) for field in
                                             ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
                                or count != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]):
                            raise ModalChatLaunchError("modal_chat_startup_source_changed")
                    finally:
                        os.close(source_fd)
                    staged[group].append({**entry, "path": str(output / "startup-inputs" / group / entry["name"])})
                os.fsync(group_fd)
            finally:
                os.close(group_fd)
        os.fsync(stage_root)
    except OSError:
        raise ModalChatLaunchError("modal_chat_startup_source_changed") from None
    finally:
        os.close(stage_root)
    return _local_source_files({"local_source": staged})


def _client(sdk, profile):
    if type(profile) is not str or _NAME.fullmatch(profile) is None:
        raise ModalChatLaunchError("modal_chat_profile_invalid")
    token_id = sdk.config.config.get("token_id", profile=profile, use_env=False)
    token_secret = sdk.config.config.get("token_secret", profile=profile, use_env=False)
    if any(type(value) is not str or not value.strip() for value in (token_id, token_secret)):
        raise ModalChatLaunchError("modal_chat_credentials_missing")
    return sdk.Client.from_credentials(token_id, token_secret)


def build_image(sdk, reference: str, configuration: Path):
    image = sdk.Image.from_registry(reference).entrypoint([])
    for name in MOUNTED_DIRECTORIES:
        image = image.add_local_dir(
            ENGINE / name, remote_path="/engine/" + name, copy=False, ignore=list(MOUNT_IGNORE)
        )
    image = image.add_local_file(ENGINE / "scripts/chat_model.py", "/engine/scripts/chat_model.py")
    return image.add_local_file(configuration, "/engine/chat.json")


def build_startup_image(image, configuration: Path, local_files=()):
    """Overlay the selected image with diagnostic source and declared files."""
    image = image.entrypoint([])
    for name in MOUNTED_DIRECTORIES:
        image = image.add_local_dir(
            ENGINE / name, remote_path="/engine/" + name, copy=False, ignore=list(MOUNT_IGNORE)
        )
    image = image.add_local_file(PROBE_SCRIPT, "/engine/scripts/probe_vllm_startup.py")
    image = image.add_local_file(configuration, "/engine/startup-probe.json")
    for path, remote, _identity in local_files:
        image = image.add_local_file(path, remote)
    return image


def _startup_stdout(raw: object, digest: str) -> dict:
    if type(raw) is bytes:
        raw = raw.decode("utf-8", "replace")
    if type(raw) is not str or len(raw.encode("utf-8")) > _LOG_LIMIT:
        raise ModalChatLaunchError("modal_chat_startup_result_invalid")
    lines = raw.splitlines()
    if len(lines) != 1:
        raise ModalChatLaunchError("modal_chat_startup_result_invalid")
    try:
        document = json.loads(lines[0], object_pairs_hook=_unique, parse_constant=_invalid)
    except (ValueError, UnicodeError, RecursionError, ModalChatLaunchError):
        raise ModalChatLaunchError("modal_chat_startup_result_invalid") from None
    if type(document) is not dict or set(document) != {"status", "result"} or document["status"] != "STARTUP_PROBE_SAVED":
        raise ModalChatLaunchError("modal_chat_startup_result_invalid")
    from scripts.probe_vllm_startup import validate_result
    try:
        return validate_result(document["result"], configuration_digest=digest)
    except ValueError:
        raise ModalChatLaunchError("modal_chat_startup_result_invalid") from None


def _log_text(value) -> str:
    if type(value) is bytes:
        value = value.decode("utf-8", "replace")
    if type(value) is not str:
        raise ModalChatLaunchError("modal_chat_sandbox_log_invalid")
    return value.encode("utf-8")[:_LOG_LIMIT].decode("utf-8", "ignore")


def _status_line(line):
    with contextlib.suppress(ValueError):
        value = json.loads(line)
        if type(value) is dict and type(value.get("status")) is str:
            return value["status"]
    return None


def parse_sandbox_stdout(text: str):
    """Return (command status codes, record lines or None, record statuses or None)."""
    lines = text.splitlines()
    statuses, records, parsed = [], [], []
    marker = lines.index(RESULT_MARKER) if RESULT_MARKER in lines else None
    for line in lines[:marker]:
        status = _status_line(line)
        if status is not None:
            statuses.append(status)
    if marker is None:
        return statuses, None, None
    for line in lines[marker + 1 :]:
        status = _status_line(line)
        if status is None:
            return statuses, None, None
        records.append(line)
        parsed.append(status)
    return statuses, records, parsed


def _reason(error: BaseException) -> str:
    code = error.args[0] if error.args else None
    return code if type(code) is str and _REASON.fullmatch(code) else "modal_chat_failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run(*, configuration, provider, attempt, profile, output, emit, sdk=None, client=None):
    """Effectful flow. Returns 0 only with a complete reply and shutdown proof."""
    lifetime = load_configuration(configuration)[0]["lifetime_seconds"]
    reference = image_reference()
    attempt = _attempt(attempt)
    directory = _output_directory(output)
    sandbox, proof = None, False
    outcome = {"returncode": None, "reason": "modal_chat_unfinished", "complete": False}
    try:
        try:
            os.stat("attempt.json", dir_fd=directory)
        except FileNotFoundError:
            pass
        else:
            raise ModalChatLaunchError("modal_chat_output_claimed")
        if sdk is None:
            sdk = importlib.import_module("modal")
        if sdk.__version__ != SDK_VERSION:
            raise ModalChatLaunchError("modal_chat_sdk_invalid")
        if client is None:
            client = _client(sdk, profile)
        app = sdk.App.lookup(
            provider.app_name,
            environment_name=provider.environment_name,
            create_if_missing=True,
            client=client,
        )
        image = build_image(sdk, reference, configuration)
        bound = lifetime + provider.startup_margin_seconds
        try:
            sandbox = sdk.Sandbox.create(
                "bash",
                "-lc",
                CONTAINER_SCRIPT,
                app=app,
                image=image,
                gpu=provider.gpu,
                cpu=provider.cpu_millicores / 1000,
                memory=provider.memory_mb,
                timeout=bound,
                idle_timeout=bound,
                name=attempt,
                client=client,
            )
        except sdk.exception.AlreadyExistsError:
            emit({"status": "MODAL_CHAT_ATTEMPT_EXISTS", "retry_authorized": False})
            return 1
        sandbox_id = sandbox.object_id
        if type(sandbox_id) is not str or not sandbox_id:
            raise ModalChatLaunchError("modal_chat_sandbox_id_invalid")
        record = {
            "schema_version": "synaptic-model-chat-modal-attempt/v1",
            "attempt": attempt,
            "app_name": provider.app_name,
            "environment_name": provider.environment_name,
            "sandbox_id": sandbox_id,
            "image_reference": reference,
            "lifetime_seconds": lifetime,
            "startup_margin_seconds": provider.startup_margin_seconds,
            "timeout_seconds": bound,
            "idle_timeout_seconds": bound,
            "created_at_utc": _utc_now(),
        }
        _write_private(directory, "attempt.json", json.dumps(record, sort_keys=True) + "\n")
        emit({"status": "MODAL_CHAT_SANDBOX_CREATED", "sandbox_id": sandbox_id})
        sandbox.wait(raise_on_termination=False)
        returncode = sandbox.returncode
        outcome["returncode"] = returncode if type(returncode) is int else None
        stdout = _log_text(sandbox.stdout.read())
        _write_private(directory, "sandbox-stdout.log", stdout)
        _write_private(directory, "sandbox-stderr.log", _log_text(sandbox.stderr.read()))
        emit({"status": "MODAL_CHAT_SANDBOX_FINISHED", "returncode": outcome["returncode"]})
        statuses, records, parsed = parse_sandbox_stdout(stdout)
        outcome.update(command_statuses=statuses, record_statuses=parsed)
        if records is not None:
            _write_private(directory, "chat-result.jsonl", "".join(line + "\n" for line in records))
        if returncode != 0:
            outcome["reason"] = "modal_chat_command_failed"
        elif records is None:
            outcome["reason"] = "modal_chat_result_unparsable"
        elif tuple(parsed) != RESULT_STATUSES:
            outcome["reason"] = "modal_chat_result_incomplete"
        else:
            outcome.update(reason=None, complete=True)
            emit({"status": "MODAL_CHAT_REPLY_SAVED"})
        if not outcome["complete"]:
            emit({"status": "MODAL_CHAT_FAILED", "reason": outcome["reason"], "retry_authorized": False})
    finally:
        try:
            if sandbox is not None:
                proof = _shutdown(sdk, client, sandbox, directory, outcome, emit)
        finally:
            os.close(directory)
    return 0 if outcome["complete"] and proof else 1


def run_startup(*, configuration, selected, digest, provider, attempt, profile,
                output, emit, sdk=None, client=None):
    """One claimed, finite probe against an explicitly selected image ID."""
    runtime = provider.startup_runtime
    if runtime is None or runtime["python_executable"] != selected["python_executable"]:
        raise ModalChatLaunchError("modal_chat_startup_runtime_invalid")
    local_files = _local_source_files(selected)
    attempt = _attempt(attempt)
    directory = _output_directory(output)
    sandbox, proof = None, False
    outcome = {"returncode": None, "reason": "modal_chat_startup_unfinished", "complete": False}
    try:
        try:
            os.stat("probe-claim.json", dir_fd=directory)
        except FileNotFoundError:
            pass
        else:
            raise ModalChatLaunchError("modal_chat_output_claimed")
        if sdk is None:
            sdk = importlib.import_module("modal")
        if sdk.__version__ != SDK_VERSION:
            raise ModalChatLaunchError("modal_chat_sdk_invalid")
        if client is None:
            client = _client(sdk, profile)
        source_hashes = _probe_source_hashes()
        bound = selected["lifetime_seconds"] + provider.startup_margin_seconds
        claim = {
            "schema_version": "synaptic-model-chat-modal-startup-claim/v1",
            "attempt": attempt,
            "configuration_sha256": digest,
            "provider_image_id": runtime["provider_image_id"],
            "python_executable": runtime["python_executable"],
            "diagnostic_source_sha256": source_hashes,
            "local_source": selected.get("local_source"),
            "app_name": provider.app_name,
            "environment_name": provider.environment_name,
            "gpu": provider.gpu,
            "timeout_seconds": bound,
            "created_at_utc": _utc_now(),
        }
        _write_private(directory, "probe-claim.json", json.dumps(claim, sort_keys=True) + "\n")
        os.fsync(directory)
        staged_files = _stage_local_source_files(selected, output, directory, local_files)
        image = sdk.Image.from_id(runtime["provider_image_id"], client=client)
        app = sdk.App.lookup(
            provider.app_name, environment_name=provider.environment_name,
            create_if_missing=True, client=client,
        )
        image = image.build(app)
        if (getattr(image, "is_hydrated", False) is not True
                or getattr(image, "object_id", None) != runtime["provider_image_id"]):
            raise ModalChatLaunchError("modal_chat_startup_image_identity_invalid")
        image = build_startup_image(image, configuration, staged_files)
        image = image.build(app)
        if _probe_source_hashes() != source_hashes:
            raise ModalChatLaunchError("modal_chat_startup_source_changed")
        staged_manifest = {"adapter": [], "tokenizer": []}
        if staged_files:
            for group in ("adapter", "tokenizer"):
                for entry in selected["local_source"][group]:
                    staged_manifest[group].append({
                        **entry, "path": str(output / "startup-inputs" / group / entry["name"]),
                    })
        if staged_files and _local_source_files({"local_source": staged_manifest}) != staged_files:
            raise ModalChatLaunchError("modal_chat_startup_source_changed")
        python = shlex.quote(runtime["python_executable"])
        script = (
            "set -euo pipefail; mkdir -m 700 /root/startup-probe; "
            f"{python} -B /engine/scripts/probe_vllm_startup.py "
            "--configuration /engine/startup-probe.json "
            "--output-directory /root/startup-probe"
        )
        try:
            sandbox = sdk.Sandbox.create(
                "bash", "-lc", script,
                app=app, image=image, gpu=provider.gpu,
                cpu=provider.cpu_millicores / 1000, memory=provider.memory_mb,
                timeout=bound, idle_timeout=bound, name=attempt, client=client,
            )
        except sdk.exception.AlreadyExistsError:
            emit({"status": "MODAL_CHAT_ATTEMPT_EXISTS", "retry_authorized": False})
            return 1
        sandbox_id = sandbox.object_id
        if type(sandbox_id) is not str or not sandbox_id:
            raise ModalChatLaunchError("modal_chat_sandbox_id_invalid")
        _write_private(directory, "probe-sandbox.json", json.dumps({
            "schema_version": "synaptic-model-chat-modal-startup-sandbox/v1",
            "attempt": attempt, "sandbox_id": sandbox_id,
            "provider_image_id": runtime["provider_image_id"],
            "configuration_sha256": digest,
            "diagnostic_source_sha256": source_hashes,
        }, sort_keys=True) + "\n")
        emit({"status": "MODAL_CHAT_STARTUP_SANDBOX_CREATED", "sandbox_id": sandbox_id})
        sandbox.wait(raise_on_termination=False)
        returncode = sandbox.returncode
        outcome["returncode"] = returncode if type(returncode) is int else None
        try:
            retained_log_bytes = _save_private_probe_stderr(directory, sandbox.stderr.read())
        except ModalChatLaunchError:
            retained_log_bytes = None
            outcome["reason"] = "modal_chat_startup_log_invalid"
        try:
            result = _startup_stdout(sandbox.stdout.read(), digest)
        except ModalChatLaunchError:
            outcome["reason"] = "modal_chat_startup_result_invalid"
        else:
            if retained_log_bytes != result["startup_log_tail_bytes"]:
                outcome["reason"] = "modal_chat_startup_log_invalid"
            _write_private(directory, "startup-probe-result.json", json.dumps(result, sort_keys=True) + "\n")
            outcome["complete"] = bool(
                outcome["reason"] != "modal_chat_startup_log_invalid"
                and result["startup_ready"] and result["cleanup_resolved"] and returncode == 0
            )
            if outcome["reason"] != "modal_chat_startup_log_invalid":
                outcome["reason"] = None if outcome["complete"] else "modal_chat_startup_failed"
            emit({"status": "MODAL_CHAT_STARTUP_RESULT_SAVED", "startup_ready": result["startup_ready"],
                  "cleanup_resolved": result["cleanup_resolved"]})
        if not outcome["complete"]:
            emit({"status": "MODAL_CHAT_FAILED", "reason": outcome["reason"], "retry_authorized": False})
    finally:
        try:
            if sandbox is not None:
                proof = _shutdown(sdk, client, sandbox, directory, outcome, emit)
        finally:
            os.close(directory)
    return 0 if outcome["complete"] and proof else 1


def _shutdown(sdk, client, sandbox, directory, outcome, emit) -> bool:
    """Terminate the exact Sandbox and read its stopped state back by id."""
    record = {
        "schema_version": "synaptic-model-chat-modal-shutdown/v1",
        "sandbox_id": sandbox.object_id,
        "terminate_exception_class": None,
        "poll_exception_class": None,
        **outcome,
    }
    try:
        sandbox.terminate(wait=True)
    except Exception as error:
        record["terminate_exception_class"] = type(error).__name__
    try:
        polled = sdk.Sandbox.from_id(sandbox.object_id, client=client).poll()
    except Exception as error:
        record["poll_exception_class"] = type(error).__name__
        polled = None
    proof = type(polled) is int
    record.update(
        poll_result=polled if proof else None,
        provider_shutdown_proof=proof,
        stopped_at_utc=_utc_now(),
    )
    emit({"status": "MODAL_CHAT_SANDBOX_STOPPED", "provider_shutdown_proof": proof})
    _write_private(directory, "shutdown.json", json.dumps(record, sort_keys=True) + "\n")
    return proof


def build_parser():
    parser = _Parser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--attempt")
    parser.add_argument("--modal-profile")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    return parser


def main(argv=None, *, sdk=None, client=None):
    stream = sys.stdout

    def emit(document):
        stream.write(json.dumps(document, separators=(",", ":")) + "\n")
        stream.flush()

    try:
        arguments = build_parser().parse_args(argv)
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            provider = load_provider(arguments.provider)
            if arguments.startup_only:
                from scripts.probe_vllm_startup import (
                    configuration_digest, load_configuration as load_probe_configuration,
                )
                selected = load_probe_configuration(arguments.configuration)
                _local_source_files(selected)
                if (provider.startup_runtime is None
                        or provider.startup_runtime["python_executable"] != selected["python_executable"]
                        or not PROBE_SCRIPT.is_file()):
                    raise ModalChatLaunchError("modal_chat_startup_runtime_invalid")
            else:
                if provider.startup_runtime is not None:
                    raise ModalChatLaunchError("modal_chat_startup_runtime_requires_mode")
                load_configuration(arguments.configuration)
            if arguments.attempt is not None:
                _attempt(arguments.attempt)
            if arguments.check:
                if arguments.startup_only:
                    emit({"status": "MODAL_CHAT_STARTUP_INPUTS_CHECKED"})
                else:
                    image_reference()
                    emit({"status": "MODAL_CHAT_INPUTS_CHECKED"})
                return 0
            if None in (arguments.attempt, arguments.modal_profile, arguments.output_directory):
                raise ModalChatLaunchError("modal_chat_arguments_invalid")
            if arguments.startup_only:
                return run_startup(
                    configuration=arguments.configuration, selected=selected,
                    digest=configuration_digest(selected), provider=provider,
                    attempt=arguments.attempt, profile=arguments.modal_profile,
                    output=arguments.output_directory, emit=emit, sdk=sdk, client=client,
                )
            return run(
                configuration=arguments.configuration,
                provider=provider,
                attempt=arguments.attempt,
                profile=arguments.modal_profile,
                output=arguments.output_directory,
                emit=emit,
                sdk=sdk,
                client=client,
            )
    except KeyboardInterrupt:
        emit({"status": "INTERRUPTED", "retry_authorized": False})
        return 130
    except Exception as error:
        emit(
            {
                "status": "MODAL_CHAT_FAILED",
                "reason": _reason(error),
                "exception_class": type(error).__name__,
                "retry_authorized": False,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
