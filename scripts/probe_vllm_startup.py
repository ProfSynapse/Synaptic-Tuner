"""Bounded, model-first vLLM startup probe; no prompt or training submission.

Model preparation uses private scratch/cache. The in-process lifetime deadline
covers preparation and startup, but a blocking Hub call is bounded by the
launching provider's finite Sandbox timeout, not interrupted by this script.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuner.inference.vllm_runtime import (
    ExplicitNetworkVLLMSource, VLLMRuntimeError, VLLMStartupDiagnostic,
    VLLMStartupSpec, _projection, start_vllm_runtime,
)


_CONFIG_LIMIT = 4096
_RESULT_LIMIT = 4096
_STARTUP_LOG_TAIL_LIMIT = 64 * 1024
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?\Z")
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}(?:[A-Za-z0-9.+-]*)\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_STARTUP_KEYS = {
    "served_model_name", "gpu_memory_utilization", "tensor_parallel_size",
    "enforce_eager", "dtype", "max_model_len", "max_num_seqs",
    "max_num_batched_tokens", "language_model_only", "max_lora_rank",
    "startup_timeout_seconds",
}


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("startup_probe_configuration_invalid")
        value[key] = item
    return value


def _invalid_constant(_value):
    raise ValueError("startup_probe_configuration_invalid")


def load_configuration(path: Path) -> dict:
    """Read one exact bounded JSON configuration without runtime side effects."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _CONFIG_LIMIT:
            raise ValueError("startup_probe_configuration_invalid")
        raw = source.read(_CONFIG_LIMIT + 1)
        after = os.fstat(source.fileno())
        if len(raw) != before.st_size or any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ValueError("startup_probe_configuration_changed")
    if len(raw) > _CONFIG_LIMIT:
        raise ValueError("startup_probe_configuration_invalid")
    try:
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("startup_probe_configuration_invalid") from None
    required = {
        "model", "revision", "expected_vllm_version", "python_executable",
        "startup", "lifetime_seconds",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("startup_probe_configuration_invalid")
    model, revision, version = value["model"], value["revision"], value["expected_vllm_version"]
    if (type(model) is not str or _MODEL.fullmatch(model) is None
            or any(".." in part or "--" in part or part.endswith((".", "-")) for part in model.split("/"))
            or type(revision) is not str or _SHA.fullmatch(revision) is None
            or type(version) is not str or _VERSION.fullmatch(version) is None):
        raise ValueError("startup_probe_configuration_invalid")
    lifetime = value["lifetime_seconds"]
    if type(lifetime) is not int or not 1 <= lifetime <= 1800:
        raise ValueError("startup_probe_configuration_invalid")
    startup = value["startup"]
    if type(startup) is not dict or set(startup) != _STARTUP_KEYS:
        raise ValueError("startup_probe_configuration_invalid")
    if type(startup["served_model_name"]) is not str or _NAME.fullmatch(startup["served_model_name"]) is None:
        raise ValueError("startup_probe_configuration_invalid")
    timeout = startup["startup_timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= lifetime:
        raise ValueError("startup_probe_configuration_invalid")
    spec = _startup_spec(value, model)
    _projection(spec, cwd=Path.cwd().resolve(), environment={})
    return value


def configuration_digest(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _startup_spec(value: dict, model_path: str) -> VLLMStartupSpec:
    selected = value["startup"]
    return VLLMStartupSpec(
        source=ExplicitNetworkVLLMSource(model_path),
        served_model_name=selected["served_model_name"],
        python_executable=value["python_executable"],
        startup_timeout_s=selected["startup_timeout_seconds"],
        gpu_memory_utilization=selected["gpu_memory_utilization"],
        tensor_parallel_size=selected["tensor_parallel_size"],
        enforce_eager=selected["enforce_eager"],
        dtype=selected["dtype"],
        max_model_len=selected["max_model_len"],
        max_num_seqs=selected["max_num_seqs"],
        max_num_batched_tokens=selected["max_num_batched_tokens"],
        language_model_only=selected["language_model_only"],
        max_lora_rank=selected["max_lora_rank"],
    )


def _private_output(path: Path) -> Path:
    if not path.is_absolute() or not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError("startup_probe_output_invalid")
    info = path.stat(follow_symlinks=False)
    if (not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077
            or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())):
        raise ValueError("startup_probe_output_invalid")
    return path


def _exclusive_json(path: Path, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > _RESULT_LIMIT:
        raise ValueError("startup_probe_result_invalid")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    if hasattr(os, "O_DIRECTORY"):
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _elapsed(started: float) -> float | None:
    value = time.monotonic() - started
    return round(value, 6) if math.isfinite(value) and 0 <= value <= 86400 else None


def _startup_log_summary(path: Path) -> tuple[int | None, int, bool]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("startup_probe_log_invalid")
        size = info.st_size
        tail_size = min(size, _STARTUP_LOG_TAIL_LIMIT)
        return (size if size <= 1_000_000_000 else None, tail_size,
                size > _STARTUP_LOG_TAIL_LIMIT)


def emit_private_startup_log_tail(output: Path) -> None:
    """Transfer at most 64 KiB to provider-retained stderr after cleanup.

    User-approved for this public-base, no-prompt/no-adapter probe. Provider
    logs and the host's downloaded copy must remain private/sensitive.
    """
    path = output / "vllm-startup.log"
    if not path.exists():
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("startup_probe_log_invalid")
        source.seek(max(0, before.st_size - _STARTUP_LOG_TAIL_LIMIT))
        tail = source.read(_STARTUP_LOG_TAIL_LIMIT)
        after = os.fstat(source.fileno())
        if len(tail) != min(before.st_size, _STARTUP_LOG_TAIL_LIMIT) or any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ValueError("startup_probe_log_changed")
    while tail:
        written = os.write(2, tail)
        if written <= 0:
            raise ValueError("startup_probe_log_transfer_invalid")
        tail = tail[written:]


def _diagnostic(value: object) -> dict | None:
    if type(value) is not VLLMStartupDiagnostic:
        return None
    if (type(value.failure) is not str or value.failure not in {"deadline", "untimely", "unknown"}
            or type(value.last_probe) is not str or value.last_probe not in {"ready", "no_connect", "http_non_200", "invalid_response", "names_mismatch", "unknown"}
            or (value.leader_alive is not None and type(value.leader_alive) is not bool)
            or (value.elapsed_seconds is not None and (type(value.elapsed_seconds) is not float
                or not math.isfinite(value.elapsed_seconds) or not 0 <= value.elapsed_seconds <= 86400))
            or (value.probe_count is not None and (type(value.probe_count) is not int
                or not 0 <= value.probe_count <= 1_000_000))):
        return None
    return {"failure": value.failure, "last_probe": value.last_probe,
            "leader_alive": value.leader_alive, "elapsed_seconds": value.elapsed_seconds,
            "probe_count": value.probe_count}


def validate_result(value: object, *, configuration_digest: str) -> dict:
    """Admit only a closed worker result before host-side retention."""
    if (type(configuration_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", configuration_digest) is None
            or type(value) is not dict or set(value) != {
                "schema_version", "configuration_sha256", "startup_ready",
                "cleanup_resolved", "preparation_seconds", "startup_diagnostic",
                "failure_stage", "startup_log_size_bytes", "startup_log_tail_bytes",
                "startup_log_truncated",
            } or value["schema_version"] != "synaptic-vllm-startup-probe-result/v1"
            or value["configuration_sha256"] != configuration_digest
            or type(value["startup_ready"]) is not bool
            or type(value["cleanup_resolved"]) is not bool
            or value["failure_stage"] not in (None, "version", "preparation", "startup", "cleanup")):
        raise ValueError("startup_probe_result_invalid")
    elapsed = value["preparation_seconds"]
    if elapsed is not None and (type(elapsed) is not float or not math.isfinite(elapsed)
                                or not 0 <= elapsed <= 86400):
        raise ValueError("startup_probe_result_invalid")
    log_size, tail_size = value["startup_log_size_bytes"], value["startup_log_tail_bytes"]
    if (log_size is not None and (type(log_size) is not int or not 0 <= log_size <= 1_000_000_000)
            or type(tail_size) is not int or not 0 <= tail_size <= _STARTUP_LOG_TAIL_LIMIT
            or type(value["startup_log_truncated"]) is not bool
            or (log_size is not None and tail_size != min(log_size, _STARTUP_LOG_TAIL_LIMIT))
            or (log_size is not None and value["startup_log_truncated"] != (log_size > _STARTUP_LOG_TAIL_LIMIT))):
        raise ValueError("startup_probe_result_invalid")
    observed = value["startup_diagnostic"]
    if observed is not None:
        if type(observed) is not dict or set(observed) != {
            "failure", "last_probe", "leader_alive", "elapsed_seconds", "probe_count",
        } or _diagnostic(VLLMStartupDiagnostic(**observed)) != observed:
            raise ValueError("startup_probe_result_invalid")
    if (value["startup_ready"] and value["cleanup_resolved"]) != (value["failure_stage"] is None):
        raise ValueError("startup_probe_result_invalid")
    return dict(value)


def execute(configuration: dict, digest: str, output: Path) -> dict:
    output = _private_output(output)
    _exclusive_json(output / "startup-probe-claim.json", {
        "schema_version": "synaptic-vllm-startup-probe-claim/v1", "configuration_sha256": digest,
    })
    started = time.monotonic()
    result = {
        "schema_version": "synaptic-vllm-startup-probe-result/v1",
        "configuration_sha256": digest,
        "startup_ready": False,
        "cleanup_resolved": False,
        "preparation_seconds": None,
        "startup_diagnostic": None,
        "failure_stage": "version",
        "startup_log_size_bytes": 0,
        "startup_log_tail_bytes": 0,
        "startup_log_truncated": False,
    }
    lease = None
    startup_log = None
    try:
        if Path(sys.executable).as_posix() != configuration["python_executable"]:
            raise ValueError("startup_probe_interpreter_mismatch")
        if metadata.version("vllm") != configuration["expected_vllm_version"]:
            raise ValueError("startup_probe_vllm_version_mismatch")
        result["failure_stage"] = "preparation"
        for name in ("model-cache", "model-destination", "model-scratch"):
            (output / name).mkdir(mode=0o700)
        from tuner.execution.providers.modal.model_snapshot import prepare_model_snapshot
        prepared = prepare_model_snapshot(
            model_ref=configuration["model"], revision=configuration["revision"], token=None,
            persistent_root=output / "model-cache", destination_root=output / "model-destination",
            scratch_root=output / "model-scratch",
        )
        result["preparation_seconds"] = _elapsed(started)
        deadline = started + configuration["lifetime_seconds"]
        if time.monotonic() >= deadline:
            raise TimeoutError("startup_probe_lifetime_expired")
        result["failure_stage"] = "startup"
        descriptor = os.open(output / "vllm-startup.log",
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        startup_log = os.fdopen(descriptor, "wb")
        lease = start_vllm_runtime(
            _startup_spec(configuration, str(prepared)), cwd=Path.cwd().resolve(),
            environment={"PATH": ":".join(dict.fromkeys((
                             str(Path(configuration["python_executable"]).parent),
                             "/usr/local/bin", "/usr/bin", "/bin"))),
                         "TMPDIR": str(output / "model-scratch"),
                         "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                         "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "VLLM_NO_USAGE_STATS": "1"},
            deadline=deadline,
            startup_log=startup_log,
        )
        result["startup_ready"] = True
    except Exception as error:
        if type(error) is VLLMRuntimeError:
            result["startup_diagnostic"] = _diagnostic(getattr(error, "startup_diagnostic", None))
            result["cleanup_resolved"] = getattr(error, "cleanup_lease", None) is None
        elif result["failure_stage"] in {"version", "preparation"}:
            result["cleanup_resolved"] = True
    finally:
        if lease is not None:
            try:
                result["cleanup_resolved"] = lease.close() is True
            except Exception:
                result["cleanup_resolved"] = False
        if startup_log is not None:
            try:
                startup_log.flush()
                os.fsync(startup_log.fileno())
            finally:
                startup_log.close()
            (result["startup_log_size_bytes"], result["startup_log_tail_bytes"],
             result["startup_log_truncated"]) = _startup_log_summary(output / "vllm-startup.log")
        if result["startup_ready"] and result["cleanup_resolved"]:
            result["failure_stage"] = None
        elif result["startup_ready"]:
            result["failure_stage"] = "cleanup"
        _exclusive_json(output / "startup-probe-result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args(argv)
    try:
        configuration = load_configuration(args.configuration)
        digest = configuration_digest(configuration)
        if args.check:
            print('{"status":"STARTUP_PROBE_CHECKED"}')
            return 0
        if args.output_directory is None:
            raise ValueError("startup_probe_output_required")
        result = execute(configuration, digest, args.output_directory)
        emit_private_startup_log_tail(args.output_directory)
        print(json.dumps({"status": "STARTUP_PROBE_SAVED", "result": result}, sort_keys=True, separators=(",", ":")))
        return 0 if result["startup_ready"] and result["cleanup_resolved"] else 1
    except Exception:
        print('{"status":"STARTUP_PROBE_UNAVAILABLE"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
