"""Owned startup of one bounded local vLLM OpenAI-compatible runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import json
import math
from pathlib import Path, PurePosixPath
import re
import socket
import sys
import threading
import time
from typing import Mapping

from Evaluator.owned_process import OwnedProcessLease, _UNKNOWN, _identity
from tuner.inference.serving_target import ServingTarget

__all__: list[str] = []

_MAX_NAME = 256
_MAX_REF = 512
_MAX_STARTUP_SECONDS = 1800.0
_MAX_PROBE_SECONDS = 30.0
_MAX_RESPONSE_BYTES = 1 << 20
_LORA_BASE_ALIAS = "synaptic-base"
_LOCAL_ENV_EXACT = frozenset(
    {
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVIDIA_VISIBLE_DEVICES",
        "VIRTUAL_ENV",
        "TMPDIR",
        "TMP",
        "TEMP",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "TORCH_COMPILE_DISABLE",
        "VLLM_USE_V1",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
)
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "VLLM_NO_USAGE_STATS": "1",
}


class VLLMRuntimeError(RuntimeError):
    """The runtime could not be started or retained safely."""


@dataclass(frozen=True, slots=True)
class ExplicitNetworkLoRA:
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class VerifiedLocalVLLMSource:
    target: ServingTarget


@dataclass(frozen=True, slots=True)
class ExplicitNetworkVLLMSource:
    model_ref: str
    revision: str | None = None
    tokenizer_ref: str | None = None
    lora: ExplicitNetworkLoRA | None = None


@dataclass(frozen=True, slots=True)
class VLLMStartupSpec:
    source: VerifiedLocalVLLMSource | ExplicitNetworkVLLMSource
    served_model_name: str
    host: str = "127.0.0.1"
    port: int = 8000
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    tokenizer_mode: str | None = None
    max_lora_rank: int = 64
    startup_timeout_s: float = 600.0
    readiness_request_timeout_s: float = 2.0
    python_executable: str = field(default_factory=lambda: sys.executable)


class VLLMRuntimeLease:
    """One vLLM endpoint and the exact process family that owns it."""

    def __init__(
        self,
        process: OwnedProcessLease,
        *,
        host: str,
        port: int,
        served_model_name: str,
    ) -> None:
        self._process = process
        self._host = host
        self._port = port
        self._served_model_name = served_model_name
        self._lock = threading.Lock()
        self._closed_result: bool | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def served_model_name(self) -> str:
        return self._served_model_name

    @property
    def cleanup_pending(self) -> bool:
        return self._process.cleanup_pending

    def close(self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0) -> bool:
        with self._lock:
            if self._closed_result is True:
                return True
            result = self._process.close(
                term_timeout=term_timeout, kill_timeout=kill_timeout
            )
            if result:
                self._closed_result = True
            return result

    def __enter__(self) -> "VLLMRuntimeLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            stopped = self.close()
        except BaseException:
            if exc_type is not None:
                return False
            raise
        if not stopped and exc_type is None:
            raise VLLMRuntimeError("vLLM process family cleanup remains unresolved")
        return False


def start_vllm_runtime(
    spec: VLLMStartupSpec,
    *,
    cwd: Path,
    environment: Mapping[str, str],
    deadline: float | None = None,
) -> VLLMRuntimeLease:
    """Validate, spawn, and wait for one explicitly scoped vLLM runtime."""
    absolute_deadline = _absolute_deadline(deadline)
    if absolute_deadline is not None:
        _deadline_now(absolute_deadline)
    projection = _projection(spec, cwd=cwd, environment=environment)
    if absolute_deadline is not None:
        _deadline_now(absolute_deadline)
    process: OwnedProcessLease | None = None
    try:
        if not _port_available(projection.host, projection.port):
            raise VLLMRuntimeError("managed vLLM loopback port is already in use")
        now = _startup_now(absolute_deadline)
        if absolute_deadline is not None:
            if now >= absolute_deadline:
                raise VLLMRuntimeError("vLLM startup deadline expired")
        startup_deadline = now + projection.startup_timeout_s
        if absolute_deadline is not None:
            startup_deadline = min(startup_deadline, absolute_deadline)
        process = _spawn(
            projection.argv,
            cwd=projection.cwd,
            environment=projection.environment,
        )
        while True:
            if not _leader_alive(process):
                raise VLLMRuntimeError("vLLM process ended before readiness")
            remaining = startup_deadline - _startup_now(absolute_deadline)
            if remaining <= 0:
                raise VLLMRuntimeError("vLLM readiness deadline expired")
            timeout = min(projection.readiness_request_timeout_s, remaining)
            if _ready(
                projection.host,
                projection.port,
                projection.expected_model_names,
                timeout,
            ):
                if _startup_now(
                    absolute_deadline
                ) >= startup_deadline or not _leader_alive(process):
                    raise VLLMRuntimeError("vLLM readiness was not timely and live")
                return VLLMRuntimeLease(
                    process,
                    host=projection.host,
                    port=projection.port,
                    served_model_name=projection.served_model_name,
                )
            _sleep(
                min(
                    0.05,
                    max(0.0, startup_deadline - _startup_now(absolute_deadline)),
                )
            )
    except BaseException as error:
        if process is not None:
            try:
                resolved = process.close()
            except BaseException:
                resolved = False
            if not resolved:
                error.cleanup_lease = process  # type: ignore[attr-defined]
        raise


def _absolute_deadline(value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise TypeError("vLLM startup deadline must be an exact number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("vLLM startup deadline must be finite") from None
    if not math.isfinite(result):
        raise ValueError("vLLM startup deadline must be finite")
    return result


def _deadline_now(deadline: float) -> float:
    now = _startup_now(deadline)
    if now >= deadline:
        raise VLLMRuntimeError("vLLM startup deadline expired")
    return now


def _startup_now(deadline: float | None) -> float:
    now = _monotonic()
    if deadline is not None:
        if type(now) not in (int, float):
            raise VLLMRuntimeError("vLLM monotonic clock is invalid")
        try:
            now = float(now)
        except (OverflowError, ValueError):
            raise VLLMRuntimeError("vLLM monotonic clock is invalid") from None
        if not math.isfinite(now):
            raise VLLMRuntimeError("vLLM monotonic clock is invalid")
    return now


@dataclass(frozen=True, slots=True)
class _Projection:
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    host: str
    port: int
    served_model_name: str
    expected_model_names: tuple[str, ...]
    startup_timeout_s: float
    readiness_request_timeout_s: float


def _projection(
    spec: VLLMStartupSpec, *, cwd: Path, environment: Mapping[str, str]
) -> _Projection:
    if type(spec) is not VLLMStartupSpec:
        raise TypeError("spec must be an exact VLLMStartupSpec")
    python_executable = _python_executable(spec.python_executable)
    if not isinstance(cwd, Path) or not cwd.is_absolute() or not cwd.is_dir():
        raise TypeError("cwd must be an existing absolute Path")
    name = _text(spec.served_model_name, "served model name", _MAX_NAME)
    if spec.host != "127.0.0.1":
        raise ValueError("managed vLLM must bind exact IPv4 loopback")
    if type(spec.port) is not int or not 1 <= spec.port <= 65535:
        raise ValueError("port is invalid")
    gpu = _finite(spec.gpu_memory_utilization, "GPU utilization", 0.01, 1.0)
    if (
        type(spec.tensor_parallel_size) is not int
        or not 1 <= spec.tensor_parallel_size <= 256
    ):
        raise ValueError("tensor parallel size is invalid")
    if type(spec.enforce_eager) is not bool:
        raise TypeError("enforce_eager must be bool")
    if spec.tokenizer_mode not in (None, "mistral"):
        raise ValueError("tokenizer mode is invalid")
    if type(spec.max_lora_rank) is not int or not 1 <= spec.max_lora_rank <= 1024:
        raise ValueError("maximum LoRA rank is invalid")
    startup = _finite(
        spec.startup_timeout_s, "startup timeout", 0.01, _MAX_STARTUP_SECONDS
    )
    probe = _finite(
        spec.readiness_request_timeout_s, "probe timeout", 0.01, _MAX_PROBE_SECONDS
    )
    env = _environment(environment)
    expected_names = (name,)
    argv = [
        python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--served-model-name",
        name,
        "--gpu-memory-utilization",
        str(gpu),
        "--tensor-parallel-size",
        str(spec.tensor_parallel_size),
    ]
    if spec.enforce_eager:
        argv.append("--enforce-eager")
    if spec.tokenizer_mode is not None:
        argv.extend(("--tokenizer-mode", spec.tokenizer_mode))
    if type(spec.source) is VerifiedLocalVLLMSource:
        target = spec.source.target
        if type(target) is not ServingTarget:
            raise TypeError("local source requires an exact ServingTarget")
        target.validate()
        retrieved = target.retrieved
        model_path = retrieved.root / retrieved.attempt / "model"
        tokenizer = retrieved.root / retrieved.attempt / "tokenizer"
        if target.base_snapshot is None:
            model = model_path
            adapter = None
        else:
            model = target.base_snapshot.root / target.base_snapshot.snapshot
            adapter = model_path
        # Snapshot primitive strings only after all retained identities validate.
        if adapter is not None:
            _lora_name(name)
            if name == _LORA_BASE_ALIAS:
                raise ValueError("LoRA served name collides with retained base alias")
            argv[argv.index("--served-model-name") + 1] = _LORA_BASE_ALIAS
            expected_names = tuple(sorted((_LORA_BASE_ALIAS, name)))
        argv.extend(("--model", str(model), "--tokenizer", str(tokenizer)))
        if adapter is not None:
            argv.extend(
                (
                    "--enable-lora",
                    "--max-lora-rank",
                    str(spec.max_lora_rank),
                    "--lora-modules",
                    f"{name}={adapter}",
                )
            )
        env = _local_environment(env)
    elif type(spec.source) is ExplicitNetworkVLLMSource:
        source = spec.source
        model_ref = _text(source.model_ref, "network model ref", _MAX_REF)
        argv.extend(("--model", model_ref))
        if source.revision is not None:
            argv.extend(("--revision", _text(source.revision, "revision", _MAX_REF)))
        if source.tokenizer_ref is not None:
            argv.extend(
                ("--tokenizer", _text(source.tokenizer_ref, "tokenizer ref", _MAX_REF))
            )
        if source.lora is not None:
            if type(source.lora) is not ExplicitNetworkLoRA:
                raise TypeError("network LoRA is invalid")
            lora_name = _lora_name(source.lora.name)
            if lora_name != name or name == _LORA_BASE_ALIAS:
                raise ValueError(
                    "network LoRA name must equal the distinct served model name"
                )
            if (
                not isinstance(source.lora.path, Path)
                or not source.lora.path.is_absolute()
                or not source.lora.path.is_dir()
            ):
                raise TypeError(
                    "network LoRA path must be an existing absolute directory"
                )
            argv[argv.index("--served-model-name") + 1] = _LORA_BASE_ALIAS
            expected_names = tuple(sorted((_LORA_BASE_ALIAS, name)))
            argv.extend(
                (
                    "--enable-lora",
                    "--max-lora-rank",
                    str(spec.max_lora_rank),
                    "--lora-modules",
                    f"{lora_name}={source.lora.path}",
                )
            )
    else:
        raise TypeError("startup source type is invalid")
    return _Projection(
        tuple(argv),
        cwd,
        env,
        spec.host,
        spec.port,
        name,
        expected_names,
        startup,
        probe,
    )


def _text(value: object, label: str, maximum: int) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > maximum
        or "\0" in value
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _python_executable(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 4096
        or "\0" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("Python executable is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Python executable is invalid") from None
    if len(encoded) > 4096:
        raise ValueError("Python executable is invalid")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or path.as_posix() != value
        or "//" in value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError("Python executable must be a canonical absolute POSIX path")
    return value


def _lora_name(value: object) -> str:
    name = _text(value, "LoRA name", 96)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", name) is None:
        raise ValueError("LoRA name is invalid")
    return name


def _finite(value: object, label: str, minimum: float, maximum: float) -> float:
    if (
        type(value) not in (int, float)
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise TypeError(f"{label} must be finite")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} is outside bounds")
    return result


def _environment(value: Mapping[str, str]) -> dict[str, str]:
    if type(value) is not dict:
        raise TypeError("environment must be an explicit dict")
    result = dict(value)
    if any(
        type(key) is not str
        or type(item) is not str
        or not key
        or "=" in key
        or "\0" in key + item
        for key, item in result.items()
    ):
        raise TypeError("environment is invalid")
    return result


def _local_environment(value: dict[str, str]) -> dict[str, str]:
    for key in value:
        upper = key.upper()
        if key not in _LOCAL_ENV_EXACT and key not in _OFFLINE_ENV:
            raise ValueError(
                "verified local environment contains a non-allowlisted name"
            )
        if any(
            word in upper
            for word in (
                "TOKEN",
                "SECRET",
                "PASSWORD",
                "CREDENTIAL",
                "PROXY",
                "API_KEY",
                "AUTH",
            )
        ) or upper in {
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "PYTHONINSPECT",
        }:
            raise ValueError("verified local environment contains a forbidden name")
    result = dict(value)
    result.update(_OFFLINE_ENV)
    return result


def _leader_alive(process: OwnedProcessLease) -> bool:
    identity = _identity(process.pid)
    return (
        identity is not None
        and identity is not _UNKNOWN
        and identity[0] != "Z"
        and identity[1] == process._pgid
        and identity[2] == process._start_time
    )


def _port_available(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _ready(
    host: str, port: int, expected_names: tuple[str, ...], timeout: float
) -> bool:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/v1/models", headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return False
        length = response.getheader("Content-Length")
        if length is not None:
            try:
                if int(length) > _MAX_RESPONSE_BYTES:
                    return False
            except ValueError:
                return False
        raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            return False
        payload = json.loads(raw)
        if (
            type(payload) is not dict
            or set(payload) != {"object", "data"}
            or payload["object"] != "list"
        ):
            return False
        data = payload["data"]
        if type(data) is not list or not 1 <= len(data) <= 256:
            return False
        names = []
        for item in data:
            if (
                type(item) is not dict
                or type(item.get("id")) is not str
                or not item["id"]
                or len(item["id"].encode("utf-8")) > _MAX_NAME
            ):
                return False
            names.append(item["id"])
        return tuple(sorted(names)) == expected_names
    except (OSError, http.client.HTTPException, ValueError, json.JSONDecodeError):
        return False
    finally:
        connection.close()


_spawn = OwnedProcessLease.spawn
_monotonic = time.monotonic
_sleep = time.sleep
