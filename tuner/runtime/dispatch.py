"""Engine-root process dispatch with explicit writable capabilities."""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol, runtime_checkable

from tuner.cloud.runtime_layout import CloudRuntimeLayout
from tuner.training.recipes import CompiledWorkload


@dataclass(frozen=True, slots=True)
class DispatchSpec:
    workload: CompiledWorkload
    layout: CloudRuntimeLayout
    entrypoint: PurePosixPath

    def __post_init__(self) -> None:
        if not isinstance(self.workload, CompiledWorkload):
            raise TypeError("workload must be a CompiledWorkload")
        if not isinstance(self.layout, CloudRuntimeLayout):
            raise TypeError("layout must be a CloudRuntimeLayout")
        entrypoint = PurePosixPath(self.entrypoint)
        if entrypoint.is_absolute() or any(
            part in {"", ".", ".."} for part in entrypoint.parts
        ):
            raise ValueError("entrypoint must be a contained engine-relative path")
        if entrypoint.as_posix() != self.workload.entrypoint:
            raise ValueError("dispatch entrypoint must match the compiled workload")
        object.__setattr__(self, "entrypoint", entrypoint)


@dataclass(frozen=True, slots=True)
class DispatchInvocation:
    argv: tuple[str, ...]
    cwd: PurePosixPath
    environment: tuple[tuple[str, str], ...]
    stdin: bytes

    @property
    def environment_map(self) -> Mapping[str, str]:
        return dict(self.environment)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise TypeError("exit_code must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("process output must be text")


def build_dispatch_invocation(spec: DispatchSpec) -> DispatchInvocation:
    if not isinstance(spec, DispatchSpec):
        raise TypeError("spec must be a DispatchSpec")
    writable = spec.layout.writable_by_name
    engine_root = spec.layout.engine.target
    project_root = spec.layout.project.target
    entrypoint = engine_root / spec.entrypoint
    _require_safe_staged_entrypoint(spec.layout.engine.source, spec.entrypoint)
    cwd = writable["tmp"].target
    environment = {
        "SYNAPTIC_ENGINE_ROOT": engine_root.as_posix(),
        "SYNAPTIC_PROJECT_ROOT": project_root.as_posix(),
        "SYNAPTIC_ARTIFACT_ROOT": writable["artifacts"].target.as_posix(),
        "SYNAPTIC_STATE_ROOT": writable["state"].target.as_posix(),
        "SYNAPTIC_TRACKING_ROOT": writable["tracking"].target.as_posix(),
        "SYNAPTIC_CACHE_ROOT": writable["cache"].target.as_posix(),
        "SYNAPTIC_TMP_ROOT": writable["tmp"].target.as_posix(),
        "SYNAPTIC_WORKLOAD_FINGERPRINT": spec.workload.fingerprint,
        "PYTHONPATH": engine_root.as_posix(),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }
    return DispatchInvocation(
        argv=(
            "python",
            entrypoint.as_posix(),
            "--canonical-workload-stdin",
        ),
        cwd=cwd,
        environment=tuple(sorted(environment.items())),
        stdin=spec.workload.canonical_bytes,
    )


def _require_safe_staged_entrypoint(
    engine_source: Path, entrypoint: PurePosixPath
) -> None:
    root = engine_source.absolute()
    candidate = root.joinpath(*entrypoint.parts)
    current = Path(root.anchor)
    for part in (*root.parts[1:], *entrypoint.parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise ValueError("dispatch entrypoint is absent from the staged engine") from exc
        redirected = current.is_symlink() or (
            hasattr(os.path, "isjunction") and os.path.isjunction(current)
        ) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
        if redirected:
            raise ValueError("dispatch entrypoint traverses redirected staging")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("dispatch entrypoint is absent from the staged engine") from exc
    if (
        resolved_root != root
        or resolved_root not in resolved.parents
        or not stat.S_ISREG(candidate.lstat().st_mode)
    ):
        raise ValueError("dispatch entrypoint is not a contained regular file")


@runtime_checkable
class ProcessRunner(Protocol):
    def run(self, invocation: DispatchInvocation) -> ProcessResult: ...


class SubprocessRunner:
    """Concrete local/container runner; provider submission is not its concern."""

    def __init__(self, *, base_environment: Mapping[str, str] | None = None) -> None:
        self._base_environment = dict(
            os.environ if base_environment is None else base_environment
        )

    def run(self, invocation: DispatchInvocation) -> ProcessResult:
        child_environment = {
            key: value
            for key, value in self._base_environment.items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"}
        }
        child_environment.update(invocation.environment_map)
        completed = subprocess.run(
            invocation.argv,
            cwd=str(invocation.cwd),
            env=child_environment,
            input=invocation.stdin,
            capture_output=True,
            check=False,
        )
        return ProcessResult(
            exit_code=completed.returncode,
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
        )


class EngineDispatcher:
    def __init__(self, runner: ProcessRunner) -> None:
        if not isinstance(runner, ProcessRunner):
            raise TypeError("runner must implement ProcessRunner")
        self._runner = runner

    def dispatch(self, spec: DispatchSpec) -> ProcessResult:
        return self._runner.run(build_dispatch_invocation(spec))


__all__ = [
    "DispatchInvocation",
    "DispatchSpec",
    "EngineDispatcher",
    "ProcessResult",
    "ProcessRunner",
    "SubprocessRunner",
    "build_dispatch_invocation",
]
