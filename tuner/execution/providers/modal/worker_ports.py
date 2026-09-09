"""Provider-free low-level contracts for the fixed Modal worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from tuner.project.execution_source import ExecutionSourceV1

from .resolution import ModalDeploymentSelectionV1


DIAGNOSTIC_CODES = frozenset({
    "artifact_layout_collision", "artifact_layout_failed", "credential_unavailable",
    "engine_clone_failed", "engine_gitlink_mismatch", "generic_failure",
    "locked_source_mismatch", "model_preparation_failed", "model_cache_commit_failed",
    "project_clone_failed", "runtime_identity_mismatch", "runtime_artifact_precondition",
    "runtime_artifact_rejected", "runtime_evidence_rejected", "runtime_invocation_rejected",
    "runtime_lock_mismatch", "runtime_trainer_failed", "runtime_unclassified_rejection",
    "runtime_workload_rejected", "runtime_workload_document_rejected",
    "runtime_workload_engine_rejected", "runtime_workload_fingerprint_rejected",
    "runtime_workload_reconstruction_rejected", "runtime_workload_roots_rejected",
    "runtime_workload_schema_rejected", "source_topology_invalid", "trainer_invocation_failed",
    "trainer_nonzero", "worker_source_path_noncanonical", "worker_control_path_noncanonical",
    "worker_source_retain_failed", "worker_source_copy_failed", "worker_closure_rejected",
})


class ModalRemotePhaseError(Exception):
    def __init__(self, returncode: int, diagnostic_code: str) -> None:
        if returncode not in {120, 121, 122, 123, 124, 125}:
            raise ValueError("remote phase returncode is invalid")
        if diagnostic_code not in DIAGNOSTIC_CODES:
            raise ValueError("remote diagnostic code is invalid")
        super().__init__(diagnostic_code)
        self.returncode = returncode
        self.diagnostic_code = diagnostic_code


@dataclass(frozen=True, slots=True)
class ModalProcessResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("returncode must be an exact integer")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise TypeError("process output must be exact bytes")
        if self.diagnostic_code is not None and self.diagnostic_code not in DIAGNOSTIC_CODES:
            raise ValueError("process diagnostic code is invalid")


class SourceMaterializer(Protocol):
    def prepare_and_verify(
        self, source: ExecutionSourceV1, deployment: ModalDeploymentSelectionV1,
    ) -> None: ...


class FixedProcessRunner(Protocol):
    def run(
        self, argv: tuple[str, str, str], *, cwd: str,
        environment: dict[str, str], stdin: bytes,
        commit_prepared: Callable[[], None],
    ) -> ModalProcessResult: ...


__all__: list[str] = []
