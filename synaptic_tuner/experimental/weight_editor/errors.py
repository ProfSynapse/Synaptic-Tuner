"""Error types. Every error carries a stable machine-readable ``code``."""

from __future__ import annotations


class EditorError(Exception):
    code = "editor_error"

    def __init__(self, message: str, *, code: str | None = None, **details):
        super().__init__(message)
        if code:
            self.code = code
        self.details = details


class ConfigError(EditorError):
    code = "config_invalid"


class CapsuleError(EditorError):
    code = "capsule_invalid"


class UnsupportedCapability(EditorError):
    """A trusted port explicitly rejects a configured capability."""

    code = "unsupported_capability"


class PortError(EditorError):
    code = "port_invalid"


class RunExists(EditorError):
    code = "run_exists"


class RunNotFound(EditorError):
    code = "run_not_found"


class LockHeld(EditorError):
    code = "lock_held"


class IdentityDrift(EditorError):
    code = "identity_drift"


class JournalCorrupt(EditorError):
    code = "journal_corrupt"


class PartialTail(JournalCorrupt):
    code = "partial_tail"


class WriteBoundaryViolation(EditorError):
    code = "write_boundary_violation"


class ExecutorRejection(EditorError):
    """Executor refuses an operation with proof that no effect started."""

    code = "executor_rejected"


class AdapterFailure(EditorError):
    code = "adapter_failure"


class AdapterCancelled(AdapterFailure):
    code = "adapter_cancelled"


class AdapterTimeout(AdapterFailure):
    code = "adapter_timeout"


class PhaseError(EditorError):
    code = "phase_closed"
