"""Authenticated lifecycle-only JobAPI.

Administrative arbitrary-job submission is intentionally absent from this
ordinary public module, regardless of the separately entitled job schema.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .execution import (
    AccessContext, CancelResult, LogCursor, LogPage, RunRef, RunStatus,
)


@runtime_checkable
class JobAPI(Protocol):
    """Lifecycle operations over Synaptic-owned runs.

    Implementations authenticate access and authorize project/run ownership on
    every call. A provider-native identifier is never accepted.
    """

    def list(
        self,
        access: AccessContext,
        *,
        cursor: LogCursor | None = None,
        limit: int = 50,
    ) -> tuple[RunRef, ...]: ...

    def show(self, access: AccessContext, run: RunRef) -> RunStatus: ...

    def logs(
        self,
        access: AccessContext,
        run: RunRef,
        *,
        cursor: LogCursor | None = None,
        limit: int = 100,
    ) -> LogPage: ...

    def cancel(self, access: AccessContext, run: RunRef) -> CancelResult: ...


__all__ = ["JobAPI"]

