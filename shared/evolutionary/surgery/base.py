"""
SurgeryOperation protocol for the Strategy pattern.

Location: shared/evolutionary/surgery/base.py
Purpose: Define the interface that all surgery operations implement.
Used by: All operation classes in surgery/operations/, LoRASurgeon orchestrator.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OperationResult, SurgeryConfig


class SurgeryOperation(Protocol):
    """Protocol for surgery operations (Strategy pattern).

    Each operation receives an adapter path, a baseline score, a work
    directory, the surgery config, and an async evaluate function. It
    returns an OperationResult describing the best variant found.
    """

    name: str

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> OperationResult:
        """Run the operation and return the result."""
        ...
