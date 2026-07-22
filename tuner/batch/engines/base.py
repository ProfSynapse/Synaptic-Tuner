"""Batch engine interface + shared data records.

Location: tuner/batch/engines/base.py
Purpose: The abstract contract every engine implements, plus the small
    dataclasses that flow between the runner and the engines. Keeping these
    engine-agnostic is what lets the persistence/checkpoint/CLI layers stay
    unaware of which backend produced a row.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class OutOfMemoryError(RuntimeError):
    """Backend-agnostic out-of-memory signal.

    The runner catches this to drive batch-size auto-halving without importing
    torch. Engines translate their native OOM (e.g. ``torch.cuda.OutOfMemory``)
    into this type.
    """


def hash_token_ids(token_ids: List[int]) -> str:
    """Hash a token sequence without persisting potentially sensitive ids."""
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class GenerateItem:
    """One generation request."""

    id: str
    prompt: str
    passthrough: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResult:
    """One generation result."""

    id: str
    completion_text: str
    completion_token_ids: List[int]
    prompt_token_ids_sha256: str
    prompt_token_len: int
    finish_reason: str
    passthrough: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureItem:
    """One capture request.

    Exactly one of ``text`` / ``token_ids`` carries the sequence. ``positions``
    maps a position name to an absolute token index into the tokenized sequence,
    or the literal string ``"last"`` for the final non-pad token.
    """

    id: str
    positions: Dict[str, Any]
    text: Optional[str] = None
    token_ids: Optional[List[int]] = None
    passthrough: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureResult:
    """One capture result.

    ``tensors`` maps ``"<position>__L<layer>"`` -> a 1-D hidden-state vector
    (as a plain nested list or a framework tensor; the persistence layer
    converts). ``n_layers`` counts hidden-state layers captured (embeddings +
    blocks when ``layers=all``), ``hidden_dim`` is the model hidden size.
    """

    id: str
    tensors: Dict[str, Any]
    n_layers: int
    hidden_dim: int
    positions: Dict[str, int]
    passthrough: Dict[str, Any] = field(default_factory=dict)


class GenerateEngine(ABC):
    """Prompts in, completions out."""

    @abstractmethod
    def generate(
        self,
        items: List[GenerateItem],
        *,
        batch_size: int,
        on_oom: Optional[Callable[[int, int], None]] = None,
    ) -> List[GenerateResult]:
        """Generate completions for ``items``.

        Implementations must honor ``batch_size`` and halve it on OOM down to 1,
        calling ``on_oom(old, new)`` on each halving (for the runner to warn).
        Results may be returned in any order; the runner keys everything by id.
        """

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release model/GPU resources. Optional."""

    def provenance(self) -> Dict[str, Any]:
        """Return runtime facts that should accompany generated artifacts."""
        return {}


class CaptureEngine(ABC):
    """Sequences in, per-layer hidden states at named positions out."""

    @abstractmethod
    def capture(
        self,
        items: List[CaptureItem],
        *,
        batch_size: int,
        on_oom: Optional[Callable[[int, int], None]] = None,
    ) -> List[CaptureResult]:
        """Capture hidden states for ``items`` (same batching/OOM contract)."""

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release model/GPU resources. Optional."""
