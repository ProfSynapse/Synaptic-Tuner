"""Verifier contract: input/output dataclasses and the Verifier protocol.

This module defines the small, stable interface that all verifiers share.
A verifier consumes a :class:`VerifierInput` (the model completion plus any
parsed/contextual signals) and produces a :class:`VerifierOutput` (a numeric
score, a pass/fail flag, and an optional detail mapping).

Typing is deliberately kept light: ``VerifierInput.parsed`` is annotated as
``Any`` (with a ``TYPE_CHECKING`` hint pointing at ``ParsedResponse``) so this
contract module never triggers a heavy import of the parsing package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shared.validation.parsing import ParsedResponse


@dataclass(frozen=True)
class VerifierInput:
    """Normalized input handed to a verifier.

    Attributes:
        completion_text: Raw model completion text being scored.
        parsed: A ``ParsedResponse`` for ``completion_text``. Typed as ``Any``
            to keep this module import-light; see the ``TYPE_CHECKING`` hint.
        prompt_text: The prompt that produced the completion, if available.
        ground_truth: Reference answer / expected tool call data.
        signals: Arbitrary extra context (e.g. flags from upstream stages).
    """

    completion_text: str
    parsed: Any
    prompt_text: str = ""
    ground_truth: Mapping[str, Any] = field(default_factory=dict)
    signals: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierOutput:
    """Result produced by a verifier.

    Attributes:
        score: Numeric score, typically in ``[0.0, 1.0]``.
        passed: Boolean pass/fail decision for the verifier.
        detail: Optional structured detail describing how the score was derived.
    """

    score: float
    passed: bool
    detail: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Verifier(Protocol):
    """Structural protocol for a verifier.

    A verifier exposes a human-readable ``name`` and a ``verify`` method that
    maps a :class:`VerifierInput` to a :class:`VerifierOutput`.
    """

    name: str

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        """Score ``sample`` and return a :class:`VerifierOutput`."""
        ...
