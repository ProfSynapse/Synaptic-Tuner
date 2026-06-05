"""Shared verifier package.

Public surface:
    - Contract types: ``VerifierInput``, ``VerifierOutput``, ``Verifier``.
    - Registry: ``VERIFIER_FACTORIES``, ``register``, ``build_verifier``.
    - Extraction: ``ExtractedAnswer``, ``extract``.
"""

from __future__ import annotations

from .contract import Verifier, VerifierInput, VerifierOutput
from .extraction import ExtractedAnswer, extract
from .registry import VERIFIER_FACTORIES, build_verifier, register

__all__ = [
    "Verifier",
    "VerifierInput",
    "VerifierOutput",
    "ExtractedAnswer",
    "extract",
    "VERIFIER_FACTORIES",
    "build_verifier",
    "register",
]
