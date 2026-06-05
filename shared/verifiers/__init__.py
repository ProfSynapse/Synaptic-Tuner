"""Shared verifier package.

Public surface:
    - Contract types: ``VerifierInput``, ``VerifierOutput``, ``Verifier``.
    - Registry: ``VERIFIER_FACTORIES``, ``register``, ``build_verifier``.
    - Extraction: ``ExtractedAnswer``, ``extract``.
"""

from __future__ import annotations

from .composite import CompositeVerifier, build_composite
from .contract import Verifier, VerifierInput, VerifierOutput
from .extraction import ExtractedAnswer, extract
from .registry import VERIFIER_FACTORIES, build_verifier, register

# Import the builtins package for its registration side-effects so that the
# registry is populated on ``import shared.verifiers``.
from . import builtins  # noqa: F401,E402  (registration side-effect)

__all__ = [
    "Verifier",
    "VerifierInput",
    "VerifierOutput",
    "ExtractedAnswer",
    "extract",
    "VERIFIER_FACTORIES",
    "build_verifier",
    "register",
    "CompositeVerifier",
    "build_composite",
]
