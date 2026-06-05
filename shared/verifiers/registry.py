"""Verifier registry: factory registration and construction by spec.

Verifier implementations register a factory under a ``type`` name via the
:func:`register` decorator. :func:`build_verifier` then constructs a verifier
from a spec mapping that carries a ``"type"`` key.

Built-in verifiers and a ``CompositeVerifier`` are intentionally NOT defined
here yet — they arrive in a later phase. This module is the minimal, functional
scaffold those builtins will plug into.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping

from .contract import Verifier

# Maps a verifier ``type`` name -> factory that builds a Verifier from a spec.
VERIFIER_FACTORIES: Dict[str, Callable[[Mapping], Verifier]] = {}


def register(type_name: str) -> Callable[[Callable[[Mapping], Verifier]], Callable[[Mapping], Verifier]]:
    """Decorator that registers a verifier factory under ``type_name``.

    Args:
        type_name: The ``type`` value used in a spec to select this factory.

    Returns:
        A decorator that registers the wrapped factory and returns it unchanged.

    Raises:
        ValueError: If ``type_name`` is empty or already registered.
    """
    if not type_name:
        raise ValueError("verifier type_name must be a non-empty string")

    def _decorator(factory: Callable[[Mapping], Verifier]) -> Callable[[Mapping], Verifier]:
        if type_name in VERIFIER_FACTORIES:
            raise ValueError(f"verifier type already registered: {type_name!r}")
        VERIFIER_FACTORIES[type_name] = factory
        return factory

    return _decorator


def build_verifier(spec: Mapping) -> Verifier:
    """Build a verifier from ``spec``.

    Args:
        spec: A mapping containing at least a ``"type"`` key naming a registered
            verifier factory. The full spec is passed through to the factory.

    Returns:
        The constructed :class:`Verifier`.

    Raises:
        ValueError: If ``spec`` has no ``"type"`` key.
        KeyError: If the ``"type"`` is not a registered factory.
    """
    try:
        type_name = spec["type"]
    except (KeyError, TypeError) as exc:
        raise ValueError("verifier spec must include a 'type' key") from exc

    factory = VERIFIER_FACTORIES.get(type_name)
    if factory is None:
        known = ", ".join(sorted(VERIFIER_FACTORIES)) or "<none registered>"
        raise KeyError(f"unknown verifier type: {type_name!r}; known types: {known}")

    return factory(spec)
