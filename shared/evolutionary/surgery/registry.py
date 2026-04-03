"""
Operation registry for surgery strategies.

Location: shared/evolutionary/surgery/registry.py
Purpose: Decorator-based static registry mapping operation names to their
         implementing classes. All operations are registered at import time.
Used by: Operation modules (decorator), LoRASurgeon (lookup), __init__.py (API).
"""

from __future__ import annotations

from typing import Dict, Type

from .base import SurgeryOperation

_OPERATION_REGISTRY: Dict[str, Type[SurgeryOperation]] = {}


def register_operation(cls: type) -> type:
    """Decorator to register an operation class by its ``name`` attribute."""
    _OPERATION_REGISTRY[cls.name] = cls
    return cls


def get_operation(name: str) -> SurgeryOperation:
    """Get an operation instance by name.

    Raises:
        ValueError: If the operation name is not registered.
    """
    if name not in _OPERATION_REGISTRY:
        raise ValueError(
            f"Unknown surgery operation: {name}. "
            f"Available: {list(_OPERATION_REGISTRY.keys())}"
        )
    return _OPERATION_REGISTRY[name]()


def list_operations() -> list[str]:
    """Return names of all registered operations."""
    return sorted(_OPERATION_REGISTRY.keys())
