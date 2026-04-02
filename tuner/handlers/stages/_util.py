"""Shared utilities for experiment stage runners.

Located at tuner/handlers/stages/_util.py.
Provides helper functions used across multiple stage runner modules.
"""

from __future__ import annotations


def _optional_backend_value(value) -> str | None:
    """Return a backend metadata value only when it is a real non-empty string."""
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None
