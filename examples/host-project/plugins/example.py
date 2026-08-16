"""Minimal trusted host plug-in examples for the reference project."""

from __future__ import annotations

from typing import Any, Mapping


def prompt(row: Mapping[str, Any]) -> str:
    """Render an example prompt from host-owned configuration data."""

    return str(row.get("prompt", row.get("text", "")))


def grade(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a deterministic example grade without engine-specific imports."""

    response = str(row.get("response", "")).strip()
    return {"passed": bool(response), "score": 1.0 if response else 0.0}
