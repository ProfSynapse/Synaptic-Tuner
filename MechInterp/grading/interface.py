"""
Pluggable grading interface.

The tuner defines only the contract: a grader is a callable that takes one row
dict (a per-row cell output record) and returns a grade dict, which the cell
merges back into that row. All grading semantics live in the project-supplied
callable, so the tuner ships no notion of what a "correct" or "refused" output
means.

A recipe names a grader as "module.path:callable". load_grader imports and
returns it. The example grader below is deliberately trivial: it marks a row as
positive when its generated text is non-empty. Real projects replace it with a
callable that inspects text, labels, or scores however they choose.
"""

from __future__ import annotations

import importlib
from typing import Callable, Protocol


class Grader(Protocol):
    """A grader maps a per-row output dict to a grade dict."""

    def __call__(self, row: dict) -> dict:  # pragma: no cover - protocol
        ...


def load_grader(spec: str) -> Callable[[dict], dict]:
    """Import a grader from a "module.path:callable" string.

    Raises ValueError if the spec is malformed and the usual import errors if the
    module or attribute is missing.
    """
    if ":" not in spec:
        raise ValueError(
            f"grader spec {spec!r} must be 'module.path:callable'"
        )
    module_path, _, attr = spec.partition(":")
    module = importlib.import_module(module_path)
    grader = getattr(module, attr)
    if not callable(grader):
        raise ValueError(f"grader {spec!r} is not callable")
    return grader


def example_grader(row: dict) -> dict:
    """Trivial example grader: positive when generated text is non-empty.

    Projects replace this with their own callable. The return dict is merged into
    the row, so keys here become row fields the gate evaluator can reference.
    """
    text = str(row.get("answer_text", "")).strip()
    positive = len(text) > 0
    return {"positive": positive, "grade": "nonempty" if positive else "empty"}
