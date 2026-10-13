"""Shared environment validation/execution primitives.

This package provides an execution runtime abstraction for evaluating
tool calls against a concrete workspace environment. It supports:

- Local temp-directory execution (default, no external dependency)
- E2B sandbox execution (remote ephemeral environments)
"""

from .types import (
    EnvironmentEpisodeTrace,
    EnvironmentIssue,
    EnvironmentStepResult,
    EnvironmentValidationResult,
    ExecutedToolCall,
)

__all__ = [
    "EnvironmentEpisodeTrace",
    "EnvironmentIssue",
    "EnvironmentSession",
    "EnvironmentStepResult",
    "ExecutedToolCall",
    "EnvironmentValidationResult",
    "EnvironmentValidator",
]


def __getattr__(name):
    """Keep optional provider dependencies out of independent runtime imports."""
    if name in {"EnvironmentSession", "EnvironmentValidator"}:
        from . import validator
        value = getattr(validator, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
