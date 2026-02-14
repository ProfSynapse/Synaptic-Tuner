"""Shared environment validation/execution primitives.

This package provides an execution runtime abstraction for evaluating
tool calls against a concrete workspace environment. It supports:

- Local temp-directory execution (default, no external dependency)
- E2B sandbox execution (remote ephemeral environments)
"""

from .types import (
    EnvironmentIssue,
    EnvironmentValidationResult,
    ExecutedToolCall,
)
from .validator import EnvironmentValidator

__all__ = [
    "EnvironmentIssue",
    "ExecutedToolCall",
    "EnvironmentValidationResult",
    "EnvironmentValidator",
]

