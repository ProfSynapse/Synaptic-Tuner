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
from .validator import EnvironmentSession, EnvironmentValidator

__all__ = [
    "EnvironmentEpisodeTrace",
    "EnvironmentIssue",
    "EnvironmentSession",
    "EnvironmentStepResult",
    "ExecutedToolCall",
    "EnvironmentValidationResult",
    "EnvironmentValidator",
]
