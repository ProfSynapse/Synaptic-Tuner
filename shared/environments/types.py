"""Datatypes for environment-backed tool execution and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentIssue:
    """Single environment validation issue."""

    level: str
    message: str


@dataclass
class ExecutedToolCall:
    """Execution record for a single tool call."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    output: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class EnvironmentValidationResult:
    """Result of running tool calls in an environment runtime."""

    passed: bool
    issues: List[EnvironmentIssue] = field(default_factory=list)
    executed_tools: List[ExecutedToolCall] = field(default_factory=list)
    assertions_run: int = 0
    snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [{"level": i.level, "message": i.message} for i in self.issues],
            "executed_tools": [tool.to_dict() for tool in self.executed_tools],
            "assertions_run": self.assertions_run,
            "snapshot": self.snapshot,
        }

