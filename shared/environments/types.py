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
class EnvironmentStepResult:
    """Execution result for one assistant turn inside an environment episode."""

    turn_index: int
    executed_tools: List[ExecutedToolCall] = field(default_factory=list)
    issues: List[EnvironmentIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "executed_tools": [tool.to_dict() for tool in self.executed_tools],
            "issues": [{"level": issue.level, "message": issue.message} for issue in self.issues],
        }


@dataclass
class EnvironmentEpisodeTrace:
    """Trace of a multi-step environment episode."""

    steps: List[EnvironmentStepResult] = field(default_factory=list)
    total_turns: int = 0
    total_tool_calls: int = 0
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "stop_reason": self.stop_reason,
        }


@dataclass
class EnvironmentValidationResult:
    """Result of running tool calls in an environment runtime."""

    passed: bool
    issues: List[EnvironmentIssue] = field(default_factory=list)
    executed_tools: List[ExecutedToolCall] = field(default_factory=list)
    assertions_run: int = 0
    snapshot: Dict[str, Any] = field(default_factory=dict)
    episode_trace: Optional[EnvironmentEpisodeTrace] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "passed": self.passed,
            "issues": [{"level": i.level, "message": i.message} for i in self.issues],
            "executed_tools": [tool.to_dict() for tool in self.executed_tools],
            "assertions_run": self.assertions_run,
            "snapshot": self.snapshot,
        }
        if self.episode_trace is not None:
            payload["episode_trace"] = self.episode_trace.to_dict()
        return payload
