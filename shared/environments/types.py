"""Datatypes for environment-backed tool execution and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentIssue:
    """Single environment validation issue."""

    level: str
    message: str
    code: Optional[str] = None
    recoverable: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "level": self.level,
            "message": self.message,
        }
        if self.code is not None:
            payload["code"] = self.code
        if self.recoverable is not None:
            payload["recoverable"] = self.recoverable
        return payload


@dataclass
class ExecutedToolCall:
    """Execution record for a single tool call."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    output: Optional[str] = None
    error: Optional[str] = None
    recoverable: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
            "output": self.output,
            "error": self.error,
        }
        if self.recoverable is not None:
            payload["recoverable"] = self.recoverable
        return payload


@dataclass
class EnvironmentStepResult:
    """Execution result for one assistant turn inside an environment episode."""

    turn_index: int
    executed_tools: List[ExecutedToolCall] = field(default_factory=list)
    issues: List[EnvironmentIssue] = field(default_factory=list)
    observation_type: str = "tool_results"
    hard_error: bool = False
    recoverable_error: bool = False
    state_changed: bool = False
    action_signature: Optional[str] = None
    issue_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "executed_tools": [tool.to_dict() for tool in self.executed_tools],
            "issues": [issue.to_dict() for issue in self.issues],
            "observation_type": self.observation_type,
            "hard_error": self.hard_error,
            "recoverable_error": self.recoverable_error,
            "state_changed": self.state_changed,
            "action_signature": self.action_signature,
            "issue_signature": self.issue_signature,
        }


@dataclass
class EnvironmentEpisodeTrace:
    """Trace of a multi-step environment episode."""

    steps: List[EnvironmentStepResult] = field(default_factory=list)
    total_turns: int = 0
    total_tool_calls: int = 0
    stop_reason: Optional[str] = None
    hard_failure: bool = False
    recovered_after_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "stop_reason": self.stop_reason,
            "hard_failure": self.hard_failure,
            "recovered_after_error": self.recovered_after_error,
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
            "issues": [i.to_dict() for i in self.issues],
            "executed_tools": [tool.to_dict() for tool in self.executed_tools],
            "assertions_run": self.assertions_run,
            "snapshot": self.snapshot,
        }
        if self.episode_trace is not None:
            payload["episode_trace"] = self.episode_trace.to_dict()
        return payload
