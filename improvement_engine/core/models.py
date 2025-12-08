"""Data models for improvement engine."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ThinkingBlock:
    """Represents a thinking block in an example."""
    goal: str
    memory: str
    requirements: List[str]
    assessment: Dict[str, bool]
    confidence: float
    plan: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal": self.goal,
            "memory": self.memory,
            "requirements": self.requirements,
            "assessment": self.assessment,
            "confidence": self.confidence,
            "plan": self.plan
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThinkingBlock':
        """Create from dictionary."""
        return cls(
            goal=data["goal"],
            memory=data["memory"],
            requirements=data["requirements"],
            assessment=data["assessment"],
            confidence=data["confidence"],
            plan=data["plan"]
        )


@dataclass
class Example:
    """Represents a complete training example."""
    conversations: List[Dict[str, str]]
    label: Optional[bool] = None  # Optional for thinking datasets
    behavior: Optional[str] = None
    pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "conversations": self.conversations
        }
        if self.label is not None:
            result["label"] = self.label
        if self.behavior:
            result["behavior"] = self.behavior
        if self.pattern:
            result["pattern"] = self.pattern
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Example':
        """Create from dictionary."""
        return cls(
            conversations=data["conversations"],
            label=data.get("label"),  # Optional for thinking datasets
            behavior=data.get("behavior"),
            pattern=data.get("pattern")
        )


@dataclass
class ImprovementResult:
    """Result of improving a single example."""
    line_number: int
    original: Example
    improved: Optional[Example]
    success: bool
    error: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.success:
            return f"✓ Line {self.line_number}: Improved"
        else:
            return f"✗ Line {self.line_number}: {self.error}"


@dataclass
class BatchResult:
    """Result of processing a batch."""
    batch_number: int
    start_line: int
    end_line: int
    results: List[ImprovementResult]
    total_processed: int
    successful: int
    failed: int
    duration_seconds: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return (self.successful / self.total_processed * 100) if self.total_processed > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Batch {self.batch_number} (lines {self.start_line}-{self.end_line}): "
            f"{self.successful}/{self.total_processed} improved "
            f"({self.success_rate:.1f}%) in {self.duration_seconds:.1f}s"
        )


@dataclass
class ImprovementConfig:
    """
    Configuration for improvement process.

    Note: API keys and host configurations are in .env file.
    Backend and model are runtime choices (CLI flags or interactive).
    """
    input_file: str
    output_file: str
    backend: str = "openrouter"  # openrouter, lmstudio, or ollama
    model: Optional[str] = None  # Optional: defaults per backend
    batch_size: int = 10
    start_line: int = 1
    end_line: Optional[int] = None
    dry_run: bool = False
    temperature: float = 0.3
    max_tokens: int = 2048

    def validate(self) -> None:
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if self.start_line < 1:
            raise ValueError("Start line must be at least 1")
        if self.end_line and self.end_line < self.start_line:
            raise ValueError("End line must be >= start line")
