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
    label: Optional[bool] = None  # KTO training label: good (true) or bad (false)
    behavior: Optional[str] = None
    pattern: Optional[str] = None
    # LLM-judged quality labels (all optional booleans)
    excellent: Optional[bool] = None  # Exemplary/reference quality
    hallucinated: Optional[bool] = None  # Makes unsupported claims
    poor_reasoning: Optional[bool] = None  # Weak/illogical reasoning
    context_mismatch: Optional[bool] = None  # Doesn't align with context

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
        # Add LLM-judged labels if present
        if self.excellent is not None:
            result["excellent"] = self.excellent
        if self.hallucinated is not None:
            result["hallucinated"] = self.hallucinated
        if self.poor_reasoning is not None:
            result["poor_reasoning"] = self.poor_reasoning
        if self.context_mismatch is not None:
            result["context_mismatch"] = self.context_mismatch
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Example':
        """Create from dictionary."""
        return cls(
            conversations=data["conversations"],
            label=data.get("label"),
            behavior=data.get("behavior"),
            pattern=data.get("pattern"),
            excellent=data.get("excellent"),
            hallucinated=data.get("hallucinated"),
            poor_reasoning=data.get("poor_reasoning"),
            context_mismatch=data.get("context_mismatch")
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


# ============================================================================
# Labeling Mode Models
# ============================================================================


@dataclass
class LabelingConfig:
    """
    Configuration for LLM-based labeling mode.

    LLM judges examples and applies boolean quality labels.
    """
    input_file: str
    output_file: str
    backend: str = "lmstudio"  # lmstudio, openrouter, or ollama
    model: Optional[str] = None  # Optional: defaults per backend
    categories_config: str = "improvement_engine/rubrics/quality_labels.yaml"
    start_line: int = 1
    end_line: Optional[int] = None
    dry_run: bool = False
    temperature: float = 0.3
    max_tokens: int = 500

    def validate(self) -> None:
        """Validate configuration."""
        if self.start_line < 1:
            raise ValueError("Start line must be at least 1")
        if self.end_line and self.end_line < self.start_line:
            raise ValueError("End line must be >= start line")


@dataclass
class LabelingResult:
    """Result of labeling a single example."""
    line_number: int
    original: Example
    labeled: Example
    tags: List[str]
    skipped: bool
    timestamp: str

    def __str__(self) -> str:
        if self.skipped:
            return f"⊘ Line {self.line_number}: Skipped"
        else:
            tags_str = ", ".join(self.tags) if self.tags else "no labels"
            return f"✓ Line {self.line_number}: Labeled [{tags_str}]"


@dataclass
class LabelingSession:
    """Tracks labeling session progress and statistics."""
    input_file: str
    output_file: str
    total_examples: int
    labeled_count: int = 0
    skipped_count: int = 0
    last_line: int = 0
    start_time: str = ""
    categories_used: Dict[str, int] = field(default_factory=dict)  # Tag -> count

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate."""
        total_processed = self.labeled_count + self.skipped_count
        return (total_processed / self.total_examples * 100) if self.total_examples > 0 else 0.0

    @property
    def labeling_rate(self) -> float:
        """Calculate labeling rate (non-skipped)."""
        total_processed = self.labeled_count + self.skipped_count
        return (self.labeled_count / total_processed * 100) if total_processed > 0 else 0.0

    def __str__(self) -> str:
        total_processed = self.labeled_count + self.skipped_count
        return (
            f"Labeling Session:\n"
            f"  Processed: {total_processed}/{self.total_examples} "
            f"({self.completion_rate:.1f}%)\n"
            f"  Labeled: {self.labeled_count} ({self.labeling_rate:.1f}%)\n"
            f"  Skipped: {self.skipped_count}\n"
            f"  Last line: {self.last_line}"
        )
