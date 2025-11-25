"""Runtime configuration primitives for the Evaluator.

This module defines configuration dataclasses for backend settings,
prompt filtering, and evaluator configuration. Uses inheritance to
reduce duplication between backend settings classes.
"""
from __future__ import annotations

import os
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .enums import BackendType


# ---------------------------------------------------------------------------
# Environment Variable Helpers
# ---------------------------------------------------------------------------

def _env_str(var_name: str, default: str) -> str:
    """Get string value from environment variable with default."""
    return os.getenv(var_name, default)


def _env_int(var_name: str, default: int) -> int:
    """Get integer value from environment variable with default."""
    raw = os.getenv(var_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be an integer") from exc


# Default getters for Ollama
def _env_ollama_host() -> str:
    return _env_str("OLLAMA_HOST", "127.0.0.1")


def _env_ollama_port() -> int:
    return _env_int("OLLAMA_PORT", 11434)


# Default getters for LM Studio
def _env_lmstudio_host() -> str:
    return _env_str("LMSTUDIO_HOST", "127.0.0.1")


def _env_lmstudio_port() -> int:
    return _env_int("LMSTUDIO_PORT", 1234)


# ---------------------------------------------------------------------------
# Backend Settings Classes
# ---------------------------------------------------------------------------

@dataclass
class BaseBackendSettings(ABC):
    """Base class for backend connection and generation parameters.

    Provides common fields shared by all backend settings:
    - model: The model identifier
    - host: Server hostname
    - port: Server port
    - temperature: Sampling temperature
    - top_p: Top-p (nucleus) sampling
    - max_tokens: Maximum output tokens
    - seed: Optional random seed for reproducibility

    Subclasses should override _default_host() and _default_port() to
    provide backend-specific defaults from environment variables.
    """

    model: str
    host: str = field(default="127.0.0.1")
    port: int = field(default=0)
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    seed: Optional[int] = None

    def base_url(self) -> str:
        """Return the base URL for the backend API."""
        return f"http://{self.host}:{self.port}"


@dataclass
class OllamaSettings(BaseBackendSettings):
    """Connection and generation parameters for Ollama.

    Environment variables:
    - OLLAMA_HOST: Server hostname (default: 127.0.0.1)
    - OLLAMA_PORT: Server port (default: 11434)
    """

    host: str = field(default_factory=_env_ollama_host)
    port: int = field(default_factory=_env_ollama_port)


@dataclass
class LMStudioSettings(BaseBackendSettings):
    """Connection and generation parameters for LM Studio.

    Environment variables:
    - LMSTUDIO_HOST: Server hostname (default: 127.0.0.1)
    - LMSTUDIO_PORT: Server port (default: 1234)
    """

    host: str = field(default_factory=_env_lmstudio_host)
    port: int = field(default_factory=_env_lmstudio_port)


# ---------------------------------------------------------------------------
# Prompt Filtering
# ---------------------------------------------------------------------------

@dataclass
class PromptFilter:
    """Filtering constraints for prompt sets.

    Attributes:
        tags: Tags that must ALL be present (AND semantics)
        limit: Maximum number of prompts to include
    """

    tags: Sequence[str] = ()
    limit: Optional[int] = None

    def matches(self, prompt_tags: Iterable[str]) -> bool:
        """Check if prompt tags satisfy the filter.

        Args:
            prompt_tags: Tags from the prompt case

        Returns:
            True if all filter tags are present in prompt_tags
        """
        if not self.tags:
            return True
        prompt_set = set(prompt_tags)
        return all(tag in prompt_set for tag in self.tags)


# ---------------------------------------------------------------------------
# Evaluator Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorConfig:
    """Full evaluator configuration.

    Attributes:
        prompts_path: Path to the prompt set file (JSON or JSONL)
        output_path: Path for JSON output (optional)
        save_markdown: Whether to save markdown report
        filter: Prompt filtering configuration
        retries: HTTP retry attempts
        request_timeout: HTTP timeout in seconds
        dry_run: Skip backend calls (for testing)
    """

    prompts_path: Path
    output_path: Optional[Path] = None
    save_markdown: bool = False
    filter: PromptFilter = field(default_factory=PromptFilter)
    retries: int = 2
    request_timeout: float = 60.0
    dry_run: bool = False

    def validate(self) -> None:
        """Validate the configuration.

        Raises:
            FileNotFoundError: If prompts_path doesn't exist
            ValueError: If configuration values are invalid
        """
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompt set not found: {self.prompts_path}")
        if self.output_path and self.output_path.is_dir():
            raise ValueError("output_path must be a file, not a directory")
        if self.retries < 0:
            raise ValueError("retries must be >= 0")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")

    def ensure_output_parent(self) -> None:
        """Create parent directory for output path if needed."""
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Path Utilities
# ---------------------------------------------------------------------------

def expand_path(raw: str) -> Path:
    """Expand user home and environment variables in a path.

    Args:
        raw: Path string that may contain ~ or $VAR

    Returns:
        Resolved absolute Path
    """
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def parse_tags(raw: Optional[str]) -> List[str]:
    """Parse comma-separated tag string.

    Args:
        raw: Comma-separated tag string (e.g., "tag1,tag2,tag3")

    Returns:
        List of trimmed, non-empty tags
    """
    if not raw:
        return []
    return [tag.strip() for tag in raw.split(",") if tag.strip()]
