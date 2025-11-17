"""Runtime configuration primitives for the Evaluator."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def _env_host() -> str:
    return os.getenv("OLLAMA_HOST", "127.0.0.1")


def _env_port() -> int:
    raw = os.getenv("OLLAMA_PORT")
    if raw is None:
        return 11434
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("OLLAMA_PORT must be an integer") from exc


@dataclass
class OllamaSettings:
    """Connection + generation parameters for Ollama."""

    model: str
    host: str = field(default_factory=_env_host)
    port: int = field(default_factory=_env_port)
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    seed: Optional[int] = None

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class PromptFilter:
    """Filtering constraints for prompt sets."""

    tags: Sequence[str] = ()
    limit: Optional[int] = None

    def matches(self, prompt_tags: Iterable[str]) -> bool:
        if not self.tags:
            return True
        prompt_set = set(prompt_tags)
        return all(tag in prompt_set for tag in self.tags)


@dataclass
class EvaluatorConfig:
    """Full evaluator configuration."""

    prompts_path: Path
    output_path: Optional[Path] = None
    save_markdown: bool = False
    filter: PromptFilter = field(default_factory=PromptFilter)
    retries: int = 2
    request_timeout: float = 60.0
    dry_run: bool = False

    def validate(self) -> None:
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompt set not found: {self.prompts_path}")
        if self.output_path and self.output_path.is_dir():
            raise ValueError("output_path must be a file, not a directory")
        if self.retries < 0:
            raise ValueError("retries must be >= 0")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")

    def ensure_output_parent(self) -> None:
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)


def expand_path(raw: str) -> Path:
    """Expand user/relative paths helper."""
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tag.strip() for tag in raw.split(",") if tag.strip()]
