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


def _is_wsl() -> bool:
    """Check if running in Windows Subsystem for Linux."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


def _get_windows_host_ip() -> str:
    """Get the Windows host IP from WSL.

    In WSL2, the Windows host is accessible via the default gateway.
    Falls back to resolv.conf nameserver if gateway detection fails.
    Returns 127.0.0.1 if not in WSL or if detection fails.
    """
    if not _is_wsl():
        return "127.0.0.1"

    import subprocess

    # Method 1: Get default gateway (most reliable for WSL2)
    try:
        result = subprocess.run(
            ["ip", "route"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split("\n"):
            if line.startswith("default"):
                parts = line.split()
                # "default via 172.x.x.1 dev eth0"
                if len(parts) >= 3 and parts[1] == "via":
                    return parts[2]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Method 2: Check /etc/resolv.conf (fallback)
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.strip().startswith("nameserver"):
                    parts = line.split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        # Skip non-local IPs (VPN, custom DNS)
                        if ip.startswith("172.") or ip.startswith("192.168."):
                            return ip
    except (FileNotFoundError, PermissionError):
        pass

    return "127.0.0.1"


def _default_host_for_windows_service() -> str:
    """Get default host for Windows-based services (LM Studio, etc.).

    Auto-detects Windows host IP when running in WSL.
    Can be overridden by setting the appropriate environment variable.
    """
    return _get_windows_host_ip()


# Default getters for Ollama
def _env_ollama_host() -> str:
    return _env_str("OLLAMA_HOST", "127.0.0.1")


def _env_ollama_port() -> int:
    return _env_int("OLLAMA_PORT", 11434)


# Default getters for LM Studio
def _env_lmstudio_host() -> str:
    # LM Studio runs on Windows, so auto-detect Windows host IP in WSL
    return _env_str("LMSTUDIO_HOST", _default_host_for_windows_service())


def _env_lmstudio_port() -> int:
    return _env_int("LMSTUDIO_PORT", 1234)


# Default getters for vLLM
def _env_vllm_host() -> str:
    return _env_str("VLLM_HOST", "127.0.0.1")


def _env_vllm_port() -> int:
    return _env_int("VLLM_PORT", 8000)


def _env_ollama_scheme() -> str:
    return _env_str("OLLAMA_SCHEME", "http")


def _env_lmstudio_scheme() -> str:
    return _env_str("LMSTUDIO_SCHEME", "http")


def _env_vllm_scheme() -> str:
    return _env_str("VLLM_SCHEME", "http")


def _env_vllm_api_key() -> str | None:
    token = os.getenv("VLLM_API_KEY")
    if token is not None:
        token = token.strip()
    return token or os.getenv("HF_TOKEN")


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
    scheme: str = field(default="http")
    api_key: str | None = field(default=None)
    auth_scheme: str = field(default="Bearer")
    host: str = field(default="127.0.0.1")
    port: int = field(default=0)
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    seed: Optional[int] = None

    def base_url(self) -> str:
        """Return the base URL for the backend API."""
        default_port = (self.scheme == "http" and self.port == 80) or (self.scheme == "https" and self.port == 443)
        if default_port:
            return f"{self.scheme}://{self.host}"
        return f"{self.scheme}://{self.host}:{self.port}"


@dataclass
class OllamaSettings(BaseBackendSettings):
    """Connection and generation parameters for Ollama.

    Environment variables:
    - OLLAMA_HOST: Server hostname (default: 127.0.0.1)
    - OLLAMA_PORT: Server port (default: 11434)
    """

    scheme: str = field(default_factory=_env_ollama_scheme)
    host: str = field(default_factory=_env_ollama_host)
    port: int = field(default_factory=_env_ollama_port)


@dataclass
class LMStudioSettings(BaseBackendSettings):
    """Connection and generation parameters for LM Studio.

    Environment variables:
    - LMSTUDIO_HOST: Server hostname (default: auto-detect Windows host in WSL)
    - LMSTUDIO_PORT: Server port (default: 1234)

    WSL Users (connecting to LM Studio on Windows):
    ------------------------------------------------
    If you get connection errors from WSL, enable local network serving:

    1. Open LM Studio on Windows
    2. Click "Developer" in the left sidebar
    3. Go to "Server" settings (or click the server icon)
    4. Toggle ON "Serve on Local Network"
    5. Note the IP address shown (e.g., 192.168.1.104)
    6. Set the environment variable:
       export LMSTUDIO_HOST=192.168.1.104

    Note: The IP address may change if your network/router assigns a new one.
    Check LM Studio's server panel for the current address.
    """

    scheme: str = field(default_factory=_env_lmstudio_scheme)
    host: str = field(default_factory=_env_lmstudio_host)
    port: int = field(default_factory=_env_lmstudio_port)


@dataclass
class VLLMSettings(BaseBackendSettings):
    """Connection and generation parameters for vLLM.

    vLLM uses an OpenAI-compatible API but may be configured with
    additional parameters like model_path for local models.

    Environment variables:
    - VLLM_HOST: Server hostname (default: 127.0.0.1)
    - VLLM_PORT: Server port (default: 8000)

    Attributes:
        model_path: Path to local model (for server startup)
        lora_adapter: Path to LoRA adapter directory (optional)
        gpu_memory_utilization: GPU memory fraction (0.0-1.0, default 0.9)
    """

    scheme: str = field(default_factory=_env_vllm_scheme)
    api_key: str | None = field(default_factory=_env_vllm_api_key)
    host: str = field(default_factory=_env_vllm_host)
    port: int = field(default_factory=_env_vllm_port)
    model_path: Optional[str] = None
    lora_adapter: Optional[str] = None
    gpu_memory_utilization: float = 0.9


@dataclass
class UnslothSettings:
    """Settings for direct Unsloth/LoRA inference.

    This loads a LoRA adapter and applies it to the base model for inference.
    No server needed - runs directly in-process.

    Attributes:
        model: Path to LoRA adapter directory (e.g., final_model/)
        max_seq_length: Maximum sequence length (default: 4096)
        load_in_4bit: Load base model in 4-bit for memory efficiency (default: True)
        temperature: Sampling temperature (0 = greedy)
        top_p: Top-p sampling
        max_tokens: Maximum tokens to generate
        seed: Optional random seed
    """

    model: str  # Path to LoRA adapter directory
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    seed: Optional[int] = None

    # Required by BackendSettings protocol but not used
    host: str = field(default="localhost")
    port: int = field(default=0)

    def base_url(self) -> str:
        """Not applicable for direct inference."""
        return f"file://{self.model}"


@dataclass
class OpenRouterSettings:
    """Settings for OpenRouter API (cloud inference).

    OpenRouter provides access to multiple model providers through a single API.
    Requires OPENROUTER_API_KEY environment variable.

    Environment variables:
    - OPENROUTER_API_KEY: Your OpenRouter API key (required)

    Attributes:
        model: Model ID (e.g., "qwen/qwen-2.5-72b-instruct", "anthropic/claude-3.5-sonnet")
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum tokens to generate
        seed: Optional random seed
    """

    model: str  # OpenRouter model ID
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    seed: Optional[int] = None

    # Required by BackendSettings protocol but not used for API
    host: str = field(default="openrouter.ai")
    port: int = field(default=443)

    def base_url(self) -> str:
        """Return OpenRouter API base URL."""
        return "https://openrouter.ai/api/v1"


def _env_mlc_host() -> str:
    return _env_str("MLC_HOST", "127.0.0.1")


def _env_mlc_port() -> int:
    return _env_int("MLC_PORT", 8080)


@dataclass
class MLCSettings:
    """Settings for MLC-LLM inference via mlc_llm serve.

    MLC-LLM runs an OpenAI-compatible server for models converted to MLC format.
    The client starts the server automatically and cleans up on exit.

    Environment variables:
    - MLC_HOST: Server hostname (default: 127.0.0.1)
    - MLC_PORT: Server port (default: 8080)

    Attributes:
        model: Path to MLC model directory (contains mlc-chat-config.json)
        host: Server hostname
        port: Server port
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum tokens to generate
        seed: Optional random seed
    """

    model: str  # Path to MLC model directory
    host: str = field(default_factory=_env_mlc_host)
    port: int = field(default_factory=_env_mlc_port)
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    seed: Optional[int] = None

    def base_url(self) -> str:
        """Return the base URL for the MLC server."""
        return f"http://{self.host}:{self.port}"


@dataclass
class LlamaCppSettings:
    """Settings for llama.cpp direct execution via llama-cli.

    Unlike other backends, llama.cpp doesn't use a server - it runs
    llama-cli directly for each inference request.

    Attributes:
        model: Path to the GGUF model file
        llama_cli_path: Path to llama-cli executable (auto-detected if not set)
        chat_template: Chat template to use (default: chatml)
        context_size: Context window size (default: 4096)
        gpu_layers: Number of layers to offload to GPU (-1 = all, default: -1)
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum tokens to generate
        seed: Optional random seed
    """

    model: str  # Path to GGUF file
    llama_cli_path: Optional[str] = None
    chat_template: str = "chatml"
    context_size: int = 4096
    gpu_layers: int = -1  # -1 = auto (all layers)
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    seed: Optional[int] = None

    # These are required by the BackendSettings protocol but not used
    host: str = field(default="localhost")
    port: int = field(default=0)

    def base_url(self) -> str:
        """Not applicable for llama.cpp (runs locally)."""
        return f"file://{self.model}"


# ---------------------------------------------------------------------------
# Judge Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalJudgeConfig:
    """Judge configuration for the Evaluator.

    Can be set globally via CLI flags, and overridden per-scenario in YAML.

    Attributes:
        enabled: Whether LLM-as-judge evaluation is active.
        mode: Composition mode ("and", "or", "judge_only") for combining
              judge and pattern-match results.
        provider: LLM provider for judge calls (defaults to eval backend).
        model: LLM model for judge calls (defaults to eval model).
        rubrics: Rubric keys to apply (empty = all available).
        rubrics_dir: Path to rubrics YAML directory.
        temperature: Sampling temperature for judge calls.
        max_tokens: Maximum tokens for judge LLM response.
        log_interactions: Whether to log judge calls for KTO training.
    """

    enabled: bool = False
    mode: str = "and"
    provider: Optional[str] = None
    model: Optional[str] = None
    rubrics: List[str] = field(default_factory=list)
    rubrics_dir: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2048
    log_interactions: bool = True


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
        judge: Judge configuration (LLM-as-judge evaluation)
    """

    prompts_path: Path
    output_path: Optional[Path] = None
    save_markdown: bool = False
    filter: PromptFilter = field(default_factory=PromptFilter)
    retries: int = 2
    request_timeout: float = 60.0
    dry_run: bool = False
    judge: EvalJudgeConfig = field(default_factory=EvalJudgeConfig)

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
