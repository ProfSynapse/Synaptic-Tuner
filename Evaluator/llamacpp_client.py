"""llama.cpp client for direct GGUF model inference.

This module provides a client that runs llama-cli directly for inference,
bypassing the need for a server. This is useful for evaluating GGUF models
without setting up Ollama or other inference servers.
"""
from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence, Optional

from .config import LlamaCppSettings
from .protocols import BackendResponse, BackendError


def _get_build_instructions() -> str:
    """Get platform-specific llama.cpp build instructions."""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        # Apple Silicon - use Metal
        return (
            "cd Trainers/llama.cpp && "
            "cmake -B build -DGGML_METAL=ON && "
            "cmake --build build --config Release"
        )
    elif system == "Darwin":
        # Intel Mac - CPU only
        return (
            "cd Trainers/llama.cpp && "
            "cmake -B build && "
            "cmake --build build --config Release"
        )
    elif system == "Linux" or system == "Windows":
        # Assume NVIDIA GPU available
        return (
            "cd Trainers/llama.cpp && "
            "cmake -B build -DGGML_CUDA=ON && "
            "cmake --build build --config Release"
        )
    else:
        return (
            "cd Trainers/llama.cpp && "
            "cmake -B build && "
            "cmake --build build --config Release"
        )


def _find_llama_cli() -> Optional[Path]:
    """Find llama-cli executable in common locations."""
    # Check common paths
    candidates = [
        # Relative to repo root (Trainers/llama.cpp)
        Path(__file__).parent.parent / "Trainers" / "llama.cpp" / "build" / "bin" / "llama-cli",
        Path(__file__).parent.parent / "Trainers" / "llama.cpp" / "llama-cli",
        # Home directory
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli",
        Path.home() / "llama.cpp" / "llama-cli",
        # System paths (if installed)
        Path("/usr/local/bin/llama-cli"),
        Path("/opt/homebrew/bin/llama-cli"),
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


class LlamaCppClient:
    """Client for llama.cpp inference via llama-cli.

    This client executes llama-cli as a subprocess for each inference request.
    It supports chat templates and various generation parameters.

    Example:
        settings = LlamaCppSettings(
            model="/path/to/model.gguf",
            chat_template="chatml"
        )
        client = LlamaCppClient(settings=settings)
        response = client.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ])
    """

    def __init__(
        self,
        settings: LlamaCppSettings,
        timeout: float = 120.0,
        retries: int = 2,
    ):
        """Initialize the llama.cpp client.

        Args:
            settings: LlamaCppSettings with model path and parameters
            timeout: Maximum time to wait for inference (seconds)
            retries: Number of retry attempts (not used, kept for interface compatibility)
        """
        self.settings = settings
        self.timeout = timeout
        self.retries = retries

        # Find llama-cli
        if settings.llama_cli_path:
            self.llama_cli = Path(settings.llama_cli_path)
            if not self.llama_cli.exists():
                raise BackendError(f"llama-cli not found at: {settings.llama_cli_path}")
        else:
            self.llama_cli = _find_llama_cli()
            if not self.llama_cli:
                build_cmd = _get_build_instructions()
                raise BackendError(
                    f"llama-cli not found. Build llama.cpp first:\n  {build_cmd}"
                )

        # Verify model exists
        model_path = Path(settings.model)
        if not model_path.exists():
            raise BackendError(f"Model not found: {settings.model}")

    def chat(self, messages: Sequence[Mapping[str, str]]) -> BackendResponse:
        """Send a chat conversation to llama-cli.

        Args:
            messages: Sequence of message dicts with 'role' and 'content' keys

        Returns:
            BackendResponse with the model's response

        Raises:
            BackendError: If llama-cli execution fails
        """
        start_time = time.time()

        # Extract system prompt and user message
        system_prompt = None
        user_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                user_messages.append(content)
            elif role == "assistant":
                # For multi-turn, we'd need to handle this differently
                # For now, we only support single-turn evaluation
                pass

        # Build the prompt - for single turn, take the last user message
        user_prompt = user_messages[-1] if user_messages else ""

        # Build command
        cmd = [
            str(self.llama_cli),
            "-m", str(self.settings.model),
            "-c", str(self.settings.context_size),
            "-n", str(self.settings.max_tokens),
            "--chat-template", self.settings.chat_template,
            "-st",  # Single turn mode
            "--no-display-prompt",  # Don't echo the prompt
            "--no-show-timings",  # Cleaner output
            "--temp", str(self.settings.temperature),
            "--top-p", str(self.settings.top_p),
        ]

        # Add GPU layers
        if self.settings.gpu_layers == -1:
            cmd.extend(["-ngl", "99"])  # Offload all layers
        elif self.settings.gpu_layers > 0:
            cmd.extend(["-ngl", str(self.settings.gpu_layers)])

        # Add seed if specified
        if self.settings.seed is not None:
            cmd.extend(["--seed", str(self.settings.seed)])

        # Add system prompt if present
        if system_prompt:
            cmd.extend(["-sys", system_prompt])

        # Add user prompt
        cmd.extend(["-p", user_prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            latency = time.time() - start_time

            if result.returncode != 0:
                # Extract error message
                error_msg = result.stderr.strip() or f"llama-cli exited with code {result.returncode}"
                raise BackendError(f"llama-cli error: {error_msg}")

            # Parse output - llama-cli outputs the response directly
            # Filter out metal initialization messages
            output_lines = []
            for line in result.stdout.split("\n"):
                # Skip llama.cpp diagnostic lines
                if any(skip in line for skip in [
                    "ggml_metal",
                    "llama_model_loader",
                    "llm_load_",
                    "llama_new_context",
                    "llama_kv_cache",
                    "llama_perf_",
                    "main: ",
                    "sampling: ",
                    "sampler ",
                    "generate: ",
                ]):
                    continue
                output_lines.append(line)

            response_text = "\n".join(output_lines).strip()

            return BackendResponse(
                message=response_text,
                raw={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "command": " ".join(cmd[:6]) + "...",  # Truncated for readability
                },
                latency_s=latency,
            )

        except subprocess.TimeoutExpired:
            raise BackendError(f"llama-cli timed out after {self.timeout}s")
        except FileNotFoundError:
            raise BackendError(f"llama-cli not found at: {self.llama_cli}")
        except Exception as e:
            raise BackendError(f"llama-cli execution failed: {e}")
