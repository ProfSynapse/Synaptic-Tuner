"""
Ollama evaluation backend.

Location: /mnt/f/Code/Toolset-Training/tuner/backends/evaluation/ollama_backend.py
Purpose: Ollama backend implementation for model evaluation
Used by: EvaluationBackendRegistry, eval_handler

This backend interfaces with Ollama via its CLI to list available models
and validate connectivity. Ollama is a local inference server that runs
quantized models optimized for consumer hardware.

Design decisions:
- Uses subprocess.run() with 10s timeout to call 'ollama list'
- Returns empty list on errors instead of raising exceptions (fail gracefully)
- Parses tabular CLI output (NAME ID SIZE MODIFIED format)
- Validates connection by attempting to list models
"""

import subprocess
from typing import List, Tuple
from .base import IEvaluationBackend


class OllamaBackend(IEvaluationBackend):
    """
    Ollama evaluation backend.

    Provides access to models served by Ollama, a local inference server
    optimized for running quantized LLMs on consumer hardware.
    """

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "ollama"

    def list_models(self) -> List[str]:
        """
        List available Ollama models via CLI.

        Executes 'ollama list' and parses the output to extract model names.
        The CLI output format is:
            NAME                ID              SIZE    MODIFIED
            model-name:tag      abc123...       4.1GB   2 hours ago

        Returns:
            List of model names (e.g., ['llama2:7b', 'mistral:latest'])
            Empty list if Ollama is not running or command fails

        Implementation notes:
        - Uses 10s timeout to prevent hanging
        - Skips header row (first line)
        - Extracts first column (model name) from each row
        - Returns empty list on any error (graceful degradation)
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return []

            models = []
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:  # Skip header
                if line.strip():
                    # Format: NAME ID SIZE MODIFIED
                    parts = line.split()
                    if parts:
                        models.append(parts[0])  # Model name
            return models
        except Exception:
            # Graceful degradation - return empty list on any error
            # (FileNotFoundError if ollama not installed, timeout, etc.)
            return []

    def validate_connection(self) -> Tuple[bool, str]:
        """
        Check if Ollama is running and accessible.

        Attempts to list models as a connectivity check. If models are
        returned, Ollama is running. If the command succeeds but no models
        are found, Ollama is running but has no models loaded.

        Returns:
            Tuple of (is_connected, error_message)
            - (True, "") if Ollama is accessible
            - (False, "Ollama running but no models found") if no models
            - (False, "Ollama not accessible: <error>") if command fails

        Implementation notes:
        - Uses list_models() to avoid duplicating logic
        - Distinguishes between "not running" and "no models loaded"
        """
        try:
            models = self.list_models()
            if models:
                return True, ""
            else:
                # Ollama might be running but have no models
                # Check if command succeeded but returned empty
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return False, "Ollama running but no models found"
                else:
                    return False, "Ollama not accessible"
        except FileNotFoundError:
            return False, "Ollama not installed (command not found)"
        except subprocess.TimeoutExpired:
            return False, "Ollama command timed out"
        except Exception as e:
            return False, f"Ollama not accessible: {e}"

    @property
    def default_host(self) -> str:
        """
        Default host for Ollama server.

        Ollama runs locally by default on loopback interface.

        Returns:
            "127.0.0.1"
        """
        return "127.0.0.1"

    @property
    def default_port(self) -> int:
        """
        Default port for Ollama server.

        Ollama's default HTTP API port.

        Returns:
            11434
        """
        return 11434
