"""
LM Studio evaluation backend.

Location: /mnt/f/Code/Toolset-Training/tuner/backends/evaluation/lmstudio_backend.py
Purpose: LM Studio backend implementation for model evaluation
Used by: EvaluationBackendRegistry, eval_handler

This backend interfaces with LM Studio via its OpenAI-compatible HTTP API
to list available models and validate connectivity. LM Studio is a desktop
application for running local LLMs with a user-friendly interface.

Design decisions:
- Uses urllib.request (stdlib) to avoid external dependencies
- Calls GET /v1/models endpoint (OpenAI-compatible API)
- Uses 5s timeout for HTTP requests (faster than subprocess)
- Returns empty list on errors instead of raising exceptions (fail gracefully)
- Parses JSON response to extract model IDs
"""

import urllib.request
import json
import sys
from pathlib import Path
from typing import List, Tuple
from .base import IEvaluationBackend


def _get_lmstudio_host() -> str:
    """
    Get the appropriate host for LM Studio based on environment.

    Priority order:
    1. LMSTUDIO_HOST environment variable (user override)
    2. WSL: Windows host IP from /etc/resolv.conf
    3. Fallback to localhost

    In WSL2, localhost points to the Linux VM, not Windows where LM Studio runs.
    """
    import os

    # Check for explicit environment variable override
    env_host = os.getenv("LMSTUDIO_HOST")
    if env_host:
        return env_host

    # Check if running in WSL
    if sys.platform != "win32" and Path("/mnt/c").exists():
        # WSL detected - try to get Windows host IP
        try:
            with open("/etc/resolv.conf") as f:
                for line in f:
                    if line.startswith("nameserver"):
                        return line.split()[1]
        except Exception:
            pass
    return "localhost"


class LMStudioBackend(IEvaluationBackend):
    """
    LM Studio evaluation backend.

    Provides access to models served by LM Studio, a desktop application
    for running local LLMs with an OpenAI-compatible API.
    """

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "lmstudio"

    def list_models(self) -> List[str]:
        """
        List available LM Studio models via HTTP API.

        Calls the OpenAI-compatible endpoint GET http://localhost:1234/v1/models
        and extracts model IDs from the response.

        Expected JSON response format:
        {
            "data": [
                {"id": "model-name-1", "object": "model", ...},
                {"id": "model-name-2", "object": "model", ...}
            ]
        }

        Returns:
            List of model IDs (e.g., ['mistral-7b-instruct', 'llama-2-7b'])
            Empty list if LM Studio is not running or request fails

        Implementation notes:
        - Uses 5s timeout for HTTP request
        - Filters out empty model IDs
        - Returns empty list on any error (graceful degradation)
        - No external dependencies (uses urllib.request from stdlib)
        """
        try:
            # LM Studio OpenAI-compatible API endpoint
            host = _get_lmstudio_host()
            req = urllib.request.Request(
                f"http://{host}:1234/v1/models",
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                models = [m.get("id", "") for m in data.get("data", [])]
                return [m for m in models if m]  # Filter out empty IDs

        except urllib.error.URLError:
            # Connection refused - LM Studio not running
            return []
        except urllib.error.HTTPError:
            # HTTP error (4xx/5xx) - server error
            return []
        except json.JSONDecodeError:
            # Invalid JSON response
            return []
        except Exception:
            # Any other error - graceful degradation
            return []

    def validate_connection(self) -> Tuple[bool, str]:
        """
        Check if LM Studio server is running and accessible.

        Attempts to list models as a connectivity check. If models are
        returned, LM Studio is running. If the request succeeds but no models
        are found, LM Studio is running but has no models loaded.

        Returns:
            Tuple of (is_connected, error_message)
            - (True, "") if LM Studio is accessible
            - (False, "LM Studio running but no models found") if no models
            - (False, "LM Studio not accessible: <error>") if request fails

        Implementation notes:
        - Uses list_models() to avoid duplicating logic
        - Provides helpful error messages for common issues
        - Distinguishes between "not running" and "no models loaded"
        """
        try:
            models = self.list_models()
            if models:
                return True, ""
            else:
                # Try to determine if server is running but has no models
                # vs server not running at all
                try:
                    host = _get_lmstudio_host()
                    req = urllib.request.Request(
                        f"http://{host}:1234/v1/models",
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=5) as response:
                        # Server responded, but no models
                        return False, "LM Studio running but no models loaded"
                except urllib.error.URLError:
                    return False, "LM Studio server not running (connection refused)"
                except Exception as e:
                    return False, f"LM Studio not accessible: {e}"

        except Exception as e:
            return False, f"LM Studio not accessible: {e}"

    @property
    def default_host(self) -> str:
        """
        Default host for LM Studio server.

        In WSL2, uses Windows host IP. Otherwise uses localhost.

        Returns:
            Host string appropriate for the current environment
        """
        return _get_lmstudio_host()

    @property
    def default_port(self) -> int:
        """
        Default port for LM Studio server.

        LM Studio's default HTTP API port for OpenAI-compatible endpoints.

        Returns:
            1234
        """
        return 1234
