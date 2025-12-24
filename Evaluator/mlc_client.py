"""MLC-LLM client for WebGPU/MLC model inference.

This module provides a client that runs mlc_llm serve as a background process
and connects via OpenAI-compatible API. This enables evaluation of MLC-converted
models without a browser.

The client:
1. Starts mlc_llm serve in background
2. Waits for server readiness
3. Sends requests via OpenAI-compatible API
4. Cleans up server on exit
"""
from __future__ import annotations

import atexit
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import requests

from .config import MLCSettings
from .protocols import BackendResponse, BackendError


# WSL patch code to avoid permission errors on Windows paths
WSL_PATCH_CODE = '''
import os
import pathlib

# Clean PATH to avoid WSL permission errors, but keep CUDA
linux_paths = [p for p in os.environ.get('PATH', '').split(':')
               if not p.startswith('/mnt/c') and not p.startswith('/mnt/f')]
# Ensure CUDA is in PATH for nvcc
cuda_paths = ['/usr/local/cuda/bin', '/usr/local/cuda-12.8/bin', '/usr/local/cuda-12/bin']
for cuda_path in cuda_paths:
    if os.path.isdir(cuda_path) and cuda_path not in linux_paths:
        linux_paths.insert(0, cuda_path)
os.environ['PATH'] = ':'.join(linux_paths)

# Patch pathlib
_original_is_dir = pathlib.Path.is_dir
def _safe_is_dir(self):
    try:
        return _original_is_dir(self)
    except PermissionError:
        return False
pathlib.Path.is_dir = _safe_is_dir

_original_stat = pathlib.Path.stat
def _safe_stat(self, *args, **kwargs):
    try:
        return _original_stat(self, *args, **kwargs)
    except PermissionError:
        raise FileNotFoundError(f"Skipped: {self}")
pathlib.Path.stat = _safe_stat
'''


def _find_mlc_python() -> Optional[str]:
    """Find Python executable with MLC-LLM installed."""
    candidates = [
        "/home/profsynapse/miniconda3/bin/python",
        os.path.expanduser("~/miniconda3/bin/python"),
        os.path.expanduser("~/anaconda3/bin/python"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _find_mlc_model(model_path: str) -> Optional[Path]:
    """Find MLC model config file.

    Args:
        model_path: Path to MLC model directory or mlc-chat-config.json

    Returns:
        Path to mlc-chat-config.json or model directory
    """
    path = Path(model_path)

    # Direct path to config
    if path.name == "mlc-chat-config.json" and path.exists():
        return path.parent

    # Directory containing config
    if path.is_dir():
        config = path / "mlc-chat-config.json"
        if config.exists():
            return path
        # Check subdirectories (e.g., model-q4f16_1-MLC/)
        for subdir in path.iterdir():
            if subdir.is_dir():
                config = subdir / "mlc-chat-config.json"
                if config.exists():
                    return subdir

    return None


class MLCClient:
    """Client for MLC-LLM inference via mlc_llm serve.

    This client manages the MLC serve process lifecycle:
    1. Starts the server on initialization
    2. Provides chat() method for inference
    3. Cleans up server on exit

    Example:
        settings = MLCSettings(
            model="/path/to/model-q4f16_1-MLC",
            port=8080
        )
        client = MLCClient(settings=settings)
        response = client.chat([
            {"role": "user", "content": "Hello!"}
        ])
    """

    def __init__(
        self,
        settings: MLCSettings,
        timeout: float = 120.0,
        retries: int = 2,
    ):
        """Initialize the MLC client and start server.

        Args:
            settings: MLCSettings with model path and parameters
            timeout: Maximum time to wait for inference (seconds)
            retries: Number of retry attempts for requests
        """
        self.settings = settings
        self.timeout = timeout
        self.retries = retries
        self._server_process: Optional[subprocess.Popen] = None
        self._session = requests.Session()

        # Find MLC Python
        self._mlc_python = _find_mlc_python()
        if not self._mlc_python:
            raise BackendError(
                "MLC-LLM Python not found. Install with:\n"
                "  pip install --pre -U -f https://mlc.ai/wheels "
                "mlc-llm-nightly-cu128 mlc-ai-nightly-cu128"
            )

        # Validate model path
        self._model_path = _find_mlc_model(settings.model)
        if not self._model_path:
            raise BackendError(
                f"MLC model not found: {settings.model}\n"
                "Expected directory with mlc-chat-config.json"
            )

        # Start server
        self._start_server()

        # Register cleanup
        atexit.register(self._cleanup)

    def _start_server(self) -> None:
        """Start the MLC serve process."""
        serve_script = WSL_PATCH_CODE + f'''
from mlc_llm.cli.serve import main
main(['{self._model_path}', '--port', '{self.settings.port}', '--host', '{self.settings.host}'])
'''

        # Start server in background
        self._server_process = subprocess.Popen(
            [self._mlc_python, "-c", serve_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.expanduser("~"),
        )

        # Wait for server to be ready
        base_url = f"http://{self.settings.host}:{self.settings.port}"
        max_wait = 60  # seconds
        start = time.time()

        while time.time() - start < max_wait:
            try:
                # Check if process died
                if self._server_process.poll() is not None:
                    stderr = self._server_process.stderr.read().decode() if self._server_process.stderr else ""
                    raise BackendError(f"MLC server failed to start: {stderr[:500]}")

                # Try to connect
                resp = self._session.get(f"{base_url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    return  # Server is ready
            except requests.exceptions.ConnectionError:
                pass
            except requests.exceptions.Timeout:
                pass

            time.sleep(0.5)

        # Timeout - cleanup and raise
        self._cleanup()
        raise BackendError(f"MLC server failed to start within {max_wait}s")

    def _cleanup(self) -> None:
        """Stop the MLC serve process."""
        if self._server_process and self._server_process.poll() is None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            except Exception:
                pass
        self._server_process = None

    def chat(self, messages: Sequence[Mapping[str, str]]) -> BackendResponse:
        """Send a chat conversation to MLC server.

        Args:
            messages: Sequence of message dicts with 'role' and 'content' keys

        Returns:
            BackendResponse with the model's response

        Raises:
            BackendError: If request fails
        """
        start_time = time.time()

        base_url = f"http://{self.settings.host}:{self.settings.port}"
        url = f"{base_url}/v1/chat/completions"

        payload = {
            "model": "default",  # MLC serve uses "default" for the loaded model
            "messages": [dict(m) for m in messages],
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
        }

        if self.settings.seed is not None:
            payload["seed"] = self.settings.seed

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                latency = time.time() - start_time

                if resp.status_code != 200:
                    raise BackendError(
                        f"MLC server returned {resp.status_code}: {resp.text[:200]}"
                    )

                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                return BackendResponse(
                    message=content,
                    raw=data,
                    latency_s=latency,
                )

            except requests.exceptions.Timeout:
                last_error = BackendError(f"Request timed out after {self.timeout}s")
            except requests.exceptions.ConnectionError as e:
                last_error = BackendError(f"Connection error: {e}")
            except json.JSONDecodeError as e:
                last_error = BackendError(f"Invalid JSON response: {e}")
            except KeyError as e:
                last_error = BackendError(f"Unexpected response format: missing {e}")
            except Exception as e:
                last_error = BackendError(f"Request failed: {e}")

            if attempt < self.retries:
                time.sleep(1)  # Brief pause before retry

        raise last_error

    def __del__(self):
        """Cleanup on garbage collection."""
        self._cleanup()
