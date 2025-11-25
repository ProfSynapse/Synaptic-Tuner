"""vLLM setup utilities for the Evaluator.

This module provides utilities for:
- Checking if vLLM is installed and ready
- Installing vLLM if needed
- Discovering training outputs (models/adapters)
- Starting and stopping the vLLM server
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default paths for training outputs
TRAINERS_DIR = Path(__file__).resolve().parent.parent / "Trainers"
SFT_OUTPUT_PATTERNS = [
    "rtx3090_sft/sft_output_rtx3090",
    "rtx3090_kto/kto_output_rtx3090",
]

# vLLM server defaults
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TrainingRun:
    """Represents a discovered training run."""
    path: Path
    name: str
    timestamp: str
    trainer_type: str  # "sft" or "kto"
    has_final_model: bool
    has_merged_16bit: bool
    has_lora: bool
    model_size: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        parts = [self.timestamp, self.trainer_type.upper()]
        if self.model_size:
            parts.append(self.model_size)
        return " - ".join(parts)

    @property
    def best_model_path(self) -> Optional[Path]:
        """Return the best available model path for inference."""
        # Prefer merged 16-bit, then final_model (LoRA)
        if self.has_merged_16bit:
            # Look for merged-16bit directory within model subdirectories
            for subdir in self.path.iterdir():
                merged = subdir / "merged-16bit"
                if merged.exists():
                    return merged
        if self.has_final_model:
            return self.path / "final_model"
        return None

    @property
    def lora_path(self) -> Optional[Path]:
        """Return LoRA adapter path if available."""
        if self.has_lora or self.has_final_model:
            final = self.path / "final_model"
            if final.exists() and (final / "adapter_config.json").exists():
                return final
        return None


@dataclass
class VLLMStatus:
    """Status of the vLLM installation and server."""
    is_installed: bool
    version: Optional[str]
    cuda_available: bool
    server_running: bool
    server_url: Optional[str]
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]


# ---------------------------------------------------------------------------
# Installation Checking
# ---------------------------------------------------------------------------

def check_vllm_installed() -> Tuple[bool, Optional[str]]:
    """Check if vLLM is installed and return version.

    Returns:
        Tuple of (is_installed, version_string)
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None


def check_cuda_available() -> Tuple[bool, Optional[str], Optional[float]]:
    """Check if CUDA is available and return GPU info.

    Returns:
        Tuple of (cuda_available, gpu_name, gpu_memory_gb)
    """
    try:
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import torch; "
                "print(torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); "
                "print(torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0)"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            available = lines[0].lower() == "true"
            gpu_name = lines[1] if len(lines) > 1 and lines[1] else None
            gpu_memory = float(lines[2]) if len(lines) > 2 and lines[2] else None
            return available, gpu_name, gpu_memory
        return False, None, None
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return False, None, None


def get_vllm_status(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> VLLMStatus:
    """Get comprehensive vLLM status.

    Args:
        host: vLLM server host
        port: vLLM server port

    Returns:
        VLLMStatus with installation and server info
    """
    is_installed, version = check_vllm_installed()
    cuda_available, gpu_name, gpu_memory = check_cuda_available()

    # Check if server is running
    server_running = False
    server_url = f"http://{host}:{port}"
    try:
        response = requests.get(f"{server_url}/v1/models", timeout=5)
        server_running = response.status_code == 200
    except requests.RequestException:
        pass

    return VLLMStatus(
        is_installed=is_installed,
        version=version,
        cuda_available=cuda_available,
        server_running=server_running,
        server_url=server_url if server_running else None,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory,
    )


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install_vllm(quiet: bool = False) -> bool:
    """Install vLLM using pip.

    Args:
        quiet: Suppress pip output

    Returns:
        True if installation succeeded
    """
    try:
        cmd = [sys.executable, "-m", "pip", "install", "vllm"]
        if quiet:
            cmd.append("-q")

        print("Installing vLLM... (this may take a few minutes)")
        result = subprocess.run(
            cmd,
            capture_output=quiet,
            timeout=600,  # 10 minute timeout
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Installation failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Training Output Discovery
# ---------------------------------------------------------------------------

def discover_training_runs(base_dir: Optional[Path] = None) -> List[TrainingRun]:
    """Discover available training runs.

    Scans the Trainers directory for SFT and KTO output directories
    and returns information about each training run.

    Args:
        base_dir: Base directory to search (defaults to Trainers/)

    Returns:
        List of TrainingRun objects, sorted by timestamp (newest first)
    """
    if base_dir is None:
        base_dir = TRAINERS_DIR

    runs: List[TrainingRun] = []

    for pattern in SFT_OUTPUT_PATTERNS:
        output_dir = base_dir / pattern
        if not output_dir.exists():
            continue

        trainer_type = "sft" if "sft" in pattern else "kto"

        # Each subdirectory is a training run (timestamp format)
        for run_dir in output_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check if it matches timestamp format (YYYYMMDD_HHMMSS)
            if not re.match(r"\d{8}_\d{6}", run_dir.name):
                continue

            # Check what's available in this run
            has_final_model = (run_dir / "final_model").exists()
            has_merged_16bit = False
            has_lora = False

            # Check for merged models in subdirectories
            for subdir in run_dir.iterdir():
                if subdir.is_dir():
                    if (subdir / "merged-16bit").exists():
                        has_merged_16bit = True
                    if (subdir / "lora").exists():
                        has_lora = True

            # Also check if final_model has LoRA adapters
            if has_final_model:
                adapter_config = run_dir / "final_model" / "adapter_config.json"
                has_lora = has_lora or adapter_config.exists()

            # Try to detect model size from config or directory name
            model_size = _detect_model_size(run_dir)

            runs.append(TrainingRun(
                path=run_dir,
                name=run_dir.name,
                timestamp=run_dir.name,
                trainer_type=trainer_type,
                has_final_model=has_final_model,
                has_merged_16bit=has_merged_16bit,
                has_lora=has_lora,
                model_size=model_size,
            ))

    # Sort by timestamp (newest first)
    runs.sort(key=lambda r: r.timestamp, reverse=True)
    return runs


def _detect_model_size(run_dir: Path) -> Optional[str]:
    """Try to detect model size from training run.

    Args:
        run_dir: Path to training run directory

    Returns:
        Model size string (e.g., "7B") or None
    """
    # Check adapter_config.json for base model info
    adapter_config = run_dir / "final_model" / "adapter_config.json"
    if adapter_config.exists():
        try:
            import json
            with open(adapter_config) as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path", "")
            # Extract size from model name
            for size in ["3b", "7b", "13b", "20b", "70b"]:
                if size in base_model.lower():
                    return size.upper()
        except Exception:
            pass
    return None


def discover_huggingface_models() -> List[str]:
    """Return list of recommended base models from HuggingFace.

    These are models known to work well with vLLM.

    Returns:
        List of model IDs
    """
    return [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/phi-2",
    ]


# ---------------------------------------------------------------------------
# Server Management
# ---------------------------------------------------------------------------

# Global to track server process
_server_process: Optional[subprocess.Popen] = None


def start_vllm_server(
    model: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    lora_modules: Optional[dict] = None,
    wait_for_ready: bool = True,
    timeout: int = 120,
    show_logs: bool = True,
) -> bool:
    """Start the vLLM server.

    Args:
        model: Model name or path
        host: Server host
        port: Server port
        gpu_memory_utilization: GPU memory fraction (0.0-1.0)
        lora_modules: Dict of lora_name -> lora_path
        wait_for_ready: Wait for server to be ready
        timeout: Timeout in seconds for server startup
        show_logs: Show live server logs during startup

    Returns:
        True if server started successfully
    """
    global _server_process

    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    # Add Mistral-specific tokenizer mode for proper [TOOL_CALLS] handling
    if "mistral" in model.lower():
        cmd.extend(["--tokenizer-mode", "mistral"])
        print("[vLLM] Using Mistral tokenizer mode for proper tool call handling")

    # Add LoRA modules if specified
    if lora_modules:
        cmd.append("--enable-lora")
        cmd.extend(["--max-lora-rank", "64"])  # Support higher LoRA ranks from training
        for name, path in lora_modules.items():
            cmd.extend(["--lora-modules", f"{name}={path}"])

    print(f"\n[vLLM] Command: {' '.join(cmd)}")

    # Set environment to disable torch.compile (avoids nvcc requirement in WSL)
    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["VLLM_USE_V1"] = "0"  # Use V0 engine which is more stable
    print("[vLLM] Disabled torch.compile for WSL compatibility\n")

    try:
        # Start server process with live output
        _server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            env=env,  # Use modified environment
        )

        if wait_for_ready:
            return _wait_for_server(host, port, timeout, show_logs=show_logs)
        return True

    except Exception as e:
        print(f"Failed to start vLLM server: {e}")
        return False


def _wait_for_server(host: str, port: int, timeout: int, show_logs: bool = True) -> bool:
    """Wait for server to become ready while showing live logs.

    Args:
        host: Server host
        port: Server port
        timeout: Timeout in seconds
        show_logs: Show live server output

    Returns:
        True if server is ready
    """
    import select

    url = f"http://{host}:{port}/v1/models"
    start_time = time.time()

    print(f"Waiting for vLLM server to start (timeout: {timeout}s)...")
    if show_logs:
        print("-" * 60)
        print("[vLLM Server Output]")
        print("-" * 60)

    while time.time() - start_time < timeout:
        # Read and display any available output from vLLM
        if show_logs and _server_process and _server_process.stdout:
            try:
                # Use select to check if there's data to read (non-blocking)
                import os
                import fcntl

                # Make stdout non-blocking
                fd = _server_process.stdout.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

                # Read available lines
                while True:
                    try:
                        line = _server_process.stdout.readline()
                        if line:
                            print(f"[vLLM] {line.rstrip()}")
                        else:
                            break
                    except (IOError, BlockingIOError):
                        break
            except Exception:
                pass  # Continue even if log reading fails

        # Check if server is ready
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                if show_logs:
                    print("-" * 60)
                print("\nvLLM server is ready!")
                return True
        except requests.RequestException:
            pass

        # Check if process died
        if _server_process and _server_process.poll() is not None:
            # Read any remaining output
            if show_logs and _server_process.stdout:
                remaining = _server_process.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        print(f"[vLLM] {line}")
                print("-" * 60)
            print("\nvLLM server process exited unexpectedly!")
            print(f"Exit code: {_server_process.returncode}")
            return False

        time.sleep(1)

    # Timeout - show any remaining output
    if show_logs:
        print("-" * 60)
    print(f"\nTimeout waiting for vLLM server (>{timeout}s)")
    print("The server may still be loading the model. Check GPU memory usage.")
    return False


def stop_vllm_server() -> bool:
    """Stop the vLLM server if running.

    Returns:
        True if server was stopped
    """
    global _server_process

    if _server_process is None:
        return True

    try:
        _server_process.terminate()
        _server_process.wait(timeout=10)
        _server_process = None
        return True
    except subprocess.TimeoutExpired:
        _server_process.kill()
        _server_process = None
        return True
    except Exception as e:
        print(f"Error stopping server: {e}")
        return False


def is_server_managed() -> bool:
    """Check if we're managing the server process.

    Returns:
        True if server was started by this module
    """
    return _server_process is not None


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def format_gpu_info(status: VLLMStatus) -> str:
    """Format GPU information for display.

    Args:
        status: VLLMStatus object

    Returns:
        Formatted string
    """
    if not status.cuda_available:
        return "No CUDA GPU detected"

    parts = []
    if status.gpu_name:
        parts.append(status.gpu_name)
    if status.gpu_memory_gb:
        parts.append(f"{status.gpu_memory_gb:.1f} GB")

    return " - ".join(parts) if parts else "CUDA available"


def estimate_memory_usage(model_size: str) -> float:
    """Estimate GPU memory usage for a model size.

    Args:
        model_size: Model size (e.g., "7B", "13B")

    Returns:
        Estimated memory in GB
    """
    # Rough estimates for fp16 with KV cache
    estimates = {
        "3B": 8.0,
        "7B": 16.0,
        "8B": 18.0,
        "13B": 28.0,
        "20B": 42.0,
        "70B": 140.0,
    }
    return estimates.get(model_size.upper(), 20.0)
