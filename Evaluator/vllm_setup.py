"""vLLM setup utilities for the Evaluator.

This module provides utilities for:
- Checking if vLLM is installed and ready
- Installing vLLM if needed
- Discovering training outputs (models/adapters)
- Resolving explicit operator-selected runtime configuration
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from shared.utilities.paths import iter_training_output_dirs
from tuner.project import ProjectContext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default paths for training outputs
TRAINERS_DIR = Path(__file__).resolve().parent.parent / "Trainers"

# The methods vLLM can actually SERVE for evaluation — intentionally NOT the
# TRAINING_METHODS SSOT (shared/utilities/paths.py). vLLM serves causal-LM adapters
# only (sft/kto); grpo/dpo/embedding/ace_step are not vLLM-servable here. Named
# distinctly so it can never be mistaken for, or drift against, the SSOT constant.
VLLM_SERVABLE_METHODS = ("sft", "kto")

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

def discover_training_runs(
    base_dir: Optional[Path] = None,
    *,
    project_context: ProjectContext | None = None,
) -> List[TrainingRun]:
    """Discover available training runs.

    Scans the Trainers directory for SFT and KTO output directories
    and returns information about each training run.

    Args:
        base_dir: Base directory to search (defaults to Trainers/)

    Returns:
        List of TrainingRun objects, sorted by timestamp (newest first)
    """
    if base_dir is None:
        base_dir = (
            project_context.artifact_root
            if project_context is not None and project_context.mode == "host"
            else TRAINERS_DIR
        )

    runs: List[TrainingRun] = []
    repo_root = base_dir.parent if base_dir.name == "Trainers" else base_dir

    for trainer_type in VLLM_SERVABLE_METHODS:
        for output_dir in iter_training_output_dirs(trainer_type, repo_root):
            if not output_dir.exists():
                continue

            for run_dir in output_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                if not re.match(r"\d{8}_\d{6}", run_dir.name):
                    continue

                has_final_model = (run_dir / "final_model").exists()
                has_merged_16bit = False
                has_lora = False

                for subdir in run_dir.iterdir():
                    if subdir.is_dir():
                        if (subdir / "merged-16bit").exists():
                            has_merged_16bit = True
                        if (subdir / "lora").exists():
                            has_lora = True

                if has_final_model:
                    adapter_config = run_dir / "final_model" / "adapter_config.json"
                    has_lora = has_lora or adapter_config.exists()

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
# Runtime Configuration
# ---------------------------------------------------------------------------

def resolve_tensor_parallel_size(model: str, requested: int = 0) -> int:
    """Resolve the existing evaluator's explicit/automatic device policy."""
    if type(requested) is not int or not 0 <= requested <= 256:
        raise ValueError("tensor parallel size must be an integer from 0 through 256")
    resolved = requested
    if resolved == 0:
        try:
            import torch

            resolved = max(int(torch.cuda.device_count()), 1) if torch.cuda.is_available() else 1
        except Exception:
            resolved = 1
    normalized = model.strip().lower()
    if resolved > 1 and any(
        marker in normalized for marker in ("bnb", "bitsandbytes", "4bit")
    ):
        resolved = 1
    return resolved


def resolve_tokenizer_mode(model: str) -> str | None:
    """Preserve the current operator-selected model policy explicitly."""
    return "mistral" if "mistral" in model.lower() else None


def network_runtime_environment() -> dict[str, str]:
    """Project explicit network-enabled evaluator settings, never log values.

    This policy is for the existing operator-selected evaluation workflows.
    It is not the credential-free verified-local inference policy.
    """
    names = (
        "PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "VIRTUAL_ENV",
        "TMPDIR", "TMP", "TEMP", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
        "HF_ENDPOINT", "HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy",
        "no_proxy", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "VLLM_USE_V1",
        "VLLM_WORKER_MULTIPROC_METHOD", "VLLM_ATTENTION_BACKEND",
        "PYTORCH_CUDA_ALLOC_CONF", "NCCL_DEBUG", "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
    )
    result = {name: os.environ[name] for name in names if name in os.environ}
    for name in ("HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        if name in result and not result[name].strip():
            del result[name]
    result["TORCH_COMPILE_DISABLE"] = "1"
    result.setdefault("VLLM_USE_V1", "1")
    return result


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
