"""llama.cpp evaluation backend.

Location: tuner/backends/evaluation/llamacpp_backend.py
Purpose: llama.cpp backend implementation for model evaluation
Used by: EvaluationBackendRegistry, eval_handler

This backend discovers GGUF models from training outputs and validates
that llama-cli is available. Unlike other backends, it doesn't connect
to a server - it runs llama-cli directly for inference.

Design decisions:
- Discovers models from canonical trainer output directories
- Validates llama-cli existence rather than server connectivity
- Returns GGUF file paths as "model names" for use with Evaluator CLI
"""

import platform
from pathlib import Path
from typing import List, Optional, Tuple

from shared.utilities.paths import iter_training_output_dirs
from .base import IEvaluationBackend


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
        # Intel Mac - CPU only or older Metal
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
        # Generic fallback
        return (
            "cd Trainers/llama.cpp && "
            "cmake -B build && "
            "cmake --build build --config Release"
        )


def _find_llama_cli(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Find llama-cli executable in common locations."""
    candidates = []

    # If repo_root provided, check Trainers/llama.cpp first
    if repo_root:
        candidates.extend([
            repo_root / "Trainers" / "llama.cpp" / "build" / "bin" / "llama-cli",
            repo_root / "Trainers" / "llama.cpp" / "llama-cli",
        ])

    # Add other common locations
    candidates.extend([
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli",
        Path.home() / "llama.cpp" / "llama-cli",
        Path("/usr/local/bin/llama-cli"),
        Path("/opt/homebrew/bin/llama-cli"),
    ])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


class LlamaCppBackend(IEvaluationBackend):
    """llama.cpp evaluation backend.

    Provides access to GGUF models for direct inference via llama-cli.
    Unlike server-based backends, this discovers local GGUF files from
    training outputs.

    Args:
        repo_root: Path to repository root (for model discovery)
    """

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize with optional repo root for model discovery."""
        self._repo_root = repo_root or self._detect_repo_root()
        self._llama_cli = _find_llama_cli(self._repo_root)

    def _detect_repo_root(self) -> Path:
        """Detect repository root from current file location."""
        # Go up from tuner/backends/evaluation/ to repo root
        current = Path(__file__).resolve()
        for _ in range(4):  # Go up 4 levels
            current = current.parent
        return current

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "llamacpp"

    def list_models(self) -> List[str]:
        """List available GGUF models from training outputs.

        Searches for GGUF files in canonical SFT, KTO, and GRPO outputs.

        Returns:
            List of GGUF file paths (absolute paths)
            Empty list if no models found

        Implementation notes:
        - Skips vocab and mmproj files
        - Returns absolute paths since llama-cli needs full paths
        - Sorts by modification time (newest first)
        """
        models = []

        for method in ("sft", "kto", "grpo", "dpo"):
            for output_dir in iter_training_output_dirs(method, self._repo_root):
                if not output_dir.exists():
                    continue

                for gguf_file in output_dir.rglob("*.gguf"):
                    name_lower = gguf_file.name.lower()
                    if "vocab" in name_lower or "mmproj" in name_lower:
                        continue

                    models.append(str(gguf_file.resolve()))

        # Sort by modification time (newest first)
        models.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)

        return models

    def validate_connection(self) -> Tuple[bool, str]:
        """Check if llama-cli is available.

        Unlike server-based backends, this checks for the llama-cli executable
        rather than a network connection.

        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") if llama-cli is found
            - (False, "error message") if llama-cli is not found
        """
        if self._llama_cli and self._llama_cli.exists():
            return True, ""
        else:
            build_cmd = _get_build_instructions()
            return False, (
                f"llama-cli not found. Build llama.cpp first:\n  {build_cmd}"
            )

    @property
    def default_host(self) -> str:
        """Not applicable for llama.cpp (runs locally)."""
        return "localhost"

    @property
    def default_port(self) -> int:
        """Not applicable for llama.cpp (runs locally)."""
        return 0

    def get_llama_cli_path(self) -> Optional[Path]:
        """Get the path to llama-cli executable."""
        return self._llama_cli

    def get_model_info(self, model_path: str) -> dict:
        """Get information about a GGUF model.

        Args:
            model_path: Path to GGUF file

        Returns:
            Dict with model info (name, size, quantization, etc.)
        """
        path = Path(model_path)
        if not path.exists():
            return {"error": f"Model not found: {model_path}"}

        # Extract info from filename and path
        name = path.stem
        size_gb = path.stat().st_size / (1024 ** 3)

        # Detect quantization from filename
        quant = None
        for q in ["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_K_S", "Q6_K", "Q4_0", "Q5_0"]:
            if q in path.name:
                quant = q
                break

        # Detect trainer type from path
        trainer_type = "unknown"
        if "sft_output" in str(path):
            trainer_type = "sft"
        elif "kto_output" in str(path):
            trainer_type = "kto"
        elif "grpo_output" in str(path):
            trainer_type = "grpo"

        return {
            "name": name,
            "path": str(path),
            "size_gb": round(size_gb, 2),
            "quantization": quant,
            "trainer_type": trainer_type,
        }
