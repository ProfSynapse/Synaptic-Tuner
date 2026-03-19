"""MLC/WebLLM evaluation backend.

Location: tuner/backends/evaluation/mlc_backend.py
Purpose: MLC backend implementation for WebLLM browser-based evaluation
Used by: EvaluationBackendRegistry, eval_handler

This backend discovers MLC-compiled WebGPU models from training outputs and
launches a browser-based evaluation interface. Unlike server-based backends,
it spawns an HTTP server and opens a browser for WebLLM inference.

Design decisions:
- Discovers models from rtx3090_sft and rtx3090_kto output directories
- Looks for directories containing mlc-chat-config.json or -MLC suffix
- Validates that model has required WebLLM files (ndarray-cache.json, config)
- Uses browser-based evaluation since MLC-LLM Python serve has version issues
"""

from pathlib import Path
from typing import List, Optional, Tuple

from shared.utilities.paths import iter_training_output_dirs
from .base import IEvaluationBackend


class MLCBackend(IEvaluationBackend):
    """MLC/WebLLM evaluation backend.

    Provides access to MLC-compiled models for browser-based WebLLM inference.
    Unlike server-based backends, this discovers local MLC models from
    training outputs and spawns a browser-based evaluation UI.

    Args:
        repo_root: Path to repository root (for model discovery)
    """

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize with optional repo root for model discovery."""
        self._repo_root = repo_root or self._detect_repo_root()

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
        return "mlc"

    def list_models(self) -> List[str]:
        """List available MLC/WebGPU models from training outputs.

        Searches for MLC model directories in:
        - Trainers/rtx3090_sft/sft_output_rtx3090/**/webgpu/*-MLC/
        - Trainers/rtx3090_kto/kto_output_rtx3090/**/webgpu/*-MLC/
        - Trainers/rtx3090_grpo/grpo_output_rtx3090/**/webgpu/*-MLC/

        Returns:
            List of MLC model directory paths (absolute paths)
            Empty list if no models found

        Implementation notes:
        - Looks for mlc-chat-config.json to identify MLC models
        - Returns absolute paths for use with WebLLM
        - Sorts by modification time (newest first)
        """
        models = []

        for method in ("sft", "kto", "grpo"):
            for output_dir in iter_training_output_dirs(method, self._repo_root):
                if not output_dir.exists():
                    continue

                for webgpu_dir in output_dir.rglob("webgpu"):
                    if not webgpu_dir.is_dir():
                        continue

                    for model_dir in webgpu_dir.iterdir():
                        if not model_dir.is_dir():
                            continue

                        config_file = model_dir / "mlc-chat-config.json"
                        if config_file.exists():
                            models.append(str(model_dir.resolve()))

        # Sort by modification time (newest first)
        models.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)

        return models

    def validate_connection(self) -> Tuple[bool, str]:
        """Check if MLC evaluation can run.

        MLC/WebLLM runs in the browser, so there's no server to connect to.
        We just check that we have models available.

        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") always - no server needed
        """
        # MLC runs in browser, no server connection needed
        return True, ""

    @property
    def default_host(self) -> str:
        """Host for HTTP server serving WebLLM files."""
        return "localhost"

    @property
    def default_port(self) -> int:
        """Default port for HTTP server."""
        return 8000

    def get_model_info(self, model_path: str) -> dict:
        """Get information about an MLC model.

        Args:
            model_path: Path to MLC model directory

        Returns:
            Dict with model info (name, architecture, quantization, etc.)
        """
        import json

        path = Path(model_path)
        if not path.exists():
            return {"error": f"Model not found: {model_path}"}

        info = {
            "name": path.name,
            "path": str(path),
            "architecture": None,
            "quantization": None,
            "trainer_type": "unknown",
        }

        # Read mlc-chat-config.json for details
        config_file = path / "mlc-chat-config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                info["architecture"] = config.get("model_type", "unknown")
                info["context_length"] = config.get("context_window_size")
            except (json.JSONDecodeError, IOError):
                pass

        # Detect quantization from model name
        name_lower = path.name.lower()
        for quant in ["q4f32_1", "q4f16_1", "q0f32", "q0f16", "q3f16_1", "q4f16_0"]:
            if quant in name_lower:
                info["quantization"] = quant.upper()
                break

        # Detect trainer type from path
        if "sft_output" in str(path):
            info["trainer_type"] = "sft"
        elif "kto_output" in str(path):
            info["trainer_type"] = "kto"
        elif "grpo_output" in str(path):
            info["trainer_type"] = "grpo"

        # Get run timestamp from parent directory structure
        # Path structure: .../YYYYMMDD_HHMMSS/model-name/webgpu/model-dir
        try:
            parent = path.parent.parent.parent
            if parent.name.count("_") == 1 and len(parent.name) == 15:
                info["timestamp"] = parent.name
        except (AttributeError, IndexError):
            pass

        return info

    def is_mlc_model(self, model_path: str) -> bool:
        """Check if a path points to an MLC model.

        Args:
            model_path: Path to check

        Returns:
            True if path is an MLC model, False otherwise
        """
        path = Path(model_path)

        # Check for -MLC suffix or webgpu in path
        if "-MLC" in str(path) or "webgpu" in str(path).lower():
            return True

        # Check for mlc-chat-config.json
        if path.is_dir():
            return (path / "mlc-chat-config.json").exists()

        return False
