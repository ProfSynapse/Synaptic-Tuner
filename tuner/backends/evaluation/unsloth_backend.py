"""Unsloth/LoRA evaluation backend.

Location: tuner/backends/evaluation/unsloth_backend.py
Purpose: Unsloth backend implementation for LoRA model evaluation
Used by: EvaluationBackendRegistry, eval_handler

This backend discovers LoRA adapters from training outputs and validates
that Unsloth is available. It runs inference directly in-process by loading
the base model and applying the LoRA adapter on top.

Design decisions:
- Discovers final_model directories from training outputs
- Validates Unsloth installation
- Returns adapter paths for use with Evaluator CLI
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from shared.utilities.paths import iter_training_run_dirs
from .base import IEvaluationBackend


class UnslothBackend(IEvaluationBackend):
    """Unsloth/LoRA evaluation backend.

    Provides access to LoRA adapters for direct inference using Unsloth.
    Discovers final_model directories from training outputs.

    Args:
        repo_root: Path to repository root (for model discovery)
    """

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize with optional repo root for model discovery."""
        self._repo_root = repo_root or self._detect_repo_root()

    def _detect_repo_root(self) -> Path:
        """Detect repository root from current file location."""
        current = Path(__file__).resolve()
        for _ in range(4):  # Go up 4 levels
            current = current.parent
        return current

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "unsloth"

    def list_models(self) -> List[str]:
        """List available LoRA adapters from training outputs.

        Searches for final_model directories containing adapter_config.json in:
        - Trainers/rtx3090_sft/sft_output_rtx3090/*/final_model/
        - Trainers/rtx3090_kto/kto_output_rtx3090/*/final_model/

        Returns:
            List of adapter directory paths (absolute paths)
            Empty list if no adapters found
        """
        models = []

        for method in ("sft", "kto", "grpo"):
            for run_dir in iter_training_run_dirs(method, self._repo_root):
                adapter_dir = run_dir / "final_model"
                adapter_config = adapter_dir / "adapter_config.json"
                if adapter_config.exists():
                    models.append(str(adapter_dir.resolve()))

        # Sort by modification time (newest first)
        models.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)

        return models

    def validate_connection(self) -> Tuple[bool, str]:
        """Check if Unsloth is available.

        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") if Unsloth is installed
            - (False, "error message") if Unsloth is not available
        """
        try:
            from unsloth import FastLanguageModel
            return True, ""
        except ImportError:
            return False, (
                "Unsloth not installed. Install with:\n"
                "  pip install unsloth\n"
                "Or run ./setup_env.sh to set up the environment"
            )

    @property
    def default_host(self) -> str:
        """Not applicable for direct inference."""
        return "localhost"

    @property
    def default_port(self) -> int:
        """Not applicable for direct inference."""
        return 0

    def get_model_info(self, adapter_path: str) -> dict:
        """Get information about a LoRA adapter.

        Args:
            adapter_path: Path to adapter directory

        Returns:
            Dict with adapter info (name, base_model, trainer_type, etc.)
        """
        path = Path(adapter_path)
        if not path.exists():
            return {"error": f"Adapter not found: {adapter_path}"}

        # Read adapter config
        config_file = path / "adapter_config.json"
        if not config_file.exists():
            return {"error": "adapter_config.json not found"}

        with open(config_file) as f:
            config = json.load(f)

        # Get adapter file size
        adapter_file = path / "adapter_model.safetensors"
        size_mb = None
        if adapter_file.exists():
            size_mb = round(adapter_file.stat().st_size / (1024 ** 2), 1)

        # Detect trainer type from path
        trainer_type = self._detect_trainer_type(path)
        source = self._detect_source(path)

        # Extract run timestamp from parent directory
        timestamp = path.parent.name if path.parent else "unknown"

        # Get base model name (truncate for display)
        base_model = config.get("base_model_name_or_path", "unknown")
        base_model_short = base_model.split("/")[-1] if "/" in base_model else base_model

        return {
            "name": f"{timestamp}_{trainer_type}",
            "path": str(path),
            "base_model": base_model,
            "base_model_short": base_model_short,
            "size_mb": size_mb,
            "trainer_type": trainer_type,
            "source": source,
            "timestamp": timestamp,
            "r": config.get("r"),  # LoRA rank
            "lora_alpha": config.get("lora_alpha"),
        }

    @staticmethod
    def _detect_trainer_type(path: Path) -> str:
        parts = [part.lower() for part in path.parts]
        markers = {
            "sft": {"sft_output", "sft_output_rtx3090", "rtx3090_sft", "sft"},
            "kto": {"kto_output", "kto_output_rtx3090", "rtx3090_kto", "kto"},
            "grpo": {"grpo_output", "grpo_output_rtx3090", "rtx3090_grpo", "grpo"},
        }
        for trainer_type, candidates in markers.items():
            if any(candidate in parts for candidate in candidates):
                if trainer_type in {"sft", "kto", "grpo"}:
                    return trainer_type
        return "unknown"

    @staticmethod
    def _detect_source(path: Path) -> str:
        parts = {part.lower() for part in path.parts}
        if "toolset-training-artifacts" in parts:
            return "bucket_pull"
        if "runs" in parts and "trainers" not in parts:
            return "cloud_artifact"
        return "local_training"
