"""
LoRA adapter save strategy.

Simply copies LoRA adapter files without merging into base model.
"""

import shutil
from pathlib import Path

from .base import BaseSaveStrategy
from ..core.types import ModelPath


class LoRASaveStrategy(BaseSaveStrategy):
    """
    Strategy for saving LoRA adapters only.

    This is the fastest and smallest option - just copies the adapter files.
    No GPU required.
    """

    @property
    def name(self) -> str:
        return "lora"

    def _execute_save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """
        Copy LoRA adapters to output directory.

        Args:
            model_path: Path to the LoRA adapters
            output_dir: Directory to save the output
            **kwargs: Unused

        Returns:
            Path to the saved adapters
        """
        save_dir = output_dir / "lora"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Copy all adapter files
        src = Path(model_path)
        shutil.copytree(str(src), str(save_dir), dirs_exist_ok=True)

        print(f"✓ LoRA adapters copied to: {save_dir}")
        return save_dir

    def estimate_size_gb(self, model_size: str) -> float:
        """
        Estimate LoRA adapter size in GB.

        LoRA adapters are very small compared to full models.

        Args:
            model_size: Model size (e.g., "7b")

        Returns:
            Estimated size in GB
        """
        sizes = {
            "3b": 0.2,
            "7b": 0.32,
            "13b": 0.5,
            "20b": 0.8,
        }
        return sizes.get(model_size.lower(), 0.32)

    def requires_gpu(self) -> bool:
        """LoRA copy doesn't require GPU."""
        return False
