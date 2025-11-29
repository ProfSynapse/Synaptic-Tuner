"""
Merged 4-bit save strategy.

Merges LoRA adapters into base model and saves with 4-bit quantization.
"""

from pathlib import Path

from .base import BaseSaveStrategy
from ..core.types import ModelPath
from ..core.exceptions import SaveError
from ..platform.gpu_memory import ensure_gpu_memory, get_required_memory
from ..platform.filesystem import copy_to_native_filesystem, cleanup_temp_directory


class Merged4BitStrategy(BaseSaveStrategy):
    """
    Strategy for saving merged 4-bit quantized models.

    Merges LoRA adapters into the base model and saves with 4-bit quantization.
    Produces smaller output with some quality trade-off.
    Requires less GPU memory than 16-bit.
    """

    @property
    def name(self) -> str:
        return "merged_4bit"

    def _execute_save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """
        Load model, merge adapters, and save with 4-bit quantization.

        Args:
            model_path: Path to the LoRA adapters
            output_dir: Directory to save the output
            **kwargs: Options including:
                - model_size: Model size for memory estimation (default: "7b")

        Returns:
            Path to the saved model
        """
        model_size = kwargs.get("model_size", "7b")

        # Check GPU memory
        required_gb = get_required_memory(model_size, "merge_4bit")
        if not ensure_gpu_memory(required_gb, "4-bit model merge"):
            raise SaveError(
                f"Insufficient GPU memory for 4-bit merge. "
                f"Try --save-method lora or free up GPU memory."
            )

        if self.model_loader is None:
            raise SaveError("Model loader required for merged save strategies")

        # Handle Windows filesystem if needed
        working_path, was_copied = copy_to_native_filesystem(
            Path(model_path),
            prefix="merge_4bit_"
        )

        try:
            save_dir = output_dir / "merged-4bit"
            save_dir.mkdir(parents=True, exist_ok=True)

            print("Loading model for 4-bit merge...")
            model, tokenizer = self.model_loader.load_model(
                str(working_path),
                load_in_4bit=True,
            )

            print("Saving merged 4-bit model...")
            self.model_loader.save_merged(
                model,
                tokenizer,
                save_dir,
                save_method="merged_4bit"
            )

            print(f"✓ Merged 4-bit model saved to: {save_dir}")
            return save_dir

        finally:
            # Cleanup temporary copy if we made one
            if was_copied and working_path.parent.exists():
                cleanup_temp_directory(working_path.parent)

    def estimate_size_gb(self, model_size: str) -> float:
        """
        Estimate 4-bit merged model size in GB.

        Args:
            model_size: Model size (e.g., "7b")

        Returns:
            Estimated size in GB
        """
        sizes = {
            "3b": 2.0,
            "7b": 3.5,
            "13b": 7.0,
            "20b": 10.0,
        }
        return sizes.get(model_size.lower(), 3.5)

    def requires_gpu(self) -> bool:
        """4-bit merge requires GPU."""
        return True
