"""
Merged 16-bit save strategy.

Merges LoRA adapters into base model and saves at full 16-bit precision.
"""

from pathlib import Path

from .base import BaseSaveStrategy
from ..core.types import ModelPath
from ..core.exceptions import SaveError
from ..platform.gpu_memory import ensure_gpu_memory, get_required_memory
from ..platform.filesystem import copy_to_native_filesystem, cleanup_temp_directory


class Merged16BitStrategy(BaseSaveStrategy):
    """
    Strategy for saving merged 16-bit models.

    Merges LoRA adapters into the base model and saves at full precision.
    Produces the largest but highest quality output.
    Requires significant GPU memory.
    """

    @property
    def name(self) -> str:
        return "merged_16bit"

    def _execute_save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """
        Load model, merge adapters, and save at 16-bit precision.

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
        required_gb = get_required_memory(model_size, "merge_16bit")
        if not ensure_gpu_memory(required_gb, "16-bit model merge"):
            raise SaveError(
                f"Insufficient GPU memory for 16-bit merge. "
                f"Try --save-method lora or free up GPU memory."
            )

        if self.model_loader is None:
            raise SaveError("Model loader required for merged save strategies")

        # Handle Windows filesystem if needed
        working_path, was_copied = copy_to_native_filesystem(
            Path(model_path),
            prefix="merge_16bit_"
        )

        try:
            save_dir = output_dir / "merged-16bit"
            save_dir.mkdir(parents=True, exist_ok=True)

            print("Loading model for 16-bit merge...")
            model, tokenizer = self.model_loader.load_model(
                str(working_path),
                load_in_4bit=False,  # Full precision for 16-bit save
            )

            print("Saving merged 16-bit model...")
            self.model_loader.save_merged(
                model,
                tokenizer,
                save_dir,
                save_method="merged_16bit"
            )

            print(f"✓ Merged 16-bit model saved to: {save_dir}")
            return save_dir

        finally:
            # Cleanup temporary copy if we made one
            if was_copied and working_path.parent.exists():
                cleanup_temp_directory(working_path.parent)

    def estimate_size_gb(self, model_size: str) -> float:
        """
        Estimate 16-bit merged model size in GB.

        Args:
            model_size: Model size (e.g., "7b")

        Returns:
            Estimated size in GB
        """
        sizes = {
            "3b": 7.0,
            "7b": 14.0,
            "13b": 26.0,
            "20b": 40.0,
        }
        return sizes.get(model_size.lower(), 14.0)

    def requires_gpu(self) -> bool:
        """16-bit merge requires GPU."""
        return True
