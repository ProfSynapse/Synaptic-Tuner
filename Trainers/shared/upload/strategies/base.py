"""
Base save strategy with template method pattern.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..core.interfaces import ISaveStrategy, IModelLoader
from ..core.types import ModelPath
from ..platform.filesystem import ensure_directory


class BaseSaveStrategy(ISaveStrategy):
    """
    Base implementation of save strategy using template method pattern.

    Subclasses implement _execute_save() with their specific save logic.
    """

    def __init__(self, model_loader: Optional[IModelLoader] = None):
        """
        Initialize strategy with optional model loader.

        Args:
            model_loader: Model loader for strategies that need to load/merge models.
                          Not needed for strategies like LoRA that just copy files.
        """
        self._model_loader = model_loader

    @property
    def model_loader(self) -> Optional[IModelLoader]:
        """Get the model loader."""
        return self._model_loader

    @model_loader.setter
    def model_loader(self, loader: IModelLoader):
        """Set the model loader."""
        self._model_loader = loader

    def save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """
        Save model using this strategy.

        Template method that handles pre/post processing.

        Args:
            model_path: Path to the source model (LoRA adapters)
            output_dir: Directory to save the output
            **kwargs: Strategy-specific options

        Returns:
            Path to the saved model directory
        """
        # Pre-save validation and setup
        self._pre_save_check(model_path, output_dir)

        # Execute the actual save
        result = self._execute_save(model_path, output_dir, **kwargs)

        # Post-save cleanup
        self._post_save_cleanup(output_dir)

        return result

    @abstractmethod
    def _execute_save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """
        Actual save implementation.

        Subclasses must implement this method.

        Args:
            model_path: Path to the source model
            output_dir: Directory to save the output
            **kwargs: Strategy-specific options

        Returns:
            Path to the saved model directory
        """
        pass

    def _pre_save_check(self, model_path: ModelPath, output_dir: Path):
        """
        Pre-save validation and setup.

        Args:
            model_path: Path to the source model
            output_dir: Directory to save the output
        """
        # Validate model path exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Ensure output directory exists
        ensure_directory(output_dir)

    def _post_save_cleanup(self, output_dir: Path):
        """
        Post-save cleanup.

        Override in subclasses if cleanup is needed.

        Args:
            output_dir: Output directory
        """
        pass

    def _get_subdirectory_name(self) -> str:
        """
        Get the subdirectory name for this format.

        Returns:
            Subdirectory name (e.g., "lora", "merged-16bit")
        """
        return self.name.replace("_", "-")
