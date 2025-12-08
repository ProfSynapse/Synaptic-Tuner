"""
Base converter abstraction.
"""

from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from ..core.interfaces import IConverter, IModelLoader
from ..core.types import ModelPath, QuantizationMethod


class BaseConverter(IConverter):
    """
    Base implementation of format converter.
    """

    def __init__(self, model_loader: Optional[IModelLoader] = None):
        """
        Initialize converter.

        Args:
            model_loader: Model loader for converters that need to load models
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

    @abstractmethod
    def convert(
        self,
        model_path: ModelPath,
        output_dir: Path,
        quantizations: Optional[List[QuantizationMethod]] = None,
        **options
    ) -> List[Path]:
        """
        Convert model to target format.

        Args:
            model_path: Path to the source model
            output_dir: Directory to save converted files
            quantizations: List of quantization methods to apply
            **options: Converter-specific options

        Returns:
            List of paths to converted files
        """
        pass

    @abstractmethod
    def supported_quantizations(self) -> List[str]:
        """Get list of supported quantization methods."""
        pass

    @abstractmethod
    def validate_environment(self) -> Tuple[bool, str]:
        """
        Validate that required tools are available.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
