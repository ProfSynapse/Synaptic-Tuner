"""
Base model loader abstraction.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

from ..upload.core.interfaces import IModelLoader
from ..upload.core.types import ModelPath


class BaseModelLoader(IModelLoader):
    """
    Base implementation of model loader.

    Provides common functionality for all model loaders.
    """

    def __init__(self, max_seq_length: int = 2048):
        """
        Initialize loader.

        Args:
            max_seq_length: Maximum sequence length for the model
        """
        self.max_seq_length = max_seq_length

    @abstractmethod
    def load_model(
        self,
        model_path: ModelPath,
        load_in_4bit: bool = True,
        **config
    ) -> Tuple[Any, Any]:
        """
        Load model and tokenizer.

        Args:
            model_path: Path to the model
            load_in_4bit: Whether to load in 4-bit quantization
            **config: Loader-specific configuration

        Returns:
            Tuple of (model, tokenizer)
        """
        pass

    @abstractmethod
    def save_merged(
        self,
        model: Any,
        tokenizer: Any,
        output_path: Path,
        save_method: str
    ) -> None:
        """
        Save merged model.

        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            output_path: Path to save the model
            save_method: Method for saving ("merged_16bit", "merged_4bit")
        """
        pass

    @abstractmethod
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass
