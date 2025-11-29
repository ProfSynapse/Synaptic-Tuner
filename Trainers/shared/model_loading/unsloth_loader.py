"""
Unsloth model loader implementation.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

from .base import BaseModelLoader
from ..upload.core.types import ModelPath
from ..upload.core.exceptions import DependencyError


class UnslothModelLoader(BaseModelLoader):
    """
    Model loader using Unsloth optimizations.

    Unsloth provides 2x faster loading and training with optimized kernels.
    """

    @property
    def name(self) -> str:
        return "unsloth"

    def __init__(self, max_seq_length: int = 2048):
        """
        Initialize Unsloth loader.

        Args:
            max_seq_length: Maximum sequence length for the model
        """
        super().__init__(max_seq_length)
        self._FastLanguageModel = None

    def _get_fast_language_model(self):
        """
        Lazily import FastLanguageModel to avoid import errors when not needed.
        """
        if self._FastLanguageModel is None:
            try:
                from unsloth import FastLanguageModel
                self._FastLanguageModel = FastLanguageModel
            except ImportError as e:
                raise DependencyError(
                    "unsloth",
                    "Install with: pip install unsloth"
                ) from e
        return self._FastLanguageModel

    def load_model(
        self,
        model_path: ModelPath,
        load_in_4bit: bool = True,
        **config
    ) -> Tuple[Any, Any]:
        """
        Load model and tokenizer using Unsloth.

        Args:
            model_path: Path to the model or HuggingFace model ID
            load_in_4bit: Whether to load in 4-bit quantization
            **config: Additional configuration

        Returns:
            Tuple of (model, tokenizer)
        """
        FastLanguageModel = self._get_fast_language_model()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=config.get("max_seq_length", self.max_seq_length),
            dtype=config.get("dtype", None),
            load_in_4bit=load_in_4bit,
        )

        return model, tokenizer

    def save_merged(
        self,
        model: Any,
        tokenizer: Any,
        output_path: Path,
        save_method: str
    ) -> None:
        """
        Save merged model using Unsloth's save_pretrained_merged.

        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            output_path: Path to save the model
            save_method: Method for saving ("merged_16bit", "merged_4bit", "lora")
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained_merged(
            str(output_path),
            tokenizer,
            save_method=save_method
        )

    def push_to_hub_merged(
        self,
        model: Any,
        tokenizer: Any,
        repo_id: str,
        token: str,
        save_method: str = "merged_16bit",
        private: bool = False
    ) -> None:
        """
        Push merged model directly to HuggingFace Hub.

        Args:
            model: The model to push
            tokenizer: The tokenizer to push
            repo_id: HuggingFace repository ID
            token: HuggingFace token
            save_method: Save method
            private: Whether to make repository private
        """
        model.push_to_hub_merged(
            repo_id,
            tokenizer,
            save_method=save_method,
            token=token,
            private=private
        )

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Args:
            model: The loaded model

        Returns:
            Dictionary with model information
        """
        info = {
            "max_seq_length": self.max_seq_length,
            "loader": self.name,
        }

        try:
            info["num_parameters"] = sum(p.numel() for p in model.parameters())
            info["trainable_parameters"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        except Exception:
            pass

        return info
