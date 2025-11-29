"""
Abstract base classes (interfaces) for the upload framework.

These interfaces define contracts that implementations must follow,
enabling dependency inversion and easy testing via mocks.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import ModelPath, RepositoryId, Credential, QuantizationMethod


class ISaveStrategy(ABC):
    """
    Strategy interface for saving models in different formats.

    Implementations:
    - LoRASaveStrategy: Copy LoRA adapters only
    - Merged16BitStrategy: Merge and save as 16-bit
    - Merged4BitStrategy: Merge and save as 4-bit quantized
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this strategy."""
        pass

    @abstractmethod
    def save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """
        Save model using this strategy.

        Args:
            model_path: Path to the source model (LoRA adapters)
            output_dir: Directory to save the output
            **kwargs: Strategy-specific options

        Returns:
            Path to the saved model directory
        """
        pass

    @abstractmethod
    def estimate_size_gb(self, model_size: str) -> float:
        """
        Estimate output size in gigabytes.

        Args:
            model_size: Model size identifier (e.g., "7b")

        Returns:
            Estimated size in GB
        """
        pass

    @abstractmethod
    def requires_gpu(self) -> bool:
        """Whether this strategy requires GPU for execution."""
        pass


class IConverter(ABC):
    """
    Interface for model format converters.

    Implementations:
    - GGUFConverter: Convert to GGUF format with quantizations
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this converter."""
        pass

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


class IUploader(ABC):
    """
    Interface for model repository uploaders.

    Implementations:
    - HuggingFaceUploader: Upload to HuggingFace Hub
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this uploader."""
        pass

    @abstractmethod
    def upload_model(
        self,
        local_path: Path,
        repo_id: RepositoryId,
        credential: Credential,
        private: bool = False,
        **options
    ) -> str:
        """
        Upload model to repository.

        Args:
            local_path: Path to the local model
            repo_id: Target repository ID
            credential: Authentication credential
            private: Whether to create private repository
            **options: Uploader-specific options

        Returns:
            URL of the uploaded model
        """
        pass

    @abstractmethod
    def upload_file(
        self,
        file_path: Path,
        repo_id: RepositoryId,
        credential: Credential,
        path_in_repo: str
    ) -> None:
        """
        Upload a single file to repository.

        Args:
            file_path: Path to the local file
            repo_id: Target repository ID
            credential: Authentication credential
            path_in_repo: Path within the repository
        """
        pass

    @abstractmethod
    def validate_credential(self, credential: Credential) -> bool:
        """Validate that the credential is valid."""
        pass


class IModelLoader(ABC):
    """
    Interface for model loading abstractions.

    Enables swapping between different loading frameworks
    (Unsloth, vanilla transformers, etc.)

    Implementations:
    - UnslothModelLoader: Load with Unsloth optimizations
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this loader."""
        pass

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


class IDocumentationGenerator(ABC):
    """
    Interface for documentation generators.

    Implementations:
    - ManifestGenerator: Generate upload manifest JSON
    - ModelCardGenerator: Generate HuggingFace model card
    - ReadmeGenerator: Generate README.md
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this generator."""
        pass

    @abstractmethod
    def generate(self, **data) -> str:
        """
        Generate documentation content.

        Args:
            **data: Data to include in documentation

        Returns:
            Generated documentation content
        """
        pass

    @abstractmethod
    def save(self, content: str, output_path: Path) -> Path:
        """
        Save documentation to file.

        Args:
            content: Documentation content
            output_path: Path to save the file

        Returns:
            Path to the saved file
        """
        pass
