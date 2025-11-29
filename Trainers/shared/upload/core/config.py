"""
Configuration dataclasses for the upload framework.

Provides structured, validated configuration objects
that replace primitive parameter passing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .types import ModelPath, RepositoryId, Credential


@dataclass
class UploadConfig:
    """
    Configuration for model upload.

    Attributes:
        model_path: Path to the saved model (LoRA adapters or merged)
        repo_id: HuggingFace repository ID (username/model-name)
        credential: HuggingFace token for authentication
        output_dir: Directory to store upload artifacts (auto-detected if None)
        private: Whether to create a private repository
    """
    model_path: ModelPath
    repo_id: RepositoryId
    credential: Credential
    output_dir: Optional[Path] = None
    private: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.model_path, str):
            self.model_path = ModelPath(Path(self.model_path).resolve())
        if self.output_dir and isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class SaveConfig:
    """
    Configuration for model saving.

    Attributes:
        strategy_name: Name of save strategy ("lora", "merged_16bit", "merged_4bit")
        save_local: Whether to save locally before uploading
        model_size: Model size for memory estimation ("3b", "7b", "13b", "20b")
    """
    strategy_name: str = "merged_16bit"
    save_local: bool = True
    model_size: str = "7b"

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_strategies = ["lora", "merged_16bit", "merged_4bit"]
        if self.strategy_name not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {self.strategy_name}. "
                f"Valid options: {valid_strategies}"
            )

        valid_sizes = ["3b", "7b", "13b", "20b"]
        if self.model_size not in valid_sizes:
            raise ValueError(
                f"Invalid model size: {self.model_size}. "
                f"Valid options: {valid_sizes}"
            )


@dataclass
class ConversionConfig:
    """
    Configuration for format conversion (e.g., GGUF).

    Attributes:
        converter_name: Name of converter to use ("gguf")
        quantizations: List of quantization methods to apply
        cleanup_temp: Whether to cleanup temporary files after conversion
        use_wsl_native: Whether to use WSL native filesystem for better I/O
    """
    converter_name: str = "gguf"
    quantizations: List[str] = field(default_factory=lambda: ["Q4_K_M", "Q5_K_M", "Q8_0"])
    cleanup_temp: bool = True
    use_wsl_native: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_converters = ["gguf"]
        if self.converter_name not in valid_converters:
            raise ValueError(
                f"Invalid converter: {self.converter_name}. "
                f"Valid options: {valid_converters}"
            )


@dataclass
class DocumentationConfig:
    """
    Configuration for documentation generation.

    Attributes:
        training_lineage_path: Path to training lineage JSON (for model card)
        include_manifest: Whether to generate upload manifest
        include_model_card: Whether to generate model card
        include_readme: Whether to generate README
        base_model: Name of the base model (for model card)
        training_method: Training method used ("sft", "kto", "lora")
    """
    training_lineage_path: Optional[Path] = None
    include_manifest: bool = True
    include_model_card: bool = True
    include_readme: bool = True
    base_model: str = ""
    training_method: str = ""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.training_lineage_path and isinstance(self.training_lineage_path, str):
            self.training_lineage_path = Path(self.training_lineage_path)


@dataclass
class FullUploadConfig:
    """
    Complete configuration combining all upload settings.

    Convenience class for passing all configuration at once.
    """
    upload: UploadConfig
    save: SaveConfig
    conversion: Optional[ConversionConfig] = None
    documentation: Optional[DocumentationConfig] = None

    @classmethod
    def from_args(cls, args) -> "FullUploadConfig":
        """
        Create configuration from CLI arguments.

        Args:
            args: Parsed argparse namespace

        Returns:
            FullUploadConfig instance
        """
        from .types import to_model_path, to_repository_id, to_credential

        upload = UploadConfig(
            model_path=to_model_path(args.model_path),
            repo_id=to_repository_id(args.repo_id),
            credential=to_credential(args.token),
            output_dir=Path(args.output_dir) if hasattr(args, 'output_dir') and args.output_dir else None,
            private=getattr(args, 'private', False),
        )

        save = SaveConfig(
            strategy_name=getattr(args, 'save_method', 'merged_16bit'),
            save_local=not getattr(args, 'no_save_local', False),
            model_size=getattr(args, 'model_size', '7b'),
        )

        conversion = None
        if getattr(args, 'create_gguf', False):
            conversion = ConversionConfig(
                converter_name="gguf",
                quantizations=getattr(args, 'gguf_quantizations', ["Q4_K_M", "Q5_K_M", "Q8_0"]),
            )

        documentation = DocumentationConfig(
            training_lineage_path=Path(args.training_lineage) if getattr(args, 'training_lineage', None) else None,
            base_model=getattr(args, 'base_model', ''),
            training_method=getattr(args, 'training_method', ''),
        )

        return cls(
            upload=upload,
            save=save,
            conversion=conversion,
            documentation=documentation,
        )
