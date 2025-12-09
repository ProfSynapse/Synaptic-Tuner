"""
Type definitions for the upload framework.

Uses NewType for type safety without runtime overhead.
"""

from pathlib import Path
from typing import NewType, Union

# Path to a saved model (LoRA adapters or merged model)
ModelPath = NewType("ModelPath", Path)

# HuggingFace repository ID (e.g., "username/model-name")
RepositoryId = NewType("RepositoryId", str)

# Authentication credential (e.g., HuggingFace token)
Credential = NewType("Credential", str)

# Quantization method name (e.g., "Q4_K_M", "Q8_0")
QuantizationMethod = NewType("QuantizationMethod", str)

# Save method name (e.g., "lora", "merged_16bit", "merged_4bit")
SaveMethod = NewType("SaveMethod", str)

# Model size identifier (e.g., "3b", "7b", "13b", "20b")
ModelSize = NewType("ModelSize", str)


def to_model_path(path: Union[str, Path]) -> ModelPath:
    """Convert a string or Path to ModelPath."""
    return ModelPath(Path(path).resolve())


def to_repository_id(repo_id: str) -> RepositoryId:
    """Convert a string to RepositoryId with validation."""
    if "/" not in repo_id:
        raise ValueError(f"Invalid repository ID: {repo_id}. Expected format: 'username/model-name'")
    return RepositoryId(repo_id)


def to_credential(token: str) -> Credential:
    """Convert a string to Credential with basic validation."""
    if not token or len(token) < 10:
        raise ValueError("Invalid credential: token too short")
    return Credential(token)
