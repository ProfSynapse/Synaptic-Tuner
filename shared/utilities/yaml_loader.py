"""YAML configuration loader.

Shared utility for loading YAML files across all modules.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary with configuration data

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(path, 'r') as f:
        return yaml.safe_load(f)
