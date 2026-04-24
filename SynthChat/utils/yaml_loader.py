"""YAML configuration loader."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml(file_path: str) -> Dict[str, Any]:
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

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load configuration from config directory.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Dictionary with configuration data
    """
    config_dir = Path(__file__).parent.parent / "config"
    config_path = config_dir / f"{config_name}.yaml"
    return load_yaml(str(config_path))
