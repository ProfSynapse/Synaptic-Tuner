"""
shared/inference/loader.py

Top-level convenience function for loading inference plugin configuration.
Provides a single entry point that handles direct YAML paths, named
profiles, and the default fallback.

Used by: tuner handlers, services/proxy, CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import InferencePluginConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path("configs/inference")


def load_inference_config(
    config_path: str | Path | None = None,
    profile: str | None = None,
    base_dir: str | Path = _DEFAULT_CONFIG_DIR,
) -> InferencePluginConfig:
    """Load inference plugin config.

    Resolution order:
        1. If *config_path* is given, load that YAML file directly.
        2. If *profile* is given, load the named profile from
           ``<base_dir>/profiles/<profile>.yaml`` (with ``extends``
           inheritance).
        3. Otherwise, load ``<base_dir>/default.yaml``.

    If the resolved file does not exist, returns a config with all defaults
    (all plugins disabled).

    Args:
        config_path: Direct path to a YAML config file.
        profile: Named profile (looks in ``<base_dir>/profiles/``).
        base_dir: Base directory for config discovery.

    Returns:
        Populated :class:`InferencePluginConfig`.
    """
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            logger.warning(
                "Inference config file not found: %s; using defaults", path,
            )
            return InferencePluginConfig()
        logger.info("Loading inference config from %s", path)
        return InferencePluginConfig.from_yaml(path)

    if profile is not None:
        base_dir = Path(base_dir)
        profile_path = base_dir / "profiles" / f"{profile}.yaml"
        if not profile_path.exists():
            logger.warning(
                "Inference profile not found: %s; using defaults",
                profile_path,
            )
            return InferencePluginConfig()
        logger.info("Loading inference profile: %s", profile)
        return InferencePluginConfig.from_profile(profile, base_dir=base_dir)

    # Fallback to default config
    base_dir = Path(base_dir)
    default_path = base_dir / "default.yaml"
    if not default_path.exists():
        logger.info(
            "Default inference config not found at %s; using defaults",
            default_path,
        )
        return InferencePluginConfig()

    logger.info("Loading default inference config from %s", default_path)
    return InferencePluginConfig.from_yaml(default_path)
