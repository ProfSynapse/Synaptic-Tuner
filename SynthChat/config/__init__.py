"""Configuration module for improvement engine.

Provides configuration loading and type-safe access to all settings.
"""

from .config_loader import (
    ConfigLoader,
    ScopeConfig,
    ScopeDefinition,
    ScopeExtractionConfig,
    ScopeMarkersConfig,
    ScopeFormatConfig,
    LLMConfig,
    JudgeConfig,
    ValidationConfig,
    OutputConfig,
)

__all__ = [
    "ConfigLoader",
    "ScopeConfig",
    "ScopeDefinition",
    "ScopeExtractionConfig",
    "ScopeMarkersConfig",
    "ScopeFormatConfig",
    "LLMConfig",
    "JudgeConfig",
    "ValidationConfig",
    "OutputConfig",
]
