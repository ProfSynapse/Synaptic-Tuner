"""Configuration module for improvement engine and format resolution.

Provides configuration loading and type-safe access to all settings,
plus loaders and resolvers for config-driven tool-call, workspace, and label formats.
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
from .format_resolver import (
    load_tool_call_formats,
    load_workspace_formats,
    load_label_mappings,
    resolve_tool_call_format,
    resolve_workspace_format,
    get_default_tool_call_format,
    get_default_label_mappings,
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
    "load_tool_call_formats",
    "load_workspace_formats",
    "load_label_mappings",
    "resolve_tool_call_format",
    "resolve_workspace_format",
    "get_default_tool_call_format",
    "get_default_label_mappings",
]
