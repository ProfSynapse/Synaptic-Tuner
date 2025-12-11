"""Configuration loader for improvement engine.

Loads scope configuration from YAML and provides type-safe access.
This eliminates all hardcoding from the codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import re

from ..utils.yaml_loader import load_yaml


@dataclass
class ScopeExtractionConfig:
    """Configuration for extracting content from a scope."""
    method: str  # "role_based", "pattern", "exclusion"
    role: Optional[str] = None
    pattern: Optional[str] = None
    patterns: Optional[Dict] = None
    flags: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


@dataclass
class ScopeMarkersConfig:
    """Markers/delimiters for a scope."""
    start: Optional[str] = None
    end: Optional[str] = None
    block_start: Optional[str] = None
    block_start_alt: Optional[str] = None
    name_prefix: Optional[str] = None
    name_prefix_alt: Optional[str] = None
    args_prefix: Optional[str] = None


@dataclass
class ScopeFormatConfig:
    """Format configuration for a scope."""
    primary: str = "text"
    fallback: Optional[str] = None


@dataclass
class ScopeDefinition:
    """Complete definition of a scope."""
    name: str
    description: str
    conversation_role: str
    extraction: ScopeExtractionConfig
    markers: ScopeMarkersConfig
    format: Optional[ScopeFormatConfig] = None


@dataclass
class LLMConfig:
    """LLM configuration for improvement engine."""
    improvement_temperature: float = 0.3
    judge_temperature: float = 0.0
    improvement_max_tokens: int = 2000
    judge_max_tokens: int = 1000


@dataclass
class JudgePromptSection:
    """Configuration for a judge prompt section."""
    name: str
    content_source: str
    optional: bool = False


@dataclass
class JudgeConfig:
    """Judge configuration."""
    feedback_field: str = "overall_feedback"
    score_field_suffix: str = "_score"
    prompt_structure: Dict = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Validation configuration."""
    schema_score_suffix: str = "_schema"
    icons: Dict[str, str] = field(default_factory=lambda: {"passed": "✅", "failed": "❌"})


@dataclass
class OutputConfig:
    """Output formatting configuration."""
    use_code_fences: bool = True
    code_fence_languages: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScopeConfig:
    """Complete scope configuration."""
    scopes: Dict[str, ScopeDefinition]
    scope_processing_order: List[str]
    llm: LLMConfig
    judge: JudgeConfig
    validation: ValidationConfig
    output: OutputConfig

    def get_scope(self, name: str) -> Optional[ScopeDefinition]:
        """Get scope definition by name."""
        return self.scopes.get(name)

    def get_extraction_pattern(self, scope_name: str, compiled: bool = False):
        """Get compiled regex pattern for scope extraction."""
        scope = self.get_scope(scope_name)
        if not scope or not scope.extraction.pattern:
            return None

        if compiled:
            flags = 0
            if scope.extraction.flags:
                for flag_name in scope.extraction.flags:
                    flags |= getattr(re, flag_name, 0)
            return re.compile(scope.extraction.pattern, flags)

        return scope.extraction.pattern

    def get_scope_markers(self, scope_name: str) -> Optional[ScopeMarkersConfig]:
        """Get markers for a scope."""
        scope = self.get_scope(scope_name)
        return scope.markers if scope else None


class ConfigLoader:
    """
    Loads scope configuration from YAML.

    Provides type-safe access to all configuration values,
    eliminating hardcoding throughout the codebase.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to scope_config.yaml (defaults to config/scope_config.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "scope_config.yaml"

        self.config_path = Path(config_path)
        self._config: Optional[ScopeConfig] = None

    def load(self) -> ScopeConfig:
        """
        Load configuration from YAML.

        Returns:
            ScopeConfig with all configuration data
        """
        if self._config is not None:
            return self._config

        data = load_yaml(self.config_path)

        # Parse scopes
        scopes = {}
        for scope_name, scope_data in data.get("scopes", {}).items():
            extraction_data = scope_data.get("extraction", {})
            extraction = ScopeExtractionConfig(
                method=extraction_data.get("method", "role_based"),
                role=extraction_data.get("role"),
                pattern=extraction_data.get("pattern"),
                patterns=extraction_data.get("patterns"),
                flags=extraction_data.get("flags"),
                exclude=extraction_data.get("exclude")
            )

            markers_data = scope_data.get("markers", {})
            markers = ScopeMarkersConfig(
                start=markers_data.get("start"),
                end=markers_data.get("end"),
                block_start=markers_data.get("block_start"),
                block_start_alt=markers_data.get("block_start_alt"),
                name_prefix=markers_data.get("name_prefix"),
                name_prefix_alt=markers_data.get("name_prefix_alt"),
                args_prefix=markers_data.get("args_prefix")
            )

            format_data = scope_data.get("format")
            scope_format = None
            if format_data:
                scope_format = ScopeFormatConfig(
                    primary=format_data.get("primary", "text"),
                    fallback=format_data.get("fallback")
                )

            scopes[scope_name] = ScopeDefinition(
                name=scope_name,
                description=scope_data.get("description", ""),
                conversation_role=scope_data.get("conversation_role", ""),
                extraction=extraction,
                markers=markers,
                format=scope_format
            )

        # Parse LLM config
        llm_data = data.get("llm", {})
        llm_config = LLMConfig(
            improvement_temperature=llm_data.get("improvement_temperature", 0.3),
            judge_temperature=llm_data.get("judge_temperature", 0.0),
            improvement_max_tokens=llm_data.get("improvement_max_tokens", 2000),
            judge_max_tokens=llm_data.get("judge_max_tokens", 1000)
        )

        # Parse judge config
        judge_data = data.get("judge", {})
        judge_config = JudgeConfig(
            feedback_field=judge_data.get("feedback_field", "overall_feedback"),
            score_field_suffix=judge_data.get("score_field_suffix", "_score"),
            prompt_structure=judge_data.get("prompt_structure", {})
        )

        # Parse validation config
        validation_data = data.get("validation", {})
        validation_config = ValidationConfig(
            schema_score_suffix=validation_data.get("schema_score_suffix", "_schema"),
            icons=validation_data.get("icons", {"passed": "✅", "failed": "❌"})
        )

        # Parse output config
        output_data = data.get("output", {})
        output_config = OutputConfig(
            use_code_fences=output_data.get("use_code_fences", True),
            code_fence_languages=output_data.get("code_fence_languages", {})
        )

        self._config = ScopeConfig(
            scopes=scopes,
            scope_processing_order=data.get("scope_processing_order", []),
            llm=llm_config,
            judge=judge_config,
            validation=validation_config,
            output=output_config
        )

        return self._config

    def reload(self) -> ScopeConfig:
        """Reload configuration from disk."""
        self._config = None
        return self.load()
