"""Evaluator module for testing tool-calling LLMs.

This module provides a complete evaluation harness for testing models
against prompt sets and validating their tool-calling behavior.

Example usage:
    from Evaluator import create_client, evaluate_cases, load_prompt_cases
    from Evaluator.config import LMStudioSettings

    settings = LMStudioSettings(model="my-model")
    client = create_client("lmstudio", settings)
    cases = load_prompt_cases(Path("prompts.json"))
    records = evaluate_cases(cases, client)
"""

# Core protocols and types
from .protocols import BackendClient, BackendError, BackendResponse, BackendSettings

# Enums
from .enums import BackendType, ResponseType, ToolCallFormat, ValidationLevel

# Factory functions
from .client_factory import (
    create_client,
    create_client_from_args,
    create_settings,
    get_supported_backends,
)

# Configuration
from .config import (
    BaseBackendSettings,
    EvaluatorConfig,
    LMStudioSettings,
    OllamaSettings,
    PromptFilter,
    expand_path,
    parse_tags,
)

# Clients
from .ollama_client import OllamaClient, OllamaError
from .lmstudio_client import LMStudioClient, LMStudioError

# Prompt handling
from .prompt_sets import PromptCase, filter_prompts, load_prompt_cases

# Validation
from .schema_validator import ValidationResult, validate_assistant_response
from .behavior_validator import (
    BehaviorIssue,
    BehaviorValidationResult,
    detect_response_type,
    validate_behavior,
)

# Response parsing
from .response_parser import (
    ParsedResponse,
    ParsedToolCall,
    parse_response,
)

# Evaluation
from .runner import EvaluationRecord, evaluate_cases

# Reporting
from .reporting import (
    aggregate_stats,
    build_run_payload,
    console_summary,
    render_markdown,
    write_json,
)

__all__ = [
    # Protocols
    "BackendClient",
    "BackendError",
    "BackendResponse",
    "BackendSettings",
    # Enums
    "BackendType",
    "ResponseType",
    "ToolCallFormat",
    "ValidationLevel",
    # Factory
    "create_client",
    "create_client_from_args",
    "create_settings",
    "get_supported_backends",
    # Config
    "BaseBackendSettings",
    "EvaluatorConfig",
    "LMStudioSettings",
    "OllamaSettings",
    "PromptFilter",
    "expand_path",
    "parse_tags",
    # Clients
    "OllamaClient",
    "OllamaError",
    "LMStudioClient",
    "LMStudioError",
    # Prompts
    "PromptCase",
    "filter_prompts",
    "load_prompt_cases",
    # Validation
    "ValidationResult",
    "validate_assistant_response",
    "BehaviorIssue",
    "BehaviorValidationResult",
    "detect_response_type",
    "validate_behavior",
    # Parsing
    "ParsedResponse",
    "ParsedToolCall",
    "parse_response",
    # Evaluation
    "EvaluationRecord",
    "evaluate_cases",
    # Reporting
    "aggregate_stats",
    "build_run_payload",
    "console_summary",
    "render_markdown",
    "write_json",
]
