"""Import-light compatibility facade for the evaluation harness."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "BackendClient": ".protocols", "BackendError": ".protocols",
    "BackendResponse": ".protocols", "BackendSettings": ".protocols",
    "BackendType": ".enums", "ResponseType": ".enums",
    "ToolCallFormat": ".enums", "ValidationLevel": ".enums",
    "create_client": ".client_factory", "create_client_from_args": ".client_factory",
    "create_settings": ".client_factory", "get_supported_backends": ".client_factory",
    "BaseBackendSettings": ".config", "EvaluatorConfig": ".config",
    "LMStudioSettings": ".config", "OllamaSettings": ".config",
    "PromptFilter": ".config", "expand_path": ".config", "parse_tags": ".config",
    "PromptCase": ".prompt_sets", "filter_prompts": ".prompt_sets",
    "load_prompt_cases": ".prompt_sets",
    "ValidationResult": ".schema_validator",
    "validate_assistant_response": ".schema_validator",
    "CorrectnessResult": "shared.verifiers.builtins.assertion_verifier",
    "evaluate_correctness": "shared.verifiers.builtins.assertion_verifier",
    "RubricValidator": ".rubric_validator",
    "RubricValidationResult": ".rubric_validator",
    "FullValidationResult": ".rubric_validator", "validate_response": ".rubric_validator",
    "ParsedResponse": "shared.validation.parsing.response_parser",
    "ParsedToolCall": "shared.validation.parsing.response_parser",
    "parse_response": "shared.validation.parsing.response_parser",
    "EvaluationRecord": ".runner", "evaluate_cases": ".runner",
    "aggregate_stats": ".reporting", "build_run_payload": ".reporting",
    "console_summary": ".reporting", "render_markdown": ".reporting",
    "write_json": ".reporting",
}

__all__ = [
    "BackendClient", "BackendError", "BackendResponse", "BackendSettings",
    "BackendType", "ResponseType", "ToolCallFormat", "ValidationLevel",
    "create_client", "create_client_from_args", "create_settings", "get_supported_backends",
    "BaseBackendSettings", "EvaluatorConfig", "LMStudioSettings", "OllamaSettings",
    "PromptFilter", "expand_path", "parse_tags", "PromptCase", "filter_prompts",
    "load_prompt_cases", "ValidationResult", "validate_assistant_response",
    "CorrectnessResult", "evaluate_correctness", "RubricValidator",
    "RubricValidationResult", "FullValidationResult", "validate_response",
    "ParsedResponse", "ParsedToolCall", "parse_response", "EvaluationRecord",
    "evaluate_cases", "aggregate_stats", "build_run_payload", "console_summary",
    "render_markdown", "write_json",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    package = __name__ if module_name.startswith(".") else None
    value = getattr(import_module(module_name, package), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))
