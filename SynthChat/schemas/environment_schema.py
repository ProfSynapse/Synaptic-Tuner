"""SynthChat Environment Schema - Canonical JSON schema for environment generation.

Location: SynthChat/schemas/environment_schema.py
Purpose: Build the JSON schema and generation prompt used when the LLM generates
         environment specifications (fixtures, assertions, system context).
Usage: Called by generator.py during environment generation stage.
"""

from typing import Any, Dict


def _non_blank_string_schema() -> Dict[str, Any]:
    return {"type": "string", "minLength": 1, "pattern": "\\S"}


def _relative_path_string_schema() -> Dict[str, Any]:
    return {
        "type": "string",
        "minLength": 1,
        "pattern": "^[A-Za-z0-9][A-Za-z0-9_ ./-]*$",
    }


def _scalar_schema() -> Dict[str, Any]:
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
            {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
            },
        ]
    }


def _assertion_schema() -> Dict[str, Any]:
    scalar = _scalar_schema()
    return {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "path_exists"},
                    "path": _relative_path_string_schema(),
                },
                "required": ["type", "path"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "path_not_exists"},
                    "path": _relative_path_string_schema(),
                },
                "required": ["type", "path"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "file_contains"},
                    "path": _relative_path_string_schema(),
                    "text": {"type": "string"},
                },
                "required": ["type", "path", "text"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "file_not_contains"},
                    "path": _relative_path_string_schema(),
                    "text": {"type": "string"},
                },
                "required": ["type", "path", "text"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "dir_contains"},
                    "path": {"type": "string"},
                    "item": _relative_path_string_schema(),
                },
                "required": ["type", "path", "item"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "frontmatter_has_key"},
                    "path": _relative_path_string_schema(),
                    "field": _non_blank_string_schema(),
                },
                "required": ["type", "path", "field"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "frontmatter_field_equals"},
                    "path": _relative_path_string_schema(),
                    "field": _non_blank_string_schema(),
                    "value": scalar,
                },
                "required": ["type", "path", "field", "value"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "frontmatter_field_contains"},
                    "path": _relative_path_string_schema(),
                    "field": _non_blank_string_schema(),
                    "value": scalar,
                },
                "required": ["type", "path", "field", "value"],
            },
        ]
    }


def _build_canonical_environment_generation_prompt(base_prompt: str) -> str:
    """Add a compact in-band contract for canonical environment generation."""
    contract_lines = [
        "Return one valid JSON object only.",
        "Top-level keys allowed: environment, system_context, task_context.",
        "environment may contain: fixture, assertions, allowed_tools, max_steps, loop, execution, mock_tool_outputs.",
        "fixture may contain inline directories, files, and notes.",
        "mock_tool_outputs may declare structured outputs for non-filesystem tools; each item may contain tool, match, output, status, error, and recoverable.",
        "Do not use fixture local_path or source unless the scenario explicitly provides a real local path.",
        "Use only plain ASCII relative paths; do not use placeholders, ellipses, angle brackets, backticks, or leading slash.",
        "notes entries may contain: path, frontmatter, body.",
        "task_context is required and must contain the hidden task anchors used to keep the environment, user request, and assertions aligned.",
        "fixture must include at least one file or note with non-empty content.",
        "Use only these assertion types:",
        "- path_exists",
        "- path_not_exists",
        "- file_contains",
        "- file_not_contains",
        "- dir_contains",
        "- frontmatter_has_key",
        "- frontmatter_field_equals",
        "- frontmatter_field_contains",
        "Do not add unsupported assertion types or extra top-level keys.",
        "Do not use markdown fences.",
    ]
    contract = "\n".join(contract_lines)
    prompt_text = str(base_prompt or "").strip()
    if not prompt_text:
        return contract
    return f"{contract}\n\nTask:\n{prompt_text}"


def _build_canonical_environment_schema() -> Dict[str, Any]:
    scalar = _scalar_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "environment": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fixture": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "directories": {
                                "type": "array",
                                "items": _relative_path_string_schema(),
                            },
                            "files": {
                                "type": "object",
                                "additionalProperties": _non_blank_string_schema(),
                                "minProperties": 1,
                            },
                            "notes": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "path": _relative_path_string_schema(),
                                        "frontmatter": {
                                            "type": "object",
                                            "additionalProperties": scalar,
                                        },
                                        "body": _non_blank_string_schema(),
                                    },
                                    "required": ["path", "body"],
                                },
                            },
                        },
                        "anyOf": [
                            {"required": ["files"]},
                            {"required": ["notes"]},
                        ],
                    },
                    "assertions": {
                        "type": "array",
                        "items": _assertion_schema(),
                        "minItems": 1,
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": _non_blank_string_schema(),
                    },
                    "max_steps": {"type": "integer", "minimum": 1},
                    "mock_tool_outputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "tool": _non_blank_string_schema(),
                                "name": _non_blank_string_schema(),
                                "command": _non_blank_string_schema(),
                                "tools": {
                                    "type": "array",
                                    "items": _non_blank_string_schema(),
                                    "minItems": 1,
                                },
                                "match": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                                "arguments": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                                "output": {},
                                "status": {
                                    "type": "string",
                                    "enum": ["ok", "error", "blocked"],
                                },
                                "error": {},
                                "recoverable": {"type": "boolean"},
                            },
                            "anyOf": [
                                {"required": ["tool"]},
                                {"required": ["name"]},
                                {"required": ["command"]},
                                {"required": ["tools"]},
                            ],
                        },
                    },
                },
                "required": ["fixture", "assertions"],
            },
            "system_context": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "assistant_instructions": {"type": "string"},
                    "available_workspaces": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "root_folder": {"type": "string"},
                            },
                        },
                    },
                    "available_prompts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "purpose": {"type": "string"},
                            },
                        },
                    },
                    "selected_workspace": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "root_folder": {"type": "string"},
                            "recent_files": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "key_files": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "preferences": {"type": "string"},
                        },
                    },
                },
                "additionalProperties": True,
            },
            "task_context": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": scalar,
            },
        },
        "required": ["environment", "task_context"],
    }
