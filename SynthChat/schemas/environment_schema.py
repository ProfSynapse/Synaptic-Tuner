"""SynthChat Environment Schema - Canonical JSON schema for environment generation.

Location: SynthChat/schemas/environment_schema.py
Purpose: Build the JSON schema and generation prompt used when the LLM generates
         environment specifications (fixtures, assertions, system context).
Usage: Called by generator.py during environment generation stage.
"""

from typing import Any, Dict


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
                    "path": {"type": "string", "minLength": 1},
                },
                "required": ["type", "path"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "path_not_exists"},
                    "path": {"type": "string", "minLength": 1},
                },
                "required": ["type", "path"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "file_contains"},
                    "path": {"type": "string", "minLength": 1},
                    "text": {"type": "string"},
                },
                "required": ["type", "path", "text"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "file_not_contains"},
                    "path": {"type": "string", "minLength": 1},
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
                    "item": {"type": "string", "minLength": 1},
                },
                "required": ["type", "path", "item"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "frontmatter_has_key"},
                    "path": {"type": "string", "minLength": 1},
                    "field": {"type": "string", "minLength": 1},
                },
                "required": ["type", "path", "field"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "frontmatter_field_equals"},
                    "path": {"type": "string", "minLength": 1},
                    "field": {"type": "string", "minLength": 1},
                    "value": scalar,
                },
                "required": ["type", "path", "field", "value"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "frontmatter_field_contains"},
                    "path": {"type": "string", "minLength": 1},
                    "field": {"type": "string", "minLength": 1},
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
        "environment may contain: fixture, assertions, allowed_tools, max_steps, loop, execution.",
        "fixture may contain: directories, files, notes, local_path, source.",
        "notes entries may contain: path, frontmatter, body.",
        "task_context should contain the hidden task anchors used to keep the environment, user request, and assertions aligned.",
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
                                "items": {"type": "string", "minLength": 1},
                            },
                            "files": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "notes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "path": {"type": "string", "minLength": 1},
                                        "frontmatter": {
                                            "type": "object",
                                            "additionalProperties": scalar,
                                        },
                                        "body": {"type": "string"},
                                    },
                                    "required": ["path"],
                                },
                            },
                        },
                    },
                    "assertions": {
                        "type": "array",
                        "items": _assertion_schema(),
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "max_steps": {"type": "integer", "minimum": 1},
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
                "additionalProperties": scalar,
            },
        },
        "required": ["environment"],
    }
