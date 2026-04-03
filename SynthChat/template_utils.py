"""SynthChat Template Utilities - Pure helper functions for template rendering and data transformation.

Location: SynthChat/template_utils.py
Purpose: Shared utility functions used by generator.py, workspace rendering, and schema builders.
         Extracted from generator.py to reduce its size and improve modularity.
Usage: from SynthChat.template_utils import _render_template_object, _deep_merge_dicts, ...
"""

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional


def _deep_merge_dicts(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(base, dict) and not isinstance(override, dict):
        return override if override is not None else base
    if not isinstance(base, dict):
        return deepcopy(override or {})
    if not isinstance(override, dict):
        return deepcopy(base)

    merged: Dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _task_context_template_vars(task_context: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(task_context, dict) or not task_context:
        return {}

    safe_task_context = _make_json_safe(task_context)
    template_vars: Dict[str, str] = {
        "task_context_json": json.dumps(safe_task_context, indent=2),
    }
    for key, value in safe_task_context.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        template_vars[f"task_{key_str}"] = rendered
    return template_vars


def _user_generation_style_instructions(scenario: Optional[Dict[str, Any]]) -> List[str]:
    task_family = scenario.get("task_family") if isinstance(scenario, dict) else None
    if not isinstance(task_family, dict):
        return []
    style = task_family.get("user_request_style")
    if not isinstance(style, dict):
        return []

    instructions: List[str] = []
    if style.get("vague_human_request"):
        instructions.append(
            "Write like a normal human user, not an operator reading internal paths or system metadata."
        )
    if style.get("require_request_form"):
        instructions.append(
            "Phrase the text as a request or question to the assistant, not as a status update, report, or completed action."
        )
    if style.get("allow_exact_paths") is False:
        instructions.append(
            "Do not mention exact file or folder paths in the user request."
        )
    if style.get("allow_exact_links") is False:
        instructions.append(
            "Do not include literal markdown links or exact linked file paths in the user request."
        )
    if style.get("avoid_exact_source_path") or style.get("avoid_exact_target_path"):
        instructions.append(
            "Avoid exact filesystem paths unless the scenario explicitly requires them."
        )
    if style.get("avoid_exact_source_path"):
        instructions.append(
            "Refer to the source file by a natural title, topic, or fuzzy description rather than its exact current path."
        )
    if style.get("avoid_exact_target_path"):
        instructions.append(
            "Refer to the destination or target location by a human folder description rather than an exact target path."
        )
    reference_mode = str(style.get("reference_mode") or "").strip().lower()
    if reference_mode == "title_only":
        instructions.append(
            "Prefer note titles, project names, or topic names over any internal path-like wording."
        )
    elif reference_mode == "folder_purpose":
        instructions.append(
            "Prefer describing folders by their purpose, such as logs, project notes, or meeting notes, rather than their exact names."
        )
    examples = style.get("examples")
    if isinstance(examples, list):
        cleaned_examples = [str(item).strip() for item in examples if str(item).strip()]
        if cleaned_examples:
            instructions.append("Use the following only as style examples. Do not copy them verbatim.")
            instructions.extend(f"- {example}" for example in cleaned_examples[:3])
    return instructions


def _render_template_object(value: Any, template_vars: Dict[str, str], task_context: Optional[Dict[str, Any]] = None) -> Any:
    if isinstance(value, dict):
        return {
            key: _render_template_object(item, template_vars, task_context)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_render_template_object(item, template_vars, task_context) for item in value]
    if isinstance(value, str):
        exact_match = re.fullmatch(r"\{task_([A-Za-z0-9_]+)\}", value.strip())
        if exact_match and isinstance(task_context, dict):
            raw_value = task_context.get(exact_match.group(1))
            if raw_value is not None:
                return deepcopy(raw_value)
        rendered = value
        for key, replacement in template_vars.items():
            rendered = rendered.replace(f"{{{key}}}", str(replacement))
        return rendered
    return deepcopy(value)


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _clean_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").strip("/")
