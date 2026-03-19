"""Generic hard-requirement builders for task families."""

from __future__ import annotations

from typing import Any, Dict, List


def build_requirements(kind: str, task_context: Dict[str, Any], family_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized = str(kind or "").strip().lower()
    if normalized == "move_file":
        return build_move_requirements(task_context, family_config)
    if normalized == "edit_file":
        return build_edit_requirements(task_context, family_config)
    if normalized == "write_new_file":
        return build_write_requirements(task_context, family_config)
    if normalized == "archive_file":
        return build_archive_requirements(task_context, family_config)
    if normalized == "answer_question":
        return build_answer_requirements(task_context, family_config)
    return []


def build_move_requirements(task_context: Dict[str, Any], family_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    family_kind = str(family_config.get("kind") or "").strip().lower()
    preserve_example_format = bool(family_config.get("preserve_example_format")) or family_kind == "promote_from_example"
    requirements = [
        {"type": "correct_target_path"},
        {"type": "remove_or_archive_source"},
    ]
    if family_config.get("preserve_source_body", True):
        requirements.append({"type": "preserve_source_body"})
    if preserve_example_format:
        requirements.append({"type": "preserve_example_format"})
        frontmatter_keys = [
            str(key).strip()
            for key in (task_context.get("example_frontmatter_keys") or [])
            if str(key).strip()
        ]
        if frontmatter_keys:
            requirements.append({"type": "target_frontmatter_keys_present", "keys": frontmatter_keys})
    if family_config.get("must_not_overwrite_existing", True):
        requirements.append({"type": "must_not_overwrite_existing"})
    return requirements


def build_edit_requirements(task_context: Dict[str, Any], family_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    requirements = [
        {"type": "update_correct_target"},
    ]
    if family_config.get("must_read_before_edit", True):
        requirements.append({"type": "must_read_before_edit"})
    if task_context.get("forbidden_concepts"):
        requirements.append({"type": "remove_forbidden_concepts"})
    if task_context.get("required_concepts"):
        requirements.append({"type": "include_required_concepts"})
    if family_config.get("preserve_structure", True):
        requirements.append({"type": "preserve_structure"})
    if family_config.get("preserve_other_fields", False):
        requirements.append({"type": "preserve_other_fields"})
    return requirements


def build_write_requirements(task_context: Dict[str, Any], family_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    requirements = [{"type": "create_target_file"}]
    if family_config.get("must_not_overwrite_existing", True):
        requirements.append({"type": "must_not_overwrite_existing"})
    return requirements


def build_archive_requirements(task_context: Dict[str, Any], family_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    requirements = [{"type": "archive_only_target"}]
    if family_config.get("must_verify_before_archive", True):
        requirements.append({"type": "must_verify_before_archive"})
    if family_config.get("must_clarify_if_multiple_candidates"):
        requirements.append({"type": "must_clarify_if_multiple_candidates"})
    return requirements


def build_answer_requirements(task_context: Dict[str, Any], family_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    requirements = [{"type": "answer_from_retrieved_content"}]
    if family_config.get("must_search_before_answer", True):
        requirements.append({"type": "must_search_before_answer"})
    if family_config.get("must_read_multiple_files"):
        requirements.append({"type": "must_read_multiple_files"})
    return requirements
