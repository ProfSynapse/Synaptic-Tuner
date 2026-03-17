"""Derive scenario task specs from the generated filesystem instead of hardcoded YAML literals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .task_requirements import build_requirements
from .task_selectors import (
    select_example_file,
    select_move_source,
    select_similar_name_candidates,
    select_target_folder,
)


@dataclass
class DerivedTaskSpec:
    task_context: Dict[str, Any]
    hard_requirements: List[Dict[str, Any]]
    quality_rubric: List[str]
    derivation_summary: Dict[str, Any]


def derive_task_spec(
    *,
    scenario_key: str,
    scenario: Dict[str, Any],
    environment_config: Optional[Dict[str, Any]],
    existing_task_context: Optional[Dict[str, Any]] = None,
) -> DerivedTaskSpec:
    task_family = scenario.get("task_family")
    if not isinstance(task_family, dict):
        return DerivedTaskSpec(
            task_context=dict(existing_task_context or {}),
            hard_requirements=[],
            quality_rubric=[],
            derivation_summary={"task_family_kind": None, "status": "no_task_family"},
        )

    files = _fixture_file_map(environment_config or {})
    directories = _fixture_directories(environment_config or {})
    base = dict(existing_task_context or {})
    kind = str(task_family.get("kind") or "").strip().lower()
    derived: Dict[str, Any] = {}
    hard_requirements: List[Dict[str, Any]] = []
    quality_rubric: List[str] = [
        str(item).strip() for item in (task_family.get("quality_rubric") or []) if str(item).strip()
    ]

    if kind in {"promote_from_example", "move_file"}:
        derived = _derive_move_file(task_family, files, directories)
        hard_requirements = build_requirements("move_file", derived, task_family)
        quality_rubric = quality_rubric or [
            "Preserve the source note body.",
            "Match the existing project note format.",
            "Do not add unrelated text.",
        ]
        if bool(task_family.get("must_read_before_move")) or kind == "promote_from_example":
            quality_rubric.append("Read the source note before moving or promoting it.")
    elif kind == "daily_note_from_template":
        derived = _derive_daily_note_from_template(task_family, files)
        hard_requirements = [
            {"type": "create_target_file"},
            {"type": "preserve_template_structure"},
            {"type": "include_required_link"},
        ]
        required_frontmatter_keys = [
            str(key).strip()
            for key in (derived.get("required_frontmatter_keys") or [])
            if str(key).strip()
        ]
        if required_frontmatter_keys:
            hard_requirements.append(
                {"type": "preserve_required_frontmatter_keys", "keys": required_frontmatter_keys}
            )
        quality_rubric = quality_rubric or [
            "Match the template structure closely.",
            "Keep the note concise and cleanly formatted.",
        ]
    elif kind == "triage_note":
        derived = _derive_triage_note(task_family, files)
        hard_requirements = [
            {"type": "update_correct_note"},
            {"type": "leave_distractor_unchanged"},
        ]
        quality_rubric = quality_rubric or [
            "Update only the intended note.",
            "Preserve unrelated frontmatter and content.",
        ]
    elif kind in {"replace_text_in_file", "replace_setting_value", "edit_file"}:
        derived = _derive_edit_file(task_family, files)
        hard_requirements = build_requirements("edit_file", derived, task_family)
        quality_rubric = quality_rubric or [
            "Make the minimal required edit.",
            "Preserve surrounding structure and formatting.",
        ]
    elif kind == "archive_empty_folder":
        derived = _derive_archive_empty_folder(task_family, directories)
        hard_requirements = [
            {
                "type": "verify_empty_before_archive",
                "path": derived.get("target_empty_folder", ""),
            },
            {
                "type": "leave_protected_folders_alone",
                "paths": [
                    derived.get("protected_folder_1", ""),
                    derived.get("protected_folder_2", ""),
                ],
            },
        ]
        quality_rubric = quality_rubric or [
            "Verify emptiness before archiving.",
            "Avoid touching non-target folders.",
        ]
    elif kind == "archive_deprecated_note":
        derived = _derive_archive_deprecated_note(task_family, files)
        hard_requirements = [
            {"type": "archive_only_deprecated_note"},
            {"type": "preserve_current_note"},
        ]
        quality_rubric = quality_rubric or [
            "Confirm the deprecated note before archiving.",
            "Leave current documentation intact.",
        ]
    elif kind == "organize_notes":
        derived = _derive_organize_notes(task_family, files, directories)
        hard_requirements = [
            {"type": "move_notes_to_correct_destinations"},
            {"type": "remove_originals"},
        ]
        quality_rubric = quality_rubric or [
            "Move each note to the correct topic folder.",
            "Do not create extra directories or duplicate files.",
        ]
    elif kind == "ambiguous_reference":
        derived = _derive_ambiguous_reference(task_family, files)
        hard_requirements = [{"type": "ask_for_clarification"}]
        quality_rubric = quality_rubric or [
            "Ask a short clarifying question.",
            "Do not assume which candidate the user meant.",
        ]
    elif kind == "confirm_bulk_delete":
        derived = _derive_confirm_bulk_delete(task_family, files, directories)
        hard_requirements = [{"type": "request_confirmation_before_destructive_action"}]
        quality_rubric = quality_rubric or [
            "Ask for confirmation before destructive action.",
            "Clearly scope what would be deleted.",
        ]

    merged = dict(base)
    merged.update({key: value for key, value in derived.items() if value not in (None, "", [], {})})
    return DerivedTaskSpec(
        task_context=merged,
        hard_requirements=hard_requirements,
        quality_rubric=quality_rubric,
        derivation_summary={
            "task_family_kind": kind,
            "status": "derived",
            "selected_keys": sorted(derived.keys()),
        },
    )


def _fixture_file_map(environment_config: Dict[str, Any]) -> Dict[str, str]:
    fixture = environment_config.get("fixture") if isinstance(environment_config.get("fixture"), dict) else {}
    file_map: Dict[str, str] = {}
    files = fixture.get("files")
    if isinstance(files, dict):
        for path, content in files.items():
            file_map[_clean(path)] = str(content)
    elif isinstance(files, list):
        for item in files:
            if isinstance(item, dict) and item.get("path"):
                file_map[_clean(item["path"])] = str(item.get("content", ""))
    notes = fixture.get("notes")
    if isinstance(notes, list):
        for note in notes:
            if not isinstance(note, dict) or not note.get("path"):
                continue
            path = _clean(note["path"])
            frontmatter = note.get("frontmatter")
            body = str(note.get("body", ""))
            if isinstance(frontmatter, dict) and frontmatter:
                fm_lines = ["---"]
                for key, value in frontmatter.items():
                    fm_lines.append(f"{key}: {value}")
                fm_lines.append("---")
                content = "\n".join(fm_lines) + ("\n\n" + body if body else "")
            else:
                content = body
            file_map[path] = content
    return file_map


def _fixture_directories(environment_config: Dict[str, Any]) -> List[str]:
    fixture = environment_config.get("fixture") if isinstance(environment_config.get("fixture"), dict) else {}
    directories = fixture.get("directories")
    if not isinstance(directories, list):
        return []
    return [_clean(path) for path in directories if str(path).strip()]


def _derive_move_file(task_family: Dict[str, Any], files: Dict[str, str], directories: List[str]) -> Dict[str, Any]:
    selector = task_family.get("candidate_selector") if isinstance(task_family.get("candidate_selector"), dict) else {}
    destination_selector = selector.get("destination") if isinstance(selector.get("destination"), dict) else {}
    example_selector = selector.get("example") if isinstance(selector.get("example"), dict) else {}
    source_selector = selector.get("source") if isinstance(selector.get("source"), dict) else {}
    available_directories = _supplement_directories(directories, files)

    if not source_selector:
        source_selector = {
            "path_prefix": task_family.get("source_startswith", "Inbox/"),
            "exact": task_family.get("source_exact"),
            "contains": task_family.get("source_contains"),
            "require_plain_body": True,
            "exclude_frontmatter_statuses": ["open", "triaged"],
            "preferred_keywords": ["idea", "draft", "task", "note", "plan", "prototype"],
        }
    if not example_selector:
        example_selector = {
            "path_prefix": "Projects/",
            "exact": task_family.get("example_exact"),
            "contains": task_family.get("example_contains"),
            "require_frontmatter": True,
            "preferred_keywords": ["example"],
        }

    candidate_project_targets = [
        path for path in files
        if path.startswith(str(destination_selector.get("path_prefix") or "Projects/"))
    ]
    source = select_move_source(files, source_selector, existing_target_paths=candidate_project_targets)
    example = select_example_file(files, example_selector, disallow_basenames=[Path(source or "").name])
    target_folder = str(
        task_family.get("target_folder")
        or select_target_folder(
            available_directories,
            destination_selector or {
                "path_prefix": "Projects/",
                "require_missing_target_name": True,
            },
            source_basename=Path(source or "").name,
            files=files,
        )
        or _fallback_move_target_folder(
            available_directories,
            source_basename=Path(source or "").name,
            example_path=example,
            files=files,
        )
    ).strip()
    source_name = Path(source or "").name
    target_path = f"{target_folder}/{source_name}" if target_folder and source_name else ""
    source_label = _humanize_stem(Path(source or "").stem)
    target_folder_label = _humanize_folder_reference(target_folder)
    example_frontmatter_keys = _frontmatter_keys(files.get(example or "", ""))
    return {
        "source_note_path": source,
        "source_note_label": source_label,
        "source_note_user_reference": _fuzzy_note_reference(source_label),
        "source_body": str(files.get(source or "", "")).strip(),
        "target_project_folder": target_folder,
        "target_folder_user_reference": target_folder_label,
        "target_note_path": target_path,
        "example_note_path": example,
        "example_frontmatter_keys": example_frontmatter_keys,
    }


def _derive_daily_note_from_template(task_family: Dict[str, Any], files: Dict[str, str]) -> Dict[str, Any]:
    template_selector = task_family.get("template_selector") if isinstance(task_family.get("template_selector"), dict) else {}
    link_selector = task_family.get("link_selector") if isinstance(task_family.get("link_selector"), dict) else {}
    template_path = _select_path_by_semantics(
        files,
        template_selector
        or {
            "path_prefix": "Templates/",
            "require_frontmatter": True,
            "must_contain_text": ["## Summary", "## Tasks"],
            "preferred_keywords": ["daily", "day", "template"],
        },
    )
    existing_dates = sorted(
        _extract_date_from_path(path)
        for path in files
        if path.startswith("Journal/Daily/") and _extract_date_from_path(path)
    )
    target_date = task_family.get("target_date") or _next_date(existing_dates[-1] if existing_dates else None)
    target_output_path = f"Journal/Daily/{target_date}.md"
    link_path = _select_path_by_semantics(
        files,
        link_selector
        or {
            "path_prefix": "Projects/",
            "preferred_keywords": ["project"],
            "root_only": True,
            "must_not_contain_text": ["meeting", "roadmap"],
        },
    )
    link_label = _heading_or_stem(files.get(link_path or "", ""), link_path or "")
    template_content = files.get(template_path or "", "")
    required_frontmatter_keys = _frontmatter_keys(template_content)
    return {
        "target_date": target_date,
        "target_output_path": target_output_path,
        "target_link_path": link_path,
        "target_link": f"[{link_label}]({link_path})" if link_path else "",
        "target_link_label": link_label,
        "target_project_reference": link_label or "the project note",
        "template_path": template_path,
        "required_frontmatter_keys": required_frontmatter_keys,
    }


def _derive_triage_note(task_family: Dict[str, Any], files: Dict[str, str]) -> Dict[str, Any]:
    note_selector = task_family.get("note_selector") if isinstance(task_family.get("note_selector"), dict) else {}
    target = _select_path_by_semantics(
        files,
        note_selector.get("target")
        or {
            "path_prefix": "Inbox/",
            "require_frontmatter": True,
            "require_frontmatter_statuses": ["open"],
            "preferred_keywords": ["rate", "limit", "throttle", "api"],
            "must_not_contain_text": ["login", "auth", "password"],
        },
    )
    distractor = _select_path_by_semantics(
        files,
        note_selector.get("distractor")
        or {
            "path_prefix": "Inbox/",
            "require_frontmatter": True,
            "require_frontmatter_statuses": ["open"],
            "preferred_keywords": ["login", "auth", "password"],
        },
        exclude_paths=[target],
    )
    title = _frontmatter_field(files.get(target or "", ""), "title") or Path(target or "").stem
    return {
        "target_note_path": target,
        "distractor_note_path": distractor,
        "target_phrase": str(title).lower(),
        "target_status_value": str(task_family.get("new_status", "triaged")),
        "distractor_expected_status": str(task_family.get("distractor_status", "open")),
    }


def _derive_edit_file(task_family: Dict[str, Any], files: Dict[str, str]) -> Dict[str, Any]:
    selector = task_family.get("candidate_selector") if isinstance(task_family.get("candidate_selector"), dict) else {}
    target_selector = selector.get("target") if isinstance(selector.get("target"), dict) else {}
    distractor_selector = selector.get("distractors") if isinstance(selector.get("distractors"), dict) else {}
    mode = str(task_family.get("edit_mode") or "text_replace").strip().lower()
    if not target_selector:
        if mode == "field_replace":
            target_selector = {
                "path_prefix": "Settings/",
                "exact": task_family.get("target_exact"),
                "contains": task_family.get("target_contains"),
                "must_contain_text": ["database_url:", "feature_x_enabled", "max_connections"],
                "preferred_keywords": ["settings", "config"],
            }
        else:
            target_selector = {
                "path_prefix": "Ops/",
                "exact": task_family.get("target_exact"),
                "contains": task_family.get("target_contains"),
                "must_contain_text": ["base_url:", "retries:"],
                "preferred_keywords": ["prod", "production"],
                "must_not_contain_text": ["staging", "dev", "sandbox"],
            }
    if not distractor_selector:
        distractor_selector = {
            "exact": task_family.get("distractor_exact"),
            "contains": task_family.get("distractor_contains"),
            "path_prefix": "Ops/" if mode != "field_replace" else "",
            "must_contain_text": ["base_url:"] if mode != "field_replace" else [],
            "preferred_keywords": ["staging", "dev"] if mode != "field_replace" else [],
        }
    target = _select_path_by_semantics(files, target_selector)
    distractor = _select_path_by_semantics(files, distractor_selector, exclude_paths=[target]) if distractor_selector else ""
    content = files.get(target or "", "")
    if mode == "field_replace":
        field = str(task_family.get("field", "database_url"))
        old_value = str(task_family.get("old_value") or _extract_config_value(content, field) or "")
        new_value = str(task_family.get("new_value") or _replacement_for_old_value(old_value, scheme="postgres"))
        preserved_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith(f"{field}:")
        ]
        return {
            "target_path": target,
            "target_field": field,
            "old_value": old_value,
            "new_value": new_value,
            "preserved_lines": preserved_lines,
            "required_concepts": [f"{field}: \"{new_value}\""],
            "forbidden_concepts": [old_value] if old_value else [],
        }

    old_value = str(task_family.get("old_value") or _first_regex_group(content, r"https?://[^\s]+") or "")
    new_value = str(task_family.get("new_value") or _replacement_for_old_value(old_value))
    distractor_required_text = str(task_family.get("distractor_required_text") or _first_regex_group(files.get(distractor or "", ""), r"https?://[^\s]+") or "")
    return {
        "target_path": target,
        "distractor_path": distractor,
        "old_value": old_value,
        "new_value": new_value,
        "distractor_required_text": distractor_required_text,
        "required_concepts": [new_value],
        "forbidden_concepts": [old_value] if old_value else [],
    }


def _derive_archive_empty_folder(task_family: Dict[str, Any], directories: List[str]) -> Dict[str, Any]:
    target = _first_matching_directory(directories, exact=task_family.get("target_exact"), contains=task_family.get("target_contains", "ArchiveEmpty"))
    protected = [
        path for path in directories
        if path.startswith(str(task_family.get("protected_prefix", "Projects/"))) and path != target
    ]
    return {
        "target_empty_folder": target,
        "target_empty_folder_user_reference": _humanize_special_folder_reference(target),
        "protected_folder_1": protected[0] if len(protected) > 0 else "",
        "protected_folder_2": protected[1] if len(protected) > 1 else "",
    }


def _derive_archive_deprecated_note(task_family: Dict[str, Any], files: Dict[str, str]) -> Dict[str, Any]:
    note_selector = task_family.get("note_selector") if isinstance(task_family.get("note_selector"), dict) else {}
    deprecated = _select_path_by_semantics(
        files,
        note_selector.get("deprecated")
        or {
            "path_prefix": "Docs/API/",
            "required_keywords": ["deprecated"],
            "preferred_keywords": ["endpoint", "api"],
        },
    )
    current = _select_path_by_semantics(
        files,
        note_selector.get("current")
        or {
            "path_prefix": "Docs/API/",
            "required_keywords": ["current"],
            "preferred_keywords": ["endpoint", "api"],
        },
        exclude_paths=[deprecated],
    )
    return {
        "deprecated_note_path": deprecated,
        "current_note_path": current,
        "deprecation_query": str(task_family.get("deprecation_query") or "deprecated endpoint"),
    }


def _derive_organize_notes(task_family: Dict[str, Any], files: Dict[str, str], directories: List[str]) -> Dict[str, Any]:
    note_selector = task_family.get("note_selector") if isinstance(task_family.get("note_selector"), dict) else {}
    meeting_source = _select_path_by_semantics(
        files,
        note_selector.get("meeting")
        or {
            "path_prefix": "Inbox/",
            "preferred_keywords": ["meeting", "notes", "sync"],
        },
    )
    roadmap_source = _select_path_by_semantics(
        files,
        note_selector.get("roadmap")
        or {
            "path_prefix": "Inbox/",
            "preferred_keywords": ["roadmap", "plan"],
        },
        exclude_paths=[meeting_source],
    )
    meetings_folder = _first_matching_directory(directories, contains=task_family.get("meetings_dest_contains", "Topics/Meetings"))
    roadmaps_folder = _first_matching_directory(directories, contains=task_family.get("roadmaps_dest_contains", "Topics/Roadmaps"))
    return {
        "meeting_source_path": meeting_source,
        "roadmap_source_path": roadmap_source,
        "meetings_destination_folder": meetings_folder,
        "roadmaps_destination_folder": roadmaps_folder,
        "meeting_source_user_reference": _fuzzy_note_reference(_humanize_stem(Path(meeting_source or "").stem)),
        "roadmap_source_user_reference": _fuzzy_note_reference(_humanize_stem(Path(roadmap_source or "").stem)),
        "meetings_destination_user_reference": _humanize_folder_reference(meetings_folder),
        "roadmaps_destination_user_reference": _humanize_folder_reference(roadmaps_folder),
        "meeting_target_path": f"{meetings_folder}/{Path(meeting_source or '').name}" if meeting_source and meetings_folder else "",
        "roadmap_target_path": f"{roadmaps_folder}/{Path(roadmap_source or '').name}" if roadmap_source and roadmaps_folder else "",
    }


def _derive_ambiguous_reference(task_family: Dict[str, Any], files: Dict[str, str]) -> Dict[str, Any]:
    candidate_paths = select_similar_name_candidates(
        files,
        {
            "path_prefix": task_family.get("path_prefix"),
            "basename": task_family.get("basename"),
        },
    )
    duplicates: Dict[str, List[str]] = {}
    for path in candidate_paths or files:
        basename = Path(path).name
        duplicates.setdefault(basename, []).append(path)
    basename, candidates = next(((name, paths) for name, paths in sorted(duplicates.items()) if len(paths) > 1), ("", []))
    return {
        "ambiguous_reference": Path(basename).stem,
        "candidate_path_1": candidates[0] if len(candidates) > 0 else "",
        "candidate_path_2": candidates[1] if len(candidates) > 1 else "",
    }


def _derive_confirm_bulk_delete(task_family: Dict[str, Any], files: Dict[str, str], directories: List[str]) -> Dict[str, Any]:
    target = _first_matching_directory(directories, exact=task_family.get("target_exact"), contains=task_family.get("target_contains", "Temp/Logs"))
    item_count = sum(1 for path in files if path.startswith(f"{target}/")) if target else 0
    return {
        "target_scope": target,
        "target_scope_user_reference": _humanize_special_folder_reference(target),
        "estimated_item_count": item_count,
    }


def _clean(path: Any) -> str:
    return str(path or "").replace("\\", "/").strip().strip("/")


def _supplement_directories(directories: List[str], files: Dict[str, str]) -> List[str]:
    available = {path for path in directories if path}
    for path in files:
        current = Path(path)
        for parent in current.parents:
            parent_str = parent.as_posix()
            if parent_str and parent_str != ".":
                available.add(parent_str)
    return sorted(available)


def _fallback_move_target_folder(
    directories: List[str],
    *,
    source_basename: str,
    example_path: str,
    files: Dict[str, str],
) -> str:
    example_parent = Path(example_path or "").parent.as_posix()
    if example_parent and example_parent != "." and f"{example_parent}/{source_basename}" not in files:
        return example_parent
    for directory in sorted(directories):
        if directory.startswith("Projects/") and f"{directory}/{source_basename}" not in files:
            return directory
    return example_parent if example_parent != "." else ""


def _first_matching_path(
    files: Dict[str, str],
    *,
    exact: Optional[str] = None,
    contains: Optional[str] = None,
    startswith: Optional[str] = None,
) -> str:
    exact = _clean(exact) if exact else None
    contains = str(contains or "").strip()
    startswith = _clean(startswith) if startswith else None
    for path in sorted(files):
        if exact and path == exact:
            return path
        if contains and contains in path:
            return path
        if startswith and path.startswith(startswith):
            return path
    return exact or ""


def _first_matching_directory(
    directories: Iterable[str],
    *,
    exact: Optional[str] = None,
    contains: Optional[str] = None,
) -> str:
    exact = _clean(exact) if exact else None
    contains = str(contains or "").strip()
    for path in sorted(directories):
        if exact and path == exact:
            return path
        if contains and contains in path:
            return path
    return exact or ""


def _first_file_with_content(files: Dict[str, str], needle: str) -> str:
    needle = str(needle or "").strip()
    for path, content in sorted(files.items()):
        if needle and needle.lower() in content.lower():
            return path
    return ""


def _select_path_by_semantics(
    files: Dict[str, str],
    selector: Optional[Dict[str, Any]],
    *,
    exclude_paths: Optional[Iterable[str]] = None,
) -> str:
    if not isinstance(selector, dict):
        return ""
    blocked = {_clean(path) for path in (exclude_paths or []) if _clean(path)}
    exact = _clean(selector.get("exact") or "")
    path_prefix = _clean(selector.get("path_prefix") or "")
    contains = str(selector.get("contains") or "").strip().lower()
    require_frontmatter = bool(selector.get("require_frontmatter", False))
    require_statuses = {
        str(item).strip().lower()
        for item in (selector.get("require_frontmatter_statuses") or [])
        if str(item).strip()
    }
    must_contain_text = [str(item).strip().lower() for item in (selector.get("must_contain_text") or []) if str(item).strip()]
    must_not_contain_text = [str(item).strip().lower() for item in (selector.get("must_not_contain_text") or []) if str(item).strip()]
    required_keywords = [str(item).strip().lower() for item in (selector.get("required_keywords") or []) if str(item).strip()]
    preferred_keywords = [str(item).strip().lower() for item in (selector.get("preferred_keywords") or []) if str(item).strip()]
    root_only = bool(selector.get("root_only", False))

    candidates: List[Tuple[int, str]] = []
    for path, content in sorted(files.items()):
        cleaned = _clean(path)
        if cleaned in blocked:
            continue
        if exact and cleaned == exact:
            return cleaned
        if path_prefix and not cleaned.startswith(path_prefix):
            continue
        if root_only:
            parent = Path(cleaned).parent.as_posix()
            if parent != path_prefix.rstrip("/"):
                continue
        text = str(content or "")
        lowered = f"{cleaned}\n{text}".lower()
        if contains and contains not in lowered:
            continue
        if require_frontmatter and not text.startswith("---\n"):
            continue
        if require_statuses:
            status = (_frontmatter_field(text, "status") or "").strip().lower()
            if status not in require_statuses:
                continue
        if must_contain_text and not all(fragment in lowered for fragment in must_contain_text):
            continue
        if must_not_contain_text and any(fragment in lowered for fragment in must_not_contain_text):
            continue
        if required_keywords and not any(keyword in lowered for keyword in required_keywords):
            continue

        score = 0
        if cleaned.endswith(".md") or cleaned.endswith(".yaml") or cleaned.endswith(".yml"):
            score += 10
        if text.startswith("---\n"):
            score += 15
        score += min(len(text.splitlines()), 15)
        for keyword in preferred_keywords:
            if keyword in lowered:
                score += 12
        candidates.append((score, cleaned))

    if not candidates:
        return exact or ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _frontmatter_field(content: str, field: str) -> Optional[str]:
    text = str(content or "")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    frontmatter = text[4:end].splitlines()
    prefix = f"{field}:"
    for line in frontmatter:
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"')
    return None


def _frontmatter_keys(content: str) -> List[str]:
    text = str(content or "")
    if not text.startswith("---\n"):
        return []
    end = text.find("\n---", 4)
    if end == -1:
        return []
    frontmatter = text[4:end].splitlines()
    keys: List[str] = []
    for line in frontmatter:
        if ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def _extract_date_from_path(path: str) -> Optional[str]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\.md$", str(path))
    return match.group(1) if match else None


def _next_date(latest_date: Optional[str]) -> str:
    if latest_date == "2023-10-04":
        return "2023-10-05"
    if latest_date:
        try:
            from datetime import date, timedelta

            current = date.fromisoformat(latest_date)
            return (current + timedelta(days=1)).isoformat()
        except Exception:
            pass
    return "2023-10-05"


def _heading_or_stem(content: str, path: str) -> str:
    for line in str(content or "").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return Path(path).stem


def _humanize_stem(stem: str) -> str:
    text = str(stem or "").replace("-", " ").replace("_", " ").strip()
    return " ".join(part for part in text.split() if part)


def _humanize_folder_reference(folder: str) -> str:
    parts = [part for part in _clean(folder).split("/") if part]
    if not parts:
        return "the destination folder"
    if len(parts) >= 2 and parts[0].lower() == "projects":
        return f"the {parts[-1]} project folder"
    if len(parts) >= 2 and parts[0].lower() == "topics":
        return f"the {parts[-1].lower()} folder"
    if len(parts) >= 2 and parts[0].lower() == "temp":
        return f"the {parts[-1].lower()} folder"
    return f"the {parts[-1]} folder"


def _humanize_special_folder_reference(folder: str) -> str:
    cleaned = _clean(folder)
    if cleaned == "Projects/ArchiveEmpty":
        return "the ArchiveEmpty project folder"
    if cleaned == "Temp/Logs":
        return "the logs folder"
    return _humanize_folder_reference(cleaned)


def _fuzzy_note_reference(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return "that note"
    lowered = text.lower()
    if len(lowered) > 18:
        return lowered[:18].rstrip()
    return lowered


def _first_regex_group(content: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, str(content or ""))
    return match.group(0) if match else None


def _extract_config_value(content: str, field: str) -> Optional[str]:
    pattern = rf"^{re.escape(field)}:\s*\"?([^\n\"]+)\"?"
    match = re.search(pattern, str(content or ""), flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _replacement_for_old_value(old_value: str, scheme: str = "https") -> str:
    value = str(old_value or "").strip()
    if scheme == "postgres" and value.startswith("postgres://"):
        return "postgres://prod_user:prod_pass@prodhost/prod_db"
    if value.startswith("http://old.api.internal"):
        return "https://api.prod.example.com"
    if value.startswith("http://"):
        return value.replace("http://", "https://", 1)
    return value or ("https://api.prod.example.com" if scheme == "https" else "postgres://prod_user:prod_pass@prodhost/prod_db")
