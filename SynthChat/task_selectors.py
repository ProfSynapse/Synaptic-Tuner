"""Generic candidate selectors for SynthChat task families."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def select_move_source(
    files: Dict[str, str],
    selector: Dict[str, object],
    *,
    existing_target_paths: Optional[Iterable[str]] = None,
) -> str:
    path_prefix = _clean(selector.get("path_prefix") or "")
    exact = _clean(selector.get("exact") or "")
    contains = str(selector.get("contains") or "").strip().lower()
    require_plain_body = bool(selector.get("require_plain_body", False))
    exclude_frontmatter_statuses = {
        str(status).strip().lower()
        for status in (selector.get("exclude_frontmatter_statuses") or [])
        if str(status).strip()
    }
    preferred_keywords = [
        str(keyword).strip().lower()
        for keyword in (selector.get("preferred_keywords") or [])
        if str(keyword).strip()
    ]
    existing_target_basenames = {Path(path).name for path in (existing_target_paths or [])}

    candidates: List[Tuple[int, str]] = []
    for path, content in sorted(files.items()):
        if exact and path == exact:
            return path
        if path_prefix and not path.startswith(path_prefix):
            continue
        if contains and contains not in path.lower() and contains not in content.lower():
            continue
        basename = Path(path).name
        if basename in existing_target_basenames:
            continue
        text = str(content or "")
        score = 0
        if require_plain_body and not text.startswith("---\n"):
            score += 100
        if text.startswith("---\n"):
            status = _frontmatter_status(text)
            if status and status.lower() in exclude_frontmatter_statuses:
                score -= 100
        if 10 <= len(text.strip()) <= 500:
            score += 20
        stem = Path(path).stem.lower()
        for keyword in preferred_keywords:
            if keyword in stem or keyword in text.lower():
                score += 15
        if path.endswith(".md"):
            score += 5
        candidates.append((score, path))

    if not candidates:
        return exact or ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def select_example_file(
    files: Dict[str, str],
    selector: Dict[str, object],
    *,
    disallow_basenames: Optional[Iterable[str]] = None,
) -> str:
    path_prefix = _clean(selector.get("path_prefix") or "")
    exact = _clean(selector.get("exact") or "")
    contains = str(selector.get("contains") or "").strip().lower()
    require_frontmatter = bool(selector.get("require_frontmatter", False))
    preferred_keywords = [
        str(keyword).strip().lower()
        for keyword in (selector.get("preferred_keywords") or [])
        if str(keyword).strip()
    ]
    blocked = {str(name) for name in (disallow_basenames or [])}

    candidates: List[Tuple[int, str]] = []
    for path, content in sorted(files.items()):
        if exact and path == exact:
            return path
        if path_prefix and not path.startswith(path_prefix):
            continue
        if contains and contains not in path.lower() and contains not in content.lower():
            continue
        if Path(path).name in blocked:
            continue
        text = str(content or "")
        score = 0
        if require_frontmatter and text.startswith("---\n"):
            score += 100
        elif require_frontmatter:
            continue
        if text.startswith("---\n"):
            lowered = text.lower()
            if "\ntitle:" in lowered:
                score += 10
            if "\nstatus:" in lowered:
                score += 30
        for keyword in preferred_keywords:
            if keyword in path.lower() or keyword in text.lower():
                score += 15
        candidates.append((score, path))

    if not candidates:
        return exact or ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def select_target_folder(
    directories: Iterable[str],
    selector: Dict[str, object],
    *,
    source_basename: str,
    files: Dict[str, str],
) -> str:
    path_prefix = _clean(selector.get("path_prefix") or "")
    exact = _clean(selector.get("exact") or "")
    contains = str(selector.get("contains") or "").strip().lower()
    require_missing_target_name = bool(selector.get("require_missing_target_name", False))

    candidates: List[Tuple[int, str]] = []
    for path in sorted(directories):
        if exact and path == exact:
            return path
        if path_prefix and not path.startswith(path_prefix):
            continue
        if contains and contains not in path.lower():
            continue
        score = 0
        if require_missing_target_name and source_basename:
            candidate_target = f"{path}/{source_basename}"
            if candidate_target in files:
                continue
            score += 50
        if path.count("/") == 1:
            score += 10
        candidates.append((score, path))

    if not candidates:
        return exact or ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def select_edit_target(files: Dict[str, str], selector: Dict[str, object]) -> str:
    exact = _clean(selector.get("exact") or "")
    contains = str(selector.get("contains") or "").strip().lower()
    path_prefix = _clean(selector.get("path_prefix") or "")
    must_contain_text = [
        str(text).strip().lower()
        for text in (selector.get("must_contain_text") or [])
        if str(text).strip()
    ]

    candidates: List[Tuple[int, str]] = []
    for path, content in sorted(files.items()):
        if exact and path == exact:
            return path
        if path_prefix and not path.startswith(path_prefix):
            continue
        text = str(content or "")
        if contains and contains not in path.lower() and contains not in text.lower():
            continue
        if must_contain_text and not all(fragment in text.lower() for fragment in must_contain_text):
            continue
        score = 0
        if path.endswith(".md") or path.endswith(".yaml") or path.endswith(".yml"):
            score += 10
        score += min(len(text.splitlines()), 20)
        candidates.append((score, path))

    if not candidates:
        return exact or ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def select_similar_name_candidates(files: Dict[str, str], selector: Dict[str, object]) -> List[str]:
    path_prefix = _clean(selector.get("path_prefix") or "")
    basename = str(selector.get("basename") or "").strip().lower()
    pairs: List[str] = []
    for path in sorted(files):
        if path_prefix and not path.startswith(path_prefix):
            continue
        if basename and basename not in Path(path).stem.lower():
            continue
        pairs.append(path)
    return pairs


def _clean(value: object) -> str:
    return str(value or "").replace("\\", "/").strip().strip("/")


def _frontmatter_status(content: str) -> str:
    if not str(content or "").startswith("---\n"):
        return ""
    lines = str(content).splitlines()[1:]
    for line in lines:
        if line.strip() == "---":
            break
        if line.strip().lower().startswith("status:"):
            return line.split(":", 1)[1].strip()
    return ""
