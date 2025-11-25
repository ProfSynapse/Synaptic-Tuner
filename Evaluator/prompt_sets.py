"""Prompt set loading utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .config import PromptFilter


@dataclass
class PromptCase:
    """Single evaluation prompt with metadata."""

    case_id: str
    question: str
    tags: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def chat_messages(self) -> List[Dict[str, str]]:
        """Return ChatML-style messages for this test case."""
        system_prompt = self.metadata.get("system")
        messages: List[Dict[str, str]] = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.question})
        return messages


def load_prompt_cases(path: Path) -> List[PromptCase]:
    """Load prompt cases from a JSON or JSONL file."""
    if path.suffix.lower() == ".jsonl":
        raw_cases = _load_jsonl(path)
    else:
        raw_cases = _load_json(path)
    return [_build_case(idx, record) for idx, record in enumerate(raw_cases)]


def filter_prompts(cases: Sequence[PromptCase], prompt_filter: PromptFilter) -> List[PromptCase]:
    """Apply tag + limit filters."""
    selected: List[PromptCase] = []
    for case in cases:
        if prompt_filter.matches(case.tags):
            selected.append(case)
        if prompt_filter.limit and len(selected) >= prompt_filter.limit:
            break
    return selected


def _build_case(idx: int, payload: Mapping[str, Any]) -> PromptCase:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Prompt entry #{idx + 1} must be an object")

    question = payload.get("question") or payload.get("prompt")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Prompt entry #{idx + 1} missing 'question' string")

    case_id = payload.get("id") or f"case_{idx + 1:04d}"
    tags = _string_list(payload.get("tags", []))
    expected = _string_list(payload.get("expected_tools", []))

    metadata = {k: v for k, v in payload.items() if k not in {"id", "question", "prompt", "tags", "expected_tools"}}
    return PromptCase(case_id=case_id, question=question.strip(), tags=tags, expected_tools=expected, metadata=metadata)


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable):
        raise ValueError("tags/expected_tools must be strings or iterable of strings")
    result = []
    for entry in value:
        if isinstance(entry, str):
            clean = entry.strip()
            if clean:
                result.append(clean)
        else:
            raise ValueError("tags/expected_tools entries must be strings")
    return result


def _load_json(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("Prompt JSON must contain a list of entries")
        return data


def _load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    entries: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {idx}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL entry on line {idx} must be an object")
            entries.append(payload)
    return entries
