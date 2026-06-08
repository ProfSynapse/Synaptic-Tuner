"""Answer extraction abstraction shared by verifiers and GRPO rewards.

This module centralizes the various ways an "answer" can be pulled out of a
model completion so that callers don't reimplement format-specific regexes.
The canonical tool-call parsing lives in ``shared.validation.parsing`` and is
reused here for the ``tool_call`` / ``tool_call_args`` modes.

Resolution order in :func:`extract`:
    1. ``output_regex`` (when provided) always wins.
    2. Otherwise, dispatch by ``mode``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from shared.validation.parsing import parse_response


@dataclass(frozen=True)
class ExtractedAnswer:
    """A normalized extraction result.

    Attributes:
        found: Whether anything was extracted.
        tool_name: Tool/function name (for tool-call modes).
        arguments: Structured payload (tool arguments, or parsed JSON block).
        answer_text: Free-text answer (for text modes / regex matches).
        raw: Raw string the answer was derived from.
    """

    found: bool
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    answer_text: str = ""
    raw: str = ""


def extract(
    text: str,
    *,
    mode: str = "tool_call",
    output_regex: str | None = None,
) -> ExtractedAnswer:
    """Extract an answer from ``text``.

    Args:
        text: The model completion to extract from.
        mode: Extraction strategy when ``output_regex`` is not supplied. One of
            ``tool_call``, ``tool_call_args``, ``last_line``, ``boxed``,
            ``json_block``, ``verbatim``.
        output_regex: If provided, takes priority over ``mode``. Matched against
            ``text``; the extracted answer is the named group ``answer`` if the
            pattern defines one, else group(1) if present, else group(0).

    Returns:
        An :class:`ExtractedAnswer`.

    Raises:
        ValueError: If ``mode`` is unknown (and no ``output_regex`` is given).
    """
    if output_regex is not None:
        return _extract_by_regex(text, output_regex)

    if mode == "tool_call" or mode == "tool_call_args":
        return _extract_tool_call(text)
    if mode == "last_line":
        return _extract_last_line(text)
    if mode == "boxed":
        return _extract_boxed(text)
    if mode == "json_block":
        return _extract_json_block(text)
    if mode == "verbatim":
        return ExtractedAnswer(found=bool(text), answer_text=text, raw=text)

    raise ValueError(f"unknown extraction mode: {mode}")


def _extract_by_regex(text: str, output_regex: str) -> ExtractedAnswer:
    """Resolve an answer via ``output_regex`` (highest priority)."""
    match = re.search(output_regex, text)
    if not match:
        return ExtractedAnswer(found=False, raw=text)

    groupdict = match.groupdict()
    if "answer" in groupdict and groupdict["answer"] is not None:
        answer = groupdict["answer"]
    elif match.lastindex:
        answer = match.group(1)
    else:
        answer = match.group(0)

    return ExtractedAnswer(found=True, answer_text=answer, raw=text)


def _extract_tool_call(text: str) -> ExtractedAnswer:
    """Extract the first tool call using the canonical parser."""
    parsed = parse_response(text)
    first = parsed.first_tool_call
    if first is None:
        return ExtractedAnswer(found=False, raw=text)
    return ExtractedAnswer(
        found=True,
        tool_name=first.name,
        arguments=first.arguments,
        raw=first.raw or "",
    )


def _extract_last_line(text: str) -> ExtractedAnswer:
    """Last non-empty, stripped line as the answer."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ExtractedAnswer(found=False, raw=text)
    return ExtractedAnswer(found=True, answer_text=lines[-1], raw=text)


def _extract_boxed(text: str) -> ExtractedAnswer:
    """Content of the last LaTeX ``\\boxed{...}`` (one level of nesting)."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return ExtractedAnswer(found=False, raw=text)

    content_start = start + len(marker)
    depth = 1
    i = content_start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return ExtractedAnswer(
                    found=True,
                    answer_text=text[content_start:i],
                    raw=text,
                )
        i += 1

    # Unbalanced — fall back to a simple flat match.
    flat = re.search(r"\\boxed\{([^{}]*)\}", text)
    if flat:
        return ExtractedAnswer(found=True, answer_text=flat.group(1), raw=text)
    return ExtractedAnswer(found=False, raw=text)


def _extract_json_block(text: str) -> ExtractedAnswer:
    """First ```json fenced block, else first balanced ``{...}`` parsed as JSON."""
    fenced = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate: Optional[str] = None
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        candidate = _first_balanced_object(text)

    if candidate is None:
        return ExtractedAnswer(found=False, raw=text)

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return ExtractedAnswer(found=False, raw=text)

    if not isinstance(parsed, dict):
        return ExtractedAnswer(found=False, raw=text)

    return ExtractedAnswer(found=True, arguments=parsed, raw=text)


def _first_balanced_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
