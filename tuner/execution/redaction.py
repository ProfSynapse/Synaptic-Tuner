"""Closed, bounded diagnostic redaction that never returns input on failure."""
from __future__ import annotations

import json
import re

_SENSITIVE_KEY = re.compile(
    r"(?ix)(?:^|[_\-\s])(?:password|passphrase|token|access[_\-\s]?token|"
    r"refresh[_\-\s]?token|client[_\-\s]?secret|api[_\-\s]?key|apikey|"
    r"authorization|secret|credential)(?:$|[_\-\s])"
)
_TEXT_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\bbasic\s+[A-Za-z0-9+/=]+"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@"),
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(
        r"(?im)\b(password|passphrase|token|access[_ -]?token|refresh[_ -]?token|"
        r"client[_ -]?secret|api[_ -]?key|apikey|authorization|secret|credential)"
        r"\s*[:=]\s*([^\r\n,;]+)"
    ),
)


def _bounded_text(value: object, limit: int) -> str:
    raw = str(value).encode("utf-8", "replace")[:limit].decode("utf-8", "ignore")
    for pattern in _TEXT_PATTERNS:
        if pattern.pattern.startswith("(?i)(https"):
            raw = pattern.sub(r"\1[REDACTED]@", raw)
        elif pattern.groups:
            raw = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", raw)
        else:
            raw = pattern.sub("[REDACTED]", raw)
    return raw


def _walk(value: object, *, depth: int, max_depth: int, max_items: int, limit: int):
    if depth > max_depth:
        return "[REDACTED:DEPTH]"
    if isinstance(value, dict):
        result = {}
        for index, (key, nested) in enumerate(value.items()):
            if index >= max_items:
                result["[TRUNCATED]"] = "[REDACTED:ITEMS]"
                break
            safe_key = _bounded_text(key, limit)
            result[safe_key] = (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(str(key))
                else _walk(nested, depth=depth + 1, max_depth=max_depth, max_items=max_items, limit=limit)
            )
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _walk(item, depth=depth + 1, max_depth=max_depth, max_items=max_items, limit=limit)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            result.append("[REDACTED:ITEMS]")
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_text(value, limit)


def redact(value: object, *, max_depth: int = 8, max_string_bytes: int = 16384, max_items: int = 256) -> str:
    """Return one bounded JSON diagnostic after structural and serialized passes."""
    try:
        if min(max_depth, max_string_bytes, max_items) < 1:
            raise ValueError
        normalized = _walk(
            value,
            depth=0,
            max_depth=max_depth,
            max_items=max_items,
            limit=max_string_bytes,
        )
        normalized = _walk(
            normalized,
            depth=0,
            max_depth=max_depth,
            max_items=max_items,
            limit=max_string_bytes,
        )
        serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if len(serialized.encode("utf-8")) > max_string_bytes * 4:
            return '"[REDACTED:SIZE]"'
        return serialized
    except Exception:
        return "[REDACTED:ERROR]"
