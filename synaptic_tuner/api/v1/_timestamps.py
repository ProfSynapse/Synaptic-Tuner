"""Private exact timestamp validation shared by staged API contracts."""

from __future__ import annotations

from datetime import datetime
import re


_RFC3339 = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


def require_rfc3339(value: str, name: str) -> str:
    if type(value) is not str or _RFC3339.fullmatch(value) is None:
        raise ValueError(f"{name} must be exact RFC3339")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        raise ValueError(f"{name} must be exact RFC3339") from None
    return value


__all__: list[str] = []
