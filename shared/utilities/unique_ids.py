from __future__ import annotations

import random
from datetime import datetime, timezone


def unique_utc_timestamp(*, now: datetime | None = None, fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return a UTC timestamp with a short nonce to avoid same-second collisions."""
    current = now or datetime.now(timezone.utc)
    base = current.strftime(fmt)
    nonce = f"{random.randrange(0, 0x10000):04x}"
    return f"{base}_{nonce}"


def unique_prefixed_id(prefix: str, *, now: datetime | None = None, fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return a prefixed unique identifier using a UTC timestamp and nonce."""
    return f"{prefix}{unique_utc_timestamp(now=now, fmt=fmt)}"
