"""Conservative resource ledger folded from journal entries (never reset)."""

from __future__ import annotations

from typing import Any, Mapping

from .canon import is_finite_nonneg
from .errors import JournalCorrupt


def clean_amount(value: Any, what: str = "amount") -> float:
    if not is_finite_nonneg(value):
        raise ValueError(f"{what} must be a finite non-negative number, got {value!r}")
    return value


class Ledger:
    """reserved + committed is the charge. Release only on proven no-effect."""

    def __init__(self, limits: Mapping[str, float] | None = None):
        self.limits = dict(limits or {})
        self.committed: dict[str, float] = {}
        self.reserved: dict[str, dict[str, float]] = {}  # ref -> resource -> amount

    def apply(self, entry: Mapping[str, Any]) -> None:
        op, ref, res = entry.get("op"), entry.get("ref"), entry.get("resource")
        try:
            amount = clean_amount(entry.get("amount"))
        except ValueError as exc:
            raise JournalCorrupt(f"ledger entry invalid: {exc}") from exc
        if op == "reserve":
            slot = self.reserved.setdefault(ref, {})
            slot[res] = slot.get(res, 0) + amount
        elif op == "release":
            slot = self.reserved.get(ref, {})
            slot[res] = max(0, slot.get(res, 0) - amount)
        elif op == "commit":
            self.committed[res] = self.committed.get(res, 0) + amount
        else:
            raise JournalCorrupt(f"unknown ledger op {op!r}")

    def outstanding(self, resource: str) -> float:
        return sum(slot.get(resource, 0) for slot in self.reserved.values())

    def charged(self, resource: str) -> float:
        return self.committed.get(resource, 0) + self.outstanding(resource)

    def exceeded_by(self, cost: Mapping[str, float]) -> list[dict]:
        out = []
        for res, amount in cost.items():
            limit = self.limits.get(res)
            if limit is not None and self.charged(res) + amount > limit:
                out.append({"resource": res, "limit": limit, "charged": self.charged(res), "requested": amount})
        return out

    def exhausted(self, resource: str, amount: float = 1) -> bool:
        limit = self.limits.get(resource)
        return limit is not None and self.charged(resource) + amount > limit

    def snapshot(self) -> dict:
        names = sorted(set(self.committed) | {r for s in self.reserved.values() for r in s} | set(self.limits))
        return {
            n: {
                "limit": self.limits.get(n),
                "committed": self.committed.get(n, 0),
                "reserved_outstanding": self.outstanding(n),
                "charged": self.charged(n),
            }
            for n in names
        }
