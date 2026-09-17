"""LLM usage and spend-reference contracts.

Location: ``synaptic_tuner/api/v1/usage.py``.

``UsageRecordV1`` is the single shape in which a paid or metered backend
reports what a run consumed. ``UsageAvailability`` is closed: a ``measured``
record carries exact token counts and, when the backend prices its calls, a
cost in minor units with its currency; an ``unavailable`` record carries no
counters at all. A facade result whose usage is ``unavailable`` omits the
``usage`` field entirely rather than publishing a null, and may claim no
``spend`` effect.

``SpendRef`` names the one spend effect claimed per ``(entity, provider
account)``: the provider, the account it was charged to, and the engine-minted
spend claim id that the ledger binds. A fully local backend has no ``SpendRef``.

Contract only: this module imports nothing from ``tuner.*`` and is registered
in both import-closure gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from ._contract import exact_fields, exact_integer, required_text


USAGE_SCHEMA_VERSION = "synaptic-usage/v1"

_CURRENCY = re.compile(r"^[A-Z]{3}$")


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return required_text(value, name)


def _optional_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return exact_integer(value, name)  # type: ignore[arg-type]


class UsageAvailability(str, Enum):
    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SpendRef:
    provider: str
    account_ref: str
    spend_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _text(self.provider, "provider"))
        object.__setattr__(self, "account_ref", _text(self.account_ref, "account_ref"))
        object.__setattr__(self, "spend_id", _text(self.spend_id, "spend_id"))

    def to_dict(self) -> dict[str, object]:
        return {"provider": self.provider, "account_ref": self.account_ref, "spend_id": self.spend_id}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SpendRef":
        value = exact_fields(value, frozenset({"provider", "account_ref", "spend_id"}), "spend_ref")
        return cls(value["provider"], value["account_ref"], value["spend_id"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class UsageRecordV1:
    schema_version: str
    availability: UsageAvailability
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_minor_units: int | None = None
    currency: str | None = None
    spend: SpendRef | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != USAGE_SCHEMA_VERSION:
            raise ValueError("unsupported usage schema version")
        if type(self.availability) is not UsageAvailability:
            raise TypeError("availability must be exact UsageAvailability")
        object.__setattr__(self, "input_tokens", _optional_integer(self.input_tokens, "input_tokens"))
        object.__setattr__(self, "output_tokens", _optional_integer(self.output_tokens, "output_tokens"))
        object.__setattr__(self, "cost_minor_units", _optional_integer(self.cost_minor_units, "cost_minor_units"))
        if self.currency is not None:
            currency = _text(self.currency, "currency")
            if _CURRENCY.fullmatch(currency) is None:
                raise ValueError("currency must be a three-letter ISO 4217 code")
        if self.spend is not None:
            if type(self.spend) is not SpendRef:
                raise TypeError("spend must be exact SpendRef or None")
            object.__setattr__(self, "spend", SpendRef.from_dict(self.spend.to_dict()))
        if (self.cost_minor_units is None) != (self.currency is None):
            raise ValueError("cost_minor_units and currency must be present together")
        if self.availability is UsageAvailability.MEASURED:
            if self.input_tokens is None or self.output_tokens is None:
                raise ValueError("measured usage requires input_tokens and output_tokens")
        else:
            if any(item is not None for item in (
                self.input_tokens, self.output_tokens, self.cost_minor_units, self.currency, self.spend,
            )):
                raise ValueError("unavailable usage carries no counters, cost or spend")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "availability": self.availability.value,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_minor_units": self.cost_minor_units,
            "currency": self.currency,
            "spend": None if self.spend is None else self.spend.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "UsageRecordV1":
        value = exact_fields(
            value,
            frozenset({
                "schema_version", "availability", "input_tokens", "output_tokens",
                "cost_minor_units", "currency", "spend",
            }),
            "usage_record",
        )
        availability = _text(value["availability"], "availability")
        try:
            parsed = UsageAvailability(availability)
        except ValueError:
            raise ValueError("unknown usage availability") from None
        spend = value["spend"]
        return cls(
            _text(value["schema_version"], "schema_version"),
            parsed,
            value["input_tokens"],  # type: ignore[arg-type]
            value["output_tokens"],  # type: ignore[arg-type]
            value["cost_minor_units"],  # type: ignore[arg-type]
            value["currency"],  # type: ignore[arg-type]
            None if spend is None else SpendRef.from_dict(spend),  # type: ignore[arg-type]
        )


__all__ = ["SpendRef", "UsageAvailability", "UsageRecordV1"]
