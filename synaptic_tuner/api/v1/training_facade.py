"""Planning and start facade; post-start semantics belong to RunsAPI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol

from ._contract import canonical_integer, digest_text, exact_fields, required_text
from .planning import ResolvedTrainingRequest, TrainingPlan
from .providers import ProviderRef
from .results import TrainingRunRef


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    request_id: str
    project_ref: str
    canonical_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", required_text(self.request_id, "request_id"))
        object.__setattr__(self, "project_ref", required_text(self.project_ref, "project_ref"))
        object.__setattr__(self, "canonical_json", required_text(self.canonical_json, "canonical_json"))


@dataclass(frozen=True, slots=True)
class AuthorizationRequirement:
    operation: str
    paid_effect: bool
    maximum_cost_minor_units: int | float | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", required_text(self.operation, "operation"))
        if not isinstance(self.paid_effect, bool):
            raise TypeError("paid_effect must be a boolean")
        amount = self.maximum_cost_minor_units
        currency = self.currency
        if (amount is None) != (currency is None):
            raise ValueError("maximum cost and currency must be supplied together")
        if amount is not None:
            object.__setattr__(
                self,
                "maximum_cost_minor_units",
                canonical_integer(amount, "maximum_cost_minor_units"),
            )
            currency = required_text(currency, "currency")  # type: ignore[arg-type]
            if len(currency) != 3 or any(not "A" <= character <= "Z" for character in currency):
                raise ValueError("currency must be an uppercase three-letter code")
            object.__setattr__(self, "currency", currency)

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "paid_effect": self.paid_effect,
            "maximum_cost_minor_units": self.maximum_cost_minor_units,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AuthorizationRequirement":
        exact_fields(
            value,
            frozenset({"operation", "paid_effect", "maximum_cost_minor_units", "currency"}),
            "authorization_requirement",
        )
        return cls(
            value["operation"], value["paid_effect"], value["maximum_cost_minor_units"],
            value["currency"],  # type: ignore[arg-type]
        )


def _timestamp(value: str, name: str) -> datetime:
    value = required_text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


@dataclass(frozen=True, slots=True)
class TrainingPreflight:
    plan_fingerprint: str
    ready: bool
    checked_at: str
    expires_at: str
    authorization: tuple[AuthorizationRequirement, ...] = ()
    diagnostic_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_fingerprint",
            digest_text(self.plan_fingerprint, "plan_fingerprint"),
        )
        if not isinstance(self.ready, bool):
            raise TypeError("ready must be a boolean")
        checked = _timestamp(self.checked_at, "checked_at")
        expires = _timestamp(self.expires_at, "expires_at")
        if expires <= checked:
            raise ValueError("expires_at must be later than checked_at")
        authorization = tuple(self.authorization)
        if any(not isinstance(item, AuthorizationRequirement) for item in authorization):
            raise TypeError("authorization must contain AuthorizationRequirement values")
        operations = tuple(item.operation for item in authorization)
        if len(operations) != len(set(operations)):
            raise ValueError("authorization operations must be unique")
        codes = tuple(required_text(code, "diagnostic_code") for code in self.diagnostic_codes)
        if len(codes) != len(set(codes)):
            raise ValueError("diagnostic_codes must be unique")
        if not self.ready and not codes:
            raise ValueError("not-ready preflight requires a diagnostic code")
        object.__setattr__(
            self,
            "authorization",
            tuple(sorted(authorization, key=lambda requirement: requirement.operation)),
        )
        object.__setattr__(self, "diagnostic_codes", codes)

    def binds(self, plan: TrainingPlan) -> bool:
        if not isinstance(plan, TrainingPlan):
            raise TypeError("plan must be TrainingPlan")
        return self.plan_fingerprint == plan.plan_fingerprint

    def is_expired(self, now: str) -> bool:
        return _timestamp(now, "now") >= _timestamp(self.expires_at, "expires_at")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "synaptic-training-preflight/v1",
            "plan_fingerprint": self.plan_fingerprint,
            "ready": self.ready,
            "checked_at": self.checked_at,
            "expires_at": self.expires_at,
            "authorization": {
                item.operation: {
                    "paid_effect": item.paid_effect,
                    "maximum_cost_minor_units": item.maximum_cost_minor_units,
                    "currency": item.currency,
                }
                for item in sorted(self.authorization, key=lambda requirement: requirement.operation)
            },
            "diagnostic_codes": list(self.diagnostic_codes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrainingPreflight":
        exact_fields(
            value,
            frozenset(
                {
                    "schema_version", "plan_fingerprint", "ready", "checked_at",
                    "expires_at", "authorization", "diagnostic_codes",
                }
            ),
            "training_preflight",
        )
        if value["schema_version"] != "synaptic-training-preflight/v1":
            raise ValueError("unsupported training preflight schema version")
        authorization = value["authorization"]
        codes = value["diagnostic_codes"]
        if not isinstance(authorization, Mapping):
            raise TypeError("authorization must be an operation-keyed object")
        if any(not isinstance(operation, str) for operation in authorization):
            raise TypeError("authorization operation keys must be strings")
        if any(not isinstance(descriptor, Mapping) for descriptor in authorization.values()):
            raise TypeError("authorization requirements must be objects")
        if not isinstance(codes, list):
            raise TypeError("diagnostic_codes must be an array")
        return cls(
            value["plan_fingerprint"], value["ready"], value["checked_at"],
            value["expires_at"],
            tuple(
                AuthorizationRequirement.from_dict(
                    {"operation": operation, **descriptor}
                )
                for operation, descriptor in sorted(authorization.items())
            ),
            tuple(codes),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TrainingStart:
    run: TrainingRunRef
    accepted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.run, TrainingRunRef):
            raise TypeError("run must be TrainingRunRef")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean")


class TrainingOperations(Protocol):
    def load(self, canonical_json: str) -> TrainingRequest: ...
    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest: ...
    def plan(self, resolved: ResolvedTrainingRequest, provider: ProviderRef) -> TrainingPlan: ...
    def preflight(self, plan: TrainingPlan) -> TrainingPreflight: ...
    def start(self, plan: TrainingPlan, preflight: TrainingPreflight) -> TrainingStart: ...


class Clock(Protocol):
    def now(self) -> str: ...


class TrainingAPI:
    __slots__ = ("_clock", "_operations")

    def __init__(self, operations: TrainingOperations, *, clock: Clock) -> None:
        self._operations = operations
        self._clock = clock

    def load(self, canonical_json: str) -> TrainingRequest:
        return self._operations.load(canonical_json)

    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest:
        return self._operations.resolve(request)

    def plan(self, resolved: ResolvedTrainingRequest, provider: ProviderRef) -> TrainingPlan:
        return self._operations.plan(resolved, provider)

    def preflight(self, plan: TrainingPlan) -> TrainingPreflight:
        if not isinstance(plan, TrainingPlan):
            raise TypeError("plan must be TrainingPlan")
        preflight = self._operations.preflight(plan)
        if not isinstance(preflight, TrainingPreflight):
            raise TypeError("operations preflight must return TrainingPreflight")
        if not preflight.binds(plan):
            raise ValueError("preflight does not bind the exact training plan")
        if preflight.is_expired(self._clock.now()):
            raise ValueError("preflight has expired")
        return preflight

    def start(self, plan: TrainingPlan, preflight: TrainingPreflight) -> TrainingStart:
        if not isinstance(plan, TrainingPlan):
            raise TypeError("plan must be TrainingPlan")
        if not isinstance(preflight, TrainingPreflight):
            raise TypeError("preflight must be TrainingPreflight")
        if not preflight.ready:
            raise ValueError("training plan did not pass preflight")
        if not preflight.binds(plan):
            raise ValueError("preflight does not bind the exact training plan")
        if preflight.is_expired(self._clock.now()):
            raise ValueError("preflight has expired")
        return self._operations.start(plan, preflight)


__all__ = [
    "AuthorizationRequirement", "Clock", "TrainingAPI", "TrainingOperations",
    "TrainingPreflight", "TrainingRequest", "TrainingStart"
]
