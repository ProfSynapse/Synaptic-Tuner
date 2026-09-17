"""One ``spend`` effect per paid run: scope, authorization, record and ledger.

Location: ``synaptic_tuner/api/v1/reference/spend.py``.

Slice 11 rules (architecture §4.4): the effect ledger records only effects
whose occurrence can be indeterminate, so LLM spend joins ``stage``,
``submit`` and ``cancel`` as the fourth and last ``EffectKind``. A run against
a paid provider claims exactly ONE spend effect per (run, provider account)
before its first LLM call and closes it once with the usage accumulated over
every call of the run:

* ``found`` when every call reported measured usage; the record carries the
  aggregated ``UsageRecordV1`` and ``provider_job_ref`` is the engine-minted
  spend claim id (providers return no correlation id today);
* ``definitely_absent`` when the run made no call;
* ``indeterminate`` when any call reported no usage: money may have moved and
  the amount is unknown, so the run fails ``spend_indeterminate`` and no
  ``found`` spend is claimed.

Local backends never claim a spend effect; usage may still be recorded on the
result. The record holds only references, model refs and counters: no provider
text, prompt or secret ever reaches it.

The ledger stores one canonical JSON document per effect under the
``effects`` storage partition of the host's ``DurableRecordStorePort``, keyed
``spend/<effect_id>``; the same key is idempotent for a claim (a crash after
the create and before the run's ``start`` resumes on the existing claim).

Used by ``reference/evaluation.py`` and ``reference/data.py``; exercised by
``tests/contract/test_reference_spend_v1.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from synaptic_tuner.api.v1.ports import ClockPort, DurableRecordStorePort, StoragePartition, StoredRecordV1
from synaptic_tuner.api.v1.usage import SpendRef, UsageAvailability, UsageRecordV1
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest, parse_canonical_object, safe_ref
from tuner.execution.foundation_v2.identities import EffectKind

SPEND_EFFECT_SCHEMA_VERSION = "synaptic-reference-spend-effect/v1"
SPEND_EFFECT_KEY_PREFIX = "spend/"
_EFFECTS = StoragePartition.EFFECTS.value
_RECORD_KEYS = frozenset({
    "schema_version", "effect_id", "kind", "entity", "scope", "authorization",
    "command_digest", "state", "provider_job_ref", "usage", "claimed_at", "closed_at",
})


class SpendEffectState(str, Enum):
    CLAIMED = "claimed"
    FOUND = "found"
    DEFINITELY_ABSENT = "definitely_absent"
    INDETERMINATE = "indeterminate"


_CLOSED_STATES = frozenset({
    SpendEffectState.FOUND, SpendEffectState.DEFINITELY_ABSENT, SpendEffectState.INDETERMINATE,
})


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class SpendScopeV1:
    """The provider account a paid backend spends against (``ExecutionScope`` for LLM spend)."""

    provider: str
    account_ref: str
    namespace_ref: str

    def __post_init__(self) -> None:
        for name in ("provider", "account_ref", "namespace_ref"):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))

    def to_dict(self) -> dict[str, object]:
        return {"provider": self.provider, "account_ref": self.account_ref, "namespace_ref": self.namespace_ref}

    @classmethod
    def from_dict(cls, value: object) -> "SpendScopeV1":
        if type(value) is not dict or set(value) != {"provider", "account_ref", "namespace_ref"}:
            raise ValueError("spend scope must carry exactly provider, account_ref and namespace_ref")
        return cls(value["provider"], value["account_ref"], value["namespace_ref"])


@dataclass(frozen=True, slots=True)
class SpendAuthorizationV1:
    """What the host authorised: the provider account, the model set and the cap.

    Its canonical bytes are the effect's ``canonical_command``; the digest is
    stored on the record so a reader can verify the claim it was made under.
    """

    provider: str
    account_ref: str
    models: tuple[str, ...]
    maximum_cost_minor_units: int | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", safe_ref(self.provider, "provider"))
        object.__setattr__(self, "account_ref", safe_ref(self.account_ref, "account_ref"))
        if type(self.models) is not tuple or not self.models:
            raise TypeError("models must be a non-empty tuple of model refs")
        models = tuple(sorted({_text(model, "model") for model in self.models}))
        object.__setattr__(self, "models", models)
        cap = self.maximum_cost_minor_units
        if cap is not None and (type(cap) is not int or cap < 0):
            raise ValueError("maximum_cost_minor_units must be a non-negative integer or None")
        if self.currency is not None:
            _text(self.currency, "currency")
        if (cap is None) != (self.currency is None):
            raise ValueError("maximum_cost_minor_units and currency are set together")

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "account_ref": self.account_ref,
            "models": list(self.models),
            "maximum_cost_minor_units": self.maximum_cost_minor_units,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SpendAuthorizationV1":
        expected = {"provider", "account_ref", "models", "maximum_cost_minor_units", "currency"}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("spend authorization must carry exactly its five fields")
        models = value["models"]
        if type(models) is not list:
            raise ValueError("models must be a list")
        return cls(
            value["provider"], value["account_ref"], tuple(models),
            value["maximum_cost_minor_units"], value["currency"],
        )

    def canonical_command(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def command_digest(self) -> str:
        return domain_digest("spend-authorization", self.canonical_command())


@dataclass(frozen=True, slots=True)
class SpendEntityV1:
    """The run the effect belongs to: family (``evaluation`` / ``data``), project and run id."""

    family: str
    project_ref: str
    run_id: str

    def __post_init__(self) -> None:
        for name in ("family", "project_ref", "run_id"):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))

    def to_dict(self) -> dict[str, object]:
        return {"family": self.family, "project_ref": self.project_ref, "run_id": self.run_id}

    @classmethod
    def from_dict(cls, value: object) -> "SpendEntityV1":
        if type(value) is not dict or set(value) != {"family", "project_ref", "run_id"}:
            raise ValueError("spend entity must carry exactly family, project_ref and run_id")
        return cls(value["family"], value["project_ref"], value["run_id"])


def spend_effect_id(entity: SpendEntityV1, scope: SpendScopeV1) -> str:
    """Deterministic id: one effect per (entity, provider account)."""
    payload = canonical_bytes({"entity": entity.to_dict(), "scope": scope.to_dict()})
    return "spend-" + domain_digest("spend-effect", payload)


def _usage_to_dict(usage: UsageRecordV1 | None) -> dict[str, object] | None:
    if usage is None:
        return None
    if type(usage) is not UsageRecordV1 or usage.availability is not UsageAvailability.MEASURED:
        raise ValueError("a spend record carries measured usage or none")
    document: dict[str, object] = {
        "schema_version": usage.schema_version,
        "availability": usage.availability.value,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_minor_units": usage.cost_minor_units,
        "currency": usage.currency,
        "spend": None if usage.spend is None else usage.spend.to_dict(),
    }
    return document


def _usage_from_dict(value: object) -> UsageRecordV1 | None:
    if value is None:
        return None
    expected = {"schema_version", "availability", "input_tokens", "output_tokens", "cost_minor_units", "currency", "spend"}
    if type(value) is not dict or set(value) != expected:
        raise ValueError("spend usage must carry exactly the usage fields")
    spend = value["spend"]
    return UsageRecordV1(
        value["schema_version"], UsageAvailability(value["availability"]),
        value["input_tokens"], value["output_tokens"], value["cost_minor_units"], value["currency"],
        None if spend is None else SpendRef.from_dict(spend),
    )


@dataclass(frozen=True, slots=True)
class SpendEffectRecordV1:
    """The durable spend effect. Closed record: ids, refs, model refs and counters only."""

    effect_id: str
    entity: SpendEntityV1
    scope: SpendScopeV1
    authorization: SpendAuthorizationV1
    state: SpendEffectState
    claimed_at: str
    provider_job_ref: str | None = None
    usage: UsageRecordV1 | None = None
    closed_at: str | None = None

    def __post_init__(self) -> None:
        if type(self.effect_id) is not str or self.effect_id != spend_effect_id(self.entity, self.scope):
            raise ValueError("effect_id must derive from the entity and scope")
        if type(self.state) is not SpendEffectState:
            raise TypeError("state must be a SpendEffectState")
        if type(self.authorization) is not SpendAuthorizationV1:
            raise TypeError("authorization must be a SpendAuthorizationV1")
        if self.authorization.provider != self.scope.provider or self.authorization.account_ref != self.scope.account_ref:
            raise ValueError("authorization must name the scope's provider account")
        _text(self.claimed_at, "claimed_at")
        closed = self.state in _CLOSED_STATES
        if closed != (self.closed_at is not None):
            raise ValueError("closed_at is set exactly when the effect is closed")
        if self.closed_at is not None:
            _text(self.closed_at, "closed_at")
        if self.state is SpendEffectState.FOUND:
            if self.usage is None or self.provider_job_ref is None:
                raise ValueError("a found spend carries usage and a provider_job_ref")
        elif self.usage is not None or self.provider_job_ref is not None:
            raise ValueError("only a found spend carries usage and a provider_job_ref")
        if self.provider_job_ref is not None:
            safe_ref(self.provider_job_ref, "provider_job_ref")
        _usage_to_dict(self.usage)

    @property
    def kind(self) -> EffectKind:
        return EffectKind.SPEND

    @property
    def spend_ref(self) -> SpendRef:
        return SpendRef(self.scope.provider, self.scope.account_ref, self.effect_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SPEND_EFFECT_SCHEMA_VERSION,
            "effect_id": self.effect_id,
            "kind": EffectKind.SPEND.value,
            "entity": self.entity.to_dict(),
            "scope": self.scope.to_dict(),
            "authorization": self.authorization.to_dict(),
            "command_digest": self.authorization.command_digest(),
            "state": self.state.value,
            "provider_job_ref": self.provider_job_ref,
            "usage": _usage_to_dict(self.usage),
            "claimed_at": self.claimed_at,
            "closed_at": self.closed_at,
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def decode(cls, raw: bytes) -> "SpendEffectRecordV1":
        document = parse_canonical_object(raw, name="spend effect record")
        if set(document) != _RECORD_KEYS:
            raise ValueError("spend effect record must carry exactly its fields")
        if document["schema_version"] != SPEND_EFFECT_SCHEMA_VERSION or document["kind"] != EffectKind.SPEND.value:
            raise ValueError("unsupported spend effect record")
        authorization = SpendAuthorizationV1.from_dict(document["authorization"])
        if document["command_digest"] != authorization.command_digest():
            raise ValueError("spend effect record command digest does not match its authorization")
        record = cls(
            document["effect_id"],  # type: ignore[arg-type]
            SpendEntityV1.from_dict(document["entity"]),
            SpendScopeV1.from_dict(document["scope"]),
            authorization,
            SpendEffectState(document["state"]),
            document["claimed_at"],  # type: ignore[arg-type]
            document["provider_job_ref"],  # type: ignore[arg-type]
            _usage_from_dict(document["usage"]),
            document["closed_at"],  # type: ignore[arg-type]
        )
        return record


class SpendLedgerV1:
    """Claims and closes spend effects in the ``effects`` partition of a record store."""

    __slots__ = ("_records", "_clock")

    def __init__(self, records: DurableRecordStorePort, clock: ClockPort) -> None:
        self._records = records
        self._clock = clock

    @staticmethod
    def _key(effect_id: str) -> str:
        return SPEND_EFFECT_KEY_PREFIX + safe_ref(effect_id, "effect_id")

    def _read(self, effect_id: str) -> tuple[SpendEffectRecordV1, int] | None:
        key = self._key(effect_id)
        stored = self._records.read(partition=_EFFECTS, key=key)
        if stored is None:
            return None
        if type(stored) is not StoredRecordV1 or stored.key != key:
            raise ValueError("spend ledger returned a foreign record")
        return SpendEffectRecordV1.decode(stored.canonical), stored.revision

    def get(self, effect_id: str) -> SpendEffectRecordV1 | None:
        found = self._read(effect_id)
        return None if found is None else found[0]

    def claim(
        self, entity: SpendEntityV1, scope: SpendScopeV1, authorization: SpendAuthorizationV1
    ) -> SpendEffectRecordV1:
        """Claim the run's single spend effect before its first call.

        A second claim for the same (entity, scope) returns the existing
        record while it is still ``claimed``; a closed record means the run
        already spent under this claim and cannot be re-opened.
        """
        record = SpendEffectRecordV1(
            spend_effect_id(entity, scope), entity, scope, authorization,
            SpendEffectState.CLAIMED, self._clock.now(),
        )
        if self._records.create(partition=_EFFECTS, key=self._key(record.effect_id), canonical=record.encode()):
            return record
        existing = self._read(record.effect_id)
        if existing is None:
            raise RuntimeError("spend effect vanished between create and read")
        current = existing[0]
        if current.state is not SpendEffectState.CLAIMED:
            raise RuntimeError("spend effect is already closed")
        if current.authorization != authorization:
            raise RuntimeError("spend effect was claimed under a different authorization")
        return current

    def close(
        self,
        effect_id: str,
        state: SpendEffectState,
        *,
        usage: UsageRecordV1 | None = None,
        provider_job_ref: str | None = None,
    ) -> SpendEffectRecordV1:
        """Close a claimed effect exactly once with the run's outcome."""
        if state not in _CLOSED_STATES:
            raise ValueError("close requires a closed state")
        found = self._read(effect_id)
        if found is None:
            raise KeyError(effect_id)
        current, revision = found
        if current.state is not SpendEffectState.CLAIMED:
            raise RuntimeError("spend effect is already closed")
        if state is SpendEffectState.FOUND and provider_job_ref is None:
            provider_job_ref = current.effect_id
        closed = SpendEffectRecordV1(
            current.effect_id, current.entity, current.scope, current.authorization, state,
            current.claimed_at, provider_job_ref, usage, self._clock.now(),
        )
        swapped = self._records.compare_and_swap(
            partition=_EFFECTS, key=self._key(effect_id), expected_revision=revision, canonical=closed.encode()
        )
        if not swapped:
            raise RuntimeError("spend effect changed underneath the close")
        return closed


__all__ = [
    "SPEND_EFFECT_KEY_PREFIX",
    "SPEND_EFFECT_SCHEMA_VERSION",
    "SpendAuthorizationV1",
    "SpendEffectRecordV1",
    "SpendEffectState",
    "SpendEntityV1",
    "SpendLedgerV1",
    "SpendScopeV1",
    "spend_effect_id",
]
