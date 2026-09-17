"""Training reference composition: request ports, host authorization bridge, service export.

Location: ``synaptic_tuner/api/v1/reference/training.py``.

``TrainingOperations`` (``api/v1/training_facade.py``) is implemented by
``CoordinatorTrainingService`` (``tuner/training/coordinator_service.py``);
this module is its public export site and builds it for the neutral
composition (``build_training_operations``).

Two host-facing pieces live here because they exist only to make training
start authorizable from the public ports:

- ``ReferenceRequestPortsV1`` bundles the host-specific request handling the
  architecture's five ports do not cover: the request ``loader``, the
  ``resolver`` to immutable inputs and the ``run_identity`` that names a run
  for a plan.
- ``ReferenceAuthorizationV1`` implements the coordinator's internal
  ``AuthorizationPortV1`` over the public ``GrantAuthorityPort``. Committing
  a preflight asks the host to ``authorize`` the preflight's requirements and
  records the host grant; every effect grant then goes back through the host
  ``bind`` hook with the exact canonical command before the engine's
  ``GrantAuthorityV2`` issues the time-bound foundation grant. Reconciliation
  grants are issued against the latest commitment for the plan. Grants expire
  at the earlier of the preflight expiry and ``maximum_grant_seconds``.
  Commitments are durable: they live in the host record store's
  ``authorization`` partition (``policy/<policy_digest>`` holds the
  commitment, ``plan/<plan_fingerprint>`` names the latest policy for the
  plan), so a host that recomposes over the same store can still issue the
  effect and reconciliation grants an in-flight run needs. A commitment
  carries the host grant *reference* only, never a credential value.

Consumed by ``synaptic_tuner/api/v1/reference/provider_family.py`` and
``synaptic_tuner/api/v1/reference/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock

from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.planning import TrainingPlan
from synaptic_tuner.api.v1.ports import (
    DurableRecordStorePort,
    GrantAuthorityPort,
    StoragePartition,
    StoredRecordV1,
)
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.coordinator_v1.coordinator import ReconciliationGrantSlotV1
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.foundation_v2.authority import GrantAuthorityV2, ReconciliationGrantContentV1
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.repository import EffectRecordV2
from tuner.training.coordinator_service import CoordinatorTrainingService


COMMITMENT_SCHEMA = "synaptic-reference-authorization-commitment/v1"
COMMITMENT_PLAN_INDEX_SCHEMA = "synaptic-reference-authorization-plan-index/v1"
_AUTHORIZATION = StoragePartition.AUTHORIZATION.value
_INDEX_ATTEMPTS = 3


class ReferenceAuthorizationError(ValueError):
    """Closed authorization refusal; carries no command or credential content."""


@dataclass(frozen=True, slots=True)
class ReferenceRequestPortsV1:
    """Host request handling: ``load``, ``resolve`` and ``for_plan``."""

    loader: object
    resolver: object
    run_identity: object


@dataclass(frozen=True, slots=True)
class _Commitment:
    plan_fingerprint: str
    preflight: TrainingPreflight
    host_grant: ExecutionGrant
    policy_digest: str
    requirement_digest: str
    expires_at_epoch: int


def _epoch(timestamp: str) -> int:
    value = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    return int(datetime.fromisoformat(value).timestamp())


def _commitment_digests(plan_fingerprint: str, preflight_document: dict, host_grant_ref: str) -> tuple[str, str]:
    """The policy and requirement digests a commitment is keyed and bound by."""
    policy = domain_digest(
        "synaptic-reference-preflight-policy/v1",
        canonical_bytes(
            {
                "plan_fingerprint": plan_fingerprint,
                "preflight": preflight_document,
                "host_grant_ref": host_grant_ref,
            }
        ),
    )
    requirement = domain_digest(
        "synaptic-reference-requirements/v1",
        canonical_bytes({"authorization": preflight_document["authorization"]}),
    )
    return policy, requirement


def _commitment_document(commitment: _Commitment) -> bytes:
    return canonical_bytes(
        {
            "schema_version": COMMITMENT_SCHEMA,
            "plan_fingerprint": commitment.plan_fingerprint,
            "preflight": commitment.preflight.to_dict(),
            "host_grant_ref": commitment.host_grant.grant_ref,
            "policy_digest": commitment.policy_digest,
            "requirement_digest": commitment.requirement_digest,
            "expires_at_epoch": commitment.expires_at_epoch,
        }
    )


def _decode_commitment(raw: bytes, policy_digest: str) -> _Commitment:
    """Rebuild a stored commitment and prove its digests re-derive from its content."""
    try:
        document = parse_canonical_object(raw, name="authorization commitment")
        if set(document) != {
            "schema_version", "plan_fingerprint", "preflight", "host_grant_ref",
            "policy_digest", "requirement_digest", "expires_at_epoch",
        } or document["schema_version"] != COMMITMENT_SCHEMA:
            raise ValueError("commitment shape")
        preflight = TrainingPreflight.from_dict(document["preflight"])
        host_grant = ExecutionGrant(document["host_grant_ref"])
        expires_at_epoch = document["expires_at_epoch"]
        if type(expires_at_epoch) is not int or expires_at_epoch != _epoch(preflight.expires_at):
            raise ValueError("commitment expiry")
        policy, requirement = _commitment_digests(
            document["plan_fingerprint"], preflight.to_dict(), host_grant.grant_ref
        )
        if (
            policy != policy_digest
            or document["policy_digest"] != policy
            or document["requirement_digest"] != requirement
            or preflight.plan_fingerprint != document["plan_fingerprint"]
        ):
            raise ValueError("commitment digests")
        commitment = _Commitment(
            document["plan_fingerprint"], preflight, host_grant, policy, requirement, expires_at_epoch
        )
    except Exception:
        raise ReferenceAuthorizationError("authorization_invalid") from None
    if _commitment_document(commitment) != raw:
        raise ReferenceAuthorizationError("authorization_invalid")
    return commitment


class ReferenceAuthorizationV1:
    """``AuthorizationPortV1`` bridging the public ``GrantAuthorityPort``."""

    def __init__(
        self,
        *,
        records: DurableRecordStorePort,
        grants: GrantAuthorityPort,
        authority: GrantAuthorityV2,
        clock,
        maximum_grant_seconds: int = 900,
        owner_ref: str = "synaptic-reference-coordinator",
    ) -> None:
        if type(authority) is not GrantAuthorityV2:
            raise TypeError("exact GrantAuthorityV2 required")
        if type(maximum_grant_seconds) is not int or not 1 <= maximum_grant_seconds <= 3600:
            raise ValueError("maximum_grant_seconds must be within 1..3600")
        self._records = records
        self._grants = grants
        self._authority = authority
        self._clock = clock
        self._maximum = maximum_grant_seconds
        self._owner_ref = safe_ref(owner_ref, "owner_ref")
        self._lock = RLock()

    # -- durable commitments -------------------------------------------------

    def _read(self, key: str) -> StoredRecordV1 | None:
        try:
            stored = self._records.read(partition=_AUTHORIZATION, key=key)
        except Exception:
            raise ReferenceAuthorizationError("authorization_invalid") from None
        if stored is None:
            return None
        if type(stored) is not StoredRecordV1 or stored.key != key:
            raise ReferenceAuthorizationError("authorization_invalid")
        return stored

    @staticmethod
    def _policy_key(policy_digest: object) -> str:
        try:
            return f"policy/{digest_text(policy_digest, 'policy_digest')}"
        except Exception:
            raise ReferenceAuthorizationError("authorization_invalid") from None

    @staticmethod
    def _plan_key(plan_fingerprint: object) -> str:
        try:
            return f"plan/{digest_text(plan_fingerprint, 'plan_fingerprint')}"
        except Exception:
            raise ReferenceAuthorizationError("authorization_invalid") from None

    def _commitment(self, policy_digest: object) -> _Commitment | None:
        stored = self._read(self._policy_key(policy_digest))
        if stored is None:
            return None
        return _decode_commitment(stored.canonical, policy_digest)

    def _policy_for_plan(self, plan_fingerprint: str) -> str | None:
        stored = self._read(self._plan_key(plan_fingerprint))
        if stored is None:
            return None
        try:
            document = parse_canonical_object(stored.canonical, name="authorization plan index")
            if set(document) != {"schema_version", "policy_digest"} or (
                document["schema_version"] != COMMITMENT_PLAN_INDEX_SCHEMA
            ):
                raise ValueError("plan index shape")
            return digest_text(document["policy_digest"], "policy_digest")
        except Exception:
            raise ReferenceAuthorizationError("authorization_invalid") from None

    @staticmethod
    def _plan_index_document(policy_digest: str) -> bytes:
        return canonical_bytes(
            {"schema_version": COMMITMENT_PLAN_INDEX_SCHEMA, "policy_digest": policy_digest}
        )

    def _store_commitment(self, commitment: _Commitment) -> None:
        """Write the commitment once and point the plan index at it (latest wins)."""
        policy_key = self._policy_key(commitment.policy_digest)
        canonical = _commitment_document(commitment)
        try:
            admitted = self._records.put_if_absent(
                partition=_AUTHORIZATION, key=policy_key, canonical=canonical
            )
        except Exception:
            raise ReferenceAuthorizationError("authorization_invalid") from None
        if admitted is not True:
            existing = self._commitment(commitment.policy_digest)
            if existing != commitment:
                raise ReferenceAuthorizationError("authorization_conflict")
        plan_key = self._plan_key(commitment.plan_fingerprint)
        index = self._plan_index_document(commitment.policy_digest)
        for _ in range(_INDEX_ATTEMPTS):
            stored = self._read(plan_key)
            try:
                if stored is None:
                    admitted = self._records.create(
                        partition=_AUTHORIZATION, key=plan_key, canonical=index
                    )
                elif stored.canonical == index:
                    return
                else:
                    admitted = self._records.compare_and_swap(
                        partition=_AUTHORIZATION,
                        key=plan_key,
                        expected_revision=stored.revision,
                        canonical=index,
                    )
            except Exception:
                raise ReferenceAuthorizationError("authorization_invalid") from None
            if admitted is True:
                return
        raise ReferenceAuthorizationError("authorization_conflict")

    # -- AuthorizationPortV1 -------------------------------------------------

    def commit_preflight(self, plan: TrainingPlan, preflight: TrainingPreflight) -> str:
        if type(plan) is not TrainingPlan or type(preflight) is not TrainingPreflight:
            raise ReferenceAuthorizationError("authorization_invalid")
        if (
            preflight.ready is not True
            or not preflight.binds(plan)
            or preflight.is_expired(self._clock.now())
        ):
            raise ReferenceAuthorizationError("authorization_invalid")
        host_grant = self._grants.authorize(preflight.authorization)
        if type(host_grant) is not ExecutionGrant:
            raise ReferenceAuthorizationError("authorization_refused")
        policy, requirement = _commitment_digests(
            plan.plan_fingerprint, preflight.to_dict(), host_grant.grant_ref
        )
        commitment = _Commitment(
            plan.plan_fingerprint, preflight, host_grant, policy, requirement, _epoch(preflight.expires_at)
        )
        with self._lock:
            self._store_commitment(commitment)
        return policy

    def _expiry(self, commitment: _Commitment, now_epoch: int) -> int:
        if type(now_epoch) is not int or now_epoch >= commitment.expires_at_epoch:
            raise ReferenceAuthorizationError("authorization_expired")
        return min(commitment.expires_at_epoch, now_epoch + self._maximum)

    def issue_effect_grant(self, command_bytes: bytes, *, preflight_digest: str, now_epoch: int):
        with self._lock:
            commitment = self._commitment(preflight_digest)
        if commitment is None or type(command_bytes) is not bytes:
            raise ReferenceAuthorizationError("authorization_invalid")
        command = parse_exact_command(command_bytes)
        if command.preparation.plan_fingerprint != commitment.plan_fingerprint:
            raise ReferenceAuthorizationError("authorization_invalid")
        expires = self._expiry(commitment, now_epoch)
        self._grants.bind(
            commitment.host_grant,
            operation=command,
            requirements=commitment.preflight.authorization,
        )
        kind = command.operation.effect.kind.value
        return self._authority.issue(
            command.canonical_bytes,
            grant_ref=f"reference-{kind}-{command.digest[:24]}",
            policy_digest=commitment.policy_digest,
            requirement_digest=commitment.requirement_digest,
            not_before_epoch=now_epoch,
            expires_at_epoch=expires,
        )

    def issue_reconciliation_grant(
        self,
        record: EffectRecordV2,
        binding: ProviderExecutionBindingV1,
        *,
        slot: ReconciliationGrantSlotV1,
        now_epoch: int,
    ):
        if (
            type(record) is not EffectRecordV2
            or type(binding) is not ProviderExecutionBindingV1
            or type(slot) is not ReconciliationGrantSlotV1
        ):
            raise ReferenceAuthorizationError("authorization_invalid")
        command = parse_exact_command(record.command_bytes)
        preparation = command.preparation
        with self._lock:
            policy = self._policy_for_plan(preparation.plan_fingerprint)
            commitment = None if policy is None else self._commitment(policy)
        if commitment is None:
            raise ReferenceAuthorizationError("authorization_invalid")
        expires = self._expiry(commitment, now_epoch)
        content = ReconciliationGrantContentV1(
            f"reference-reconcile-{slot.generation}-{slot.ownership_epoch}-{command.digest[:16]}",
            command.digest,
            command.operation.effect.effect_id,
            preparation.preparation_digest,
            binding.reconciliation_adapter_digest,
            preparation.provider.provider_id,
            preparation.provider.profile_ref,
            preparation.scope.account_ref,
            preparation.scope.namespace_ref,
            self._owner_ref,
            slot.generation,
            slot.ownership_epoch,
            commitment.policy_digest,
            commitment.requirement_digest,
            now_epoch,
            expires,
            self._authority.epoch,
            self._authority.revocation_generation,
        )
        return self._authority.issue_reconciliation(content)


def build_training_operations(
    *, loader, resolver, planning, planning_store, coordinator, clock
) -> CoordinatorTrainingService:
    """The ``TrainingOperations`` implementation over a composed coordinator."""
    return CoordinatorTrainingService(
        loader=loader,
        resolver=resolver,
        planning=planning,
        planning_store=planning_store,
        coordinator=coordinator,
        clock=clock,
    )


__all__ = [
    "COMMITMENT_PLAN_INDEX_SCHEMA",
    "COMMITMENT_SCHEMA",
    "CoordinatorTrainingService",
    "ReferenceAuthorizationError",
    "ReferenceAuthorizationV1",
    "ReferenceRequestPortsV1",
    "build_training_operations",
]
