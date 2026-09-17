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

Consumed by ``synaptic_tuner/api/v1/reference/provider_family.py`` and
``synaptic_tuner/api/v1/reference/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import RLock

from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.planning import TrainingPlan
from synaptic_tuner.api.v1.ports import GrantAuthorityPort
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.coordinator_v1.coordinator import ReconciliationGrantSlotV1
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.foundation_v2.authority import GrantAuthorityV2, ReconciliationGrantContentV1
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest, safe_ref
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.repository import EffectRecordV2
from tuner.training.coordinator_service import CoordinatorTrainingService


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


class ReferenceAuthorizationV1:
    """``AuthorizationPortV1`` bridging the public ``GrantAuthorityPort``."""

    def __init__(
        self,
        *,
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
        self._grants = grants
        self._authority = authority
        self._clock = clock
        self._maximum = maximum_grant_seconds
        self._owner_ref = safe_ref(owner_ref, "owner_ref")
        self._by_policy: dict[str, _Commitment] = {}
        self._by_plan: dict[str, str] = {}
        self._lock = RLock()

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
        document = preflight.to_dict()
        policy = domain_digest(
            "synaptic-reference-preflight-policy/v1",
            canonical_bytes(
                {
                    "plan_fingerprint": plan.plan_fingerprint,
                    "preflight": document,
                    "host_grant_ref": host_grant.grant_ref,
                }
            ),
        )
        requirement = domain_digest(
            "synaptic-reference-requirements/v1",
            canonical_bytes({"authorization": document["authorization"]}),
        )
        commitment = _Commitment(
            plan.plan_fingerprint, preflight, host_grant, policy, requirement, _epoch(preflight.expires_at)
        )
        with self._lock:
            existing = self._by_policy.get(policy)
            if existing is None:
                self._by_policy[policy] = commitment
            elif existing != commitment:
                raise ReferenceAuthorizationError("authorization_conflict")
            self._by_plan[plan.plan_fingerprint] = policy
        return policy

    def _expiry(self, commitment: _Commitment, now_epoch: int) -> int:
        if type(now_epoch) is not int or now_epoch >= commitment.expires_at_epoch:
            raise ReferenceAuthorizationError("authorization_expired")
        return min(commitment.expires_at_epoch, now_epoch + self._maximum)

    def issue_effect_grant(self, command_bytes: bytes, *, preflight_digest: str, now_epoch: int):
        with self._lock:
            commitment = self._by_policy.get(preflight_digest)
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
            policy = self._by_plan.get(preparation.plan_fingerprint)
            commitment = None if policy is None else self._by_policy.get(policy)
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
    "CoordinatorTrainingService",
    "ReferenceAuthorizationError",
    "ReferenceAuthorizationV1",
    "ReferenceRequestPortsV1",
    "build_training_operations",
]
