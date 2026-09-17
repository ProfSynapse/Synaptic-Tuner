"""Provider family contract and the provider-neutral coordinator composition.

Location: ``synaptic_tuner/api/v1/reference/provider_family.py``.

A provider family is everything one provider (Modal, Docker, the fake
conformance provider, ...) contributes to the training coordinator: its
descriptor, planning and preparation adapters, run reader, effect executor
and reconciliation adapter resolvers, evidence issuance and the three
authenticators the coordinator consults when it replays provider evidence.
``compose_family_coordinator`` wires one family into the reviewed
``coordinator_v1`` machinery (effect broker, reconciliation service, composed
foundation, coordinator, training service, run operations) without knowing
which provider it is. It was extracted from
``tuner/execution/providers/modal/coordinator_composition.py``; that module
now builds a Modal ``ProviderFamilyV1`` (keeping its exact-type and retained
binding checks) and calls this function.

Port completeness is checked statically with ``require_methods``
(``inspect.getattr_static``) so a hostile property on a port object is never
invoked during composition.

Consumers: ``synaptic_tuner/api/v1/reference/__init__.py``
(``compose_reference_host``) and the Modal family builder. Imports no
``api/v1/persistence.py`` and no ``ProjectContext``.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import getattr_static

from synaptic_tuner.api.v1.ports import (
    ClockPort,
    DurableRecordStorePort,
    DurableStreamStorePort,
    GrantAuthorityPort,
    SecretResolverPort,
)
from synaptic_tuner.api.v1.providers import ProviderDescriptor
from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.coordinator_v1.foundation import ComposedEffectFoundationV1
from tuner.execution.coordinator_v1.operations import RunOperationsV1
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.training.coordinator_service import CoordinatorTrainingService

from .runs import build_run_operations
from .training import build_training_operations


@dataclass(frozen=True, slots=True)
class ProviderFamilyV1:
    """One provider's contribution to the coordinator.

    ``planning`` exposes ``describe``, ``context`` and ``preflight``;
    ``preparation`` exposes ``resolve``, ``prepare`` and ``payload`` (the
    binding resolver and materializer are one object, as in Modal);
    ``reader`` exposes ``observe``, ``logs``, ``artifacts`` and
    ``iter_artifact_bytes``. The two resolvers are what the effect broker and
    reconciliation service consume; the family builder owns the executor and
    adapter they resolve to. ``evidence_authority`` issues reader evidence
    (``observation``, ``log_page``) and is consumed by family builders that
    construct their own reader, never by this module.

    ``recovery_verifier`` (``verify_quiescence``, ``verify_finality``) and
    ``quiescence_evidence`` (``obtain``) are the provider's proofs that an
    indeterminate effect is quiescent or absent; ``None`` composes the
    unavailable implementations, which refuse every proof.
    ``foundation_retention`` (``retain(core)``) lets a family wrap the composed
    foundation, as Modal does with its retention delegate.
    """

    descriptor: ProviderDescriptor
    planning: object
    preparation: object
    reader: object
    executor_resolver: object
    reconciliation_resolver: object
    evidence_authority: object
    artifact_verifier: object
    observation_authenticator: object
    log_authenticator: object
    recovery_verifier: object | None = None
    quiescence_evidence: object | None = None
    foundation_retention: object | None = None

    def __post_init__(self) -> None:
        if type(self.descriptor) is not ProviderDescriptor:
            raise TypeError("descriptor must be exact ProviderDescriptor")


@dataclass(frozen=True, slots=True)
class ReferenceHostPortsV1:
    """The five public host ports a reference host supplies."""

    records: DurableRecordStorePort
    streams: DurableStreamStorePort
    clock: ClockPort
    grants: GrantAuthorityPort
    secrets: SecretResolverPort


@dataclass(frozen=True, slots=True)
class CoordinatorStoresV1:
    """The five internal coordinator store ports."""

    planning_store: object
    workflow_store: object
    preparation_store: object
    execution_grant_store: object
    reconciliation_grant_store: object


@dataclass(frozen=True, slots=True)
class FoundationPortsV1:
    """Foundation authorities and the effect repository the coordinator trusts."""

    effect_repository: object
    grant_authority: object
    receipt_authority: object
    invalid_evidence_authority: object
    assessment_authority: object
    foundation_authenticator: object
    trusted_quiescence_evidence: object


@dataclass(frozen=True, slots=True)
class CoordinatorRequestPortsV1:
    """Host-specific request handling plus the authorization, cursor and clock ports."""

    loader: object
    resolver: object
    run_identity: object
    authorization: object
    cursor_authority: object
    clock: object


@dataclass(frozen=True, slots=True)
class FamilyCoordinatorCompositionV1:
    training: CoordinatorTrainingService
    runs: RunOperationsV1
    coordinator: TrainingCoordinatorV1
    foundation: object


def require_methods(value: object, *names: str) -> None:
    """Statically require callable members without touching instance properties."""
    missing = object()
    if any(
        (member := getattr_static(type(value), name, missing)) is missing or not callable(member)
        for name in names
    ):
        raise TypeError("coordinator composition port is incomplete")


_REPOSITORY_METHODS = (
    "get", "begin_dispatch", "consume_attempt", "complete_dispatch",
    "complete_invalid_dispatch", "relinquish", "orphan",
    "acquire_reconciliation", "complete_reconciliation",
    "interrupt_reconciliation", "interrupt_invalid_reconciliation",
    "prove_quiescence",
)


def check_composition_ports(
    *,
    family: ProviderFamilyV1,
    stores: CoordinatorStoresV1,
    foundation: FoundationPortsV1,
    requests: CoordinatorRequestPortsV1,
) -> None:
    """Exact bundle types plus the static method sweep over every port."""
    if (
        type(family) is not ProviderFamilyV1
        or type(stores) is not CoordinatorStoresV1
        or type(foundation) is not FoundationPortsV1
        or type(requests) is not CoordinatorRequestPortsV1
    ):
        raise TypeError("exact coordinator composition bundles are required")
    checks = [
        (family.planning, ("describe", "context", "preflight")),
        (family.preparation, ("resolve", "prepare", "payload")),
        (family.reader, ("observe", "logs", "artifacts", "iter_artifact_bytes")),
        (family.executor_resolver, ("resolve",)),
        (family.reconciliation_resolver, ("resolve",)),
        (family.evidence_authority, ("observation", "log_page")),
        (family.artifact_verifier, ("verify", "replay", "authenticate")),
        (family.observation_authenticator, ("authenticate",)),
        (family.log_authenticator, ("authenticate",)),
        (foundation.effect_repository, _REPOSITORY_METHODS),
        (foundation.grant_authority, ("authenticate", "verify", "verify_reconciliation")),
        (foundation.receipt_authority, ("issue", "verify")),
        (foundation.invalid_evidence_authority, ("issue", "verify")),
        (foundation.assessment_authority, ("assess", "authenticate")),
        (foundation.foundation_authenticator, ("authenticate_grant",)),
        (foundation.trusted_quiescence_evidence, ("obtain",)),
        (requests.loader, ("load",)),
        (requests.resolver, ("resolve",)),
        (requests.run_identity, ("for_plan",)),
        (requests.authorization, ("commit_preflight", "issue_effect_grant", "issue_reconciliation_grant")),
        (requests.cursor_authority, ("issue", "verify")),
        (requests.clock, ("now", "now_iso", "now_epoch")),
        (stores.planning_store, ("put_plan_if_absent", "get_plan", "put_context_if_absent", "get_context")),
        (stores.workflow_store, ("create", "get", "get_by_plan", "list_page", "is_descendant", "compare_and_swap")),
        (stores.preparation_store, ("put_if_absent", "get")),
        (stores.execution_grant_store, ("put_if_absent", "get")),
        (stores.reconciliation_grant_store, ("put_if_absent", "get")),
    ]
    if family.recovery_verifier is not None:
        checks.append((family.recovery_verifier, ("verify_quiescence", "verify_finality")))
    if family.quiescence_evidence is not None:
        checks.append((family.quiescence_evidence, ("obtain",)))
    if family.foundation_retention is not None:
        checks.append((family.foundation_retention, ("retain",)))
    for value, names in checks:
        require_methods(value, *names)


def compose_family_coordinator(
    *,
    family: ProviderFamilyV1,
    stores: CoordinatorStoresV1,
    foundation: FoundationPortsV1,
    requests: CoordinatorRequestPortsV1,
) -> FamilyCoordinatorCompositionV1:
    """Wire one provider family into the coordinator without provider I/O."""
    check_composition_ports(family=family, stores=stores, foundation=foundation, requests=requests)
    broker = EffectBrokerV2(
        foundation.effect_repository,
        family.executor_resolver,
        foundation.grant_authority,
        foundation.receipt_authority,
        foundation.invalid_evidence_authority,
    )
    reconciliation = ReconciliationServiceV1(
        foundation.effect_repository,
        foundation.grant_authority,
        family.reconciliation_resolver,
        foundation.receipt_authority,
        foundation.invalid_evidence_authority,
    )
    core = ComposedEffectFoundationV1(
        foundation.effect_repository,
        broker,
        reconciliation,
        grant_authority=foundation.grant_authority,
        receipt_authority=foundation.receipt_authority,
        invalid_evidence_authority=foundation.invalid_evidence_authority,
        assessment_authority=foundation.assessment_authority,
        trusted_quiescence_evidence=foundation.trusted_quiescence_evidence,
    )
    effect_foundation = core
    if family.foundation_retention is not None:
        effect_foundation = family.foundation_retention.retain(core)
        require_methods(effect_foundation, "get", "execute", "reconcile", "recover_orphan")
    coordinator = TrainingCoordinatorV1(
        family.planning,
        stores.planning_store,
        stores.workflow_store,
        stores.preparation_store,
        stores.execution_grant_store,
        stores.reconciliation_grant_store,
        family.preparation,
        family.preparation,
        requests.authorization,
        effect_foundation,
        foundation.foundation_authenticator,
        requests.clock,
        requests.run_identity,
    )
    training = build_training_operations(
        loader=requests.loader,
        resolver=requests.resolver,
        planning=family.planning,
        planning_store=stores.planning_store,
        coordinator=coordinator,
        clock=requests.clock,
    )
    runs = build_run_operations(
        planning=family.planning,
        planning_store=stores.planning_store,
        workflow_store=stores.workflow_store,
        coordinator=coordinator,
        foundation=effect_foundation,
        foundation_authenticator=foundation.foundation_authenticator,
        assessment_authenticator=foundation.assessment_authority,
        reader=family.reader,
        observation_authenticator=family.observation_authenticator,
        log_authenticator=family.log_authenticator,
        artifact_verifier=family.artifact_verifier,
        cursor_authority=requests.cursor_authority,
        clock=requests.clock,
    )
    return FamilyCoordinatorCompositionV1(training, runs, coordinator, effect_foundation)


__all__ = [
    "CoordinatorRequestPortsV1",
    "CoordinatorStoresV1",
    "FamilyCoordinatorCompositionV1",
    "FoundationPortsV1",
    "ProviderFamilyV1",
    "ReferenceHostPortsV1",
    "check_composition_ports",
    "compose_family_coordinator",
    "require_methods",
]
