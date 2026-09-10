"""Provider-I/O-free composition of the inactive Modal coordinator."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from inspect import getattr_static

from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.coordinator_v1.foundation import ComposedEffectFoundationV1
from tuner.execution.coordinator_v1.operations import TrainingOperationsV1
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.training.coordinator_service import CoordinatorTrainingService
from tuner.training.recipes import RecipeRegistry
from tuner.project.execution_source import ExecutionSourceV1

from .binding import ModalClientBinding
from .coordinator_adapter import ModalPreparationAdapter
from .coordinator_effects import ModalFoundationEffectExecutor, ModalFoundationReconciliationAdapter, ModalFoundationExecutorResolver, ModalFoundationReconciliationResolver
from .coordinator_factories import modal_coordinator_registration
from .coordinator_preflight import ModalOperationalPreflightAdapter
from .coordinator_reader import ModalCoordinatorRunReader
from .coordinator_read_transport import ModalFoundationReadTransport
from .coordinator_retention import ModalFoundationRetentionDelegate
from .coordinator_transport import ModalFoundationHostTransport
from .facade import ExplicitModal154ReadFacade
from .resolution import ModalDeploymentSelectionV1, VerifiedModalDeploymentIdentityV1


@dataclass(frozen=True, slots=True)
class ModalFoundationCompositionPorts:
    effect_repository: object
    grant_authority: object
    receipt_authority: object
    invalid_evidence_authority: object
    assessment_authority: object
    foundation_authenticator: object
    trusted_quiescence_evidence: object
    binding_authority: object
    stage_authority: object
    launch_authority: object
    binding_catalog: object
    stage_catalog: object
    launch_catalog: object
    retained_inputs: object


@dataclass(frozen=True, slots=True)
class ModalCoordinatorStorePorts:
    planning_store: object
    workflow_store: object
    preparation_store: object
    execution_grant_store: object
    reconciliation_grant_store: object


@dataclass(frozen=True, slots=True)
class ModalCoordinatorComposition:
    training: CoordinatorTrainingService
    runs: TrainingOperationsV1
    registration: object
    foundation: ModalFoundationRetentionDelegate
    coordinator: TrainingCoordinatorV1


def _methods(value: object, *names: str) -> None:
    missing = object()
    if any((member := getattr_static(type(value), name, missing)) is missing
           or not callable(member) for name in names):
        raise TypeError("Modal coordinator composition port is incomplete")


def compose_modal_coordinator(
    *, preparation: ModalPreparationAdapter,
    operational_preflight: ModalOperationalPreflightAdapter,
    facade: ExplicitModal154ReadFacade,
    deployment: VerifiedModalDeploymentIdentityV1,
    recipes: RecipeRegistry,
    evidence_authority: object,
    evidence_verifier: object,
    observation_authenticator: object,
    log_authenticator: object,
    artifact_verifier: object,
    cursor_authority: object,
    observed_at: str,
    loader: object,
    resolver: object,
    authorization: object,
    clock: object,
    run_identity: object,
    foundation_ports: ModalFoundationCompositionPorts,
    stores: ModalCoordinatorStorePorts,
) -> ModalCoordinatorComposition:
    """Compose reviewed components without provider I/O or global registration."""
    if type(preparation) is not ModalPreparationAdapter or type(operational_preflight) is not ModalOperationalPreflightAdapter:
        raise TypeError("exact Modal preparation and operational preflight are required")
    if operational_preflight._preparation is not preparation:
        raise ValueError("operational preflight does not retain the exact preparation")
    if operational_preflight._facade is not facade:
        raise ValueError("operational preflight does not retain the exact facade")
    if preparation._clock is not clock or operational_preflight._clock is not clock:
        raise ValueError("Modal coordinator clocks must be the same exact object")
    if (
        type(facade) is not ExplicitModal154ReadFacade
        or type(deployment) is not VerifiedModalDeploymentIdentityV1
        or type(recipes) is not RecipeRegistry
        or type(foundation_ports) is not ModalFoundationCompositionPorts
        or type(stores) is not ModalCoordinatorStorePorts
    ):
        raise TypeError("exact Modal coordinator composition inputs are required")
    context, binding, plan = preparation._snapshot()
    snapshot = parse_canonical_object(preparation.snapshot(), name="preparation snapshot")
    retained_selection = ModalDeploymentSelectionV1.from_dict(
        snapshot["configuration"]["selection"]
    )
    if deployment.selection != retained_selection:
        raise ValueError("deployment differs from retained preparation")
    expected_client = ModalClientBinding(
        retained_selection.account_ref, retained_selection.workspace_ref,
        retained_selection.environment_ref, retained_selection.client_ref,
        retained_selection.sdk_version,
    )
    if facade.binding != expected_client or operational_preflight._deployment_bytes != canonical_bytes(deployment.to_dict()):
        raise ValueError("operational client or deployment differs")
    source = ExecutionSourceV1.from_dict(parse_canonical_object(
        operational_preflight._source_bytes, name="operational execution source",
    ))
    if (
        source.canonical_bytes != operational_preflight._source_bytes
        or source.fingerprint != plan.basis.source_digest
        or source.deployment_member_sha256
        != hashlib.sha256(canonical_bytes(deployment.to_dict())).hexdigest()
        or operational_preflight._quote.body.quote_digest != binding.quote_digest
    ):
        raise ValueError("operational source or quote differs")
    for value, names in (
        (loader, ("load",)), (resolver, ("resolve",)),
        (authorization, ("commit_preflight", "issue_effect_grant", "issue_reconciliation_grant")),
        (clock, ("now", "now_iso", "now_epoch")), (run_identity, ("for_plan",)),
        (foundation_ports.effect_repository, (
            "get", "begin_dispatch", "consume_attempt", "complete_dispatch",
            "complete_invalid_dispatch", "relinquish", "orphan",
            "acquire_reconciliation", "complete_reconciliation",
            "interrupt_reconciliation", "interrupt_invalid_reconciliation",
            "prove_quiescence",
        )),
        (foundation_ports.grant_authority, ("authenticate", "verify", "verify_reconciliation")),
        (foundation_ports.receipt_authority, ("issue", "verify")),
        (foundation_ports.invalid_evidence_authority, ("issue", "verify")),
        (foundation_ports.assessment_authority, ("assess", "authenticate")),
        (foundation_ports.foundation_authenticator, ("authenticate_grant",)),
        (foundation_ports.trusted_quiescence_evidence, ("obtain",)),
        (foundation_ports.binding_authority, ("authenticate",)),
        (foundation_ports.stage_authority, ("sign", "verify")),
        (foundation_ports.launch_authority, ("sign", "verify")),
        (foundation_ports.binding_catalog, ("resolve", "publish_if_absent")),
        (foundation_ports.stage_catalog, ("resolve", "publish_if_absent")),
        (foundation_ports.launch_catalog, ("resolve", "publish_if_absent")),
        (foundation_ports.retained_inputs, ("resolve",)),
        (evidence_authority, ("observation", "log_page")),
        (evidence_verifier, ("verify",)),
        (observation_authenticator, ("authenticate",)),
        (log_authenticator, ("authenticate",)),
        (artifact_verifier, ("verify", "replay", "authenticate")),
        (cursor_authority, ("issue", "verify")),
        (stores.planning_store, ("put_plan_if_absent", "get_plan", "put_context_if_absent", "get_context")),
        (stores.workflow_store, ("create", "get", "get_by_plan", "list_page", "is_descendant", "compare_and_swap")),
        (stores.preparation_store, ("put_if_absent", "get")),
        (stores.execution_grant_store, ("put_if_absent", "get")),
        (stores.reconciliation_grant_store, ("put_if_absent", "get")),
    ):
        _methods(value, *names)
    transport = ModalFoundationHostTransport(
        facade=facade, deployment=deployment,
        stage_source=foundation_ports.stage_catalog,
        launch_source=foundation_ports.launch_catalog,
        binding_authority=foundation_ports.binding_authority,
        foundation_authenticator=foundation_ports.foundation_authenticator,
        assessment_authenticator=foundation_ports.assessment_authority,
        stage_verifier=foundation_ports.stage_authority,
        launch_verifier=foundation_ports.launch_authority, recipes=recipes,
    )
    executor = ModalFoundationEffectExecutor(
        profile_ref=context.provider.profile_ref, account_ref=binding.scope.account_ref,
        namespace_ref=binding.scope.namespace_ref,
        catalog=foundation_ports.binding_catalog,
        authority=foundation_ports.binding_authority, transport=transport,
    )
    reconciliation_adapter = ModalFoundationReconciliationAdapter(
        profile_ref=context.provider.profile_ref, account_ref=binding.scope.account_ref,
        namespace_ref=binding.scope.namespace_ref,
        catalog=foundation_ports.binding_catalog,
        authority=foundation_ports.binding_authority, transport=transport,
    )
    broker = EffectBrokerV2(
        foundation_ports.effect_repository, ModalFoundationExecutorResolver(executor),
        foundation_ports.grant_authority, foundation_ports.receipt_authority,
        foundation_ports.invalid_evidence_authority,
    )
    reconciliation = ReconciliationServiceV1(
        foundation_ports.effect_repository, foundation_ports.grant_authority,
        ModalFoundationReconciliationResolver(reconciliation_adapter),
        foundation_ports.receipt_authority, foundation_ports.invalid_evidence_authority,
    )
    core = ComposedEffectFoundationV1(
        foundation_ports.effect_repository, broker, reconciliation,
        grant_authority=foundation_ports.grant_authority,
        receipt_authority=foundation_ports.receipt_authority,
        invalid_evidence_authority=foundation_ports.invalid_evidence_authority,
        assessment_authority=foundation_ports.assessment_authority,
        trusted_quiescence_evidence=foundation_ports.trusted_quiescence_evidence,
    )
    retained = ModalFoundationRetentionDelegate(
        core, foundation_authenticator=foundation_ports.foundation_authenticator,
        assessment_authority=foundation_ports.assessment_authority,
        binding_authority=foundation_ports.binding_authority,
        stage_authority=foundation_ports.stage_authority,
        launch_authority=foundation_ports.launch_authority,
        binding_catalog=foundation_ports.binding_catalog,
        stage_catalog=foundation_ports.stage_catalog,
        launch_catalog=foundation_ports.launch_catalog,
        retained_inputs=foundation_ports.retained_inputs,
    )
    read_transport = ModalFoundationReadTransport(
        facade=facade, deployment=deployment,
        launch_source=foundation_ports.launch_catalog,
        binding_authority=foundation_ports.binding_authority,
        foundation_authenticator=foundation_ports.foundation_authenticator,
        assessment_authenticator=foundation_ports.assessment_authority,
        stage_verifier=foundation_ports.stage_authority,
        launch_verifier=foundation_ports.launch_authority,
        evidence_verifier=evidence_verifier, recipes=recipes,
    )
    reader = ModalCoordinatorRunReader(
        catalog=foundation_ports.binding_catalog,
        binding_authority=foundation_ports.binding_authority,
        foundation_authenticator=foundation_ports.foundation_authenticator,
        assessment_authenticator=foundation_ports.assessment_authority,
        evidence_authority=evidence_authority, transport=read_transport,
        observed_at=observed_at,
    )
    coordinator = TrainingCoordinatorV1(
        operational_preflight, stores.planning_store, stores.workflow_store,
        stores.preparation_store, stores.execution_grant_store,
        stores.reconciliation_grant_store, preparation, preparation,
        authorization, retained, foundation_ports.foundation_authenticator,
        clock, run_identity,
    )
    training = CoordinatorTrainingService(
        loader=loader, resolver=resolver, planning=operational_preflight,
        planning_store=stores.planning_store, coordinator=coordinator, clock=clock,
    )
    runs = TrainingOperationsV1(
        operational_preflight, stores.planning_store, stores.workflow_store,
        coordinator, retained, foundation_ports.foundation_authenticator,
        foundation_ports.assessment_authority, reader,
        observation_authenticator, log_authenticator, artifact_verifier,
        cursor_authority, clock,
    )
    registration = modal_coordinator_registration(
        preparation, executor, reconciliation_adapter, reader,
    )
    return ModalCoordinatorComposition(training, runs, registration, retained, coordinator)


__all__: list[str] = []
