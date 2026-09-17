"""Provider-I/O-free composition of the inactive Modal coordinator.

Location: ``tuner/execution/providers/modal/coordinator_composition.py``.

``compose_modal_coordinator`` builds the Modal ``ProviderFamilyV1`` (transport,
executor, reconciliation adapter, read transport, reader, retention delegate)
and hands it to the provider-neutral ``compose_family_coordinator`` in
``synaptic_tuner/api/v1/reference/provider_family.py``. The Modal exact-type
and retained-binding checks stay here: they are load-bearing Modal invariants
(one retained preparation, one clock, one deployment, one quote), not
composition boilerplate. Its return type ``ModalCoordinatorComposition`` is
unchanged for ``examples/modal_chat/host.py`` and the composition tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from synaptic_tuner.api.v1.reference.provider_family import (
    CoordinatorRequestPortsV1,
    CoordinatorStoresV1,
    FoundationPortsV1,
    ProviderFamilyV1,
    compose_family_coordinator,
    require_methods,
)
from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.coordinator_v1.operations import RunOperationsV1
from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
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
    runs: RunOperationsV1
    registration: object
    foundation: ModalFoundationRetentionDelegate
    coordinator: TrainingCoordinatorV1


class _ModalFoundationRetention:
    """``ProviderFamilyV1.foundation_retention``: wrap the core in the Modal delegate."""

    def __init__(self, ports: ModalFoundationCompositionPorts) -> None:
        self._ports = ports

    def retain(self, core) -> ModalFoundationRetentionDelegate:
        ports = self._ports
        return ModalFoundationRetentionDelegate(
            core, foundation_authenticator=ports.foundation_authenticator,
            assessment_authority=ports.assessment_authority,
            binding_authority=ports.binding_authority,
            stage_authority=ports.stage_authority,
            launch_authority=ports.launch_authority,
            binding_catalog=ports.binding_catalog,
            stage_catalog=ports.stage_catalog,
            launch_catalog=ports.launch_catalog,
            retained_inputs=ports.retained_inputs,
        )


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
    # Modal-only ports; the neutral composition sweeps the shared ones.
    for value, names in (
        (foundation_ports.binding_authority, ("authenticate",)),
        (foundation_ports.stage_authority, ("sign", "verify")),
        (foundation_ports.launch_authority, ("sign", "verify")),
        (foundation_ports.binding_catalog, ("resolve", "publish_if_absent")),
        (foundation_ports.stage_catalog, ("resolve", "publish_if_absent")),
        (foundation_ports.launch_catalog, ("resolve", "publish_if_absent")),
        (foundation_ports.retained_inputs, ("resolve",)),
        (evidence_authority, ("observation", "log_page")),
        (evidence_verifier, ("verify",)),
    ):
        require_methods(value, *names)
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
    family = ProviderFamilyV1(
        descriptor=preparation.describe(context.provider),
        planning=operational_preflight,
        preparation=preparation,
        reader=reader,
        executor_resolver=ModalFoundationExecutorResolver(executor),
        reconciliation_resolver=ModalFoundationReconciliationResolver(reconciliation_adapter),
        evidence_authority=evidence_authority,
        artifact_verifier=artifact_verifier,
        observation_authenticator=observation_authenticator,
        log_authenticator=log_authenticator,
        foundation_retention=_ModalFoundationRetention(foundation_ports),
    )
    composed = compose_family_coordinator(
        family=family,
        stores=CoordinatorStoresV1(
            stores.planning_store, stores.workflow_store, stores.preparation_store,
            stores.execution_grant_store, stores.reconciliation_grant_store,
        ),
        foundation=FoundationPortsV1(
            foundation_ports.effect_repository, foundation_ports.grant_authority,
            foundation_ports.receipt_authority, foundation_ports.invalid_evidence_authority,
            foundation_ports.assessment_authority, foundation_ports.foundation_authenticator,
            foundation_ports.trusted_quiescence_evidence,
        ),
        requests=CoordinatorRequestPortsV1(
            loader, resolver, run_identity, authorization, cursor_authority, clock,
        ),
    )
    if type(composed.foundation) is not ModalFoundationRetentionDelegate:
        raise TypeError("Modal composition must retain through the Modal delegate")
    registration = modal_coordinator_registration(
        preparation, executor, reconciliation_adapter, reader,
    )
    return ModalCoordinatorComposition(
        composed.training, composed.runs, registration, composed.foundation, composed.coordinator,
    )


__all__: list[str] = []
