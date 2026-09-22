"""Provider-I/O-free composition for packaged Modal training.

This module wires the explicit-client transport to the existing Foundation
effect and reconciliation ports.  Construction only validates and retains
collaborators; provider reads begin at an explicit effect, lookup, completion
observation, or artifact-streaming call.
"""

from __future__ import annotations

from dataclasses import dataclass

from tuner.execution.foundation_v2.canonical import safe_ref

from .facade import EXACT_MODAL_SDK_VERSION, ExplicitModal154ReadFacade
from .packaged_binding import (
    ModalPackagedBindingCatalog,
    ModalPackagedRuntimeFactsV1,
)
from .packaged_deployment import ModalPackagedDeploymentObserver
from .packaged_dispatch import ModalPackagedDispatchVerifier
from .packaged_effects import (
    ModalPackagedBindingAuthority,
    ModalPackagedFoundationEffectExecutor,
    ModalPackagedFoundationExecutorResolver,
    ModalPackagedFoundationReconciliationAdapter,
    ModalPackagedFoundationReconciliationResolver,
)
from .packaged_reader import ModalPackagedReader
from .packaged_staging import ModalPackagedInputStager
from .packaged_transport import (
    ModalPackagedCallCatalog,
    ModalPackagedDispatchSource,
    ModalPackagedHostTransport,
    ModalPackagedStageReceiptCatalog,
    ModalPackagedStageSource,
)
from .packaged_worker import (
    ModalPackagedEvidenceSigner,
    ModalPackagedWorker,
    ModalPackagedWorkerRoots,
    PackagedTrainerExecutor,
)


def _requires(value: object, *members: str) -> None:
    if any(not callable(getattr(value, member, None)) for member in members):
        raise TypeError("packaged Modal composition collaborator is incomplete")


@dataclass(frozen=True, slots=True)
class ModalPackagedCatalogPorts:
    """Consumer-owned retained catalogs; none is read at construction."""

    bindings: ModalPackagedBindingCatalog
    stage_receipts: ModalPackagedStageReceiptCatalog
    calls: ModalPackagedCallCatalog

    def __post_init__(self) -> None:
        _requires(self.bindings, "resolve")
        _requires(self.stage_receipts, "resolve", "publish_if_absent")
        _requires(self.calls, "resolve", "publish_if_absent")


@dataclass(frozen=True, slots=True)
class ModalPackagedSourcePorts:
    """Consumer-owned one-use stage material and signed-dispatch sources."""

    stages: ModalPackagedStageSource
    dispatches: ModalPackagedDispatchSource

    def __post_init__(self) -> None:
        _requires(self.stages, "resolve")
        _requires(self.dispatches, "resolve")


@dataclass(frozen=True, slots=True)
class ModalPackagedAuthorityPorts:
    """Authentication authorities retained by host effects and workers."""

    bindings: ModalPackagedBindingAuthority
    dispatches: ModalPackagedDispatchVerifier

    def __post_init__(self) -> None:
        _requires(self.bindings, "authenticate")
        _requires(self.dispatches, "verify")


@dataclass(frozen=True, slots=True)
class ModalPackagedComposition:
    """Fully wired host adapter plus an offline worker-construction seam."""

    transport: ModalPackagedHostTransport
    effect_executor: ModalPackagedFoundationEffectExecutor
    reconciliation_adapter: ModalPackagedFoundationReconciliationAdapter
    executor_resolver: ModalPackagedFoundationExecutorResolver
    reconciliation_resolver: ModalPackagedFoundationReconciliationResolver
    reader: ModalPackagedReader
    trainer: PackagedTrainerExecutor
    clock: object
    catalogs: ModalPackagedCatalogPorts
    sources: ModalPackagedSourcePorts
    authorities: ModalPackagedAuthorityPorts

    def worker(
        self,
        *,
        expected_facts: ModalPackagedRuntimeFactsV1,
        evidence_signer: ModalPackagedEvidenceSigner,
        roots: ModalPackagedWorkerRoots,
    ) -> ModalPackagedWorker:
        """Build the remote callable without opening storage or provider state."""
        return ModalPackagedWorker(
            expected_facts=expected_facts,
            dispatch_verifier=self.authorities.dispatches,
            trainer_executor=self.trainer,
            evidence_signer=evidence_signer,
            roots=roots,
        )


def compose_modal_packaged_adapter(
    *,
    profile_ref: str,
    account_ref: str,
    namespace_ref: str,
    sdk: object,
    client: object,
    facade: ExplicitModal154ReadFacade,
    deployment_observer: ModalPackagedDeploymentObserver,
    stager: ModalPackagedInputStager,
    catalogs: ModalPackagedCatalogPorts,
    sources: ModalPackagedSourcePorts,
    authorities: ModalPackagedAuthorityPorts,
    clock: object,
    trainer: PackagedTrainerExecutor,
    reader: ModalPackagedReader,
) -> ModalPackagedComposition:
    """Compose exact packaged components without touching Modal or a catalog."""
    profile = safe_ref(profile_ref, "profile_ref")
    account = safe_ref(account_ref, "account_ref")
    namespace = safe_ref(namespace_ref, "namespace_ref")
    if getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION or client is None:
        raise TypeError("exact SDK and explicit Modal client required")
    if type(facade) is not ExplicitModal154ReadFacade \
            or type(deployment_observer) is not ModalPackagedDeploymentObserver \
            or type(stager) is not ModalPackagedInputStager \
            or type(reader) is not ModalPackagedReader \
            or type(catalogs) is not ModalPackagedCatalogPorts \
            or type(sources) is not ModalPackagedSourcePorts \
            or type(authorities) is not ModalPackagedAuthorityPorts:
        raise TypeError("exact packaged Modal composition inputs required")
    if facade.client is not client or deployment_observer.client is not client \
            or facade.binding != deployment_observer.client_binding \
            or stager._facade is not facade \
            or reader._facade is not facade \
            or reader._observer is not deployment_observer:
        raise ValueError("packaged Modal collaborators do not share one client")
    if facade.binding.account_ref != account:
        raise ValueError("packaged Modal account differs from the explicit client")
    if not callable(clock) and not callable(getattr(clock, "now", None)):
        raise TypeError("packaged Modal clock is required")
    _requires(trainer, "execute")

    transport = ModalPackagedHostTransport(
        sdk=sdk,
        client=client,
        facade=facade,
        deployment_observer=deployment_observer,
        stager=stager,
        stage_source=sources.stages,
        dispatch_source=sources.dispatches,
        stage_receipts=catalogs.stage_receipts,
        call_catalog=catalogs.calls,
        dispatch_verifier=authorities.dispatches,
    )
    effect_executor = ModalPackagedFoundationEffectExecutor(
        profile_ref=profile,
        account_ref=account,
        namespace_ref=namespace,
        catalog=catalogs.bindings,
        authority=authorities.bindings,
        transport=transport,
    )
    reconciliation_adapter = ModalPackagedFoundationReconciliationAdapter(
        profile_ref=profile,
        account_ref=account,
        namespace_ref=namespace,
        catalog=catalogs.bindings,
        authority=authorities.bindings,
        transport=transport,
    )
    return ModalPackagedComposition(
        transport,
        effect_executor,
        reconciliation_adapter,
        ModalPackagedFoundationExecutorResolver(effect_executor),
        ModalPackagedFoundationReconciliationResolver(reconciliation_adapter),
        reader,
        trainer,
        clock,
        catalogs,
        sources,
        authorities,
    )


__all__ = [
    "ModalPackagedAuthorityPorts",
    "ModalPackagedCatalogPorts",
    "ModalPackagedComposition",
    "ModalPackagedSourcePorts",
    "compose_modal_packaged_adapter",
]
