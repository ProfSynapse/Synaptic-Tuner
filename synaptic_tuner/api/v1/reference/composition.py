"""``compose_reference_host``: the provider-neutral reference host over the public ports.

Location: ``synaptic_tuner/api/v1/reference/composition.py``.

This is the body behind ``synaptic_tuner.api.v1.reference.compose_reference_host``.
It lives in its own module because ``reference/__init__.py`` must stay free of
``tuner.*`` imports: both import-closure gates
(``tests/contract/test_provider_neutral_foundation_v1.py`` and
``tests/contract/test_public_host_ports_v1.py``) prove that importing the
contract-only ``reference/stores.py`` loads no engine module, and importing a
submodule always executes the package ``__init__``. The package resolves the
composition names lazily (PEP 562) so the documented host spelling still works.

Composing a host is two calls, because provider families whose readers
authenticate foundation evidence need the engine authorities before the
family exists::

    authority = compose_reference_authority(
        clock=ports.clock, secrets=ports.secrets, authority_secret=SecretRef("env", "..."))
    family = build_my_family(authority.foundation_authenticator, authority.assessment_authority)
    composition = compose_reference_host(
        family=family, ports=ports, requests=requests, authority=authority)
    api = composition.api()

Passing ``data=ReferenceDataPortsV1(...)`` (SynthChat config, scenario and
rubric directories plus an LLM client factory the host owns) also composes the
Data family (``data.py``) over the same record and stream stores.

The five coordinator stores, the foundation effect ledger and the host
authorization commitments are all durable through the host record and stream
stores (``repositories.py``, ``training.py``), so a second host composed over
the same stores after ``start`` drives ``outcome``, ``cancel`` and
``reconcile`` for a run the first host began. Only the publication store is
process-local in this version.

The evaluation family (``evaluation.py``) is composed only when the host
passes ``evaluation=ReferenceEvaluationPortsV1(...)``: it needs a scenario
root, a backend registry and an artifact sink the engine cannot invent. Without
it ``APIHost.evaluation`` stays ``None`` and the ``Evaluator`` stack is never
imported.
"""

from __future__ import annotations

from dataclasses import dataclass

from synaptic_tuner.api.v1.host import APIHost, HostPorts

from .artifacts import compose_reference_artifacts
from .authority import (
    ReferenceAuthorityV1,
    UnavailableQuiescenceEvidenceV1,
    UnavailableRecoveryVerifierV1,
    compose_reference_authority,
)
from .provider_family import (
    CoordinatorRequestPortsV1,
    CoordinatorStoresV1,
    FoundationPortsV1,
    ProviderFamilyV1,
    ReferenceHostPortsV1,
    compose_family_coordinator,
    require_methods,
)
from .repositories import (
    DurableEffectRepositoryV1,
    DurableExecutionGrantStoreV1,
    DurablePlanningStoreV1,
    DurablePreparationStoreV1,
    DurableReconciliationGrantStoreV1,
    DurableWorkflowStoreV1,
)
from .data import ReferenceDataPortsV1, build_data_operations
from .training import ReferenceAuthorizationV1, ReferenceRequestPortsV1


@dataclass(frozen=True, slots=True)
class ReferenceComposition:
    """Every composed operations object plus the coordinator internals a host may inspect."""

    training: object
    runs: object
    artifacts: object
    evaluation: object | None
    chat: None
    data: object | None
    pipelines: object | None
    coordinator: object
    foundation: object
    stores: CoordinatorStoresV1
    authority: ReferenceAuthorityV1

    def api(self) -> APIHost:
        return APIHost(
            HostPorts(
                training=self.training,
                runs=self.runs,
                artifacts=self.artifacts,
                evaluation=self.evaluation,
                chat=self.chat,
                data=self.data,
                pipelines=self.pipelines,
                clock=self.authority.clock,
            )
        )


def compose_reference_stores(
    *,
    ports: ReferenceHostPortsV1,
    authority: ReferenceAuthorityV1,
    family: ProviderFamilyV1,
) -> CoordinatorStoresV1:
    """The five coordinator store ports over the host record and stream stores."""
    records = ports.records
    require_methods(records, "create", "read", "compare_and_swap", "put_if_absent", "list_page")
    require_methods(ports.streams, "append", "read_page")
    return CoordinatorStoresV1(
        DurablePlanningStoreV1(records),
        DurableWorkflowStoreV1(
            records,
            ports.streams,
            foundation_authenticator=authority.foundation_authenticator,
            assessment_authenticator=authority.assessment_authority,
            observation_authenticator=family.observation_authenticator,
            artifact_verifier=family.artifact_verifier,
        ),
        DurablePreparationStoreV1(records),
        DurableExecutionGrantStoreV1(records, authority.grant_authority),
        DurableReconciliationGrantStoreV1(records, authority.grant_authority),
    )


def compose_reference_host(
    *,
    family: ProviderFamilyV1,
    ports: ReferenceHostPortsV1,
    requests: ReferenceRequestPortsV1,
    authority: ReferenceAuthorityV1,
    maximum_grant_seconds: int = 900,
    evaluation: object | None = None,
    data: ReferenceDataPortsV1 | None = None,
    pipelines: object | None = None,
) -> ReferenceComposition:
    """Compose the Training, Runs, Artifacts and, when given, Evaluation and Data families.

    The evaluation family is composed when the host supplies its ports; the
    data family needs SynthChat locations and an LLM client factory the host
    owns (``ReferenceDataPortsV1``). Without them the slot stays pending on
    the composed host.
    """
    if (
        type(family) is not ProviderFamilyV1
        or type(ports) is not ReferenceHostPortsV1
        or type(requests) is not ReferenceRequestPortsV1
        or type(authority) is not ReferenceAuthorityV1
    ):
        raise TypeError("exact reference composition inputs are required")
    if data is not None and type(data) is not ReferenceDataPortsV1:
        raise TypeError("data must be exact ReferenceDataPortsV1 or None")
    require_methods(ports.streams, "append", "read_page")
    require_methods(ports.grants, "authorize", "bind")
    evaluation_operations = None
    if evaluation is not None:
        # Imported here so a host without an evaluation family never loads ``Evaluator``.
        from .evaluation import ReferenceEvaluationPortsV1, compose_reference_evaluation

        if type(evaluation) is not ReferenceEvaluationPortsV1:
            raise TypeError("evaluation must be exact ReferenceEvaluationPortsV1")
        evaluation_operations = compose_reference_evaluation(
            records=ports.records, streams=ports.streams, clock=authority.clock, evaluation=evaluation,
        )
    if pipelines is not None:
        # Imported here so a host without a pipelines family never loads its driver.
        from .pipelines import ReferencePipelinePortsV1

        if type(pipelines) is not ReferencePipelinePortsV1:
            raise TypeError("pipelines must be exact ReferencePipelinePortsV1")
    recovery = family.recovery_verifier
    if recovery is None:
        recovery = UnavailableRecoveryVerifierV1()
    quiescence = family.quiescence_evidence
    if quiescence is None:
        quiescence = UnavailableQuiescenceEvidenceV1()
    repository = DurableEffectRepositoryV1(
        ports.records,
        receipt_authority=authority.receipt_authority,
        invalid_evidence_authority=authority.invalid_evidence_authority,
        recovery_verifier=recovery,
        finality_verifier=recovery,
        grant_authority=authority.grant_authority,
    )
    foundation_ports = FoundationPortsV1(
        repository,
        authority.grant_authority,
        authority.receipt_authority,
        authority.invalid_evidence_authority,
        authority.assessment_authority,
        authority.foundation_authenticator,
        quiescence,
    )
    stores = compose_reference_stores(ports=ports, authority=authority, family=family)
    authorization = ReferenceAuthorizationV1(
        records=ports.records,
        grants=ports.grants,
        authority=authority.grant_authority,
        clock=authority.clock,
        maximum_grant_seconds=maximum_grant_seconds,
    )
    composed = compose_family_coordinator(
        family=family,
        stores=stores,
        foundation=foundation_ports,
        requests=CoordinatorRequestPortsV1(
            requests.loader,
            requests.resolver,
            requests.run_identity,
            authorization,
            authority.cursor_authority,
            authority.clock,
        ),
    )
    pipeline_operations = None
    if pipelines is not None:
        from .pipelines import compose_reference_pipelines

        pipeline_operations = compose_reference_pipelines(
            records=ports.records, streams=ports.streams, clock=authority.clock,
            training=composed.training, runs=composed.runs, evaluation=evaluation_operations,
            pipelines=pipelines,
        )
    return ReferenceComposition(
        training=composed.training,
        runs=composed.runs,
        artifacts=compose_reference_artifacts(authority=authority),
        evaluation=evaluation_operations,
        chat=None,
        data=None if data is None else build_data_operations(
            records=ports.records, streams=ports.streams, clock=authority.clock, ports=data,
        ),
        pipelines=pipeline_operations,
        coordinator=composed.coordinator,
        foundation=composed.foundation,
        stores=stores,
        authority=authority,
    )


__all__ = [
    "ReferenceComposition",
    "compose_reference_host",
    "compose_reference_stores",
]
