"""Provider-free composition tests for the packaged Modal adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.packaged_composition import (
    ModalPackagedAuthorityPorts,
    ModalPackagedCatalogPorts,
    ModalPackagedComposition,
    ModalPackagedSourcePorts,
    compose_modal_packaged_adapter,
)
from tuner.execution.providers.modal.packaged_deployment import (
    ModalPackagedDeploymentObserver,
)
from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader
from tuner.execution.providers.modal.packaged_staging import ModalPackagedInputStager
from tuner.execution.providers.modal.packaged_worker import (
    ModalPackagedWorker,
    ModalPackagedWorkerRoots,
)

from tests.execution.providers.test_modal_packaged_binding import _binding
from tests.execution.providers.test_modal_sdk154_adapter import SDK


class Port:
    def __init__(self, events, label):
        self.events, self.label = events, label

    def resolve(self, *args, **kwargs):
        self.events.append((self.label + ".resolve", args, kwargs))

    def publish_if_absent(self, *args, **kwargs):
        self.events.append((self.label + ".publish", args, kwargs))
        return True


class Authority:
    def __init__(self, events, label):
        self.events, self.label = events, label

    def authenticate(self, *args, **kwargs):
        self.events.append((self.label + ".authenticate", args, kwargs))
        return True

    def verify(self, *args, **kwargs):
        self.events.append((self.label + ".verify", args, kwargs))
        return True


class Trainer:
    def __init__(self, events):
        self.events = events

    def execute(self, **kwargs):
        self.events.append(("trainer.execute", kwargs))


class DeploymentReader:
    def __init__(self, events):
        self.events = events

    def observe(self, **kwargs):
        self.events.append(("deployment.observe", kwargs))


def _components():
    binding = _binding()
    command, facts = binding.command, binding.provider_facts
    client, events = object(), []
    scope = lambda supplied: events.append(("scope", supplied)) or (
        facts.account_ref, facts.workspace_ref, facts.environment_ref,
        facts.client_ref,
    )
    facade = ExplicitModal154ReadFacade(
        facts.client_binding, sdk=SDK, client=client, scope_observer=scope,
        deployment_observer=lambda **kwargs: events.append(("facade.deployment", kwargs)),
        volume_names={
            facts.control_volume_id: "control-name",
            facts.artifact_volume_id: "artifact-name",
        },
    )
    deployment = ModalPackagedDeploymentObserver(
        sdk=SDK, client=client, client_binding=facts.client_binding,
        reader=DeploymentReader(events),
    )
    stager = ModalPackagedInputStager(facade)
    evidence = Authority(events, "evidence")
    reader = ModalPackagedReader(
        facade=facade, deployment_observer=deployment,
        verifier=evidence, key_ref="evidence-key",
    )
    catalogs = ModalPackagedCatalogPorts(
        Port(events, "bindings"), Port(events, "stage_receipts"),
        Port(events, "calls"),
    )
    sources = ModalPackagedSourcePorts(
        Port(events, "stages"), Port(events, "dispatches"),
    )
    authorities = ModalPackagedAuthorityPorts(
        Authority(events, "bindings"), Authority(events, "dispatches"),
    )
    trainer = Trainer(events)
    arguments = dict(
        profile_ref=command.preparation.provider.profile_ref,
        account_ref=command.preparation.scope.account_ref,
        namespace_ref=command.preparation.scope.namespace_ref,
        sdk=SDK, client=client, facade=facade,
        deployment_observer=deployment, stager=stager, catalogs=catalogs,
        sources=sources, authorities=authorities, clock=lambda: 1,
        trainer=trainer, reader=reader,
    )
    return binding, arguments, events, trainer


def test_composes_exact_foundation_ports_without_provider_or_catalog_io() -> None:
    _, arguments, events, trainer = _components()
    composed = compose_modal_packaged_adapter(**arguments)
    assert type(composed) is ModalPackagedComposition
    assert composed.transport is composed.effect_executor._transport
    assert composed.transport is composed.reconciliation_adapter._transport
    assert composed.executor_resolver._executor is composed.effect_executor
    assert composed.reconciliation_resolver._adapter is composed.reconciliation_adapter
    assert composed.trainer is trainer
    assert events == []


def test_worker_factory_is_inert_and_retains_exact_offline_seam(tmp_path: Path) -> None:
    binding, arguments, events, trainer = _components()
    composed = compose_modal_packaged_adapter(**arguments)
    roots = ModalPackagedWorkerRoots(
        (tmp_path / "control").resolve(), (tmp_path / "artifacts").resolve(),
        (tmp_path / "cache").resolve(),
    )
    signer = Authority(events, "signer")
    signer.sign = lambda *args: events.append(("signer.sign", args)) or b"tag"
    worker = composed.worker(
        expected_facts=binding.provider_facts,
        evidence_signer=signer,
        roots=roots,
    )
    assert type(worker) is ModalPackagedWorker
    assert worker._executor is trainer
    assert worker._verifier is arguments["authorities"].dispatches
    assert events == []


@pytest.mark.parametrize("fault", ("sdk", "account", "client", "stager", "reader"))
def test_incompatible_configuration_is_rejected_without_fallback_or_io(fault: str) -> None:
    _, arguments, events, _ = _components()
    if fault == "sdk":
        arguments["sdk"] = type("WrongSDK", (), {"__version__": "1.5.3"})
    elif fault == "account":
        arguments["account_ref"] = "other-account"
    elif fault == "client":
        arguments["client"] = object()
    elif fault == "stager":
        other = _components()[1]
        arguments["stager"] = other["stager"]
    else:
        other = _components()[1]
        arguments["reader"] = other["reader"]
    with pytest.raises((TypeError, ValueError)):
        compose_modal_packaged_adapter(**arguments)
    assert events == []


def test_unknown_mode_or_implicit_defaults_are_not_an_adapter_surface() -> None:
    _, arguments, events, _ = _components()
    arguments["mode"] = "legacy"
    with pytest.raises(TypeError):
        compose_modal_packaged_adapter(**arguments)
    assert events == []
