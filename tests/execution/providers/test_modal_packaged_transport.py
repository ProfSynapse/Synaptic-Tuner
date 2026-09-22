"""Provider-free transport tests using the existing Modal SDK fakes."""

from __future__ import annotations

import pytest

from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.packaged_deployment import (
    ModalPackagedDeploymentObserver,
)
from tuner.execution.providers.modal.packaged_dispatch import (
    build_modal_packaged_dispatch,
)
from tuner.execution.providers.modal.packaged_staging import ModalPackagedInputStager
from tuner.execution.providers.modal.packaged_transport import ModalPackagedHostTransport

from tests.execution.providers.test_modal_packaged_deployment import Reader
from tests.execution.providers.test_modal_packaged_binding import _binding
from tests.execution.providers.test_modal_packaged_dispatch import (
    PREPARED_PAYLOAD,
    Auth,
    _case,
    _rebound_receipt,
)
from tests.execution.providers.test_modal_sdk154_adapter import (
    FakeFunction,
    FakeFunctionCall,
    FakeVolume,
    SDK,
)


class Source:
    def __init__(self, value, events, label):
        self.value, self.events, self.label = value, events, label
    def resolve(self, digest):
        self.events.append((self.label, digest))
        return self.value


class Catalog:
    def __init__(self, events, label):
        self.values = {}
        self.events, self.label = events, label
        self.fail_publish = False
    def publish_if_absent(self, digest, value):
        self.events.append((f"{self.label}.publish", digest, value))
        if self.fail_publish:
            raise RuntimeError("private catalog failure")
        prior = self.values.setdefault(digest, value)
        return prior == value
    def resolve(self, digest):
        self.events.append((f"{self.label}.resolve", digest))
        return self.values.get(digest)


def _transport():
    binding, receipt, workload, policy = _case()
    auth = Auth()
    dispatch = build_modal_packaged_dispatch(
        binding, receipt, workload, policy, auth, key_ref="dispatch-key",
    )
    facts = binding.provider_facts
    client = object()
    FakeVolume.calls = []
    FakeVolume.registry = {
        "control-name": FakeVolume(facts.control_volume_id),
        "artifact-name": FakeVolume(facts.artifact_volume_id),
    }
    FakeFunction.calls = []
    FakeFunction.spawn_calls = []
    FakeFunction.fail = False
    FakeFunctionCall.calls = []
    facade = ExplicitModal154ReadFacade(
        facts.client_binding, sdk=SDK, client=client,
        scope_observer=lambda supplied: (
            facts.account_ref, facts.workspace_ref, facts.environment_ref,
            facts.client_ref,
        ) if supplied is client else (),
        deployment_observer=lambda **_: None,
        volume_names={
            facts.control_volume_id: "control-name",
            facts.artifact_volume_id: "artifact-name",
        },
    )
    reader = Reader(facts)
    observer = ModalPackagedDeploymentObserver(
        sdk=SDK, client=client, client_binding=facts.client_binding,
        reader=reader,
    )
    events = []
    stage_receipts = Catalog(events, "stage")
    calls = Catalog(events, "call")
    transport = ModalPackagedHostTransport(
        sdk=SDK, client=client, facade=facade,
        deployment_observer=observer,
        stager=ModalPackagedInputStager(facade),
        stage_source=Source(None, events, "stage_source"),
        dispatch_source=Source(dispatch, events, "dispatch_source"),
        stage_receipts=stage_receipts, call_catalog=calls,
        dispatch_verifier=auth,
    )
    return binding, transport, calls, events, reader


def test_submit_observes_binding_then_spawns_once_and_retains_exact_call_id() -> None:
    binding, transport, calls, events, reader = _transport()
    command = binding.command
    outcome = transport.execute_once(binding, command)
    assert outcome.disposition is ObservationDisposition.FOUND
    assert outcome.provider_ref == "fc-1"
    assert len(reader.calls) == 1
    assert len(FakeFunction.spawn_calls) == 1
    assert FakeFunction.spawn_calls[0] == (transport._dispatch_source.value,)
    assert calls.values == {command.digest: "fc-1"}
    assert events == [
        ("dispatch_source", command.digest),
        ("call.publish", command.digest, "fc-1"),
        ("call.resolve", command.digest),
    ]


def test_spawn_failure_is_indeterminate_after_exactly_one_attempt() -> None:
    binding, transport, calls, _, _ = _transport()
    FakeFunction.fail = True
    outcome = transport.execute_once(binding, binding.command)
    assert outcome.disposition is ObservationDisposition.INDETERMINATE
    assert len(FakeFunction.spawn_calls) == 1
    assert calls.values == {}


def test_call_catalog_failure_after_spawn_is_ambiguous_and_never_replayed() -> None:
    binding, transport, calls, _, _ = _transport()
    calls.fail_publish = True
    outcome = transport.execute_once(binding, binding.command)
    assert outcome.disposition is ObservationDisposition.INDETERMINATE
    assert len(FakeFunction.spawn_calls) == 1
    assert calls.values == {}


def test_reconciliation_uses_only_retained_call_id_without_spawn() -> None:
    binding, transport, calls, _, _ = _transport()
    calls.values[binding.command_digest] = "fc-retained"
    outcome = transport.lookup_once(binding, binding.command)
    assert outcome.disposition is ObservationDisposition.FOUND
    assert outcome.provider_ref == "fc-retained"
    assert FakeFunction.spawn_calls == []
    assert FakeFunctionCall.calls == []


def test_operational_binding_substitution_stops_before_function_lookup() -> None:
    binding, transport, _, _, _ = _transport()
    other = _case()[0]
    # The separately parsed commands are semantically identical, so use a
    # stage command to prove effect-type substitution is rejected pre-provider.
    stage = __import__(
        "tests.execution.providers.test_modal_packaged_binding",
        fromlist=["_binding"],
    )._binding().command
    try:
        transport.execute_once(binding, stage)
    except ValueError:
        pass
    else:  # pragma: no cover - explicit security assertion
        raise AssertionError("effect substitution was admitted")
    assert FakeFunction.calls == []
    assert FakeFunction.spawn_calls == []


@pytest.mark.parametrize("fault", ("effect", "volume", "size", "digest"))
def test_stage_reconciliation_rejects_valid_but_substituted_receipt(
    fault: str,
) -> None:
    stage_binding = _binding(PREPARED_PAYLOAD)
    receipt = _case()[1]
    if fault == "effect":
        receipt = _rebound_receipt(receipt, stage="other-effect")
    elif fault == "volume":
        receipt = _rebound_receipt(receipt, volume="vo-other")
    elif fault == "size":
        receipt = _rebound_receipt(receipt, size=receipt.size_bytes + 1)
    else:
        receipt = _rebound_receipt(receipt, digest="0" * 64)
    with pytest.raises(ValueError, match="receipt mismatch"):
        ModalPackagedHostTransport._stage_receipt(
            stage_binding, stage_binding.command, receipt,
        )
