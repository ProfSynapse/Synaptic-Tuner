"""Actual pinned SDK cancellation contract against controlled no-network RPCs."""
from importlib.metadata import version
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize("rpc_failure", [False, True])
def test_real_sdk_cancel_identity_ack_and_uncertain_failure(monkeypatch, rpc_failure):
    modal = pytest.importorskip("modal")
    from google.protobuf.empty_pb2 import Empty
    from modal_proto import api_pb2
    from modal._utils.async_utils import synchronizer
    from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1, build_cancel_command
    from tuner.execution.foundation_v2.identities import EffectKind
    from tuner.execution.foundation_v2.references import CancellationRefV1, ProviderRunRefV1
    from tuner.execution.providers.modal.packaged_transport import ModalPackagedHostTransport
    from tuner.execution.providers.modal.coordinator_effects import ModalEffectOutcome
    from tuner.execution.foundation_v2.observations import ObservationDisposition
    from tests.execution.providers.test_modal_packaged_binding import _binding

    assert version("modal") == "1.5.4"
    calls = []
    class Stub:
        async def FunctionCallCancel(self, request, **kwargs):
            calls.append((request.function_call_id, request.terminate_containers))
            if rpc_failure:
                raise RuntimeError("controlled response loss")
            return Empty()
    client = modal.Client("http://127.0.0.1:1", api_pb2.CLIENT_TYPE_CONTAINER, None)
    synchronizer._translate_in(client)._stub = Stub()
    call = modal.FunctionCall.from_id("fc-owned", client=client)
    assert not call.is_hydrated
    assert call.hydrate(client) is call
    assert call.client is client and call.object_id == "fc-owned"
    if rpc_failure:
        with pytest.raises(RuntimeError):
            call.cancel(terminate_containers=True)
    else:
        assert call.cancel(terminate_containers=True) is None
    assert calls == [("fc-owned", True)]
    bound = _binding()
    preparation = bound.command.preparation
    command = build_cancel_command(
        preparation, "a" * 64,
        CanonicalProviderPayloadV1.build("modal", f"{EffectKind.CANCEL.value}-payload/v2", preparation.workload_digest),
        bound.command.executor, CancellationRefV1(ProviderRunRefV1("fc-owned"), "b" * 64),
    )
    # Full binding/store authentication is covered by the public scripted tests;
    # this control isolates the actual production SDK hydrate/cancel closure.
    transport = ModalPackagedHostTransport.__new__(ModalPackagedHostTransport)
    transport._sdk, transport._client = modal, client
    monkeypatch.setattr(ModalPackagedHostTransport, "_binding", lambda self, binding, parsed: bound)
    outcome = transport.execute_once(bound, command)
    expected = (ModalEffectOutcome(ObservationDisposition.INDETERMINATE) if rpc_failure
                else ModalEffectOutcome(ObservationDisposition.FOUND, "fc-owned"))
    assert outcome == expected
    assert calls == [("fc-owned", True)] * 2
    wrong_client = modal.Client("http://127.0.0.1:2", api_pb2.CLIENT_TYPE_CONTAINER, None)
    class WrongCalls:
        @staticmethod
        def from_id(reference, *, client):
            assert reference == "fc-owned"
            return modal.FunctionCall._new_hydrated(reference, wrong_client, None)
    transport._sdk = SimpleNamespace(FunctionCall=WrongCalls)
    assert transport.execute_once(bound, command) == ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
    assert calls == [("fc-owned", True)] * 2
