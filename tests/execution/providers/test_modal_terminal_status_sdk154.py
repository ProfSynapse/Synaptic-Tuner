"""Actual pinned SDK bridge/protobuf controls; no provider connections."""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

modal = pytest.importorskip("modal")
pytestmark = pytest.mark.skipif(
    modal.__version__ != "1.5.4", reason="dedicated Modal 1.5.4 lane"
)

from modal.client import _Client
from modal._utils.async_utils import synchronizer
from modal_proto import api_pb2

from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.facade import (
    ExplicitModal154ReadFacade,
    ModalFunctionCallState,
)


def _facade(response):
    raw = _Client("http://unused.invalid", api_pb2.CLIENT_TYPE_CLIENT, None)
    raw._owner_pid = os.getpid()
    calls = []

    async def loop():
        return asyncio.get_running_loop()

    sdk_loop = synchronizer.create_blocking(loop)()

    class Method:
        client = raw

        async def direct(self, request, *, timeout):
            assert asyncio.get_running_loop() is sdk_loop
            assert self.client is raw
            calls.append((request, timeout))
            if isinstance(response, Exception):
                raise response
            return response

        async def __call__(self, *_args, **_kwargs):
            pytest.fail("retrying SDK call used")

    raw._stub = SimpleNamespace(FunctionGetOutputs=Method())
    client = synchronizer._translate_out(raw)
    facade = ExplicitModal154ReadFacade(
        ModalClientBinding("account", "workspace", "environment", "client", "1.5.4"),
        sdk=modal,
        client=client,
        scope_observer=lambda _: ("account", "workspace", "environment", "client"),
        deployment_observer=lambda **_: None,
        volume_names={"vo-control": "control"},
    )
    return facade, calls, raw


def _response(status):
    return api_pb2.FunctionGetOutputsResponse(
        outputs=[
            api_pb2.FunctionGetOutputsItem(
                idx=0,
                result=api_pb2.GenericResult(status=status, data=b"never deserialize"),
            )
        ]
    )


@pytest.mark.parametrize(
    "status,expected",
    [
        (api_pb2.GenericResult.GENERIC_STATUS_SUCCESS, ModalFunctionCallState.RETURNED),
        (
            api_pb2.GenericResult.GENERIC_STATUS_TERMINATED,
            ModalFunctionCallState.CANCELLED,
        ),
        (api_pb2.GenericResult.GENERIC_STATUS_FAILURE, ModalFunctionCallState.FAILED),
        (
            api_pb2.GenericResult.GENERIC_STATUS_INIT_FAILURE,
            ModalFunctionCallState.FAILED,
        ),
        (api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT, ModalFunctionCallState.FAILED),
        (
            api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE,
            ModalFunctionCallState.UNKNOWN,
        ),
        (999, ModalFunctionCallState.UNKNOWN),
    ],
)
def test_exact_nonconsuming_request_and_same_sdk_loop(status, expected, monkeypatch):
    monkeypatch.setattr(
        modal.FunctionCall, "get", lambda *_a, **_k: pytest.fail("deserialized output")
    )
    facade, calls, _ = _facade(_response(status))
    assert (
        facade.observe_known_call_status("fc-known", deadline=time.monotonic() + 5)
        is expected
    )
    assert len(calls) == 1
    request, timeout = calls[0]
    assert type(request) is api_pb2.FunctionGetOutputsRequest
    assert request.function_call_id == "fc-known"
    assert request.timeout == 0 and request.last_entry_id == "0-0"
    assert request.clear_on_success is False
    assert request.start_idx == request.end_idx == 0
    assert not request.input_jwts and request.requested_at > 0
    assert 0 < timeout <= 5


@pytest.mark.parametrize(
    "response,expected",
    [
        (
            api_pb2.FunctionGetOutputsResponse(num_unfinished_inputs=1),
            ModalFunctionCallState.PENDING,
        ),
        (api_pb2.FunctionGetOutputsResponse(), ModalFunctionCallState.UNKNOWN),
        (
            api_pb2.FunctionGetOutputsResponse(num_unfinished_inputs=-1),
            ModalFunctionCallState.UNKNOWN,
        ),
        (RuntimeError("private provider text"), ModalFunctionCallState.UNKNOWN),
        (object(), ModalFunctionCallState.UNKNOWN),
    ],
)
def test_pending_expired_and_transport_are_distinct(response, expected):
    facade, _, _ = _facade(response)
    assert (
        facade.observe_known_call_status("fc-known", deadline=time.monotonic() + 5)
        is expected
    )


@pytest.mark.parametrize(
    "mutation", ["many", "index", "indices", "missing", "unfinished", "size"]
)
def test_ambiguous_or_oversized_response_is_unknown(mutation):
    response = _response(api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
    if mutation == "many":
        response.outputs.add().CopyFrom(response.outputs[0])
    elif mutation == "index":
        response.outputs[0].idx = 1
    elif mutation == "indices":
        response.idxs.append(1)
    elif mutation == "missing":
        response.outputs[0].ClearField("result")
    elif mutation == "unfinished":
        response.num_unfinished_inputs = 1
    else:
        response.outputs[0].result.data = b"x" * 1_048_576
    facade, _, _ = _facade(response)
    assert (
        facade.observe_known_call_status("fc-known", deadline=time.monotonic() + 5)
        is ModalFunctionCallState.UNKNOWN
    )


@pytest.mark.parametrize("mutation", ["snapshot", "process", "expired"])
def test_no_fresh_global_client_or_expired_rpc(mutation):
    facade, calls, raw = _facade(
        _response(api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
    )
    deadline = time.monotonic() + 5
    if mutation == "snapshot":
        raw._snapshotted = True
    elif mutation == "process":
        raw._owner_pid = -1
    else:
        deadline = time.monotonic() - 1
    assert (
        facade.observe_known_call_status("fc-known", deadline=deadline)
        is ModalFunctionCallState.UNKNOWN
    )
    assert calls == []


@pytest.mark.parametrize("mutation", ["hydrated_id", "stub_client"])
def test_exact_hydrated_call_and_stub_identity(mutation, monkeypatch):
    facade, calls, raw = _facade(
        _response(api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
    )
    if mutation == "stub_client":
        raw._stub.FunctionGetOutputs.client = object()
    else:
        original = modal.FunctionCall.from_id
        monkeypatch.setattr(
            modal.FunctionCall, "from_id", lambda _ref, **kw: original("fc-other", **kw)
        )
    assert (
        facade.observe_known_call_status("fc-known", deadline=time.monotonic() + 5)
        is ModalFunctionCallState.UNKNOWN
    )
    assert calls == []


def test_original_deadline_bounds_sdk_loop_wait():
    facade, calls, raw = _facade(
        _response(api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
    )

    async def delayed(request, *, timeout):
        calls.append(request)
        await asyncio.sleep(1)
        pytest.fail("expired response accepted")

    raw._stub.FunctionGetOutputs.direct = delayed
    started = time.monotonic()
    assert (
        facade.observe_known_call_status("fc-known", deadline=started + 0.05)
        is ModalFunctionCallState.UNKNOWN
    )
    assert len(calls) == 1 and time.monotonic() - started < 0.5


def test_call_identity_substitution_during_rpc_is_unknown(monkeypatch):
    facade, calls, raw = _facade(
        _response(api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
    )
    original_factory = modal.FunctionCall.from_id
    held = []

    def factory(ref, **kwargs):
        call = original_factory(ref, **kwargs)
        held.append(synchronizer._translate_in(call))
        return call

    monkeypatch.setattr(modal.FunctionCall, "from_id", factory)
    original_direct = raw._stub.FunctionGetOutputs.direct

    async def substitute(request, *, timeout):
        response = await original_direct(request, timeout=timeout)
        held[0]._object_id = "fc-other"
        return response

    raw._stub.FunctionGetOutputs.direct = substitute
    assert (
        facade.observe_known_call_status("fc-known", deadline=time.monotonic() + 5)
        is ModalFunctionCallState.UNKNOWN
    )
    assert len(calls) == 1
