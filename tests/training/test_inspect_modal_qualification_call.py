"""Provider-free proof for the retained, read-only qualification call diagnostic."""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickletools
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inspect_modal_qualification_call.py"
_SPEC = importlib.util.spec_from_file_location("inspect_modal_qualification_call", _SCRIPT)
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)

_FACTS = "a" * 64
_DISPATCH = "b" * 64
_CALL = "fc-exact-call"
_CLAIM_REF = "qualify-" + _FACTS


def _journal(tmp_path, *, call_id=_CALL, catalog_digest=None, claim_digest=None,
             catalog_ref=diagnostic._CATALOG):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "modal-host.sqlite3"
    claim = canonical_bytes({
        "schema_version": "synaptic-modal-host-cpu-qualification-claim/v1",
        "effect_id": "cpu-qual-test",
        "runtime_release_digest": "c" * 64,
        "deployment_facts_digest": _FACTS,
        "dispatch_digest": _DISPATCH,
    })
    payload = call_id.encode("ascii")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE attempts(namespace_ref TEXT,attempt_ref TEXT,digest TEXT,evidence BLOB)"
        )
        connection.execute(
            "CREATE TABLE catalogs(namespace_ref TEXT,catalog_ref TEXT,item_ref TEXT,payload BLOB,digest TEXT)"
        )
        connection.execute(
            "INSERT INTO attempts VALUES(?,?,?,?)",
            (diagnostic._NAMESPACE, _CLAIM_REF,
             claim_digest or hashlib.sha256(claim).hexdigest(), claim),
        )
        connection.execute(
            "INSERT INTO catalogs VALUES(?,?,?,?,?)",
            (diagnostic._NAMESPACE, catalog_ref, _DISPATCH,
             payload, catalog_digest or hashlib.sha256(payload).hexdigest()),
        )
    path.chmod(0o600)
    return path


@pytest.mark.skipif(diagnostic.os.name != "posix", reason="private POSIX journal")
def test_exact_claim_and_catalog_are_read_only(tmp_path):
    path = _journal(tmp_path)
    before = set(path.parent.iterdir())
    assert diagnostic.read_retained_call(path, _CLAIM_REF, _CALL) == _CALL
    assert set(path.parent.iterdir()) == before
    for claim_ref, call_id in (
        (_CLAIM_REF, "fc-fabricated"),
        ("qualify-" + "d" * 64, _CALL),
        ("invalid-claim", _CALL),
        (_CLAIM_REF, "bad-call"),
    ):
        with pytest.raises(diagnostic.DiagnosticUnavailable):
            diagnostic.read_retained_call(path, claim_ref, call_id)
    assert set(path.parent.iterdir()) == before


@pytest.mark.skipif(diagnostic.os.name != "posix", reason="private POSIX journal")
@pytest.mark.parametrize("override", [
    {"catalog_digest": "d" * 64},
    {"claim_digest": "d" * 64},
    {"catalog_ref": "another-catalog"},
])
def test_malformed_evidence_is_rejected(tmp_path, override):
    path = _journal(tmp_path, **override)
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_retained_call(path, _CLAIM_REF, _CALL)


@pytest.mark.skipif(diagnostic.os.name != "posix", reason="private POSIX journal")
def test_no_symlink_or_broad_permissions(tmp_path):
    path = _journal(tmp_path)
    link = path.parent / "link.sqlite3"
    link.symlink_to(path)
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_retained_call(link, _CLAIM_REF, _CALL)
    path.chmod(0o644)
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_retained_call(path, _CLAIM_REF, _CALL)


class _OpaqueResult:
    def __init__(self, status, data=b"opaque", data_blob_id=""):
        self.status = status
        self.data = data
        self.data_blob_id = data_blob_id

    def __getattr__(self, name):
        raise AssertionError("payload field must never be read: " + name)


class _Response:
    def __init__(self, outputs, unfinished):
        self.outputs = outputs
        self.num_unfinished_inputs = unfinished


class _GenericResult:
    GENERIC_STATUS_SUCCESS = 1
    GENERIC_STATUS_FAILURE = 2
    GENERIC_STATUS_TERMINATED = 3
    GENERIC_STATUS_TIMEOUT = 4
    GENERIC_STATUS_INIT_FAILURE = 5
    GENERIC_STATUS_INTERNAL_FAILURE = 6
    GENERIC_STATUS_IDLE_TIMEOUT = 7
    GENERIC_STATUS_MEMORY_MANAGER_EVICTION = 8


class _Proto:
    GenericResult = _GenericResult
    FunctionGetOutputsResponse = _Response
    DATA_FORMAT_PICKLE = 1

    @staticmethod
    def FunctionGetOutputsRequest(**kwargs):
        return SimpleNamespace(**kwargs)


@pytest.mark.parametrize("response,category", [
    (_Response([], 1), "PENDING"),
    (_Response([], 0), "OUTPUT_EXPIRED"),
    (_Response([SimpleNamespace(idx=0, data_format=1, result=_OpaqueResult(1))], 0), "PROVIDER_SUCCESS_UNKNOWN"),
    (_Response([SimpleNamespace(idx=0, data_format=1, result=_OpaqueResult(2))], 0), "PROVIDER_FAILURE"),
    (_Response([SimpleNamespace(idx=0, data_format=1, result=_OpaqueResult(7))], 0), "PROVIDER_FAILURE"),
    (_Response([SimpleNamespace(idx=0, data_format=1, result=_OpaqueResult(0))], 0), "INVALID_RESPONSE"),
    (_Response([SimpleNamespace(idx=1, data_format=1, result=_OpaqueResult(1))], 0), "INVALID_RESPONSE"),
    (_Response([], -1), "INVALID_RESPONSE"),
])
def test_one_raw_metadata_request_only(response, category):
    calls = []

    class _Stub:
        async def FunctionGetOutputs(self, request, *, retry, timeout):
            calls.append((request, retry, timeout))
            return response

        def __getattr__(self, name):
            raise AssertionError("unexpected provider method: " + name)

    client = SimpleNamespace(stub=_Stub())
    assert asyncio.run(diagnostic.inspect_call(
        client, _CALL, _Proto, lambda _value: b"fixed-local-result",
    )) == category
    assert len(calls) == 1
    request, retry, timeout = calls[0]
    assert retry is None and timeout == 15
    assert request.function_call_id == _CALL
    assert request.timeout == 0
    assert request.last_entry_id == "0-0"
    assert request.clear_on_success is False
    assert request.start_idx == request.end_idx == 0
    assert request.max_values == 1
    assert type(request.requested_at) is float


def test_raw_transport_error_is_closed():
    calls = []

    class _Stub:
        async def FunctionGetOutputs(self, request, *, retry, timeout):
            calls.append((request, retry, timeout))
            raise RuntimeError("secret provider response")

    assert asyncio.run(diagnostic.inspect_call(
        SimpleNamespace(stub=_Stub()), _CALL, _Proto,
        lambda _value: b"fixed-local-result",
    )) == "POLL_UNAVAILABLE"
    assert len(calls) == 1


@pytest.mark.parametrize("status,stage,expected,index", [
    ("completed", None, "WORKER_COMPLETED", 0),
    ("failed", None, "WORKER_FAILED", 1),
    ("failed", "PARENT_SETUP", "WORKER_PARENT_SETUP", 2),
    ("failed", "INSTALLED_CHILD", "WORKER_INSTALLED_CHILD", 3),
    ("failed", "PARENT_RELEASE", "WORKER_PARENT_RELEASE", 4),
    ("failed", "CHILD_RESULT", "WORKER_CHILD_RESULT", 5),
])
def test_exact_fixed_serialized_results_only(monkeypatch, status, stage, expected, index):
    monkeypatch.setattr(diagnostic, "_pinned_python", lambda: True)
    seen = []

    def serialize(value):
        seen.append(tuple(value.items()))
        return ("fixed:" + value["status_code"] + ":"
                + value.get("failure_stage", "")).encode("ascii")

    output = SimpleNamespace(
        idx=0, data_format=_Proto.DATA_FORMAT_PICKLE,
        result=_OpaqueResult(1, data=("fixed:" + status + ":"
                                      + (stage or "")).encode("ascii")),
    )

    class _Stub:
        async def FunctionGetOutputs(self, _request, *, retry, timeout):
            assert retry is None and timeout == 15
            return _Response([output], 0)

    result = asyncio.run(diagnostic.inspect_call(
        SimpleNamespace(stub=_Stub()), _CALL, _Proto, serialize,
    ))
    assert result == expected
    assert seen == [
        (("schema_version", diagnostic.QUALIFICATION_RESULT_SCHEMA),
         ("status_code", "completed")),
        (("schema_version", diagnostic.QUALIFICATION_RESULT_SCHEMA),
         ("status_code", "failed")),
        (("schema_version", diagnostic.QUALIFICATION_RESULT_SCHEMA),
         ("status_code", "failed"), ("failure_stage", "PARENT_SETUP")),
        (("schema_version", diagnostic.QUALIFICATION_RESULT_SCHEMA),
         ("status_code", "failed"), ("failure_stage", "INSTALLED_CHILD")),
        (("schema_version", diagnostic.QUALIFICATION_RESULT_SCHEMA),
         ("status_code", "failed"), ("failure_stage", "PARENT_RELEASE")),
        (("schema_version", diagnostic.QUALIFICATION_RESULT_SCHEMA),
         ("status_code", "failed"), ("failure_stage", "CHILD_RESULT")),
    ][:index + 1]


@pytest.mark.parametrize("data_format,data,blob", [
    (99, b"fixed:completed", ""),
    (1, b"fixed:completed", "bl-private"),
    (1, b"x" * (diagnostic._MAX_FIXED_RESULT + 1), ""),
    (1, b"unrecognized", ""),
])
def test_unsupported_or_hostile_success_stays_unknown(
    monkeypatch, data_format, data, blob,
):
    monkeypatch.setattr(diagnostic, "_pinned_python", lambda: True)
    output = SimpleNamespace(
        idx=0, data_format=data_format,
        result=_OpaqueResult(1, data=data, data_blob_id=blob),
    )

    class _Stub:
        async def FunctionGetOutputs(self, _request, *, retry, timeout):
            return _Response([output], 0)

    result = asyncio.run(diagnostic.inspect_call(
        SimpleNamespace(stub=_Stub()), _CALL, _Proto,
        lambda value: ("fixed:" + value["status_code"]).encode("ascii"),
    ))
    assert result == "PROVIDER_SUCCESS_UNKNOWN"


def test_hostile_pickle_is_never_deserialized(monkeypatch):
    monkeypatch.setattr(diagnostic, "_pinned_python", lambda: True)
    executed = []
    monkeypatch.setattr(os, "system", lambda command: executed.append(command) or 0)
    hostile = b"cos\nsystem\n(S'echo should-not-run'\ntR."
    assert "REDUCE" in {opcode.name for opcode, _argument, _position
                        in pickletools.genops(hostile)}
    output = SimpleNamespace(
        idx=0, data_format=_Proto.DATA_FORMAT_PICKLE,
        result=_OpaqueResult(1, data=hostile),
    )

    class _Stub:
        async def FunctionGetOutputs(self, _request, *, retry, timeout):
            return _Response([output], 0)

    assert asyncio.run(diagnostic.inspect_call(
        SimpleNamespace(stub=_Stub()), _CALL, _Proto,
        lambda value: ("fixed:" + value["status_code"]).encode("ascii"),
    )) == "PROVIDER_SUCCESS_UNKNOWN"
    assert executed == []


def test_other_python_cannot_classify_fixed_bytes(monkeypatch):
    monkeypatch.setattr(diagnostic, "_pinned_python", lambda: False)
    output = SimpleNamespace(
        idx=0, data_format=_Proto.DATA_FORMAT_PICKLE,
        result=_OpaqueResult(1, data=b"fixed:completed"),
    )

    class _Stub:
        async def FunctionGetOutputs(self, _request, *, retry, timeout):
            return _Response([output], 0)

    assert asyncio.run(diagnostic.inspect_call(
        SimpleNamespace(stub=_Stub()), _CALL, _Proto,
        lambda _value: (_ for _ in ()).throw(AssertionError("must not serialize")),
    )) == "PROVIDER_SUCCESS_UNKNOWN"


def test_pinned_runtime_requires_cpython(monkeypatch):
    monkeypatch.setattr(
        diagnostic, "sys",
        SimpleNamespace(
            implementation=SimpleNamespace(name="pypy"),
            version_info=(3, 11, 14),
        ),
    )
    assert diagnostic._pinned_python() is False


def test_invalid_journal_never_imports_modal(tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__
    seen = []

    def guarded_import(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            seen.append(name)
            raise AssertionError("Modal must not load before journal admission")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    code = diagnostic.main([
        "--journal", str(tmp_path / "missing.sqlite3"),
        "--claim-ref", _CLAIM_REF, "--call-id", _CALL,
        "--modal-profile", "synaptic-labs",
    ])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["result"] == "JOURNAL_INVALID"
    assert seen == []


def test_main_closed_output_and_named_credentials(monkeypatch, capsys):
    monkeypatch.setattr(diagnostic, "read_retained_call", lambda *_args: _CALL)
    modal = ModuleType("modal")
    modal.__version__ = "1.5.4"
    client = object()
    calls = []

    def from_credentials(token_id, token_secret):
        calls.append((token_id, token_secret))
        return client

    modal.Client = SimpleNamespace(from_credentials=from_credentials)
    config_module = ModuleType("modal.config")
    utils_module = ModuleType("modal._utils")
    async_utils_module = ModuleType("modal._utils.async_utils")
    bridge_calls = []

    def create_blocking(fn):
        bridge_calls.append(fn)

        def blocking(*args):
            bridge_calls.append(args)
            return asyncio.run(fn(*args))

        return blocking

    async_utils_module.synchronizer = SimpleNamespace(create_blocking=create_blocking)

    def credential(key, *, profile, use_env):
        assert profile == "named-profile" and use_env is False
        return "private-credential"

    config_module.config = SimpleNamespace(get=credential)
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setitem(sys.modules, "modal.config", config_module)
    monkeypatch.setitem(sys.modules, "modal._utils", utils_module)
    monkeypatch.setitem(sys.modules, "modal._utils.async_utils", async_utils_module)
    serializer_module = ModuleType("modal._serialization")
    serializer = lambda _value: b"trusted-fixed-result"
    serializer_module.serialize = serializer
    monkeypatch.setitem(sys.modules, "modal._serialization", serializer_module)
    async def inspect_call(*_args):
        return "PROVIDER_SUCCESS_UNKNOWN"

    proto_module = ModuleType("modal_proto")
    proto_module.api_pb2 = _Proto
    monkeypatch.setitem(sys.modules, "modal_proto", proto_module)
    monkeypatch.setattr(diagnostic, "inspect_call", inspect_call)
    code = diagnostic.main([
        "--journal", "/unused/private.sqlite3", "--claim-ref", _CLAIM_REF,
        "--call-id", _CALL, "--modal-profile", "named-profile",
    ])
    output = capsys.readouterr()
    assert code == 0
    assert json.loads(output.out) == {
        "schema_version": "synaptic-modal-qualification-call-diagnostic/v1",
        "authority": "DIAGNOSTIC_ONLY",
        "result": "PROVIDER_SUCCESS_UNKNOWN",
    }
    assert output.err == ""
    assert _CALL not in output.out and "private" not in output.out
    assert calls == [("private-credential", "private-credential")]
    assert bridge_calls == [inspect_call, (client, _CALL, _Proto, serializer)]
