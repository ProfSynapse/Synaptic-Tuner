"""Provider-free tests for the claim-bound packaged-call diagnostic."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.training.modal_host_effects import _encode_binding
from tuner.execution.providers.modal.packaged_worker import PACKAGED_WORKER_FAILURE_STAGES
from tuner.training.modal_host_reader import _FIXED_WORKER_FAILURE_STAGES

from tests.execution.providers.test_modal_packaged_binding import _binding
from tests.execution.providers.test_modal_packaged_dispatch import _case


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inspect_modal_packaged_call.py"
_SPEC = importlib.util.spec_from_file_location("inspect_modal_packaged_call", _SCRIPT)
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)
_CALL = "fc-packaged-call"


def test_worker_failure_stage_allowlists_agree_across_all_readers():
    assert PACKAGED_WORKER_FAILURE_STAGES == _FIXED_WORKER_FAILURE_STAGES
    assert PACKAGED_WORKER_FAILURE_STAGES == frozenset(diagnostic._WORKER_FAILURE_STAGES)


def _journal(tmp_path, *, binding=None, claim_override=None, call_override=None,
             catalog_name=None):
    binding = binding or _case()[0]
    command_digest = binding.command_digest
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "modal-host.sqlite3"
    claim = canonical_bytes(claim_override or {
        "schema_version": "synaptic-modal-packaged-host-attempt/v1",
        "command_digest": command_digest,
        "binding_digest": binding.authenticated_binding_digest,
    })
    raw_binding = _encode_binding(binding)
    call = canonical_bytes(call_override or {"provider_job_ref": _CALL})
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE attempts(namespace_ref TEXT,attempt_ref TEXT,digest TEXT,evidence BLOB)")
        db.execute("CREATE TABLE catalogs(namespace_ref TEXT,catalog_ref TEXT,item_ref TEXT,payload BLOB,digest TEXT)")
        db.execute("INSERT INTO attempts VALUES(?,?,?,?)", (
            diagnostic._NAMESPACE, command_digest,
            hashlib.sha256(claim).hexdigest(), claim,
        ))
        for name, raw in ((diagnostic._BINDINGS, raw_binding),
                          (catalog_name or diagnostic._CALLS, call)):
            db.execute("INSERT INTO catalogs VALUES(?,?,?,?,?)", (
                diagnostic._NAMESPACE, name, command_digest, raw,
                hashlib.sha256(raw).hexdigest(),
            ))
    path.chmod(0o600)
    return path, command_digest


@pytest.mark.skipif(os.name != "posix", reason="private POSIX journal")
def test_real_submit_binding_claim_and_call_are_required(tmp_path):
    path, claim_ref = _journal(tmp_path)
    before = set(path.parent.iterdir())
    assert diagnostic.read_retained_call(path, claim_ref, _CALL) == _CALL
    assert set(path.parent.iterdir()) == before
    for ref, call_id in (("a" * 64, _CALL), (claim_ref, "fc-other"),
                         ("invalid", _CALL)):
        with pytest.raises(diagnostic.DiagnosticUnavailable):
            diagnostic.read_retained_call(path, ref, call_id)
    assert set(path.parent.iterdir()) == before


@pytest.mark.skipif(os.name != "posix", reason="private POSIX journal")
@pytest.mark.parametrize("mutation", ["bad_binding_digest", "wrong_call", "wrong_catalog"])
def test_claim_binding_and_catalog_cannot_be_swapped(tmp_path, mutation):
    binding = _case()[0]
    overrides = {}
    if mutation == "bad_binding_digest":
        overrides["claim_override"] = {
            "schema_version": "synaptic-modal-packaged-host-attempt/v1",
            "command_digest": binding.command_digest, "binding_digest": "a" * 64,
        }
    elif mutation == "wrong_call":
        overrides["call_override"] = {"provider_job_ref": "fc-other"}
    else:
        overrides["catalog_name"] = "another-catalog"
    path, ref = _journal(tmp_path, binding=binding, **overrides)
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_retained_call(path, ref, _CALL)


@pytest.mark.skipif(os.name != "posix", reason="private POSIX journal")
def test_rejects_link_and_broad_permissions(tmp_path):
    path, ref = _journal(tmp_path)
    link = path.parent / "link.sqlite3"
    link.symlink_to(path)
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_retained_call(link, ref, _CALL)
    path.chmod(0o644)
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_retained_call(path, ref, _CALL)


@pytest.mark.skipif(os.name != "posix", reason="private POSIX journal")
def test_stage_command_cannot_be_treated_as_a_submitted_call(tmp_path):
    path, ref = _journal(tmp_path, binding=_binding(b"packaged-prepared-input"))
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_retained_call(path, ref, _CALL)


class _Result:
    def __init__(self, status, data=b"opaque", blob=""):
        self.status, self.data, self.data_blob_id = status, data, blob


class _Response:
    def __init__(self, outputs, unfinished):
        self.outputs, self.num_unfinished_inputs = outputs, unfinished


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


@pytest.mark.parametrize("response,expected", [
    (_Response([], 1), "PENDING"),
    (_Response([], 0), "OUTPUT_EXPIRED"),
    (_Response([SimpleNamespace(idx=0, data_format=1, result=_Result(1))], 0),
     "PROVIDER_SUCCESS_UNKNOWN"),
    (_Response([SimpleNamespace(idx=0, data_format=1, result=_Result(2))], 0),
     "PROVIDER_FAILURE"),
    (_Response([SimpleNamespace(idx=1, data_format=1, result=_Result(1))], 0),
     "INVALID_RESPONSE"),
])
def test_one_non_consuming_raw_poll(monkeypatch, response, expected):
    monkeypatch.setattr(diagnostic, "_pinned_python", lambda: False)
    calls = []

    class _Stub:
        async def FunctionGetOutputs(self, request, *, retry, timeout):
            calls.append((request, retry, timeout))
            return response

    result = asyncio.run(diagnostic.inspect_call(
        SimpleNamespace(stub=_Stub()), _CALL, _Proto, lambda _: b"fixed",
    ))
    assert result == expected
    assert len(calls) == 1
    request, retry, timeout = calls[0]
    assert retry is None and timeout == 15
    assert request.function_call_id == _CALL
    assert request.timeout == 0 and request.clear_on_success is False
    assert request.last_entry_id == "0-0"
    assert request.start_idx == request.end_idx == 0 and request.max_values == 1


def test_only_exact_locally_serialized_failure_is_classified(monkeypatch):
    monkeypatch.setattr(diagnostic, "_pinned_python", lambda: True)
    fixed = b"trusted-fixed-failure"
    serialize_calls = []

    def serialize(document):
        serialize_calls.append(document)
        return fixed

    def classify(data, *, blob="", data_format=1):
        output = SimpleNamespace(data_format=data_format, result=_Result(1, data, blob))
        return diagnostic._classify_fixed_failure(output, _Proto, serialize)

    assert classify(fixed) == "WORKER_FAILED"
    assert serialize_calls == [{
        "schema_version": diagnostic._RESULT_SCHEMA, "effect_id": "unavailable",
        "status_code": "failed", "completion_sha256": "0" * 64,
    }]
    for data, options in ((b"untrusted", {}), (fixed, {"blob": "bl-opaque"}),
                          (fixed, {"data_format": 99})):
        assert classify(data, **options) == "PROVIDER_SUCCESS_UNKNOWN"


@pytest.mark.parametrize("stage", diagnostic._WORKER_FAILURE_STAGES)
def test_exact_v2_failure_bytes_report_only_fixed_stage(monkeypatch, stage):
    monkeypatch.setattr(diagnostic, "_pinned_python", lambda: True)
    document = {
        "schema_version": diagnostic._RESULT_SCHEMA_V2,
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
        "failure_stage": stage,
    }
    # This serializer is deterministic; the diagnostic still compares opaque
    # bytes and never parses the returned provider payload.
    serialize = canonical_bytes
    output = SimpleNamespace(
        data_format=_Proto.DATA_FORMAT_PICKLE,
        result=_Result(_GenericResult.GENERIC_STATUS_SUCCESS,
                       serialize(document)),
    )
    assert diagnostic._classify_fixed_failure(
        output, _Proto, serialize,
    ) == f"WORKER_{stage}"


@pytest.mark.parametrize("mutation", [
    {"effect_id": "other"}, {"status_code": "completed"},
    {"completion_sha256": "a" * 64}, {"failure_stage": "SFT_OTHER"},
    {"schema_version": "other"}, {"extra": "private data"},
])
def test_v2_result_near_misses_remain_unclassified(monkeypatch, mutation):
    monkeypatch.setattr(diagnostic, "_pinned_python", lambda: True)
    document = {
        "schema_version": diagnostic._RESULT_SCHEMA_V2,
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
        "failure_stage": "SFT_TRAINER",
    }
    document.update(mutation)
    output = SimpleNamespace(
        data_format=_Proto.DATA_FORMAT_PICKLE,
        result=_Result(_GenericResult.GENERIC_STATUS_SUCCESS,
                       canonical_bytes(document)),
    )
    assert diagnostic._classify_fixed_failure(
        output, _Proto, canonical_bytes,
    ) == "PROVIDER_SUCCESS_UNKNOWN"


def test_invalid_journal_prevents_provider_import(tmp_path, monkeypatch, capsys):
    imported = []
    original = __import__("builtins").__import__

    def guarded(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            imported.append(name)
            raise AssertionError("provider imported before journal admission")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    code = diagnostic.main([
        "--journal", str(tmp_path / "missing.sqlite3"),
        "--claim-ref", "a" * 64, "--call-id", _CALL,
        "--modal-profile", "named-profile",
    ])
    assert code == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "synaptic-modal-packaged-call-diagnostic/v1",
        "authority": "DIAGNOSTIC_ONLY", "result": "JOURNAL_INVALID",
    }
    assert imported == []
