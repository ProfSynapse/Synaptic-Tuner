"""Provider-free tests for the claim-bound packaged-call diagnostic."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.training.modal_host_effects import (
    ModalPackagedMarkerMaterial, _encode_binding, _encode_marker_materials,
)
from tuner.execution.providers.modal.packaged_dispatch import ModalPackagedVolumeMarker
from tuner.execution.providers.modal.bounded_volume_read import BoundedVolumeReadError
from tuner.execution.providers.modal.packaged_worker import (
    PACKAGED_WORKER_FAILURE_STAGES, packaged_worker_failure,
)
from tuner.training.modal_host_reader import _FIXED_WORKER_FAILURE_STAGES

from tests.execution.providers.test_modal_packaged_binding import _binding
from tests.execution.providers.test_modal_packaged_dispatch import _case


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inspect_modal_packaged_call.py"
_SPEC = importlib.util.spec_from_file_location("inspect_modal_packaged_call", _SCRIPT)
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)
_CALL = "fc-packaged-call"


def _evaluation_document(binding):
    return {
        "schema_version": "synaptic-modal-packaged-evaluation/v1",
        "effect_id": binding.command.operation.effect.effect_id,
        "command_digest": binding.command_digest,
        "provider_job_ref": _CALL,
        "execution_binding_digest": binding.execution_binding.binding_digest,
        "evaluation": {"schema_version": "synaptic-post-training-evaluation/v1",
                       "status": "completed", "gate_passed": True, "response": "private café"},
    }


@pytest.mark.parametrize("ascii_only", [True, False])
def test_evaluation_metadata_projects_encoding_without_raw_content(ascii_only):
    binding = _case()[0]
    raw = json.dumps(_evaluation_document(binding), sort_keys=True, separators=(",", ":"),
                     ensure_ascii=ascii_only).encode("utf-8")
    metadata = diagnostic.evaluation_metadata(raw, b"unverified tag", binding, _CALL)
    assert metadata["result"] == "METADATA_READ"
    assert metadata["mac_authentication"] == "UNVERIFIED"
    assert metadata["canonical_ascii_matches"] is ascii_only
    assert metadata["canonical_utf8_matches"] is not ascii_only
    assert metadata["schema_matches"] and metadata["evaluation_schema_matches"]
    assert metadata["status"] == "completed" and metadata["gate_passed"] is True
    assert all(metadata["binding_matches"].values())
    assert metadata["record_size_bytes"] == len(raw)
    assert metadata["record_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "private" not in json.dumps(metadata)


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"evaluation":{}} trailing',
    b'{"evaluation":{},"evaluation":{}}', b'{"evaluation":{"gate_passed":NaN}}'])
def test_evaluation_bad_json_is_closed(raw):
    metadata = diagnostic.evaluation_metadata(raw, b"tag", _case()[0], _CALL)
    assert metadata["result"] == "JSON_INVALID"
    assert metadata["mac_authentication"] == "UNVERIFIED"
    assert "binding_matches" not in metadata


def test_evaluation_mismatches_and_unknown_values_are_not_echoed():
    binding = _case()[0]
    document = _evaluation_document(binding)
    document.update(effect_id="private unrelated", schema_version="private schema")
    document["evaluation"].update(status="private status", gate_passed=1)
    metadata = diagnostic.evaluation_metadata(json.dumps(document).encode(), b"tag", binding, _CALL)
    assert metadata["status"] == "UNKNOWN" and metadata["gate_passed"] is None
    assert metadata["schema_matches"] is False
    assert metadata["binding_matches"]["effect_id"] is False
    assert "private" not in json.dumps(metadata)


@pytest.mark.parametrize("failure,expected", [(None, None), ("deadline", "deadline"),
    ("startup_failed", "startup_failed"), ("private provider details", "UNKNOWN")])
@pytest.mark.parametrize("count,expected_count", [(0, 0), (32, 32), (33, None), (-1, None), (True, None)])
def test_evaluation_failure_and_counts_are_closed(failure, expected, count, expected_count):
    binding = _case()[0]
    document = _evaluation_document(binding)
    document["evaluation"].update(failure_code=failure, case_count=count, passed_count=count)
    metadata = diagnostic.evaluation_metadata(json.dumps(document).encode(), b"tag", binding, _CALL)
    assert metadata["failure_code"] == expected
    assert metadata["case_count"] == expected_count and metadata["passed_count"] == expected_count
    assert "private" not in json.dumps(metadata)


def _evaluation_transport(raw, tag=b"tag", *, size_delta=0, metadata_change=None, encoding=None):
    requests = []
    content_by_url = {"https://provider.example/record": raw,
                      "https://provider.example/mac": tag}
    class Stub:
        async def VolumeGetFile2(self, request, *, retry, timeout):
            requests.append((request, retry, timeout))
            is_record = request.path.endswith(".json")
            data = raw if is_record else tag
            values = dict(size=len(data) + size_delta, start=0, len=len(data) + size_delta,
                          get_urls=("https://provider.example/record" if is_record else "https://provider.example/mac",))
            values.update(metadata_change or {})
            return SimpleNamespace(**values)
    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        def get(self, url, *, allow_redirects):
            assert allow_redirects is False
            data = content_by_url[url]
            class Body:
                remaining = data
                async def read(self, maximum):
                    chunk, self.remaining = self.remaining[:maximum], self.remaining[maximum:]
                    return chunk
            class Block:
                status = 200
                headers = {"Content-Encoding": encoding} if encoding else {}
                content = Body()
                async def __aenter__(self): return self
                async def __aexit__(self, *_args): pass
            return Block()
    return SimpleNamespace(stub=Stub()), Session, requests


def _inspect_evaluation(raw, **kwargs):
    binding = _case()[0]
    client, session, requests = _evaluation_transport(raw, **kwargs)
    class MarkerReader:
        async def read_exact(self, **_kwargs): return b"marker"
    proto = SimpleNamespace(VolumeGetFile2Request=lambda **values: SimpleNamespace(**values))
    result = asyncio.run(diagnostic.inspect_evaluation_metadata(
        client, binding, _materials(binding), _CALL, proto,
        marker_reader=MarkerReader(), session_factory=session,
    ))
    return result, requests, binding


def test_evaluation_reads_exact_derived_paths_with_record_and_mac_budgets():
    raw = json.dumps(_evaluation_document(_case()[0])).encode()
    result, requests, binding = _inspect_evaluation(raw)
    assert result["result"] == "METADATA_READ"
    assert len(requests) == 2
    for (request, retry, timeout), leaf, bound in zip(requests, ("record.json", "record.mac"),
                                                   (16 * 1024 * 1024, 128)):
        assert request.volume_id == binding.provider_facts.artifact_volume_id
        assert request.path == diagnostic.operation_path(
            binding.command.operation.effect.effect_id, "evaluation") + "/" + leaf
        assert (request.start, request.len, retry, timeout) == (0, bound + 1, None, 15)


def test_evaluation_accepts_mac_at_exact_budget():
    result, _, _ = _inspect_evaluation(b'{"evaluation":{}}', tag=b"x" * 128)
    assert result["result"] == "METADATA_READ"
    assert result["mac_size_bytes"] == 128
    assert result["mac_authentication"] == "UNVERIFIED"


def _training_completion_fixture():
    from tests.execution.providers.test_modal_packaged_reader import _reader
    from tuner.execution.providers.modal.contracts import operation_path, provider_entry_identity
    binding, _, _, control, _, _ = _reader()
    prefix = operation_path(binding.command.operation.effect.effect_id, "evidence")
    document = json.loads(control.files[prefix + "/packaged-completion.json"])
    document["provider_job_ref"] = _CALL
    names = {"workload_record": "workload.json", "training_lineage": "training_lineage.json",
             "training_metrics": "training_metrics.json", "final_model": "final_model.tar",
             "tokenizer": "tokenizer.tar"}
    for member in document["members"]:
        member["path"] = operation_path(document["effect_id"], "output", names[member["role"]])
        member["provider_entry_id"] = provider_entry_identity(
            binding.provider_facts.artifact_volume_id, member["path"], member["size"])
    return binding, document


def test_training_completion_metadata_fixture_projects_only_closed_fields():
    from tuner.execution.providers.modal.coordinator_producer import MODAL_TRAINING_ARTIFACT_BOUNDS_V1
    assert diagnostic._MAX_ARTIFACT_BYTES == MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes
    assert diagnostic._MAX_ARTIFACT_TOTAL_BYTES == MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_total_bytes
    binding, document = _training_completion_fixture()
    result = diagnostic.training_completion_metadata(canonical_bytes(document), b"private MAC", binding, _CALL)
    assert result["result"] == "METADATA_READ" and result["mac_authentication"] == "UNVERIFIED"
    assert result["schema_matches"] and result["fields_match"] and result["digest_fields_valid"]
    assert result["inventory_matches"] and all(result["binding_matches"].values())
    assert "members" not in result and "private" not in json.dumps(result)


def test_completion_cli_authenticates_then_reads_control_only(monkeypatch, tmp_path, capsys):
    binding, document = _training_completion_fixture()
    client, session, requests = _evaluation_transport(canonical_bytes(document))
    events = []
    def retained(*_args):
        events.append("authenticated")
        return binding, _materials(binding), _CALL
    monkeypatch.setattr(diagnostic, "read_retained_probe", retained)
    def credentials(*_args):
        assert events == ["authenticated"]
        events.append("client")
        return client
    modal = ModuleType("modal")
    modal.__version__ = "1.5.4"
    modal.Client = SimpleNamespace(from_credentials=credentials)
    config = ModuleType("modal.config")
    config.config = SimpleNamespace(get=lambda *_args, **_kwargs: "private credential")
    async_utils = ModuleType("modal._utils.async_utils")
    async_utils.synchronizer = SimpleNamespace(create_blocking=lambda function:
        lambda *args, **kwargs: asyncio.run(function(*args, **kwargs)))
    for name, module in [("modal", modal), ("modal.config", config),
                         ("modal._utils.async_utils", async_utils)]:
        monkeypatch.setitem(sys.modules, name, module)
    class MarkerReader:
        async def read_exact(self, **_kwargs): return b"marker"
    monkeypatch.setattr(diagnostic, "BoundedModalVolumeReader", lambda **_kwargs: MarkerReader())
    inspect = diagnostic.inspect_training_completion_metadata
    async def with_session(*args, **kwargs):
        return await inspect(*args, session_factory=session, **kwargs)
    monkeypatch.setattr(diagnostic, "inspect_training_completion_metadata", with_session)
    assert diagnostic.main(["--journal", str(tmp_path / "journal"), "--claim-ref", "a" * 64,
        "--call-id", _CALL, "--modal-profile", "named-profile",
        "--inspect-training-completion-metadata"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "synaptic-modal-packaged-training-completion-diagnostic/v1"
    assert output["authority"] == "DIAGNOSTIC_ONLY" and output["mac_authentication"] == "UNVERIFIED"
    assert output["result"]["inventory_matches"] and output["result"]["result"] == "METADATA_READ"
    assert events == ["authenticated", "client"] and len(requests) == 2
    assert all(request.volume_id == binding.provider_facts.control_volume_id
               for request, _, _ in requests)


@pytest.mark.parametrize("mutation", ["path", "role", "duplicate", "size_text", "size_bool",
    "size_zero", "size_large", "digest", "entry", "extra", "count", "total"])
def test_training_completion_inventory_near_misses_never_echo_content(mutation):
    binding, document = _training_completion_fixture()
    member = document["members"][0]
    if mutation == "path": member["path"] = "private/../../hostile"
    elif mutation == "role": member["role"] = "private role"
    elif mutation == "duplicate": member["role"] = document["members"][1]["role"]
    elif mutation == "size_text": member["size"] = "private size"
    elif mutation == "size_bool": member["size"] = True
    elif mutation == "size_zero": member["size"] = 0
    elif mutation == "size_large": member["size"] = diagnostic._MAX_ARTIFACT_BYTES + 1
    elif mutation == "digest": member["sha256"] = "private digest"
    elif mutation == "entry": member["provider_entry_id"] = "private entry"
    elif mutation == "extra": member["private field"] = "private value"
    elif mutation == "count": document["members"].pop()
    elif mutation == "total":
        from tuner.execution.providers.modal.contracts import provider_entry_identity
        for item in document["members"]:
            item["size"] = 60 * 1024 * 1024
            item["provider_entry_id"] = provider_entry_identity(binding.provider_facts.artifact_volume_id,
                                                               item["path"], item["size"])
    result = diagnostic.training_completion_metadata(canonical_bytes(document), b"tag", binding, _CALL)
    assert result.get("inventory_matches", False) is False
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"members":[],"members":[]}',
                                 b'{"members":NaN}', b'{"members":1.5}'])
def test_training_completion_bad_json_is_closed(raw):
    result = diagnostic.training_completion_metadata(raw, b"tag", _case()[0], _CALL)
    assert result["result"] == "JSON_INVALID"


@pytest.mark.parametrize("fault", [None, "oversized", "truncated", "marker_before", "marker_after", "provider"])
def test_training_completion_exact_control_reads_bounds_and_marker_revalidation(fault):
    binding, document = _training_completion_fixture()
    raw = canonical_bytes(document)
    client, session, requests = _evaluation_transport(raw,
        size_delta=1 if fault == "truncated" else 0,
        metadata_change={"size": diagnostic._MAX_COMPLETION_RECORD_BYTES + 1} if fault == "oversized" else None)
    class MarkerReader:
        count = 0
        async def read_exact(self, **_kwargs):
            self.count += 1
            if fault == "marker_before" or fault == "marker_after" and self.count == 4:
                raise BoundedVolumeReadError("private marker")
            return b"marker"
    if fault == "provider":
        async def reject(*_args, **_kwargs): raise RuntimeError("private provider")
        client.stub.VolumeGetFile2 = reject
    proto = SimpleNamespace(VolumeGetFile2Request=lambda **values: SimpleNamespace(**values))
    result = asyncio.run(diagnostic.inspect_training_completion_metadata(
        client, binding, _materials(binding), _CALL, proto,
        marker_reader=MarkerReader(), session_factory=session))
    assert result["mac_authentication"] == "UNVERIFIED"
    if fault is None:
        assert result["inventory_matches"] and result["result"] == "METADATA_READ"
        assert len(requests) == 2
    else:
        assert result["result"] in {"MARKER_UNAVAILABLE", "FILE_METADATA_INVALID", "FILE_SIZE_INVALID", "FILE_UNAVAILABLE"}
    for (request, retry, timeout), leaf, maximum in zip(requests,
        ("packaged-completion.json", "packaged-completion.mac"), (64 * 1024, 128)):
        assert request.volume_id == binding.provider_facts.control_volume_id
        assert request.path == diagnostic.operation_path(binding.command.operation.effect.effect_id, "evidence") + "/" + leaf
        assert (request.start, request.len, retry, timeout) == (0, maximum + 1, None, 15)
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("error", [RuntimeError("private provider response"),
    diagnostic.DiagnosticUnavailable("private provider response")])
def test_evaluation_provider_errors_never_escape_projection(error):
    binding = _case()[0]
    class Stub:
        async def VolumeGetFile2(self, *_args, **_kwargs): raise error
    class Reader:
        async def read_exact(self, **_kwargs): return b"marker"
    result = asyncio.run(diagnostic.inspect_evaluation_metadata(
        SimpleNamespace(stub=Stub()), binding, _materials(binding), _CALL,
        SimpleNamespace(VolumeGetFile2Request=lambda **kwargs: SimpleNamespace(**kwargs)),
        marker_reader=Reader(), session_factory=lambda: None,
    ))
    assert result == {"result": "FILE_UNAVAILABLE", "mac_authentication": "UNVERIFIED"}


@pytest.mark.parametrize("options,code", [
    ({"size_delta": 1}, "FILE_SIZE_INVALID"),
    ({"size_delta": -1}, "FILE_SIZE_INVALID"),
    ({"metadata_change": {"size": 16 * 1024 * 1024 + 1}}, "FILE_METADATA_INVALID"),
    ({"metadata_change": {"start": True}}, "FILE_METADATA_INVALID"),
    ({"metadata_change": {"get_urls": ()}}, "FILE_METADATA_INVALID"),
    ({"metadata_change": {"get_urls": ("http://private.example",)}}, "FILE_UNAVAILABLE"),
    ({"encoding": "gzip"}, "FILE_UNAVAILABLE"),
    ({"tag": b"x" * 129}, "FILE_METADATA_INVALID"),
])
def test_evaluation_transport_rejects_invalid_sizes_urls_encoding_and_mac(options, code):
    result, _, _ = _inspect_evaluation(b'{"evaluation":{}}', **options)
    assert result == {"result": code, "mac_authentication": "UNVERIFIED"}


@pytest.mark.parametrize("later", [False, True])
def test_evaluation_marker_mismatch_blocks_or_discards_metadata(later):
    binding = _case()[0]
    client, session, requests = _evaluation_transport(b'{"evaluation":{}}')
    class Reader:
        count = 0
        async def read_exact(self, **kwargs):
            self.count += 1
            if self.count == (3 if later else 1):
                raise BoundedVolumeReadError("private provider exception")
            return b"marker"
    result = asyncio.run(diagnostic.inspect_evaluation_metadata(
        client, binding, _materials(binding), _CALL,
        SimpleNamespace(VolumeGetFile2Request=lambda **values: SimpleNamespace(**values)),
        marker_reader=Reader(), session_factory=session,
    ))
    assert result == {"result": "MARKER_UNAVAILABLE", "mac_authentication": "UNVERIFIED"}
    assert len(requests) == (2 if later else 0)


def test_invalid_evaluation_journal_is_closed_before_provider_import(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(diagnostic, "read_retained_probe", lambda *args: (_ for _ in ()).throw(
        diagnostic.DiagnosticUnavailable("JOURNAL_INVALID")))
    assert diagnostic.main(["--journal", str(tmp_path / "missing.sqlite3"),
        "--claim-ref", "a" * 64, "--call-id", _CALL, "--inspect-evaluation-metadata",
        "--modal-profile", "named-profile"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output == {"schema_version": "synaptic-modal-packaged-evaluation-diagnostic/v1",
                      "authority": "DIAGNOSTIC_ONLY", "mac_authentication": "UNVERIFIED",
                      "result": "JOURNAL_INVALID"}


def test_worker_failure_stage_allowlists_agree_across_all_readers():
    assert PACKAGED_WORKER_FAILURE_STAGES == _FIXED_WORKER_FAILURE_STAGES
    assert PACKAGED_WORKER_FAILURE_STAGES == frozenset(diagnostic._WORKER_FAILURE_STAGES)
    runtime_phases = (
        "CONFIG", "MODEL_SNAPSHOT", "MODEL_LIBRARY_LOAD", "MODEL_SOURCE",
        "TOKENIZER_SOURCE", "MODEL_FINALIZE", "LOSS_GUARD", "DATA_PREP",
        "LORA_ATTACH", "TRAINER_SETUP", "TRAIN_CALL", "SAVE", "POST_SAVE", "BOOTSTRAP_ENV",
        "TORCH_IMPORT", "UNSLOTH_IMPORT", "TRAINER_IMPORT",
    )
    assert {
        f"SFT_TRAINER_CHILD_EXEC_RUNTIME_{phase}" for phase in runtime_phases
    } <= PACKAGED_WORKER_FAILURE_STAGES


def test_fixed_worker_failure_results_fit_pinned_modal_serializer():
    serialize = pytest.importorskip("modal._serialization").serialize
    sizes = [len(serialize(packaged_worker_failure(stage)))
             for stage in PACKAGED_WORKER_FAILURE_STAGES]
    assert max(sizes) <= diagnostic._MAX_FIXED_RESULT


def _journal(tmp_path, *, binding=None, claim_override=None, call_override=None,
             catalog_name=None, marker_rows=False):
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
        if marker_rows:
            raw = _encode_marker_materials(_materials(binding))
            db.execute("INSERT INTO catalogs VALUES(?,?,?,?,?)", (
                diagnostic._NAMESPACE, diagnostic._MARKERS, command_digest,
                raw, hashlib.sha256(raw).hexdigest(),
            ))
    path.chmod(0o600)
    return path, command_digest


def _materials(binding):
    facts = binding.provider_facts
    roles = (("control", facts.control_volume_id),
             ("artifacts", facts.artifact_volume_id))
    if facts.model_cache_volume_id is not None:
        roles += (("model_cache", facts.model_cache_volume_id),)
    return tuple(
        ModalPackagedMarkerMaterial(
            ModalPackagedVolumeMarker(role, volume_id,
                                      ".synaptic-volume-marker-" + f"{index:032x}",
                                      hashlib.sha256(bytes([index]) * 32).hexdigest()),
            bytes([index]) * 32,
        ) for index, (role, volume_id) in enumerate(roles, 1)
    )


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


@pytest.mark.parametrize("label", ["SUCCESS", "FAILURE", "TERMINATED", "TIMEOUT",
    "INIT_FAILURE", "INTERNAL_FAILURE", "IDLE_TIMEOUT", "MEMORY_MANAGER_EVICTION",
    "UNKNOWN", "UNKNOWN_ZERO"])
@pytest.mark.parametrize("include", [False, True])
def test_provider_status_actual_protobuf_one_poll(monkeypatch, label, include):
    from modal_proto import api_pb2
    monkeypatch.setattr(diagnostic, "_pinned_python", lambda: False)
    status = (0 if label == "UNKNOWN_ZERO" else 987654 if label == "UNKNOWN" else
              getattr(api_pb2.GenericResult, "GENERIC_STATUS_" + label))
    if label == "UNKNOWN_ZERO": label = "UNKNOWN"
    response = api_pb2.FunctionGetOutputsResponse(outputs=[api_pb2.FunctionGetOutputsItem(
        idx=0, result=api_pb2.GenericResult(status=status, exception="private exception",
            traceback="private traceback", data_blob_id="private blob"))])
    calls = []
    class Stub:
        async def FunctionGetOutputs(self, request, *, retry, timeout):
            calls.append((request, retry, timeout))
            return response
    client = SimpleNamespace(stub=Stub())
    result = asyncio.run(diagnostic.inspect_call(client, _CALL, api_pb2, None,
                                                include_provider_status=include))
    expected = ("PROVIDER_SUCCESS_UNKNOWN" if label == "SUCCESS" else
                "INVALID_RESPONSE" if label == "UNKNOWN" else "PROVIDER_FAILURE")
    assert result == ({"result": expected, "provider_status": label} if include else expected)
    assert len(calls) == 1
    request, retry, timeout = calls[0]
    assert (request.function_call_id, request.timeout, request.clear_on_success,
            request.last_entry_id, request.start_idx, request.end_idx, request.max_values,
            retry, timeout) == (_CALL, 0, False, "0-0", 0, 0, 1, None, 15)
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("kind,expected", [("pending", "PENDING"),
    ("expired", "OUTPUT_EXPIRED"), ("index", "INVALID_RESPONSE"),
    ("multiple", "INVALID_RESPONSE"), ("negative", "INVALID_RESPONSE"),
    ("wrong_type", "INVALID_RESPONSE"), ("unavailable", "POLL_UNAVAILABLE")])
def test_provider_status_absent_for_unavailable_or_malformed_protobuf(kind, expected):
    from modal_proto import api_pb2
    response = api_pb2.FunctionGetOutputsResponse(num_unfinished_inputs=1 if kind == "pending" else 0)
    if kind in {"index", "multiple"}:
        response.outputs.add(idx=1 if kind == "index" else 0)
        if kind == "multiple": response.outputs.add(idx=0)
    if kind == "negative": response.num_unfinished_inputs = -1
    if kind == "wrong_type": response = object()
    class Stub:
        async def FunctionGetOutputs(self, *_args, **_kwargs):
            if kind == "unavailable": raise RuntimeError("private failure")
            return response
    result = asyncio.run(diagnostic.inspect_call(SimpleNamespace(stub=Stub()), _CALL,
        api_pb2, None, include_provider_status=True))
    assert result == {"result": expected, "provider_status": None}


def test_provider_failure_projection_never_reads_hostile_payload_fields():
    class HostileResult:
        status = _GenericResult.GENERIC_STATUS_TIMEOUT
        @property
        def exception(self): pytest.fail("exception read")
        @property
        def traceback(self): pytest.fail("traceback read")
        @property
        def data(self): pytest.fail("data read")
        @property
        def data_blob_id(self): pytest.fail("blob read")
        def __getattr__(self, name):
            raise AssertionError("payload field must not be read: " + name)
    response = _Response([SimpleNamespace(idx=0, result=HostileResult())], 0)
    class Stub:
        async def FunctionGetOutputs(self, *_args, **_kwargs): return response
    assert asyncio.run(diagnostic.inspect_call(SimpleNamespace(stub=Stub()), _CALL,
        _Proto, None, include_provider_status=True)) == {
            "result": "PROVIDER_FAILURE", "provider_status": "TIMEOUT"}


@pytest.mark.parametrize("include", [False, True])
def test_provider_status_cli_default_compatibility_and_authentication(monkeypatch, tmp_path, capsys, include):
    from modal_proto import api_pb2
    events = []
    def retained(*_args):
        events.append("authenticated")
        return _CALL
    monkeypatch.setattr(diagnostic, "read_retained_call", retained)
    class Stub:
        async def FunctionGetOutputs(self, *_args, **_kwargs):
            events.append("poll")
            return api_pb2.FunctionGetOutputsResponse(outputs=[api_pb2.FunctionGetOutputsItem(
                idx=0, result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT))])
    def credentials(*_args):
        assert events == ["authenticated"]
        events.append("client")
        return SimpleNamespace(stub=Stub())
    modal = ModuleType("modal")
    modal.__version__ = "1.5.4"
    modal.Client = SimpleNamespace(from_credentials=credentials)
    config = ModuleType("modal.config")
    config.config = SimpleNamespace(get=lambda *_args, **_kwargs: "private credential")
    async_utils = ModuleType("modal._utils.async_utils")
    async_utils.synchronizer = SimpleNamespace(create_blocking=lambda function:
        lambda *args, **kwargs: asyncio.run(function(*args, **kwargs)))
    serialization = ModuleType("modal._serialization")
    serialization.serialize = lambda _document: b"fixed"
    for name, module in [("modal", modal), ("modal.config", config),
        ("modal._utils.async_utils", async_utils), ("modal._serialization", serialization)]:
        monkeypatch.setitem(sys.modules, name, module)
    args = ["--journal", str(tmp_path / "journal"), "--claim-ref", "a" * 64,
            "--call-id", _CALL, "--modal-profile", "named-profile"]
    if include: args.append("--include-provider-status")
    assert diagnostic.main(args) == 0
    expected = {"schema_version": "synaptic-modal-packaged-call-diagnostic/v1",
                "authority": "DIAGNOSTIC_ONLY", "result": "PROVIDER_FAILURE"}
    if include: expected["provider_status"] = "TIMEOUT"
    assert json.loads(capsys.readouterr().out) == expected
    assert events == ["authenticated", "client", "poll"]


@pytest.mark.parametrize("mode", ["--inspect-markers", "--inspect-evaluation-metadata",
                                  "--probe-final-model-first-chunk", None])
def test_provider_status_cli_rejects_modes_and_unauthenticated_claim_before_client(monkeypatch, tmp_path, capsys, mode):
    def reject(*_args):
        if mode is not None: pytest.fail("incompatible option reached journal")
        raise diagnostic.DiagnosticUnavailable("JOURNAL_INVALID")
    monkeypatch.setattr(diagnostic, "read_retained_call", reject)
    monkeypatch.setattr(diagnostic, "read_retained_probe", reject)
    monkeypatch.setattr(diagnostic, "read_retained_markers", reject)
    modal = ModuleType("modal")
    modal.Client = SimpleNamespace(from_credentials=lambda *_args: pytest.fail("client created"))
    monkeypatch.setitem(sys.modules, "modal", modal)
    args = ["--journal", str(tmp_path / "journal"), "--claim-ref", "a" * 64,
            "--modal-profile", "named-profile", "--include-provider-status"]
    args.extend(["--inspect-markers"] if mode == "--inspect-markers" else ["--call-id", _CALL])
    if mode not in {None, "--inspect-markers"}: args.append(mode)
    assert diagnostic.main(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["result"] == ("INPUT_INVALID" if mode else "JOURNAL_INVALID")
    if mode is None: assert output["provider_status"] is None


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
    {"failure_stage": "ENTRYPOINT_MOUNT_CONTROL_DIR_EXTRA"},
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


@pytest.mark.parametrize("completion", [False, True])
def test_invalid_journal_prevents_provider_import(tmp_path, monkeypatch, capsys, completion):
    imported = []
    original = __import__("builtins").__import__

    def guarded(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            imported.append(name)
            raise AssertionError("provider imported before journal admission")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    args = [
        "--journal", str(tmp_path / "missing.sqlite3"),
        "--claim-ref", "a" * 64, "--call-id", _CALL,
        "--modal-profile", "named-profile",
    ]
    if completion: args.append("--inspect-training-completion-metadata")
    code = diagnostic.main(args)
    assert code == 1
    expected = {
        "schema_version": ("synaptic-modal-packaged-training-completion-diagnostic/v1" if completion
                           else "synaptic-modal-packaged-call-diagnostic/v1"),
        "authority": "DIAGNOSTIC_ONLY", "result": "JOURNAL_INVALID",
    }
    if completion: expected["mac_authentication"] = "UNVERIFIED"
    assert json.loads(capsys.readouterr().out) == expected
    assert imported == []


@pytest.mark.parametrize("mode", ["--inspect-markers", "--inspect-evaluation-metadata",
    "--probe-final-model-first-chunk", "--include-provider-status"])
def test_completion_cli_incompatible_modes_fail_before_journal(tmp_path, monkeypatch, capsys, mode):
    def reject(*_args): pytest.fail("incompatible mode reached journal")
    for name in ("read_retained_probe", "read_retained_call", "read_retained_markers"):
        monkeypatch.setattr(diagnostic, name, reject)
    args = ["--journal", str(tmp_path / "journal"), "--claim-ref", "a" * 64,
            "--modal-profile", "named-profile", "--inspect-training-completion-metadata", mode]
    if mode != "--inspect-markers": args.extend(["--call-id", _CALL])
    assert diagnostic.main(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["result"] == "INPUT_INVALID"
    assert output["mac_authentication"] == "UNVERIFIED"


def test_completion_metadata_binding_and_digest_mismatches_are_not_echoed():
    binding, document = _training_completion_fixture()
    document.update(command_digest="private digest", provider_job_ref="private call",
                    schema_version="private schema", terminal_sha256="private terminal")
    result = diagnostic.training_completion_metadata(canonical_bytes(document), b"tag", binding, _CALL)
    assert result["result"] == "METADATA_READ"
    assert result["schema_matches"] is False and result["digest_fields_valid"] is False
    assert result["binding_matches"]["command_digest"] is False
    assert result["binding_matches"]["provider_job_ref"] is False
    assert "private" not in json.dumps(result)


@pytest.mark.skipif(os.name != "posix", reason="private POSIX journal")
def test_markers_need_no_call_and_leave_journal_untouched(tmp_path):
    path, ref = _journal(tmp_path, marker_rows=True)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM catalogs WHERE catalog_ref=?", (diagnostic._CALLS,))
    before = path.read_bytes()
    expected = _materials(_case()[0])
    assert diagnostic.read_retained_markers(path, ref) == expected
    assert path.read_bytes() == before
    assert tuple(path.parent.iterdir()) == (path,)


@pytest.mark.skipif(os.name != "posix", reason="private POSIX journal")
@pytest.mark.parametrize("mutation", ["wrong_role", "wrong_volume", "duplicate_name",
                                      "wrong_value", "wrong_digest", "missing_catalog"])
def test_marker_catalog_tampering_fails_closed(tmp_path, mutation):
    path, ref = _journal(tmp_path, marker_rows=True)
    with sqlite3.connect(path) as db:
        if mutation == "missing_catalog":
            db.execute("DELETE FROM catalogs WHERE catalog_ref=?", (diagnostic._MARKERS,))
        else:
            raw = db.execute("SELECT payload FROM catalogs WHERE catalog_ref=?",
                             (diagnostic._MARKERS,)).fetchone()[0]
            document = json.loads(raw)
            items = document["items"]
            if mutation == "wrong_role":
                items[0]["role"] = "artifacts"
            elif mutation == "wrong_volume":
                items[0]["volume_id"] = "vo-other"
            elif mutation == "duplicate_name":
                items[1]["marker_name"] = items[0]["marker_name"]
            elif mutation == "wrong_value":
                items[0]["value_hex"] = "00" * 32
            else:
                items[0]["value_sha256"] = "a" * 64
            raw = canonical_bytes(document)
            db.execute("UPDATE catalogs SET payload=?,digest=? WHERE catalog_ref=?",
                       (raw, hashlib.sha256(raw).hexdigest(), diagnostic._MARKERS))
    with pytest.raises(diagnostic.DiagnosticUnavailable, match="JOURNAL_INVALID"):
        diagnostic.read_retained_markers(path, ref)


@pytest.mark.skipif(os.name != "posix", reason="private POSIX journal")
def test_invalid_marker_journal_prevents_provider_import(tmp_path, monkeypatch, capsys):
    imported = []
    original = __import__("builtins").__import__

    def guarded(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            imported.append(name)
            raise AssertionError("provider imported before marker admission")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    code = diagnostic.main([
        "--journal", str(tmp_path / "missing.sqlite3"),
        "--claim-ref", "a" * 64, "--inspect-markers",
        "--modal-profile", "named-profile",
    ])
    assert code == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "synaptic-modal-packaged-marker-diagnostic/v1",
        "authority": "DIAGNOSTIC_ONLY", "result": "JOURNAL_INVALID",
    }
    assert imported == []


def test_marker_reads_are_exact_and_classified_without_payloads():
    materials = _materials(_case()[0])
    calls = []

    class Reader:
        async def read_exact(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise BoundedVolumeReadError("modal_volume_digest_mismatch")
            return b"opaque bytes are never reported"

    results = asyncio.run(diagnostic.inspect_markers(Reader(), materials))
    assert results == {"control": "MATCH", "artifacts": "UNAVAILABLE"}
    for call, material in zip(calls, materials):
        assert call == {"volume_id": material.commitment.volume_id,
                        "path": material.commitment.marker_name,
                        "expected_size": 32,
                        "expected_sha256": material.commitment.value_sha256,
                        "max_bytes": 32}


def test_not_found_requires_exact_grpc_cause(monkeypatch):
    class Status:
        NOT_FOUND = object()
        INTERNAL = object()

    class GRPCError(Exception):
        def __init__(self, status):
            self.status = status

    package = ModuleType("grpclib")
    package.__path__ = []
    constants = ModuleType("grpclib.const")
    constants.Status = Status
    exceptions = ModuleType("grpclib.exceptions")
    exceptions.GRPCError = GRPCError
    for name, module in (("grpclib", package), ("grpclib.const", constants),
                         ("grpclib.exceptions", exceptions)):
        monkeypatch.setitem(sys.modules, name, module)

    def failure(cause):
        try:
            raise cause
        except Exception:
            try:
                raise BoundedVolumeReadError("modal_volume_range_unavailable") from None
            except BoundedVolumeReadError as error:
                return error

    assert diagnostic._proven_not_found(failure(GRPCError(Status.NOT_FOUND)))
    assert not diagnostic._proven_not_found(failure(GRPCError(Status.INTERNAL)))
    assert not diagnostic._proven_not_found(failure(ValueError("untrusted")))
    assert not diagnostic._proven_not_found(
        BoundedVolumeReadError("modal_volume_digest_mismatch"))


@pytest.mark.skipif(os.name != "posix", reason="private POSIX journal")
def test_probe_admission_requires_same_claim_call_and_markers(tmp_path):
    path, ref = _journal(tmp_path, marker_rows=True)
    before = path.read_bytes()
    binding, materials, call = diagnostic.read_retained_probe(path, ref, _CALL)
    assert binding.command_digest == ref
    assert materials == _materials(binding)
    assert call == _CALL
    assert path.read_bytes() == before
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_retained_probe(path, ref, "fc-other")
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM catalogs WHERE catalog_ref=?", (diagnostic._MARKERS,))
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_retained_probe(path, ref, _CALL)


@pytest.mark.parametrize("size,expected", [
    (0, "FIRST_BLOCK_EMPTY"),
    (1024 * 1024, "FIRST_BLOCK_LE_1M"),
    (1024 * 1024 + 1, "FIRST_BLOCK_GT_1M"),
    (2 * 1024 * 1024, "FIRST_BLOCK_GT_1M"),
])
def test_probe_reads_only_first_bound_block_without_returning_bytes(size, expected):
    binding = _case()[0]
    requests, reads, urls = [], [], []

    class Body:
        remaining = size

        async def read(self, maximum):
            reads.append(maximum)
            count = min(self.remaining, maximum, 8192)
            self.remaining -= count
            return b"private"[:1] * count

    class Block:
        status = 200
        headers = {}
        content = Body()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, url, *, allow_redirects):
            urls.append((url, allow_redirects))
            return Block()

    class Stub:
        async def VolumeGetFile2(self, request, *, retry, timeout):
            requests.append((request, retry, timeout))
            return SimpleNamespace(size=max(size, 1), start=0, len=max(size, 1),
                                   get_urls=("https://provider.example/first",
                                             "https://provider.example/unused"))

    class Proto:
        @staticmethod
        def VolumeGetFile2Request(**kwargs):
            return SimpleNamespace(**kwargs)

    category = asyncio.run(diagnostic.inspect_first_artifact_block(
        SimpleNamespace(stub=Stub()), binding, Proto, session_factory=Session,
    ))
    assert category == expected
    assert len(requests) == 1
    request, retry, timeout = requests[0]
    assert retry is None and timeout == 15
    assert request.volume_id == binding.provider_facts.artifact_volume_id
    assert request.path == diagnostic.operation_path(
        binding.command.operation.effect.effect_id, "output", "final_model.tar",
    )
    assert urls == [("https://provider.example/first", False)]
    assert all(0 < count <= 64 * 1024 for count in reads)
    assert size - Block.content.remaining <= diagnostic._FIRST_BLOCK_LIMIT + 1


@pytest.mark.parametrize("change", [
    {"size": 193 * 1024 * 1024}, {"start": 1}, {"len": 2},
    {"get_urls": ()}, {"get_urls": ("http://provider.example/first",)},
    {"get_urls": ("https://user@provider.example/first",)},
])
def test_probe_rejects_unbounded_or_untrusted_metadata_before_http(change):
    binding = _case()[0]
    metadata = dict(size=1, start=0, len=1,
                    get_urls=("https://provider.example/first",))
    metadata.update(change)

    class Stub:
        async def VolumeGetFile2(self, *_args, **_kwargs):
            return SimpleNamespace(**metadata)

    def no_session():
        raise AssertionError("invalid metadata reached HTTP")

    category = asyncio.run(diagnostic.inspect_first_artifact_block(
        SimpleNamespace(stub=Stub()), binding,
        SimpleNamespace(VolumeGetFile2Request=lambda **kwargs: SimpleNamespace(**kwargs)),
        session_factory=no_session,
    ))
    assert category in {"BLOCK_METADATA_INVALID", "FIRST_BLOCK_UNAVAILABLE"}


def test_probe_rejects_encoded_block_without_reading_body():
    binding = _case()[0]

    class Block:
        status = 200
        headers = {"Content-Encoding": "gzip"}

        class content:
            @staticmethod
            async def read(_maximum):
                raise AssertionError("encoded body must not be read")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, *_args, **_kwargs):
            return Block()

    class Stub:
        async def VolumeGetFile2(self, *_args, **_kwargs):
            return SimpleNamespace(size=2, start=0, len=2,
                                   get_urls=("https://provider.example/first",))

    category = asyncio.run(diagnostic.inspect_first_artifact_block(
        SimpleNamespace(stub=Stub()), binding,
        SimpleNamespace(VolumeGetFile2Request=lambda **kwargs: SimpleNamespace(**kwargs)),
        session_factory=Session,
    ))
    assert category == "FIRST_BLOCK_UNAVAILABLE"


def test_invalid_probe_journal_prevents_provider_import(tmp_path, monkeypatch, capsys):
    imported = []
    original = __import__("builtins").__import__

    def guarded(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            imported.append(name)
            raise AssertionError("provider imported before probe admission")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    code = diagnostic.main([
        "--journal", str(tmp_path / "missing.sqlite3"),
        "--claim-ref", "a" * 64, "--call-id", _CALL,
        "--probe-final-model-first-chunk", "--modal-profile", "named-profile",
    ])
    assert code == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "synaptic-modal-packaged-artifact-probe-diagnostic/v1",
        "authority": "DIAGNOSTIC_ONLY", "result": "JOURNAL_INVALID",
    }
    assert imported == []


def test_pinned_synchronizer_bridges_probe_without_volume_hydration():
    synchronizer = pytest.importorskip("modal._utils.async_utils").synchronizer
    binding = _case()[0]
    requests = []

    class Body:
        calls = 0

        async def read(self, maximum):
            self.calls += 1
            assert maximum <= 64 * 1024
            return b"x" if self.calls == 1 else b""

    class Block:
        status = 200
        headers = {}
        content = Body()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, url, *, allow_redirects):
            assert url == "https://provider.example/first"
            assert allow_redirects is False
            return Block()

    class Stub:
        async def VolumeGetFile2(self, request, *, retry, timeout):
            requests.append((request, retry, timeout))
            return SimpleNamespace(size=1, start=0, len=1,
                                   get_urls=("https://provider.example/first",))

    proto = SimpleNamespace(VolumeGetFile2Request=lambda **kwargs: SimpleNamespace(**kwargs))
    category = synchronizer.create_blocking(diagnostic.inspect_first_artifact_block)(
        SimpleNamespace(stub=Stub()), binding, proto, session_factory=Session,
    )
    assert category == "FIRST_BLOCK_LE_1M"
    assert len(requests) == 1
    request, retry, timeout = requests[0]
    assert (request.volume_id, request.path, retry, timeout) == (
        binding.provider_facts.artifact_volume_id,
        diagnostic.operation_path(binding.command.operation.effect.effect_id,
                                  "output", "final_model.tar"), None, 15,
    )


def _phase_line(**changes):
    record = dict(schema_version=diagnostic._PHASE_SCHEMA, phase="TRAINER_EXECUTE",
                  edge="START", elapsed_ms=0, request_ordinal=None)
    record.update(changes)
    return diagnostic._PHASE_PREFIX + json.dumps(record, separators=(",", ":")) + "\n"


def _phase_response(lines, *, function="fu-worker", call=_CALL):
    proto = pytest.importorskip("modal_proto.api_pb2")
    return proto.AppFetchLogsResponse(batches=[proto.TaskLogsBatch(
        function_id=function, items=[proto.TaskLogs(data=line, function_call_id=call,
                                                  file_descriptor=proto.FILE_DESCRIPTOR_STDOUT)
                                    for line in lines])])


def _inspect_trace(response):
    proto = pytest.importorskip("modal_proto.api_pb2")
    requests = []
    class Stub:
        async def AppFetchLogs(self, request, *, retry, timeout):
            requests.append((request, retry, timeout))
            if isinstance(response, Exception):
                raise response
            return response
    return asyncio.run(diagnostic.inspect_phase_trace(
        SimpleNamespace(stub=Stub()), _case()[0], _CALL, proto)), requests


def test_phase_trace_actual_pinned_protobuf_and_one_exact_request():
    import importlib.metadata
    proto = pytest.importorskip("modal_proto.api_pb2")
    assert importlib.metadata.version("modal") == "1.5.4"
    assert importlib.metadata.version("protobuf") == "6.33.6"
    fields = proto.AppFetchLogsRequest.DESCRIPTOR.fields_by_name
    assert set(fields) == {"app_id", "since", "until", "limit", "source",
                           "function_id", "function_call_id", "task_id", "sandbox_id", "search_text"}
    assert fields["limit"].type == fields["limit"].TYPE_UINT32
    assert proto.AppFetchLogsResponse.DESCRIPTOR.fields_by_name["batches"].message_type.full_name == "modal.client.TaskLogsBatch"
    result, requests = _inspect_trace(_phase_response([
        "private path prompt case response credential\n", _phase_line(),
        _phase_line(phase="CHAT_REQUEST", edge="RETURN", elapsed_ms=86400000, request_ordinal=32)]))
    assert len(requests) == 1
    request, retry, timeout = requests[0]
    assert type(request) is proto.AppFetchLogsRequest
    assert request == proto.AppFetchLogsRequest(app_id="ap-owned", function_id="fu-worker",
                                               function_call_id=_CALL, limit=256)
    assert (retry, timeout) == (None, 15)
    assert result["result"] == "TRACE_READ" and result["reason"] == "SNAPSHOT_ONLY"
    assert result["completeness"] == "INCONCLUSIVE"
    assert result["wire_byte_cap"] is False and result["byte_bounds"] == "AFTER_RECEPTION"
    assert result["records"] == [
        dict(phase="TRAINER_EXECUTE", edge="START", elapsed_ms=0, request_ordinal=None),
        dict(phase="CHAT_REQUEST", edge="RETURN", elapsed_ms=86400000, request_ordinal=32)]
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("phase", sorted(diagnostic._PHASES))
@pytest.mark.parametrize("edge", ["START", "RETURN", "ERROR"])
def test_phase_closed_vocabulary(phase, edge):
    result, _ = _inspect_trace(_phase_response([_phase_line(
        phase=phase, edge=edge, request_ordinal=1 if phase == "CHAT_REQUEST" else None)]))
    assert result["records"][0]["phase"] == phase
    assert result["records"][0]["edge"] == edge


@pytest.mark.parametrize("change", [
    {"schema_version": "private"}, {"phase": "private"}, {"edge": "private"},
    {"phase": []}, {"edge": []}, {"elapsed_ms": True}, {"elapsed_ms": -1},
    {"elapsed_ms": 86400001}, {"elapsed_ms": 1.5}, {"elapsed_ms": "private"},
    {"request_ordinal": 1}, {"private": "credential"},
    {"phase": "CHAT_REQUEST", "request_ordinal": True},
    {"phase": "CHAT_REQUEST", "request_ordinal": None},
    {"phase": "CHAT_REQUEST", "request_ordinal": 0},
    {"phase": "CHAT_REQUEST", "request_ordinal": 33},
    {"phase": "CHAT_REQUEST", "request_ordinal": 1.5},
])
def test_phase_invalid_labels_types_and_ranges_never_echo(change):
    result, requests = _inspect_trace(_phase_response([_phase_line(**change)]))
    assert result["result"] == "TRACE_INCONCLUSIVE"
    assert result["reason"] == "MALFORMED" and result["records"] == []
    assert len(requests) == 1 and "private" not in json.dumps(result)


@pytest.mark.parametrize("line", [
    _phase_line().replace('"edge":"START"', '"edge":"START","edge":"RETURN"'),
    _phase_line().replace('"elapsed_ms":0', '"elapsed_ms":NaN'),
    _phase_line().replace('"request_ordinal":null', ''),
    _phase_line().rstrip("\n"), _phase_line().replace("\n", "\r\n"),
    "SYNAPTIC_PHASE private\n", "SYNAPTIC_PHASE []\n",
    "SYNAPTIC_PHASE " + " " * 513 + "\n",
    _phase_line().replace('"elapsed_ms":0', '"elapsed_ms":\n0'),
    _phase_line().replace('"phase":"TRAINER_EXECUTE"', '"phase":"private\\nresponse"'),
])
def test_phase_hostile_matching_lines_are_inconclusive(line):
    result, _ = _inspect_trace(_phase_response([line]))
    assert result["reason"] == "MALFORMED" and result["records"] == []
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("function,call", [("fu-other", _CALL), ("", _CALL),
                                            ("fu-worker", "fc-other"), ("fu-worker", "")])
def test_phase_response_identity_mismatch_is_closed(function, call):
    result, _ = _inspect_trace(_phase_response([_phase_line()], function=function, call=call))
    assert result["reason"] == "MALFORMED" and result["records"] == []


@pytest.mark.parametrize("kind", ["message", "aggregate", "entry_over", "entries_cap", "records_cap"])
def test_phase_received_bounds_and_caps(kind):
    lines = {"message": ["private" * 1024],
             "aggregate": ["x" * 2048] * 256,
             "entry_over": ["ignored\n"] * 257,
             "entries_cap": ["ignored\n"] * 255 + [_phase_line()],
             "records_cap": [_phase_line() * 16] * 16}[kind]
    result, requests = _inspect_trace(_phase_response(lines))
    assert len(requests) == 1
    assert result["result"] == "TRACE_INCONCLUSIVE"
    assert result["reason"] == ("CAPPED" if kind.endswith("cap") else "RECEIVED_BOUND_EXCEEDED")
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("response", [RuntimeError("private credential"), SimpleNamespace(batches=[])])
def test_phase_provider_failure_or_wrong_message_is_closed(response):
    result, requests = _inspect_trace(response)
    assert len(requests) == 1 and result["records"] == []
    assert result["reason"] in {"UNAVAILABLE", "MALFORMED"}
    assert "private" not in json.dumps(result)


def test_phase_unknown_lines_and_missing_trace_are_inconclusive():
    result, _ = _inspect_trace(_phase_response(["private\n prefix SYNAPTIC_PHASE private\n"]))
    assert result["reason"] == "MISSING" and result["records"] == []


@pytest.mark.parametrize("mode", ["--inspect-markers", "--inspect-evaluation-metadata",
    "--probe-final-model-first-chunk", "--include-provider-status", "--inspect-training-completion-metadata"])
def test_phase_cli_mutually_exclusive_before_authentication(monkeypatch, tmp_path, capsys, mode):
    def reject(*_args): pytest.fail("invalid mode reached authentication")
    monkeypatch.setattr(diagnostic, "read_retained_probe", reject)
    args = ["--journal", str(tmp_path / "journal"), "--claim-ref", "a" * 64,
            "--modal-profile", "profile", "--inspect-phase-trace", mode]
    if mode != "--inspect-markers": args.extend(["--call-id", _CALL])
    assert diagnostic.main(args) == 1
    assert json.loads(capsys.readouterr().out)["result"] == "INPUT_INVALID"


@pytest.mark.parametrize("fault", ["journal", "app", "function", None])
@pytest.mark.parametrize("serving", [False, True])
def test_phase_cli_authentication_and_filters_precede_credentials(monkeypatch, tmp_path, capsys, fault, serving):
    proto = pytest.importorskip("modal_proto.api_pb2")
    events, requests = [], []
    binding = _case()[0]
    if fault in {"app", "function"}:
        facts = SimpleNamespace(app_id="ap-owned", function_id="fu-worker")
        setattr(facts, "app_id" if fault == "app" else "function_id", "")
        binding = SimpleNamespace(provider_facts=facts)
    def retained(*_args):
        events.append("authenticated")
        if fault == "journal": raise diagnostic.DiagnosticUnavailable("JOURNAL_INVALID")
        return binding, (), _CALL
    monkeypatch.setattr(diagnostic, "read_retained_probe", retained)
    def credential(*_args, **_kwargs):
        assert events[0] == "authenticated" and fault is None
        events.append("credential")
        return "private credential"
    class Stub:
        async def AppFetchLogs(self, request, **kwargs):
            requests.append((request, kwargs))
            return _phase_response([_serving_line() if serving else _phase_line()])
    def client(*_args):
        assert events == ["authenticated", "credential", "credential"]
        events.append("client")
        return SimpleNamespace(stub=Stub())
    modal = ModuleType("modal")
    modal.__version__ = "1.5.4"
    modal.Client = SimpleNamespace(from_credentials=client)
    config = ModuleType("modal.config")
    config.config = SimpleNamespace(get=credential)
    utils = ModuleType("modal._utils.async_utils")
    utils.synchronizer = SimpleNamespace(create_blocking=lambda function:
        lambda *args, **kwargs: asyncio.run(function(*args, **kwargs)))
    for name, module in [("modal", modal), ("modal.config", config), ("modal._utils.async_utils", utils)]:
        monkeypatch.setitem(sys.modules, name, module)
    code = diagnostic.main(["--journal", str(tmp_path / "journal"), "--claim-ref", "a" * 64,
        "--call-id", _CALL, "--modal-profile", "profile",
        "--inspect-serving-metrics" if serving else "--inspect-phase-trace"])
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == ("synaptic-modal-serving-diagnostic/v1" if serving
                                        else "synaptic-modal-packaged-phase-diagnostic/v1")
    assert output["authority"] == "DIAGNOSTIC_ONLY" and "private" not in json.dumps(output)
    if fault:
        assert code == 1 and output["result"] == "JOURNAL_INVALID"
        assert events == ["authenticated"] and requests == []
    else:
        assert code == 0 and len(requests) == 1
        assert output["result"]["completeness"] == "INCONCLUSIVE"


def test_phase_invalid_journal_prevents_provider_import(monkeypatch, tmp_path, capsys):
    original = __import__("builtins").__import__
    def guarded(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal.") or name == "modal_proto":
            pytest.fail("provider imported before retained authentication")
        return original(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded)
    assert diagnostic.main(["--journal", str(tmp_path / "missing"), "--claim-ref", "a" * 64,
        "--call-id", _CALL, "--modal-profile", "profile", "--inspect-phase-trace"]) == 1
    assert json.loads(capsys.readouterr().out)["result"] == "JOURNAL_INVALID"


@pytest.mark.parametrize("ordinal", [1, 32])
def test_phase_request_ordinal_valid_boundaries(ordinal):
    result, _ = _inspect_trace(_phase_response([
        _phase_line(phase="CHAT_REQUEST", request_ordinal=ordinal)]))
    assert result["records"][0]["request_ordinal"] == ordinal


@pytest.mark.parametrize("extra", [0, 1])
def test_phase_record_byte_boundary(extra):
    line = _phase_line()
    line = line[:-1] + " " * (512 - len(line[:-1].encode()) + extra) + "\n"
    result, _ = _inspect_trace(_phase_response([line]))
    assert result["reason"] == ("MALFORMED" if extra else "SNAPSHOT_ONLY")


@pytest.mark.parametrize("extra", [0, 1])
def test_phase_message_byte_boundary_includes_protobuf_metadata(extra):
    response = _phase_response([_phase_line()])
    item = response.batches[0].items[0]
    # The container-name field contributes bytes despite not being projected.
    item.container_name = "private"
    while item.ByteSize() < diagnostic._PHASE_MESSAGE_BYTES + extra:
        item.container_name += "x"
    assert item.ByteSize() == diagnostic._PHASE_MESSAGE_BYTES + extra
    result, _ = _inspect_trace(response)
    assert result["reason"] == ("RECEIVED_BOUND_EXCEEDED" if extra else "SNAPSHOT_ONLY")
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("extra", [0, 1])
def test_phase_response_byte_boundary_includes_unprojected_metadata(extra):
    response = _phase_response([_phase_line()])
    response.batches[0].task_id = "private" + "x" * 261000
    # Adjust the top-level batch field rather than the per-message data budget.
    response.batches[0].task_id += "x" * (
        diagnostic._PHASE_RESPONSE_BYTES + extra - response.ByteSize())
    assert response.ByteSize() == diagnostic._PHASE_RESPONSE_BYTES + extra
    result, _ = _inspect_trace(response)
    assert result["reason"] == ("RECEIVED_BOUND_EXCEEDED" if extra else "SNAPSHOT_ONLY")
    assert "private" not in json.dumps(result)


def test_phase_batch_count_is_bounded_even_without_entries():
    proto = pytest.importorskip("modal_proto.api_pb2")
    response = proto.AppFetchLogsResponse(batches=[proto.TaskLogsBatch()] * 257)
    result, _ = _inspect_trace(response)
    assert result["reason"] == "RECEIVED_BOUND_EXCEEDED"


def test_phase_rpc_has_one_bounded_wait(monkeypatch):
    calls = []
    original = asyncio.wait_for
    async def waiting(awaitable, *, timeout):
        calls.append(timeout)
        return await original(awaitable, timeout=timeout)
    monkeypatch.setattr(diagnostic.asyncio, "wait_for", waiting)
    result, requests = _inspect_trace(_phase_response([_phase_line()]))
    assert calls == [16] and len(requests) == 1
    assert result["reason"] == "SNAPSHOT_ONLY"


def test_phase_pinned_rpc_is_unary_and_synchronizer_bridges_without_helpers():
    import inspect
    grpc = pytest.importorskip("modal_proto.api_grpc")
    proto = pytest.importorskip("modal_proto.api_pb2")
    source = inspect.getsource(grpc.ModalClientStub.__init__)
    method = source.split("self.AppFetchLogs = ", 1)[1].split("self.AppGetByDeploymentName", 1)[0]
    assert "grpclib.client.UnaryUnaryMethod(" in method
    assert "'/modal.client.ModalClient/AppFetchLogs'" in method
    assert "modal_proto.api_pb2.AppFetchLogsRequest" in method
    assert "modal_proto.api_pb2.AppFetchLogsResponse" in method
    synchronizer = pytest.importorskip("modal._utils.async_utils").synchronizer
    calls = []
    class Stub:
        async def AppFetchLogs(self, request, *, retry, timeout):
            calls.append((request, retry, timeout))
            return _phase_response([_phase_line()])
    result = synchronizer.create_blocking(diagnostic.inspect_phase_trace)(
        SimpleNamespace(stub=Stub()), _case()[0], _CALL, proto)
    assert len(calls) == 1 and result["result"] == "TRACE_READ"


def test_phase_parser_matches_real_runtime_emitter_contract():
    from tuner.execution.providers.modal.packaged_worker import PACKAGED_PHASES, _PackagedPhaseTrace
    assert diagnostic._PHASES == PACKAGED_PHASES
    lines = []
    trace = _PackagedPhaseTrace(clock=lambda: 1.0, sink=lines.append)
    trace.emit("TRAINER_EXECUTE", "START")
    trace.emit("CHAT_REQUEST", "RETURN", 32)
    result, _ = _inspect_trace(_phase_response([line + "\n" for line in lines]))
    assert result["result"] == "TRACE_READ" and len(result["records"]) == 2


def test_phase_reassembles_real_emitter_print_write_chunks():
    import contextlib
    from tuner.execution.providers.modal.packaged_worker import _PackagedPhaseTrace
    chunks = []
    sink = SimpleNamespace(write=lambda value: chunks.append(value), flush=lambda: None)
    trace = _PackagedPhaseTrace(clock=lambda: 1.0)
    with contextlib.redirect_stdout(sink):
        trace.emit("TRAINER_EXECUTE", "START")
        trace.emit("CHAT_REQUEST", "RETURN", 1)
    assert len(chunks) == 4 and chunks[1] == chunks[3] == "\n"
    result, requests = _inspect_trace(_phase_response(chunks))
    assert len(requests) == 1 and result["result"] == "TRACE_READ"
    assert result["records"] == [
        dict(phase="TRAINER_EXECUTE", edge="START", elapsed_ms=0, request_ordinal=None),
        dict(phase="CHAT_REQUEST", edge="RETURN", elapsed_ms=0, request_ordinal=1)]


@pytest.mark.parametrize("cut", range(1, len(diagnostic._PHASE_PREFIX) + 2))
def test_phase_prefix_and_json_fragments_reassemble(cut):
    line = _phase_line()
    result, _ = _inspect_trace(_phase_response([line[:cut], line[cut:]]))
    assert result["result"] == "TRACE_READ" and len(result["records"]) == 1


@pytest.mark.parametrize("cut", range(1, len(diagnostic._PHASE_PREFIX) + 1))
def test_phase_truncated_matching_prefix_is_not_proven_absence(cut):
    result, _ = _inspect_trace(_phase_response([diagnostic._PHASE_PREFIX[:cut]]))
    assert result["reason"] == "MALFORMED"
    assert result["validation_stage"] == "LINE_FRAMING" and result["records"] == []


@pytest.mark.parametrize("field", ["fd", "task", "container", "batch_input", "item_input"])
def test_phase_interleaved_streams_reassemble_independently(field):
    proto = pytest.importorskip("modal_proto.api_pb2")
    first = _phase_line(phase="TRAINER_EXECUTE")
    second = _phase_line(phase="VLLM_PREPARE")
    response = _phase_response([first[:-1], second[:-1], "\n", "\n"])
    batch = response.batches[0]
    if field in {"task", "batch_input"}:
        # Batch-level identity requires four independent protobuf batches.
        response = proto.AppFetchLogsResponse(batches=[
            proto.TaskLogsBatch(function_id=batch.function_id, items=[item])
            for item in batch.items])
        for index, source in enumerate(response.batches):
            setattr(source, "task_id" if field == "task" else "input_id",
                    "private-a" if index in {0, 2} else "private-b")
    else:
        for index, item in enumerate(batch.items):
            second_stream = index in {1, 3}
            if field == "fd":
                item.file_descriptor = proto.FILE_DESCRIPTOR_STDERR if second_stream else proto.FILE_DESCRIPTOR_STDOUT
            else:
                setattr(item, "container_id" if field == "container" else "input_id",
                        "private-b" if second_stream else "private-a")
    result, requests = _inspect_trace(response)
    assert len(requests) == 1 and result["result"] == "TRACE_READ"
    assert [item["phase"] for item in result["records"]] == ["TRAINER_EXECUTE", "VLLM_PREPARE"]
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("field", ["fd", "task", "container", "batch_input", "item_input"])
def test_phase_mismatched_stream_fragment_cannot_complete_record(field):
    proto = pytest.importorskip("modal_proto.api_pb2")
    line = _phase_line()
    response = _phase_response([line[:-1], "\n"])
    if field in {"task", "batch_input"}:
        batch = response.batches[0]
        response = proto.AppFetchLogsResponse(batches=[
            proto.TaskLogsBatch(function_id=batch.function_id, items=[item]) for item in batch.items])
        setattr(response.batches[1], "task_id" if field == "task" else "input_id", "private")
    else:
        item = response.batches[0].items[1]
        if field == "fd": item.file_descriptor = proto.FILE_DESCRIPTOR_STDERR
        else: setattr(item, "container_id" if field == "container" else "input_id", "private")
    result, _ = _inspect_trace(response)
    assert result["reason"] == "MALFORMED" and result["validation_stage"] == "LINE_FRAMING"
    assert result["records"] == [] and "private" not in json.dumps(result)


@pytest.mark.parametrize("identity", ["function", "call", "fd_unknown", "fd_unspecified", "fd_info"])
@pytest.mark.parametrize("fragment", ["SYN", diagnostic._PHASE_PREFIX, _phase_line()])
def test_phase_wrong_identity_matching_fragments_are_never_discarded(identity, fragment):
    proto = pytest.importorskip("modal_proto.api_pb2")
    response = _phase_response([fragment])
    if identity == "function": response.batches[0].function_id = "fu-private"
    elif identity == "call": response.batches[0].items[0].function_call_id = "fc-private"
    else:
        response.batches[0].items[0].file_descriptor = {
            "fd_unknown": 99, "fd_unspecified": proto.FILE_DESCRIPTOR_UNSPECIFIED,
            "fd_info": proto.FILE_DESCRIPTOR_INFO}[identity]
    result, _ = _inspect_trace(response)
    assert result["reason"] == "MALFORMED" and result["validation_stage"] == "RESPONSE_IDENTITY"
    assert result["records"] == [] and "private" not in json.dumps(result)


def test_phase_oversized_fragmented_record_is_rejected_without_echo():
    line = _phase_line().rstrip("\n")
    result, _ = _inspect_trace(_phase_response([line, " " * (513 - len(line)), "\n"]))
    assert result["reason"] == "MALFORMED" and result["validation_stage"] == "PHASE_RECORD"
    assert result["records"] == []


def test_phase_fragmented_hostile_json_is_rejected_without_echo():
    line = _phase_line(phase="private prompt")
    result, _ = _inspect_trace(_phase_response([line[:20], line[20:100], line[100:]]))
    assert result["reason"] == "MALFORMED" and result["validation_stage"] == "PHASE_RECORD"
    assert result["records"] == [] and "private" not in json.dumps(result)


def test_phase_unknown_oversized_line_is_discarded_until_its_delimiter():
    result, _ = _inspect_trace(_phase_response([
        "private" * 400, _phase_line(), _phase_line()]))
    assert result["result"] == "TRACE_READ" and len(result["records"]) == 1
    assert "private" not in json.dumps(result)


def test_phase_malformed_response_shape_stage_is_closed():
    result, _ = _inspect_trace(SimpleNamespace(batches=[]))
    assert result["validation_stage"] == "RESPONSE_SHAPE"


def test_phase_stream_uses_effective_batch_identity_for_missing_item_fields():
    line = _phase_line()
    response = _phase_response([line[:-1], "\n"])
    batch = response.batches[0]
    batch.task_id, batch.input_id = "private-container", "private-input"
    batch.items[0].container_id, batch.items[0].input_id = batch.task_id, batch.input_id
    # The second chunk uses the batch-level fallback rather than item fields.
    result, _ = _inspect_trace(response)
    assert result["result"] == "TRACE_READ" and len(result["records"]) == 1
    assert "private" not in json.dumps(result)


def test_phase_omitted_call_id_continuation_remains_fail_closed():
    proto = pytest.importorskip("modal_proto.api_pb2")
    line = _phase_line()
    response = _phase_response([line[:-1]])
    # The real protobuf permits an omitted/default-empty call ID, without
    # presence tracking. The exact request filter does not supply its identity.
    assert not proto.TaskLogs.DESCRIPTOR.fields_by_name["function_call_id"].has_presence
    response.batches[0].items.append(proto.TaskLogs(
        data="\n", file_descriptor=proto.FILE_DESCRIPTOR_STDOUT))
    assert response.batches[0].items[1].function_call_id == ""
    result, requests = _inspect_trace(response)
    assert len(requests) == 1 and result["result"] == "TRACE_INCONCLUSIVE"
    assert result["reason"] == "MALFORMED" and result["validation_stage"] == "LINE_FRAMING"
    assert result["records"] == []


def _serving_line(**changes):
    record = dict(schema_version="synaptic-modal-serving-diagnostic/v1", kind="METRICS",
                  elapsed_ms=0, running_requests=1, waiting_requests=2,
                  generation_tokens=3, kv_cache_usage=0.5, cleanup_resolved=None)
    record.update(changes)
    return "SYNAPTIC_SERVING " + json.dumps(record, separators=(",", ":")) + "\n"


def _cleanup_line(resolved):
    return _serving_line(kind="CLEANUP", running_requests=None, waiting_requests=None,
                         generation_tokens=None, kv_cache_usage=None, cleanup_resolved=resolved)


def _inspect_serving(response):
    proto = pytest.importorskip("modal_proto.api_pb2")
    requests = []
    class Stub:
        async def AppFetchLogs(self, request, *, retry, timeout):
            requests.append((request, retry, timeout))
            if isinstance(response, Exception): raise response
            return response
    result = asyncio.run(diagnostic.inspect_serving_metrics(
        SimpleNamespace(stub=Stub()), _case()[0], _CALL, proto))
    return result, requests


@pytest.mark.parametrize("resolved", [False, True])
def test_serving_one_exact_protobuf_request_and_numeric_projection(resolved):
    proto = pytest.importorskip("modal_proto.api_pb2")
    result, requests = _inspect_serving(_phase_response([
        _phase_line(), "private label model prompt response\n", _serving_line(), _cleanup_line(resolved)]))
    assert len(requests) == 1
    request, retry, timeout = requests[0]
    assert type(request) is proto.AppFetchLogsRequest
    assert request == proto.AppFetchLogsRequest(app_id="ap-owned", function_id="fu-worker",
                                               function_call_id=_CALL, limit=256)
    assert (retry, timeout) == (None, 15)
    assert result["result"] == "SERVING_READ" and result["completeness"] == "INCONCLUSIVE"
    assert result["reason"] == "SNAPSHOT_ONLY" and result["wire_byte_cap"] is False
    assert result["byte_bounds"] == "AFTER_RECEPTION"
    assert result["records"] == [
        dict(kind="METRICS", elapsed_ms=0, running_requests=1, waiting_requests=2,
             generation_tokens=3, kv_cache_usage=0.5, cleanup_resolved=None),
        dict(kind="CLEANUP", elapsed_ms=0, running_requests=None, waiting_requests=None,
             generation_tokens=None, kv_cache_usage=None, cleanup_resolved=resolved)]
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("field", ["running_requests", "waiting_requests", "generation_tokens"])
@pytest.mark.parametrize("value", [0, 2**53 - 1])
def test_serving_counter_valid_bounds(field, value):
    result, _ = _inspect_serving(_phase_response([_serving_line(**{field: value})]))
    assert result["records"][0][field] == value


@pytest.mark.parametrize("field", ["running_requests", "waiting_requests", "generation_tokens"])
@pytest.mark.parametrize("value", [-1, 2**53, True, False, 1.0, None, "private", float("nan"), float("inf")])
def test_serving_counter_invalid_types_and_bounds(field, value):
    result, _ = _inspect_serving(_phase_response([_serving_line(**{field: value})]))
    assert result["reason"] == "MALFORMED" and result["validation_stage"] == "SERVING_RECORD"
    assert result["records"] == [] and "private" not in json.dumps(result)


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, 0.125])
def test_serving_kv_fraction_valid(value):
    result, _ = _inspect_serving(_phase_response([_serving_line(kv_cache_usage=value)]))
    assert result["records"][0]["kv_cache_usage"] == value


@pytest.mark.parametrize("value", [-0.01, 1.01, True, False, None, "private",
                                   float("nan"), float("inf"), float("-inf"), 10**309])
def test_serving_kv_fraction_invalid_and_nonfinite(value):
    result, _ = _inspect_serving(_phase_response([_serving_line(kv_cache_usage=value)]))
    assert result["reason"] == "MALFORMED" and result["records"] == []
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("changes", [
    {"schema_version": "private"}, {"kind": "private"}, {"kind": []},
    {"elapsed_ms": True}, {"elapsed_ms": -1}, {"elapsed_ms": 86400001},
    {"elapsed_ms": 1.0}, {"elapsed_ms": None}, {"cleanup_resolved": True},
    {"private": "model label"}, {"kind": "CLEANUP", "cleanup_resolved": True},
])
def test_serving_exact_schema_and_kind_invariants(changes):
    result, _ = _inspect_serving(_phase_response([_serving_line(**changes)]))
    assert result["reason"] == "MALFORMED" and result["records"] == []
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("value", [None, 0, 1, "private", [], {}])
def test_serving_cleanup_resolution_requires_exact_bool(value):
    result, _ = _inspect_serving(_phase_response([_cleanup_line(value)]))
    assert result["reason"] == "MALFORMED" and result["records"] == []
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("field", ["running_requests", "waiting_requests", "generation_tokens", "kv_cache_usage"])
def test_serving_cleanup_metrics_must_all_be_null(field):
    line = _cleanup_line(False).replace('"' + field + '":null', '"' + field + '":0')
    result, _ = _inspect_serving(_phase_response([line]))
    assert result["reason"] == "MALFORMED" and result["records"] == []


@pytest.mark.parametrize("line", [
    _serving_line().replace('"kind":"METRICS"', '"kind":"METRICS","kind":"CLEANUP"'),
    _serving_line().replace('"generation_tokens":3,', ''),
    _serving_line().replace('"kv_cache_usage":0.5', '"kv_cache_usage":1e999'),
    _serving_line().replace('"elapsed_ms":0', '"elapsed_ms":\n0'),
    'SYNAPTIC_SERVING []\n', 'SYNAPTIC_SERVING private\n',
])
def test_serving_hostile_json_is_closed_without_echo(line):
    result, _ = _inspect_serving(_phase_response([line]))
    assert result["reason"] == "MALFORMED" and result["records"] == []
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("cut", [1, 8, len("SYNAPTIC_SERVING "), 60, -1])
def test_serving_actual_protobuf_chunk_reassembly(cut):
    line = _serving_line()
    result, requests = _inspect_serving(_phase_response([line[:cut], line[cut:]]))
    assert len(requests) == 1 and result["result"] == "SERVING_READ"
    assert len(result["records"]) == 1


@pytest.mark.parametrize("fault", ["call", "container", "input", "fd"])
def test_serving_interleaved_wrong_stream_cannot_complete_record(fault):
    proto = pytest.importorskip("modal_proto.api_pb2")
    line = _serving_line()
    response = _phase_response([line[:-1], "\n"])
    item = response.batches[0].items[1]
    if fault == "call": item.function_call_id = ""
    elif fault == "container": item.container_id = "private"
    elif fault == "input": item.input_id = "private"
    else: item.file_descriptor = proto.FILE_DESCRIPTOR_STDERR
    result, _ = _inspect_serving(response)
    assert result["reason"] == "MALFORMED" and result["validation_stage"] == "LINE_FRAMING"
    assert result["records"] == [] and "private" not in json.dumps(result)


@pytest.mark.parametrize("fragment", ["SYN", "SYNAPTIC_SERVING ", _serving_line()])
def test_serving_wrong_call_matching_fragment_rejects(fragment):
    result, _ = _inspect_serving(_phase_response([fragment], call="fc-private"))
    assert result["validation_stage"] == "RESPONSE_IDENTITY" and result["records"] == []
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("lines,reason", [
    ([], "MISSING"), ([_phase_line()], "MISSING"), (["SYNAPTIC_SERV"], "MALFORMED"),
    ([_serving_line()] * 65, "CAPPED"),
    (["ignored\n"] * 255 + [_serving_line()], "CAPPED"),
    (["private" * 1024], "RECEIVED_BOUND_EXCEEDED"),
    (["x" * 2048] * 256, "RECEIVED_BOUND_EXCEEDED"),
    (["ignored\n"] * 257, "RECEIVED_BOUND_EXCEEDED"),
])
def test_serving_missing_truncated_capped_and_received_bounds(lines, reason):
    result, requests = _inspect_serving(_phase_response(lines))
    assert len(requests) == 1 and result["result"] == "SERVING_INCONCLUSIVE"
    assert result["reason"] == reason and result["completeness"] == "INCONCLUSIVE"
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("extra", [0, 1])
def test_serving_record_byte_bound_includes_prefix(extra):
    line = _serving_line().rstrip("\n")
    line += " " * (512 - len(line.encode()) + extra)
    result, _ = _inspect_serving(_phase_response([line, "\n"]))
    assert result["reason"] == ("MALFORMED" if extra else "SNAPSHOT_ONLY")


def test_phase_mode_ignores_serving_records_and_preserves_projection():
    result, _ = _inspect_trace(_phase_response([_serving_line(), _cleanup_line(False), _phase_line()]))
    baseline, _ = _inspect_trace(_phase_response([_phase_line()]))
    assert result == baseline


@pytest.mark.parametrize("mode", ["--inspect-markers", "--inspect-evaluation-metadata",
    "--probe-final-model-first-chunk", "--include-provider-status",
    "--inspect-training-completion-metadata", "--inspect-phase-trace"])
def test_serving_cli_exclusive_modes_reject_before_authentication(monkeypatch, tmp_path, capsys, mode):
    def reject(*_args): pytest.fail("invalid serving mode reached authentication")
    monkeypatch.setattr(diagnostic, "read_retained_probe", reject)
    args = ["--journal", str(tmp_path / "journal"), "--claim-ref", "a" * 64,
            "--modal-profile", "profile", "--inspect-serving-metrics", mode]
    if mode != "--inspect-markers": args.extend(["--call-id", _CALL])
    assert diagnostic.main(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert output == {"schema_version": "synaptic-modal-serving-diagnostic/v1",
                      "authority": "DIAGNOSTIC_ONLY", "result": "INPUT_INVALID"}


def test_serving_invalid_journal_prevents_provider_import(monkeypatch, tmp_path, capsys):
    """Serving mode admits retained authentication before provider imports."""
    original = __import__("builtins").__import__
    def guarded(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal.") or name == "modal_proto":
            pytest.fail("provider imported before serving admission")
        return original(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded)
    assert diagnostic.main(["--journal", str(tmp_path / "missing"), "--claim-ref", "a" * 64,
        "--call-id", _CALL, "--modal-profile", "profile", "--inspect-serving-metrics"]) == 1
    assert json.loads(capsys.readouterr().out)["result"] == "JOURNAL_INVALID"


def test_serving_real_emitter_chunks_and_schema_parity():
    import contextlib
    from tuner.execution.providers.modal.packaged_worker import _PackagedPhaseTrace
    chunks = []
    sink = SimpleNamespace(write=lambda value: chunks.append(value), flush=lambda: None)
    trace = _PackagedPhaseTrace(clock=lambda: 1.0)
    with contextlib.redirect_stdout(sink):
        trace.emit_serving("METRICS", dict(running_requests=1, waiting_requests=0,
            generation_tokens=2**53 - 1, kv_cache_usage=0.25))
        trace.emit_serving("CLEANUP", dict(cleanup_resolved=False))
    assert len(chunks) == 4 and chunks[1] == chunks[3] == "\n"
    result, requests = _inspect_serving(_phase_response(chunks))
    assert len(requests) == 1 and result["result"] == "SERVING_READ"
    assert [record["kind"] for record in result["records"]] == ["METRICS", "CLEANUP"]
    assert result["records"][0]["generation_tokens"] == 2**53 - 1
    assert result["records"][1]["cleanup_resolved"] is False


@pytest.mark.parametrize("include_cleanup", [False, True])
def test_serving_real_emitter_exact_metric_limit_is_readable(include_cleanup):
    from tuner.execution.providers.modal.packaged_worker import _PackagedPhaseTrace
    lines = []
    trace = _PackagedPhaseTrace(clock=lambda: 1.0, sink=lines.append)
    for _ in range(65):
        trace.emit_serving("METRICS", dict(running_requests=0, waiting_requests=0,
            generation_tokens=0, kv_cache_usage=0))
    if include_cleanup:
        trace.emit_serving("CLEANUP", dict(cleanup_resolved=True))
    assert len(lines) == 64 + int(include_cleanup)
    # Real print framing: record and newline are separate protobuf log items.
    result, requests = _inspect_serving(_phase_response([chunk for line in lines for chunk in (line, "\n")]))
    assert len(requests) == 1 and result["result"] == "SERVING_READ"
    assert len(result["records"]) == 64 + int(include_cleanup)
    assert result["reason"] == "SNAPSHOT_ONLY" and "validation_stage" not in result
    assert result["completeness"] == "INCONCLUSIVE"
    assert result["wire_byte_cap"] is False and result["byte_bounds"] == "AFTER_RECEPTION"
    assert all(record["kind"] == "METRICS" for record in result["records"][:64])
    if include_cleanup:
        assert result["records"][-1]["kind"] == "CLEANUP"
        assert result["records"][-1]["cleanup_resolved"] is True


def test_serving_effective_id_fallback_and_interleaving_preserved():
    proto = pytest.importorskip("modal_proto.api_pb2")
    first, second = _serving_line(), _cleanup_line(False)
    response = _phase_response([first[:-1], second[:-1], "\n", "\n"])
    batch = response.batches[0]
    batch.task_id, batch.input_id = "private-container", "private-input"
    for index, item in enumerate(batch.items):
        if index in {1, 3}: item.file_descriptor = proto.FILE_DESCRIPTOR_STDERR
    batch.items[0].container_id, batch.items[0].input_id = batch.task_id, batch.input_id
    result, _ = _inspect_serving(response)
    assert result["result"] == "SERVING_READ"
    assert [record["kind"] for record in result["records"]] == ["METRICS", "CLEANUP"]
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("response", [RuntimeError("private provider"), SimpleNamespace(batches=[])])
def test_serving_provider_failure_and_shape_are_closed(response):
    result, requests = _inspect_serving(response)
    assert len(requests) == 1 and result["records"] == []
    assert result["result"] == "SERVING_INCONCLUSIVE"
    assert result["reason"] in {"UNAVAILABLE", "MALFORMED"}
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("extra", [0, 1])
def test_serving_response_byte_boundary_metadata_is_not_projected(extra):
    response = _phase_response([_serving_line()])
    response.batches[0].task_id = "private" + "x" * 261000
    response.batches[0].task_id += "x" * (
        diagnostic._PHASE_RESPONSE_BYTES + extra - response.ByteSize())
    assert response.ByteSize() == diagnostic._PHASE_RESPONSE_BYTES + extra
    result, _ = _inspect_serving(response)
    assert result["reason"] == ("RECEIVED_BOUND_EXCEEDED" if extra else "SNAPSHOT_ONLY")
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("extra", [0, 1])
def test_serving_message_byte_boundary_includes_unprojected_metadata(extra):
    response = _phase_response([_serving_line()])
    item = response.batches[0].items[0]
    item.container_name = "private"
    while item.ByteSize() < diagnostic._PHASE_MESSAGE_BYTES + extra:
        item.container_name += "x"
    assert item.ByteSize() == diagnostic._PHASE_MESSAGE_BYTES + extra
    result, _ = _inspect_serving(response)
    assert result["reason"] == ("RECEIVED_BOUND_EXCEEDED" if extra else "SNAPSHOT_ONLY")
    assert "private" not in json.dumps(result)
