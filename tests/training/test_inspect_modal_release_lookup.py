"""Provider-free checks for the private one-lookup diagnostic."""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inspect_modal_release_lookup.py"
_SPEC = importlib.util.spec_from_file_location("inspect_modal_release_lookup", _SCRIPT)
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)

_DIGEST = "a" * 64
_CLAIM_REF = "deploy-" + _DIGEST


@pytest.mark.skipif(diagnostic.os.name != "posix", reason="private POSIX journal")
def test_exact_claim_read_is_read_only(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    database = private / "modal-host.sqlite3"
    claim = canonical_bytes({
        "schema_version": "synaptic-modal-host-deploy-claim/v1",
        "release_digest": _DIGEST,
        "capture_digest": "b" * 64,
        "quote_digest": "c" * 64,
        "deployment_name": "synaptic-training-abc",
    })
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE attempts(namespace_ref TEXT,attempt_ref TEXT,digest TEXT,evidence BLOB)"
        )
        connection.execute(
            "INSERT INTO attempts VALUES(?,?,?,?)",
            ("standalone-training", _CLAIM_REF, hashlib.sha256(claim).hexdigest(), claim),
        )
    database.chmod(0o600)
    initial = set(private.iterdir())
    assert diagnostic.read_deploy_claim(database, _CLAIM_REF) == "synaptic-training-abc"
    assert set(private.iterdir()) == initial
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_deploy_claim(database, "deploy-" + "d" * 64)
    assert set(private.iterdir()) == initial


@pytest.mark.skipif(diagnostic.os.name != "posix", reason="private POSIX journal")
def test_oversized_claim_fails_before_blob_selection(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    database = private / "modal-host.sqlite3"
    raw = b"x" * (diagnostic._MAX_CLAIM + 1)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE attempts(namespace_ref TEXT,attempt_ref TEXT,digest TEXT,evidence BLOB)"
        )
        connection.execute(
            "INSERT INTO attempts VALUES(?,?,?,?)",
            ("standalone-training", _CLAIM_REF, hashlib.sha256(raw).hexdigest(), raw),
        )
    database.chmod(0o600)
    with pytest.raises(diagnostic.DiagnosticUnavailable):
        diagnostic.read_deploy_claim(database, _CLAIM_REF)
    assert {item.name for item in private.iterdir()} == {database.name}


class _Proto:
    APP_STATE_DEPLOYED = 2
    APP_STATE_STOPPED = 3

    @staticmethod
    def AppGetByDeploymentNameRequest(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def AppLifecycle():
        return SimpleNamespace(app_state=0, version=0)


class _Stub:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error
        self.calls = []

    async def AppGetByDeploymentName(self, request):
        self.calls.append((request.name, request.environment_name))
        if self.error:
            raise self.error
        return self.response

    def __getattr__(self, name):
        raise AssertionError("unexpected provider method: " + name)


def _response(environment="", current="", previous="", state=0, version=0):
    return SimpleNamespace(
        environment_name=environment,
        app_id=current,
        previous_app_id=previous,
        lifecycle=SimpleNamespace(app_state=state, version=version),
    )


@pytest.mark.parametrize("echo,category", [("", "EMPTY"), ("main", "MATCH")])
def test_absent_shape_with_empty_or_matching_environment(echo, category):
    stub = _Stub(_response(environment=echo))
    result = asyncio.run(diagnostic.inspect_lookup(
        SimpleNamespace(stub=stub), "exact-name", "main", _Proto,
    ))
    assert result == {"result": "ABSENT", "environment_echo": category}
    assert stub.calls == [("exact-name", "main")]


def test_transport_and_invalid_response_never_mutate():
    failed = _Stub(error=RuntimeError("private provider text"))
    assert asyncio.run(diagnostic.inspect_lookup(
        SimpleNamespace(stub=failed), "exact-name", "main", _Proto,
    )) == {"result": "TRANSPORT_ERROR"}
    invalid = _Stub(_response(environment="other"))
    assert asyncio.run(diagnostic.inspect_lookup(
        SimpleNamespace(stub=invalid), "exact-name", "main", _Proto,
    )) == {"result": "ENVIRONMENT_MISMATCH"}
    assert failed.calls == invalid.calls == [("exact-name", "main")]


def test_deployed_and_stopped_shape_only_exposes_booleans():
    for response, category in (
        (_response("main", "ap-sensitive", "", 2, 1), "DEPLOYED"),
        (_response("main", "", "ap-sensitive", 3, 2), "STOPPED"),
    ):
        result = asyncio.run(diagnostic.inspect_lookup(
            SimpleNamespace(stub=_Stub(response)), "exact-name", "main", _Proto,
        ))
        assert result["result"] == category
        assert "ap-sensitive" not in str(result)


def test_present_app_with_empty_environment_is_not_success():
    result = asyncio.run(diagnostic.inspect_lookup(
        SimpleNamespace(stub=_Stub(_response(
            "", "ap-sensitive", "", _Proto.APP_STATE_DEPLOYED, 1,
        ))), "exact-name", "main", _Proto,
    ))
    assert result == {"result": "ENVIRONMENT_UNCONFIRMED"}


def test_main_invalid_claim_never_imports_modal(tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__
    seen = []

    def guarded_import(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            seen.append(name)
            raise AssertionError("Modal must not load before claim admission")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    code = diagnostic.main([
        "--journal", str(tmp_path / "missing.sqlite3"),
        "--claim-ref", _CLAIM_REF,
        "--environment", "main", "--modal-profile", "synaptic-labs",
    ])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["result"] == "JOURNAL_INVALID"
    assert seen == []


def test_main_missing_named_credentials_never_creates_client(monkeypatch, capsys):
    monkeypatch.setattr(diagnostic, "read_deploy_claim", lambda *_: "exact-name")
    modal = ModuleType("modal")
    modal.__version__ = "1.5.4"

    class _Client:
        @staticmethod
        def from_credentials(*_args):
            raise AssertionError("client must not be created")

    modal.Client = _Client
    config_module = ModuleType("modal.config")
    config_module.config = SimpleNamespace(get=lambda *_args, **_kwargs: "")
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setitem(sys.modules, "modal.config", config_module)
    code = diagnostic.main([
        "--journal", "/unused/private.sqlite3", "--claim-ref", _CLAIM_REF,
        "--environment", "main", "--modal-profile", "synaptic-labs",
    ])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["result"] == "CREDENTIAL_UNAVAILABLE"


@pytest.mark.parametrize("outcome,expected,exit_code", [
    (None, "READER_ABSENT", 0),
    (object(), "READER_PRESENT", 0),
    (RuntimeError("secret provider payload"), "READER_UNAVAILABLE", 1),
])
def test_production_reader_mode_is_closed_and_single_read(
    monkeypatch, capsys, outcome, expected, exit_code,
):
    monkeypatch.setattr(diagnostic, "read_deploy_claim", lambda *_: "exact-name")
    modal = ModuleType("modal")
    modal.__version__ = "1.5.4"
    client = object()
    modal.Client = SimpleNamespace(from_credentials=lambda *_: client)
    config_module = ModuleType("modal.config")
    config_module.config = SimpleNamespace(get=lambda *_args, **_kwargs: "credential")
    utils_module = ModuleType("modal._utils")
    async_module = ModuleType("modal._utils.async_utils")
    async_module.synchronizer = SimpleNamespace(create_blocking=lambda fn: fn)
    proto_module = ModuleType("modal_proto")
    proto_module.api_pb2 = object()
    reader_module = ModuleType("tuner.execution.providers.modal.runtime_release_deployment")
    calls = []

    class Reader:
        def __init__(self, *, sdk):
            assert sdk is modal

        def observe(self, *, client, app_name, environment_name):
            calls.append((client, app_name, environment_name))
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    reader_module.ExplicitModal154ReleaseDeploymentReader = Reader
    for key, module in (
        ("modal", modal), ("modal.config", config_module),
        ("modal._utils", utils_module),
        ("modal._utils.async_utils", async_module),
        ("modal_proto", proto_module),
        ("tuner.execution.providers.modal.runtime_release_deployment", reader_module),
    ):
        monkeypatch.setitem(sys.modules, key, module)
    code = diagnostic.main([
        "--journal", "/unused/private.sqlite3", "--claim-ref", _CLAIM_REF,
        "--environment", "main", "--modal-profile", "synaptic-labs",
        "--production-reader",
    ])
    result = json.loads(capsys.readouterr().out)
    assert code == exit_code
    assert result["result"] == expected
    assert "secret" not in str(result)
    assert calls == [(client, "exact-name", "main")]
