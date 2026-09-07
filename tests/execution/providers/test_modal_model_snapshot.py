from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tuner.execution.providers.modal.model_snapshot import prepare_model_snapshot


REVISION = "a" * 40
MODEL = "fixture/tiny"


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    files = {"config.json": b'{"model_type":"fixture"}', "model.safetensors": b"fixture weights"}
    siblings = []
    for name, content in files.items():
        lfs = name.endswith("safetensors")
        siblings.append(SimpleNamespace(
            rfilename=name, size=len(content),
            blob_id=hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
            lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()) if lfs else None,
        ))
    info = SimpleNamespace(sha=REVISION, siblings=siblings)
    calls = []

    class API:
        def __init__(self, **kwargs):
            calls.append(("api", kwargs))

        def model_info(self, repo_id, **kwargs):
            calls.append(("info", {"repo_id": repo_id, **kwargs}))
            return info

    def download(**kwargs):
        calls.append(("download", kwargs))
        for name in kwargs["allow_patterns"]:
            destination = kwargs["local_dir"] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(files[name])
        return str(kwargs["local_dir"])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API, snapshot_download=download))
    roots = {name: tmp_path / name for name in ("persistent_root", "destination_root", "scratch_root")}
    for path in roots.values():
        path.mkdir()
    return SimpleNamespace(files=files, info=info, calls=calls, roots=roots)


def prepare(fixture, **overrides):
    return prepare_model_snapshot(model_ref=MODEL, revision=REVISION, token="fixture-credential", **(fixture.roots | overrides))


def test_cache_miss_downloads_privately_then_reuses_verified_files(fixture, tmp_path):
    result = prepare(fixture)
    assert {p.name: p.read_bytes() for p in result.iterdir()} == fixture.files
    download = [args for name, args in fixture.calls if name == "download"]
    assert len(download) == 1
    assert download[0]["revision"] == REVISION
    assert download[0]["endpoint"] == "https://huggingface.co"
    assert download[0]["token"] == "fixture-credential"
    assert download[0]["local_dir"].is_relative_to(fixture.roots["scratch_root"])
    assert download[0]["cache_dir"].is_relative_to(fixture.roots["scratch_root"])
    second = tmp_path / "second"
    second.mkdir()
    reused = prepare(fixture, destination_root=second)
    assert {p.name: p.read_bytes() for p in reused.iterdir()} == fixture.files
    assert sum(name == "download" for name, _ in fixture.calls) == 1
    assert list(fixture.roots["scratch_root"].iterdir()) == []


def test_partial_cache_downloads_only_missing_members(fixture):
    cached = fixture.roots["persistent_root"] / "models--fixture--tiny" / "snapshots" / REVISION
    cached.mkdir(parents=True)
    (cached / "config.json").write_bytes(fixture.files["config.json"])
    prepare(fixture)
    assert [args["allow_patterns"] for name, args in fixture.calls if name == "download"] == [["model.safetensors"]]


def test_zero_byte_repository_member_is_verified_and_reused(fixture, tmp_path):
    fixture.files["empty.txt"] = b""
    fixture.info.siblings.append(SimpleNamespace(
        rfilename="empty.txt", size=0, lfs=None,
        blob_id=hashlib.sha1(b"blob 0\0").hexdigest(),
    ))
    assert (prepare(fixture) / "empty.txt").read_bytes() == b""
    second = tmp_path / "second"
    second.mkdir()
    assert (prepare(fixture, destination_root=second) / "empty.txt").read_bytes() == b""
    assert sum(name == "download" for name, _ in fixture.calls) == 1


@pytest.mark.parametrize("mutation", ["revision", "digest", "size", "path", "duplicate", "missing_digest"])
def test_metadata_and_download_must_match_exactly(fixture, mutation):
    if mutation == "revision":
        fixture.info.sha = "b" * 40
    elif mutation == "digest":
        fixture.info.siblings[0].blob_id = "b" * 40
    elif mutation == "size":
        fixture.info.siblings[0].size += 1
    elif mutation == "path":
        fixture.info.siblings[0].rfilename = "../outside"
    elif mutation == "duplicate":
        fixture.info.siblings.append(fixture.info.siblings[0])
    else:
        fixture.info.siblings[1].lfs.sha256 = None
    with pytest.raises(ValueError, match="^model preparation failed$"):
        prepare(fixture)
    assert list(fixture.roots["destination_root"].iterdir()) == []


@pytest.mark.parametrize("mutation", ["symlink", "ancestor", "corruption"])
def test_hostile_persistent_cache_never_reaches_sdk_or_trainer(fixture, tmp_path, mutation):
    cached = fixture.roots["persistent_root"] / "models--fixture--tiny" / "snapshots" / REVISION
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "config.json"
    target.write_bytes(fixture.files["config.json"])
    if mutation == "ancestor":
        (fixture.roots["persistent_root"] / "models--fixture--tiny").symlink_to(outside, target_is_directory=True)
    else:
        cached.mkdir(parents=True)
        if mutation == "symlink":
            (cached / "config.json").symlink_to(target)
        else:
            (cached / "config.json").write_bytes(b"x" * len(fixture.files["config.json"]))
    with pytest.raises(ValueError, match="^model preparation failed$"):
        prepare(fixture)
    assert not any(name == "download" for name, _ in fixture.calls)
    assert target.read_bytes() == fixture.files["config.json"]
    assert list(fixture.roots["destination_root"].iterdir()) == []


def test_sdk_exception_and_output_are_closed(fixture, monkeypatch, capsys):
    def fail(**kwargs):
        print("fixture credential diagnostic")
        print("fixture provider response", file=sys.stderr)
        raise RuntimeError("fixture raw provider failure")
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", fail)
    with pytest.raises(ValueError, match="^model preparation failed$") as failure:
        prepare(fixture)
    assert failure.value.__suppress_context__
    assert capsys.readouterr() == ("", "")


def test_blank_token_disables_implicit_credential_lookup(fixture):
    prepare_model_snapshot(model_ref=MODEL, revision=REVISION, token="  ", **fixture.roots)
    assert all(kwargs["token"] is False for _, kwargs in fixture.calls)


def test_download_symlink_is_rejected(fixture, monkeypatch, tmp_path):
    original = sys.modules["huggingface_hub"].snapshot_download
    def download(**kwargs):
        result = original(**kwargs)
        target = Path(result) / "config.json"
        target.unlink()
        outside = tmp_path / "outside-config"
        outside.write_bytes(fixture.files["config.json"])
        target.symlink_to(outside)
        return result
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", download)
    with pytest.raises(ValueError, match="^model preparation failed$"):
        prepare(fixture)


def test_existing_destination_is_never_overwritten(fixture):
    result = prepare(fixture)
    original = (result / "config.json").read_bytes()
    with pytest.raises(ValueError, match="^model preparation failed$"):
        prepare(fixture)
    assert (result / "config.json").read_bytes() == original


def test_real_runner_prepares_before_credential_free_offline_child(fixture, tmp_path, monkeypatch):
    import json
    from tuner.execution.providers.modal import runtime

    mount = tmp_path / "volume"
    cache = mount / "run-fixture" / "cache"
    cache.mkdir(parents=True)
    scratch = tmp_path / "worker-private"
    path_type = Path
    monkeypatch.setattr(runtime, "Path", lambda value: {
        "/workspace/run": mount, "/workspace/model-preparation": scratch,
    }.get(str(value), path_type(value)))
    monkeypatch.setenv("MODEL_CREDENTIAL", "fixture-credential")
    monkeypatch.setenv("EVIDENCE_CREDENTIAL", "fixture-evidence")
    snapshot = cache / "model" / "models--fixture--tiny" / "snapshots" / REVISION
    child_calls = []
    def child(argv, **kwargs):
        assert (snapshot / "model.safetensors").read_bytes() == fixture.files["model.safetensors"]
        assert "MODEL_CREDENTIAL" not in kwargs["env"]
        assert "EVIDENCE_CREDENTIAL" not in kwargs["env"]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        child_calls.append(kwargs)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runtime.subprocess, "run", child)
    environment = {
        "SYNAPTIC_CACHE_ROOT": str(cache),
        "SYNAPTIC_MODEL_SNAPSHOT": str(snapshot),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "EVIDENCE_CREDENTIAL": "must-be-removed",
    }
    workload = json.dumps({"configuration": {"document": {"model": {"ref": MODEL, "revision": REVISION}}}}).encode()
    runner = runtime.SubprocessSftRunner(
        secret_keys=("MODEL_CREDENTIAL", "EVIDENCE_CREDENTIAL"),
        model_token_key="MODEL_CREDENTIAL", timeout_seconds=10,
    )
    result = runner.run(("/python", "/runtime.py", "--canonical-workload-stdin"), cwd=str(tmp_path), environment=environment, stdin=workload, commit_prepared=lambda: child_calls.append("committed"))
    assert result.returncode == 0 and len(child_calls) == 2
    assert child_calls[0] == "committed"
    assert any(name == "download" for name, _ in fixture.calls)


def test_runner_preparation_failure_never_launches_child(monkeypatch):
    from tuner.execution.providers.modal import runtime
    from tuner.execution.providers.modal.remote import ModalRemotePhaseError

    monkeypatch.setenv("MODEL_CREDENTIAL", "fixture")
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    runner = runtime.SubprocessSftRunner(secret_keys=("MODEL_CREDENTIAL",), model_token_key="MODEL_CREDENTIAL", timeout_seconds=10)
    with pytest.raises(ModalRemotePhaseError) as failure:
        runner.run(("/python", "/runtime.py", "--canonical-workload-stdin"), cwd="/tmp", environment={}, stdin=b"malformed", commit_prepared=lambda: None)
    assert failure.value.diagnostic_code == "model_preparation_failed"
    assert calls == []


def test_cache_commit_failure_never_launches_child(monkeypatch):
    from tuner.execution.providers.modal import runtime
    from tuner.execution.providers.modal.remote import ModalRemotePhaseError

    monkeypatch.setenv("MODEL_CREDENTIAL", "fixture")
    calls = []
    monkeypatch.setattr(runtime.SubprocessSftRunner, "_prepare_model", lambda *args: None)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    def commit():
        raise RuntimeError("fixture private provider response")
    runner = runtime.SubprocessSftRunner(secret_keys=("MODEL_CREDENTIAL",), model_token_key="MODEL_CREDENTIAL", timeout_seconds=10)
    with pytest.raises(ModalRemotePhaseError) as failure:
        runner.run(("/python", "/runtime.py", "--canonical-workload-stdin"), cwd="/tmp", environment={}, stdin=b"fixture", commit_prepared=commit)
    assert failure.value.diagnostic_code == "model_cache_commit_failed"
    assert calls == []
