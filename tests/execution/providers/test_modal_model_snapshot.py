from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tuner.execution.providers.modal import model_snapshot
from tuner.execution.providers.modal.model_snapshot import (
    ModelSnapshotPreparationError, prepare_model_snapshot,
)
from tuner.project.execution_source import ExecutionSourceV1
from tests.training.test_training_service import _execution_source

REVISION = "a" * 40
MODEL = "fixture/tiny"


def test_authenticated_deployment_path_is_retained_in_execution_source(tmp_path):
    from tests.execution.providers.test_modal_source_resolution import (
        _context,
        _deployment,
        _finalizer,
        _source,
    )
    from tuner.execution.providers.modal.resolution import ModalDeploymentSelectionV1

    locked = _source()
    document = _deployment().to_dict()
    document["runtime_environment"]["PATH"] = "/usr/bin:/bin"
    resolved = _finalizer(locked).finalize(
        locked,
        context=_context(tmp_path),
        deployment=ModalDeploymentSelectionV1.from_dict(document),
        audience_ref="project/run-1",
    )
    assert resolved.execution_source.environment["PATH"] == "/usr/bin:/bin"


def _source_for_run(
    run_id: str, *, cache_run_id: str | None = None, path: str | None = None
) -> ExecutionSourceV1:
    document = _execution_source("vendor/engine").to_dict()
    document["run_id"] = run_id
    selected = cache_run_id or run_id
    roots = document["runtime"]["roots"]
    for name, value in roots.items():
        roots[name] = value.replace("run-service", selected)
    variables = document["runtime"]["environment"]["variables"]
    for name, value in variables.items():
        variables[name] = value.replace("run-service", selected)
    if path is not None:
        variables["PATH"] = path
    return ExecutionSourceV1.from_dict(document)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    files = {
        "config.json": b'{"model_type":"fixture"}',
        "model.safetensors": b"fixture weights",
    }
    siblings = []
    for name, content in files.items():
        lfs = name.endswith("safetensors")
        siblings.append(
            SimpleNamespace(
                rfilename=name,
                size=len(content),
                blob_id=hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content
                ).hexdigest(),
                lfs=(
                    SimpleNamespace(sha256=hashlib.sha256(content).hexdigest())
                    if lfs
                    else None
                ),
            )
        )
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

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=API, snapshot_download=download),
    )
    roots = {
        name: tmp_path / name
        for name in ("persistent_root", "destination_root", "scratch_root")
    }
    for path in roots.values():
        path.mkdir()
    return SimpleNamespace(files=files, info=info, calls=calls, roots=roots)


def prepare(fixture, **overrides):
    return prepare_model_snapshot(
        **({"model_ref": MODEL, "revision": REVISION, "token": "fixture-credential"}
           | fixture.roots | overrides),
    )


class _BoundCache:
    def __init__(self, root: Path):
        self.root = root
        self.published = []

    def claim_directory(self, relative_path: str) -> None:
        (self.root / relative_path).mkdir()

    def copy_in_exclusive(
        self, relative_path: str, source_path: str, *,
        expected_size: int, expected_sha256: str, maximum: int,
    ) -> None:
        source = Path(source_path)
        content = source.read_bytes()
        assert len(content) == expected_size <= maximum
        assert hashlib.sha256(content).hexdigest() == expected_sha256
        destination = self.root / relative_path
        with destination.open("xb") as output:
            output.write(content)
        self.published.append((relative_path, expected_sha256))


@pytest.mark.parametrize("stage", sorted(model_snapshot.MODEL_SNAPSHOT_PREPARATION_STAGES))
def test_model_preparation_stages_are_fixed_and_private(fixture, monkeypatch, tmp_path, stage):
    kwargs = {}
    if stage == "SDK_ADMISSION":
        monkeypatch.setattr(model_snapshot, "_bind_hub_api", lambda *_: (_ for _ in ()).throw(RuntimeError("PRIVATE_SENTINEL")))
    elif stage == "INPUT":
        kwargs["model_ref"] = "invalid/three/parts"
    elif stage == "WORKSPACE_SETUP":
        kwargs["destination_root"] = tmp_path / "absent"
    elif stage == "METADATA_FETCH":
        monkeypatch.setattr(sys.modules["huggingface_hub"].HfApi, "model_info",
                            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("PRIVATE_SENTINEL")))
    elif stage == "METADATA_VALIDATION":
        fixture.info.sha = "b" * 40
    elif stage == "DOWNLOAD":
        monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download",
                            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("PRIVATE_SENTINEL")))
    elif stage == "VERIFICATION":
        fixture.info.siblings[0].blob_id = "b" * 40
    elif stage == "PERSISTENT_PUBLICATION":
        kwargs["persistent_binding"] = _BoundCache(tmp_path / "absent-cache")
    else:
        original = model_snapshot.copy_regular
        def reject_at_boundary(source_root, source, destination_root, destination, **options):
            if ((stage == "DESTINATION_COPY" and destination_root == fixture.roots["destination_root"])
                    or (stage == "DESTINATION_VERIFICATION" and destination_root.name == "verification")):
                raise RuntimeError("PRIVATE_SENTINEL")
            return original(source_root, source, destination_root, destination, **options)
        monkeypatch.setattr(model_snapshot, "copy_regular", reject_at_boundary)
    with pytest.raises(ModelSnapshotPreparationError) as caught:
        prepare(fixture, **kwargs)
    assert caught.value.stage == stage
    assert str(caught.value) == "model preparation failed"
    assert caught.value.__cause__ is None


def test_bound_cache_publishes_every_verified_member_from_private_scratch(
    fixture, tmp_path
):
    cache = _BoundCache(tmp_path / "bound-cache")
    cache.root.mkdir()
    result = prepare(fixture, persistent_binding=cache)
    assert {p.name: p.read_bytes() for p in result.iterdir()} == fixture.files
    assert sorted(path for path, _ in cache.published) == [
        f"models--fixture--tiny/snapshots/{REVISION}/{name}"
        for name in sorted(fixture.files)
    ]
    for path, digest in cache.published:
        assert hashlib.sha256((cache.root / path).read_bytes()).hexdigest() == digest
    assert [args["allow_patterns"] for name, args in fixture.calls if name == "download"] == [
        list(fixture.files)
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux descriptor-relative mount semantics")
@pytest.mark.parametrize("umask", (0o022, 0o002))
def test_real_volume_binding_publishes_verified_model_fixture(fixture, monkeypatch, tmp_path, umask):
    from tuner.execution.providers.modal import volume_root_binding as bound
    nested_name, nested_content = "nested/config.json", b"nested verified bytes"
    fixture.files[nested_name] = nested_content
    fixture.info.siblings.append(SimpleNamespace(
        rfilename=nested_name, size=len(nested_content), lfs=None,
        blob_id=hashlib.sha1(f"blob {len(nested_content)}\0".encode() + nested_content).hexdigest(),
    ))
    target_parent = tmp_path / "__modal" / "volumes"
    target = target_parent / "vo-test"
    target.mkdir(parents=True)
    mount = tmp_path / "mnt" / "model-cache"
    mount.parent.mkdir()
    mount.symlink_to(target, target_is_directory=True)
    marker_name = ".synaptic-test-marker"
    marker = b"m" * 32
    (target / marker_name).write_bytes(marker)
    monkeypatch.setattr(bound, "_PROVIDER_VOLUME_ROOT", str(target_parent))
    original_trust = bound._trusted_dir
    def allow_test_tmp(info):
        if info.st_uid == 0 and info.st_mode & 0o1000:
            return
        original_trust(info)
    monkeypatch.setattr(bound, "_trusted_dir", allow_test_tmp)
    original_umask = os.umask(umask)
    try:
        with bound.VolumeRootBinding.bind(
            root_path=str(mount), volume_id="vo-test", marker_name=marker_name,
            marker_sha256=hashlib.sha256(marker).hexdigest(),
        ) as binding:
            snapshot = prepare(fixture, persistent_root=mount, persistent_binding=binding)
            assert all((snapshot / name).read_bytes() == content for name, content in fixture.files.items())
    finally:
        os.umask(original_umask)
    published = target / "models--fixture--tiny" / "snapshots" / REVISION
    assert all((published / name).read_bytes() == content for name, content in fixture.files.items())


def test_bound_cache_never_opens_provider_symlink_path(fixture, tmp_path):
    cache = _BoundCache(tmp_path / "bound-cache")
    cache.root.mkdir()
    mounted = tmp_path / "modal-mounted-cache"
    mounted.symlink_to(cache.root, target_is_directory=True)
    result = prepare(fixture, persistent_root=mounted, persistent_binding=cache)
    assert {p.name: p.read_bytes() for p in result.iterdir()} == fixture.files
    assert len(cache.published) == len(fixture.files)
    assert list(fixture.roots["persistent_root"].iterdir()) == []
    assert list(fixture.roots["scratch_root"].iterdir()) == []


def test_bound_cache_collision_fails_without_adopting_existing_files(fixture, tmp_path):
    cache = _BoundCache(tmp_path / "bound-cache")
    existing = cache.root / "models--fixture--tiny" / "snapshots" / REVISION
    existing.mkdir(parents=True)
    (existing / "config.json").write_bytes(fixture.files["config.json"])
    with pytest.raises(ValueError, match="^model preparation failed$"):
        prepare(fixture, persistent_binding=cache)
    assert cache.published == []
    assert (existing / "config.json").read_bytes() == fixture.files["config.json"]
    assert list(fixture.roots["destination_root"].iterdir()) == []


def test_bound_cache_rejects_wrong_git_blob_before_any_volume_write(fixture, tmp_path):
    cache = _BoundCache(tmp_path / "bound-cache")
    cache.root.mkdir()
    fixture.info.siblings[0].blob_id = "b" * 40
    with pytest.raises(ValueError, match="^model preparation failed$"):
        prepare(fixture, persistent_binding=cache)
    assert list(cache.root.iterdir()) == []
    assert cache.published == []
    assert list(fixture.roots["destination_root"].iterdir()) == []


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
    cached = (
        fixture.roots["persistent_root"]
        / "models--fixture--tiny"
        / "snapshots"
        / REVISION
    )
    cached.mkdir(parents=True)
    (cached / "config.json").write_bytes(fixture.files["config.json"])
    prepare(fixture)
    assert [
        args["allow_patterns"] for name, args in fixture.calls if name == "download"
    ] == [["model.safetensors"]]


def test_zero_byte_repository_member_is_verified_and_reused(fixture, tmp_path):
    fixture.files["empty.txt"] = b""
    fixture.info.siblings.append(
        SimpleNamespace(
            rfilename="empty.txt",
            size=0,
            lfs=None,
            blob_id=hashlib.sha1(b"blob 0\0").hexdigest(),
        )
    )
    assert (prepare(fixture) / "empty.txt").read_bytes() == b""
    second = tmp_path / "second"
    second.mkdir()
    assert (prepare(fixture, destination_root=second) / "empty.txt").read_bytes() == b""
    assert sum(name == "download" for name, _ in fixture.calls) == 1


@pytest.mark.parametrize(
    "mutation", ["revision", "digest", "size", "path", "duplicate", "missing_digest"]
)
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
def test_hostile_persistent_cache_never_reaches_sdk_or_trainer(
    fixture, tmp_path, mutation
):
    cached = (
        fixture.roots["persistent_root"]
        / "models--fixture--tiny"
        / "snapshots"
        / REVISION
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "config.json"
    target.write_bytes(fixture.files["config.json"])
    if mutation == "ancestor":
        (fixture.roots["persistent_root"] / "models--fixture--tiny").symlink_to(
            outside, target_is_directory=True
        )
    else:
        cached.mkdir(parents=True)
        if mutation == "symlink":
            (cached / "config.json").symlink_to(target)
        else:
            (cached / "config.json").write_bytes(
                b"x" * len(fixture.files["config.json"])
            )
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
    prepare_model_snapshot(
        model_ref=MODEL, revision=REVISION, token="  ", **fixture.roots
    )
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


def test_real_runner_prepares_before_credential_free_offline_child(
    fixture, tmp_path, monkeypatch
):
    import json
    from tuner.execution.providers.modal import runtime

    mount = tmp_path / "volume"
    cache = mount / "modal-chat-20260914-e" / "cache"
    cache.mkdir(parents=True)
    scratch = tmp_path / "worker-private"
    path_type = Path

    def mapped_path(value):
        text = str(value)
        if text == "/workspace/model-preparation":
            return scratch
        if text == "/workspace/run":
            return mount
        if text.startswith("/workspace/run/"):
            return mount / text.removeprefix("/workspace/run/")
        return path_type(value)

    monkeypatch.setattr(runtime, "Path", mapped_path)
    monkeypatch.setenv("MODEL_CREDENTIAL", "fixture-credential")
    monkeypatch.setenv("EVIDENCE_CREDENTIAL", "fixture-evidence")
    monkeypatch.setenv("PATH", "/hostile/ambient/path")
    snapshot = cache / "model" / "models--fixture--tiny" / "snapshots" / REVISION
    child_calls = []

    def child(argv, **kwargs):
        assert (snapshot / "model.safetensors").read_bytes() == fixture.files[
            "model.safetensors"
        ]
        assert "MODEL_CREDENTIAL" not in kwargs["env"]
        assert "EVIDENCE_CREDENTIAL" not in kwargs["env"]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert kwargs["env"]["PATH"] == "/usr/bin:/bin"
        child_calls.append(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", child)
    source = _source_for_run("modal-chat-20260914-e", path="/usr/bin:/bin")
    assert source.environment["PATH"] == "/usr/bin:/bin"
    environment = dict(source.environment)
    environment.update(
        {
            "SYNAPTIC_MODEL_SNAPSHOT": "/workspace/run/modal-chat-20260914-e/cache/model/models--fixture--tiny/snapshots/"
            + REVISION,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "EVIDENCE_CREDENTIAL": "must-be-removed",
        }
    )
    workload = json.dumps(
        {
            "configuration": {
                "document": {"model": {"ref": MODEL, "revision": REVISION}}
            },
            "execution_source": source.to_dict(),
        }
    ).encode()
    runner = runtime.SubprocessSftRunner(
        secret_keys=("MODEL_CREDENTIAL", "EVIDENCE_CREDENTIAL"),
        model_token_key="MODEL_CREDENTIAL",
        timeout_seconds=10,
    )
    result = runner.run(
        ("/python", "/runtime.py", "--canonical-workload-stdin"),
        cwd=str(tmp_path),
        environment=environment,
        stdin=workload,
        commit_prepared=lambda: child_calls.append("committed"),
    )
    assert result.returncode == 0 and len(child_calls) == 2
    assert child_calls[0] == "committed"
    assert any(name == "download" for name, _ in fixture.calls)


@pytest.mark.parametrize("changed", ("run", "cache", "source_cache"))
def test_runner_denies_changed_run_cache_binding_before_download_or_child(
    fixture, tmp_path, monkeypatch, changed
):
    import json
    from tuner.execution.providers.modal import runtime
    from tuner.execution.providers.modal.worker_ports import ModalRemotePhaseError

    mount = tmp_path / "volume"
    cache = mount / "modal-chat-20260914-e" / "cache"
    cache.mkdir(parents=True)
    path_type = Path
    monkeypatch.setattr(
        runtime,
        "Path",
        lambda value: (
            mount / str(value).removeprefix("/workspace/run/")
            if str(value).startswith("/workspace/run/")
            else mount if str(value) == "/workspace/run" else path_type(value)
        ),
    )
    monkeypatch.setenv("MODEL_CREDENTIAL", "fixture-credential")
    source = _source_for_run(
        "different-run" if changed == "run" else "modal-chat-20260914-e",
        cache_run_id="other-run" if changed == "source_cache" else None,
    )
    selected_cache = (
        "/workspace/run/other-run/cache"
        if changed == "cache"
        else "/workspace/run/modal-chat-20260914-e/cache"
    )
    environment = {
        "SYNAPTIC_CACHE_ROOT": selected_cache,
        "SYNAPTIC_MODEL_SNAPSHOT": selected_cache
        + "/model/models--fixture--tiny/snapshots/"
        + REVISION,
    }
    workload = json.dumps(
        {
            "configuration": {
                "document": {"model": {"ref": MODEL, "revision": REVISION}}
            },
            "execution_source": source.to_dict(),
        }
    ).encode()
    children = []
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *args, **kwargs: children.append(args)
    )
    runner = runtime.SubprocessSftRunner(
        secret_keys=("MODEL_CREDENTIAL",),
        model_token_key="MODEL_CREDENTIAL",
        timeout_seconds=10,
    )
    with pytest.raises(ModalRemotePhaseError) as failure:
        runner.run(
            ("/python", "/runtime.py", "--canonical-workload-stdin"),
            cwd=str(tmp_path),
            environment=environment,
            stdin=workload,
            commit_prepared=lambda: None,
        )
    assert failure.value.diagnostic_code == "model_preparation_failed"
    assert not any(name == "download" for name, _ in fixture.calls)
    assert children == []


def test_runner_preparation_failure_never_launches_child(monkeypatch):
    from tuner.execution.providers.modal import runtime
    from tuner.execution.providers.modal.worker_ports import ModalRemotePhaseError

    monkeypatch.setenv("MODEL_CREDENTIAL", "fixture")
    calls = []
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )
    runner = runtime.SubprocessSftRunner(
        secret_keys=("MODEL_CREDENTIAL",),
        model_token_key="MODEL_CREDENTIAL",
        timeout_seconds=10,
    )
    with pytest.raises(ModalRemotePhaseError) as failure:
        runner.run(
            ("/python", "/runtime.py", "--canonical-workload-stdin"),
            cwd="/tmp",
            environment={},
            stdin=b"malformed",
            commit_prepared=lambda: None,
        )
    assert failure.value.diagnostic_code == "model_preparation_failed"
    assert calls == []


def test_cache_commit_failure_never_launches_child(monkeypatch):
    from tuner.execution.providers.modal import runtime
    from tuner.execution.providers.modal.worker_ports import ModalRemotePhaseError

    monkeypatch.setenv("MODEL_CREDENTIAL", "fixture")
    calls = []
    monkeypatch.setattr(
        runtime.SubprocessSftRunner, "_prepare_model", lambda *args: None
    )
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )

    def commit():
        raise RuntimeError("fixture private provider response")

    runner = runtime.SubprocessSftRunner(
        secret_keys=("MODEL_CREDENTIAL",),
        model_token_key="MODEL_CREDENTIAL",
        timeout_seconds=10,
    )
    with pytest.raises(ModalRemotePhaseError) as failure:
        runner.run(
            ("/python", "/runtime.py", "--canonical-workload-stdin"),
            cwd="/tmp",
            environment={},
            stdin=b"fixture",
            commit_prepared=commit,
        )
    assert failure.value.diagnostic_code == "model_cache_commit_failed"
    assert calls == []
