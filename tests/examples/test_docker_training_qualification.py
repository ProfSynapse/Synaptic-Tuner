"""Shared consumer/real public Docker host and SQLite; no daemon/model effects."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.training.test_docker_host_composition import host_fixture, native_values  # noqa: F401
from tests.execution.providers.docker_provider_v1.test_local_source import material  # noqa: F401
from tests.execution.providers.docker_provider_v1.conftest import profile  # noqa: F401
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunOperationError, RunOperationCode
from tuner.cloud.hf_training_image_lock import CommandSpec

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "qualification_launch", ROOT / "examples/training_qualification/launch.py"
)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
import docker_host


def _locked_roots():
    return SimpleNamespace(
        roots={
            "engine": "/source/engine",
            "project": "/source/project",
            **{
                name: "/artifacts/run/" + name
                for name in ("artifacts", "state", "tracking", "cache", "tmp")
            },
        }
    )


def test_cached_model_start_prepares_all_locked_writable_roots(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir(mode=0o700)
    cache = artifact / "run" / "cache"
    cache.parent.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    model = cache / "verified-model"
    model.write_bytes(b"existing verified model bytes")
    before = (model.stat().st_ino, model.read_bytes())

    docker_host._prepare_writable_roots(_locked_roots(), artifact, "/artifacts")

    assert all(
        (artifact / "run" / name).is_dir()
        for name in ("artifacts", "state", "tracking", "cache", "tmp")
    )
    assert (model.stat().st_ino, model.read_bytes()) == before


@pytest.mark.parametrize("hostile", ["ancestor_link", "leaf_link", "loose_mode"])
def test_cached_model_root_preparation_rejects_hostile_paths(tmp_path, hostile):
    artifact = tmp_path / "artifact"
    artifact.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    run = artifact / "run"
    if hostile == "ancestor_link":
        run.symlink_to(outside, target_is_directory=True)
    else:
        run.mkdir(mode=0o700)
        leaf = run / "artifacts"
        if hostile == "leaf_link":
            leaf.symlink_to(outside, target_is_directory=True)
        else:
            leaf.mkdir(mode=0o755)

    with pytest.raises(ValueError, match="docker_runtime_roots_invalid"):
        docker_host._prepare_writable_roots(_locked_roots(), artifact, "/artifacts")


def test_cached_model_root_preparation_rejects_unmapped_root(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir(mode=0o700)
    source = _locked_roots()
    source.roots["tracking"] = "/source/tracking"

    with pytest.raises(ValueError, match="docker_runtime_roots_invalid"):
        docker_host._prepare_writable_roots(source, artifact, "/artifacts")
    assert not list(artifact.iterdir())


def test_cached_model_root_preparation_rejects_overlap(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir(mode=0o700)
    source = _locked_roots()
    source.roots["tracking"] = "/artifacts/run/cache/nested"

    with pytest.raises(ValueError, match="docker_runtime_roots_invalid"):
        docker_host._prepare_writable_roots(source, artifact, "/artifacts")
    assert not list(artifact.iterdir())


def test_runtime_root_platform_refuses_before_filesystem_mutation(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "artifact"
    artifact.mkdir(mode=0o700)
    monkeypatch.setattr(
        docker_host.os,
        "supports_dir_fd",
        os.supports_dir_fd - {os.mkdir},
    )

    with pytest.raises(ValueError, match="docker_runtime_roots_platform_unavailable"):
        docker_host._prepare_writable_roots(_locked_roots(), artifact, "/artifacts")
    assert not list(artifact.iterdir())
    state = tmp_path / "attempt"
    with pytest.raises(ValueError, match="docker_runtime_roots_platform_unavailable"):
        with docker_host.open_host({}, state_root=state, mode="start"):
            pytest.fail("unsupported platform reached host composition")
    assert not state.exists()


def committed(path, raw):
    path.write_bytes(raw)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


@pytest.fixture
def configured(host_fixture, monkeypatch):  # noqa: F811
    _, kwargs, commands, state, roles, _, tmp_path = host_fixture
    p = kwargs["profile"]
    config = {
        "provider": "docker",
        "state_root": str(tmp_path / "attempt"),
        "request_file": committed(
            tmp_path / "request.json", kwargs["material"].request_bytes
        ),
        "host": {
            "authority_secret": kwargs["authority_secret"].to_dict(),
            "evidence_secret": kwargs["evidence_secret"].to_dict(),
            "grant_ref": "cpu-configured-grant",
            "allowed_requirements": [
                {
                    "operation": name,
                    "paid_effect": False,
                    "maximum_cost_minor_units": None,
                    "currency": None,
                }
                for name in ("training.start", "training.cancel", "training.reconcile")
            ],
        },
        "docker": {
            "profile_file": committed(
                tmp_path / "profile.json", json.dumps(p.to_dict()).encode()
            ),
            "material_file": committed(
                tmp_path / "material.json", kwargs["material"].canonical_bytes
            ),
            "image_profile_file": {
                "path": "/required/original/derived-profile.yaml",
                "sha256": "0" * 64,
                "size_bytes": 1,
            },
            "image_verification_file": {
                "path": "/required/original/verification.json",
                "sha256": "0" * 64,
                "size_bytes": 1,
            },
            "executable": str(
                kwargs["cli_configuration"]["authority"]._authority.executable
            ),
            "config_directory": str(
                kwargs["cli_configuration"]["authority"]._authority.config_directory
            ),
            "endpoint": "unix:///fixture.sock",
            "daemon_id": "fixture-daemon",
            "source_host_root": str(kwargs["source_inputs"]["source_root"]),
            "source_container_root": "/source",
            "artifact_host_root": str(kwargs["artifact_root"]),
            "artifact_container_root": "/artifacts",
            "control_container_root": "/source/control",
            "prepared_publication": "/required/original/prepared.jsonl",
            "role_paths": roles,
            "command_seconds": 120,
            "read_seconds": 60,
        },
    }
    monkeypatch.setenv(
        kwargs["authority_secret"].name, "original-host-secret-" + "a" * 40
    )
    monkeypatch.setenv(
        kwargs["evidence_secret"].name, "original-evidence-secret-" + "b" * 40
    )
    original = docker_host.compose_docker_reference_host

    def compose(**options):
        host = original(**options)
        run = TrainingRunRef(kwargs["material"].run_id, "project")

        def claims():
            workflow = host.composition.stores.workflow_store.get(run)
            return tuple(
                kind
                for kind, intent in (
                    ("stage", workflow.stage),
                    ("submit", workflow.submit),
                )
                if intent is not None
                and host.composition.foundation.get(intent.effect_id).attempt_count == 1
            )

        state["claims"] = claims
        return host

    monkeypatch.setattr(docker_host, "compose_docker_reference_host", compose)
    options = {
        "runner": kwargs["cli_configuration"]["runner"],
        "launch_authority": kwargs["cli_configuration"]["authority"],
        "source_inputs": kwargs["source_inputs"],
    }
    return config, kwargs, commands, state, roles, options


def test_shared_preview_public_start_native_five_and_cold_retained(
    configured, monkeypatch
):
    config, kwargs, commands, state, roles, options = configured

    def no_resolution(*args, **kwargs):
        pytest.fail("pure preview resolved credential or contacted Docker")

    original_resolve = launch.NamedSecrets.resolve
    monkeypatch.setattr(launch.NamedSecrets, "resolve", no_resolution)
    preview = launch.preview(config)
    assert (
        preview["public_preflight"] == "deferred_until_host_composition"
        and not commands
    )
    monkeypatch.setattr(launch.NamedSecrets, "resolve", original_resolve)
    result = launch.launch(config, adapter_options=options)
    run = TrainingRunRef.from_dict(result["start"]["run"])
    root = Path(config["state_root"])
    retained = launch.load_config(root / "config.json")
    assert (
        result["start"]["accepted"]
        and len([c for c in commands if c[:2] == ("container", "create")]) == 1
    )
    writable = kwargs["artifact_root"] / "run"
    assert all(
        (writable / name).is_dir()
        for name in ("artifacts", "state", "tracking", "cache", "tmp")
    )
    # A later cold read authenticates retained state; it must not prepare roots.
    (writable / "tracking").rmdir()
    # Original compiled documents/request may be removed: cold uses exact copies.
    for ref in (
        config["request_file"],
        config["docker"]["profile_file"],
        config["docker"]["material_file"],
    ):
        Path(ref["path"]).unlink()
    assert (
        launch.known(retained, "observe", adapter_options=options)["state"] == "running"
    )
    assert not (writable / "tracking").exists()
    state["status"] = "exited"
    (kwargs["artifact_root"] / "native").mkdir(mode=0o700)
    values = native_values(kwargs, monkeypatch)
    for role, relative in roles.items():
        (kwargs["artifact_root"] / relative).write_bytes(values[role])
    assert (
        launch.known(retained, "observe", adapter_options=options)["state"]
        == "succeeded"
    )
    report = launch.known(retained, "verify", adapter_options=options)
    assert report["verified"] and set(report) == {"run", "verified", "checked_at"}
    output = root / "downloads"
    downloaded = launch.known(
        retained, "download", destination=output, adapter_options=options
    )
    assert len(downloaded["paths"]) == 5
    for role in roles:
        assert (output / (role + ".artifact")).read_bytes() == values[role]
    before = len(commands)
    assert (
        launch.known(retained, "show", adapter_options=options)["state"] == "succeeded"
    )
    assert len([c for c in commands if c[:2] == ("container", "create")]) == 1
    # Cold composition does identity reads only; no stage/create/start replay.
    assert all(
        c[:2] not in {("container", "create"), ("container", "start")}
        for c in commands[before:]
    )
    with SQLiteDurableStoreV1(root / "workflow.sqlite3", create=False) as store:
        assert store.list_page(
            partition="effects", prefix="", after_key=None, limit=100
        ).records
    assert run.to_dict() == preview["run"]


def bind_tiny_model_receipt(config, kwargs):
    _, rich, _, _, _ = docker_host._components(config)
    model = rich.resolved_config.to_dict()["model"]
    artifact = kwargs["artifact_root"]
    snapshot = (artifact / "run/cache/model"
                / ("models--" + model["ref"].replace("/", "--"))
                / "snapshots" / model["revision"])
    current = artifact
    for part in snapshot.relative_to(artifact).parts:
        current /= part
        current.mkdir(mode=0o700)
    member = snapshot / "synthetic-config.json"
    member.write_bytes(b'{"fixture":true}')
    receipt = {
        "schema_version": "synaptic-docker-model-preparation/v1",
        "model": model,
        "configuration_sha256": hashlib.sha256(
            rich.resolved_config.canonical_json.encode()
        ).hexdigest(),
        "runtime_declaration_sha256": hashlib.sha256(
            rich.execution_context.canonical_json.encode()
        ).hexdigest(),
        "artifact_host_root": str(artifact),
        "snapshot": str(snapshot),
        "files": docker_host.model_inventory(artifact, snapshot),
    }
    receipt_path = artifact.parent / "model-preparation.json"
    raw = json.dumps(receipt, sort_keys=True).encode()
    config["docker"]["model_preparation_file"] = committed(receipt_path, raw)
    return artifact, snapshot, member, receipt_path, raw


def test_model_first_public_start_rejects_changed_bytes_before_create(configured):
    config, kwargs, commands, _, _, options = configured
    _, _, member, _, _ = bind_tiny_model_receipt(config, kwargs)
    member.write_bytes(b"changed bytes")
    assert launch.preview(config)["run"]
    with pytest.raises(ValueError):
        launch.launch(config, adapter_options=options)
    assert not [c for c in commands if c[:2] == ("container", "create")]
    assert Path(config["state_root"]).exists()
    with pytest.raises(FileExistsError):
        launch.launch(config, adapter_options=options)


def test_model_first_public_start_and_cold_reads_without_weights(
    configured, monkeypatch
):
    config, kwargs, commands, state, roles, options = configured
    artifact, snapshot, member, receipt_path, raw = bind_tiny_model_receipt(config, kwargs)
    assert launch.preview(config)["run"]
    receipt_path.write_bytes(b"changed receipt")
    with pytest.raises(ValueError):
        launch.launch(config, adapter_options=options)
    assert not [c for c in commands if c[:2] == ("container", "create")]
    assert not Path(config["state_root"]).exists()
    receipt_path.write_bytes(raw)
    result = launch.launch(config, adapter_options=options)
    assert result["start"]["accepted"]
    assert len([c for c in commands if c[:2] == ("container", "create")]) == 1
    retained = launch.load_config(Path(config["state_root"]) / "config.json")
    retained_receipt = Path(retained["docker"]["model_preparation_file"]["path"])
    assert retained_receipt != receipt_path and retained_receipt.read_bytes() == raw
    receipt_path.unlink()
    member.unlink()
    snapshot.rmdir()
    assert launch.preview(retained)["run"] == result["start"]["run"]
    assert launch.known(retained, "observe", adapter_options=options)["state"] == "running"
    state["status"] = "exited"
    (artifact / "native").mkdir(mode=0o700)
    values = native_values(kwargs, monkeypatch)
    for role, relative in roles.items():
        (artifact / relative).write_bytes(values[role])
    assert launch.known(retained, "observe", adapter_options=options)["state"] == "succeeded"
    assert launch.known(retained, "verify", adapter_options=options)["verified"]
    output = Path(config["state_root"]) / "downloads"
    downloaded = launch.known(
        retained, "download", destination=output, adapter_options=options
    )
    assert len(downloaded["paths"]) == 5
    for role in roles:
        assert (output / (role + ".artifact")).read_bytes() == values[role]
    wrong_run = json.loads(json.dumps(retained))
    wrong_run["run"]["run_id"] = "another-run"
    with pytest.raises(ValueError):
        launch.known(wrong_run, "observe", adapter_options=options)
    retained_receipt.write_bytes(b"changed receipt")
    with pytest.raises(ValueError):
        launch.known(retained, "observe", adapter_options=options)
    with pytest.raises(ValueError):
        launch.preview(retained)
    assert len([c for c in commands if c[:2] == ("container", "create")]) == 1


def test_unknown_submit_and_existing_attempt_never_create_again(configured):
    config, _, commands, state, _, options = configured
    state["unknown_create"] = True
    result = launch.launch(config, adapter_options=options)
    assert result["start"]["accepted"]
    with pytest.raises(FileExistsError):
        launch.launch(config, adapter_options=options)
    retained = launch.load_config(Path(config["state_root"]) / "config.json")
    launch.known(retained, "show", adapter_options=options)
    assert len([c for c in commands if c[:2] == ("container", "create")]) == 1


def test_selected_daemon_command_spec_forwarding_never_uses_ambient(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://forbidden.example:2375")
    seen = []
    spec = CommandSpec(
        (
            "/absolute/docker",
            "--config",
            "/private/config",
            "image",
            "inspect",
            "sha256:" + "a" * 64,
        ),
        {"PATH": "/usr/bin"},
        10,
        1024,
    )
    run = docker_host.selected_runner(
        "unix:///selected.sock", lambda value: seen.append(value)
    )
    run(spec)
    assert seen[0].argv[:5] == (
        "/absolute/docker",
        "--config",
        "/private/config",
        "--host",
        "unix:///selected.sock",
    )
    assert seen[0].argv[5:] == spec.argv[3:]
    assert seen[0].env == spec.env and seen[0].timeout_seconds == spec.timeout_seconds
    with pytest.raises(ValueError):
        docker_host.selected_runner("tcp://remote:2375")
    with pytest.raises(ValueError):
        run(
            CommandSpec(
                (
                    "/absolute/docker",
                    "--config",
                    "/private/config",
                    "--host",
                    "unix:///other.sock",
                ),
                {},
                10,
                1024,
            )
        )


def test_known_cancel_ack_preserves_completion_and_no_second_stop(configured):
    config, _, commands, _, _, options = configured
    launch.launch(config, adapter_options=options)
    retained = launch.load_config(Path(config["state_root"]) / "config.json")
    first = launch.known(
        retained, "cancel", reason="explicit CPU fixture stop", adapter_options=options
    )
    assert first["state"] == "cancel_requested"
    launch.known(
        retained, "cancel", reason="explicit CPU fixture stop", adapter_options=options
    )
    # The fixture exits with code0: do not manufacture causal cancellation.
    assert (
        launch.known(retained, "observe", adapter_options=options)["state"]
        == "succeeded"
    )
    assert len([c for c in commands if c[:2] == ("container", "stop")]) == 1
    assert len([c for c in commands if c[:2] == ("container", "create")]) == 1


def test_unpromoted_profile_capabilities_refuse_cancel_reconcile(configured):
    config, _, commands, _, _, options = configured
    path = Path(config["docker"]["profile_file"]["path"])
    document = json.loads(path.read_bytes())
    document["descriptor"]["capabilities"]["cancel"] = False
    document["descriptor"]["capabilities"]["reconcile"] = False
    config["docker"]["profile_file"] = committed(path, json.dumps(document).encode())
    launch.launch(config, adapter_options=options)
    retained = launch.load_config(Path(config["state_root"]) / "config.json")
    for action in ("cancel", "reconcile"):
        with pytest.raises(RunOperationError) as captured:
            launch.known(
                retained, action, reason="CPU refusal control", adapter_options=options
            )
        assert captured.value.code is RunOperationCode.CAPABILITY_UNAVAILABLE
    assert not [c for c in commands if c[:2] == ("container", "stop")]
    assert len([c for c in commands if c[:2] == ("container", "create")]) == 1


def test_changed_material_commitment_refuses_before_state_or_secret(
    configured, monkeypatch
):
    config, _, commands, _, _, _ = configured
    Path(config["docker"]["material_file"]["path"]).write_bytes(b"{}")
    monkeypatch.setattr(
        launch.NamedSecrets,
        "resolve",
        lambda *_: pytest.fail("secret before input admission"),
    )
    with pytest.raises(ValueError, match="commitment mismatch"):
        launch.launch(config)
    assert not Path(config["state_root"]).exists() and not commands
