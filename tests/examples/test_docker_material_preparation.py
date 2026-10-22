# ruff: noqa: F811
"""Genuine Git/publication/compiler and no-daemon Docker preparation controls."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import contextmanager
import pytest

from tests.examples.test_hf_material_preparation import configured as source_configured  # noqa: F401
from tests.execution.providers.docker_provider_v1.conftest import profile  # noqa: F401
from tests.runtime.test_packaged_runtime_releases import _release
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
from tuner.training import default_recipe_registry
from tuner.execution.providers.modal.model_snapshot import (
    ModelSnapshotPreparationError, prepare_model_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "docker_prepare_example", ROOT / "examples/training_qualification/prepare_docker.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.fixture
def configured(source_configured, profile, tmp_path):
    c, verifier, auth, clock, remote = source_configured
    model = json.loads(Path(c["resolved_config"]).read_bytes())["model"]
    release = _release(
        image_ref="fixture-image@sha256:" + "a" * 64,
        image_digest="a" * 64,
        compatible_models=((model["ref"], model["revision"]),),
        compatible_dataset_formats=("syntunia-sft-row/v1",),
    )
    path = tmp_path / "release.json"
    path.write_bytes(release.canonical_bytes())
    c["runtime_release"] = str(path)
    c["resources"] = {
        "accelerator": "cpu",
        "accelerator_count": 1,
        "timeout_seconds": 3600,
    }
    c["profile"] = profile.to_dict()
    for key in (
        "schema_version",
        "image",
        "workload",
        "resource_digest",
        "quote_digest",
        "secret_requirements_digest",
    ):
        del c["profile"][key]
    del c["profile"]["artifacts"]["roles"]
    c["consumer"] = {
        "provider": "docker",
        "state_root": str(tmp_path / "attempt"),
        "host": c["consumer"]["host"],
        "docker": {
            "image_profile_file": {
                "path": "/required/derived-profile.yaml",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            "image_verification_file": {
                "path": "/required/verified.json",
                "sha256": "b" * 64,
                "size_bytes": 1,
            },
            "executable": "/usr/bin/docker",
            "config_directory": str(tmp_path / "docker-config"),
            "endpoint": "unix:///fixture.sock",
            "daemon_id": "fixture-daemon",
            "source_host_root": str(tmp_path / "source"),
            "source_container_root": "/source",
            "artifact_host_root": str(tmp_path / "artifacts"),
            "artifact_container_root": "/artifacts",
            "control_container_root": "/source/control",
            "role_paths": {r: r + ".json" for r in helper.ROLES},
            "command_seconds": 120,
            "read_seconds": 60,
        },
    }
    c["consumer"]["host"]["allowed_requirements"] = [
        {
            "operation": "training.start",
            "paid_effect": False,
            "maximum_cost_minor_units": None,
            "currency": None,
        }
    ]
    roots = {"engine": "/source/engine", "project": "/source/project"}
    roots.update(
        {
            n: "/artifacts/run/" + n
            for n in ("state", "tracking", "cache", "tmp", "artifacts")
        }
    )
    env = c["runtime_environment"]
    env["PYTHONPATH"] = roots["engine"]
    for name, value in roots.items():
        env[
            "SYNAPTIC_"
            + ("ARTIFACT" if name == "artifacts" else name.upper())
            + "_ROOT"
        ] = value
    env["HF_HOME"], env["TRANSFORMERS_CACHE"] = (
        roots["cache"] + "/huggingface",
        roots["cache"] + "/transformers",
    )
    return c, verifier, auth, clock, remote


def test_real_source_publication_compiler_and_pure_docker_preview(configured):
    c, verifier, auth, clock, remote = configured
    result = helper.prepare(c, pushed_sources=verifier, authenticator=auth, clock=clock)
    output = Path(c["output_root"])
    consumer = helper.launch.load_config(output / "consumer.json")
    material = CoordinatorResolvedMaterial.parse(
        (output / "material.json").read_bytes(), default_recipe_registry()
    )
    assert helper.docker_host.preview(consumer) == result["preview"]
    assert material.request_bytes == (output / "request.json").read_bytes()
    assert result["public_preflight"] == "deferred_until_host_composition"
    assert len(remote.calls) == 2
    assert not Path(consumer["state_root"]).exists()
    assert not Path(consumer["docker"]["source_host_root"]).exists()
    with pytest.raises(ValueError, match="used"):
        helper.prepare(c, pushed_sources=verifier, authenticator=auth, clock=clock)


@pytest.mark.parametrize("change", ["format", "model", "source", "input"])
def test_wrong_commitments_refuse_before_consumer_publication(configured, change):
    c, verifier, auth, clock, _ = configured
    if change in {"format", "model"}:
        value = json.loads(Path(c["runtime_release"]).read_bytes())
        if change == "format":
            value["compatibility"]["dataset_formats"] = ["syntunia-sft-row/v2"]
        else:
            value["compatibility"]["models"][0]["revision"] = "2" * 40
        Path(c["runtime_release"]).write_text(json.dumps(value))
    elif change == "source":
        (Path(c["project"]["root"]) / "dirty.txt").write_text("changed")
    else:
        (Path(c["prepared_publication"]) / "dataset.jsonl").chmod(0o600)
        (Path(c["prepared_publication"]) / "dataset.jsonl").write_text("{}\n")
    with pytest.raises(Exception):
        helper.prepare(c, pushed_sources=verifier, authenticator=auth, clock=clock)
    assert not (Path(c["output_root"]) / "consumer.json").exists()


def test_pinned_model_preparation_disjoint_cache_and_one_attempt(configured):
    c, verifier, auth, clock, _ = configured
    helper.prepare(c, pushed_sources=verifier, authenticator=auth, clock=clock)
    calls = []

    def model_preparer(**kwargs):
        calls.append(kwargs)
        target = (
            kwargs["destination_root"]
            / "model"
            / ("models--" + kwargs["model_ref"].replace("/", "--"))
            / "snapshots"
            / kwargs["revision"]
        )
        current = kwargs["destination_root"]
        for part in target.relative_to(current).parts:
            current /= part
            current.mkdir(mode=0o700)
        (target / "synthetic-config.json").write_text('{"fixture":true}')
        return target

    result = helper.prepare_model(c, model_preparer=model_preparer, sdk_preflight=lambda: None)
    assert calls[0]["token"] is False
    assert (
        calls[0]["revision"]
        == json.loads(Path(c["resolved_config"]).read_bytes())["model"]["revision"]
    )
    assert result["provider_qualified"] is result["source_mutation"] is False
    assert not list(Path(c["consumer"]["docker"]["source_host_root"]).iterdir())
    assert Path(result["snapshot"]).is_relative_to(
        Path(c["consumer"]["docker"]["artifact_host_root"])
    )
    with pytest.raises(FileExistsError):
        helper.prepare_model(c, model_preparer=model_preparer, sdk_preflight=lambda: None)
    assert len(calls) == 1


def test_model_first_claim_precedes_source_and_receipt_binds_fresh_prepare(configured):
    c, verifier, auth, clock, _ = configured
    c["model_output_root"] = str(Path(c["output_root"]).parent / "model-attempt")
    selected = c["consumer"]["docker"]
    source = Path(selected["source_host_root"])
    artifact = Path(selected["artifact_host_root"])
    calls = []

    def model_preparer(**kwargs):
        assert not source.exists()
        assert not Path(c["output_root"]).exists()
        assert (Path(c["model_output_root"]) / "model-preparation-attempt.json").exists()
        calls.append(kwargs)
        target = (kwargs["destination_root"] / "model"
                  / ("models--" + kwargs["model_ref"].replace("/", "--"))
                  / "snapshots" / kwargs["revision"])
        target.mkdir(parents=True, mode=0o700)
        (target / "synthetic-config.json").write_text('{"fixture":true}')
        return target

    receipt = helper.prepare_model(
        c, model_preparer=model_preparer, sdk_preflight=lambda: None,
    )
    assert receipt["schema_version"] == "synaptic-docker-model-preparation/v1"
    assert not source.exists()
    assert len(calls) == 1
    with pytest.raises(ValueError):
        helper.prepare_model(c, model_preparer=model_preparer, sdk_preflight=lambda: None)
    assert len(calls) == 1
    member = (artifact / "run/cache/model"
              / ("models--" + calls[0]["model_ref"].replace("/", "--"))
              / "snapshots" / calls[0]["revision"] / "synthetic-config.json")
    original = member.read_bytes()
    member.write_bytes(b"changed before source")
    with pytest.raises(ValueError):
        helper.prepare(c, pushed_sources=verifier, authenticator=auth, clock=clock)
    assert not Path(c["output_root"]).exists()
    member.write_bytes(original)
    helper.prepare(c, pushed_sources=verifier, authenticator=auth, clock=clock)
    consumer = helper.launch.load_config(Path(c["output_root"]) / "consumer.json")
    assert "model_preparation_file" in consumer["docker"]
    assert not source.exists()
    member.write_text("changed")
    assert helper.docker_host.preview(consumer)["run"]
    with pytest.raises(ValueError):
        helper.docker_host._components(consumer, live_model_bytes=True)


def test_model_first_failure_consumes_only_fresh_model_claim(configured):
    c, _, _, _, _ = configured
    c["model_output_root"] = str(Path(c["output_root"]).parent / "model-attempt")
    selected = c["consumer"]["docker"]

    def fail(**kwargs):
        assert (Path(c["model_output_root"]) / "model-preparation-attempt.json").exists()
        raise RuntimeError("private fixture failure")

    with pytest.raises(RuntimeError):
        helper.prepare_model(c, model_preparer=fail, sdk_preflight=lambda: None)
    assert not Path(c["output_root"]).exists()
    assert not Path(selected["source_host_root"]).exists()
    with pytest.raises(ValueError):
        helper.prepare_model(c, model_preparer=fail, sdk_preflight=lambda: None)


def test_model_first_rejects_source_created_during_expensive_preparation(configured):
    c, _, _, _, _ = configured
    c["model_output_root"] = str(Path(c["output_root"]).parent / "model-attempt")
    source = Path(c["consumer"]["docker"]["source_host_root"])

    def drift(**kwargs):
        source.mkdir(mode=0o700)
        target = (kwargs["destination_root"] / "model"
                  / ("models--" + kwargs["model_ref"].replace("/", "--"))
                  / "snapshots" / kwargs["revision"])
        target.mkdir(parents=True, mode=0o700)
        (target / "synthetic-config.json").write_text('{"fixture":true}')
        return target

    with pytest.raises(ValueError, match="docker_source_changed"):
        helper.prepare_model(c, model_preparer=drift, sdk_preflight=lambda: None)
    assert (Path(c["model_output_root"]) / "model-preparation-attempt.json").exists()
    assert not (Path(c["model_output_root"]) / "model-preparation.json").exists()


@pytest.mark.parametrize("seed_state", [
    "verified", "corrupt", "symlink", "revision_symlink", "mutated",
    "absent_revision", "cold",
])
def test_model_first_seed_is_independently_verified(configured, monkeypatch, seed_state):
    c, _, _, _, _ = configured
    c["model_output_root"] = str(Path(c["output_root"]).parent / "model-attempt")
    model = json.loads(Path(c["resolved_config"]).read_bytes())["model"]
    content = b'{"fixture":true}'
    name = "synthetic-config.json"
    seed = Path(c["output_root"]).parent / "retained-cache"
    relative = (Path("models--" + model["ref"].replace("/", "--"))
                / "snapshots" / model["revision"])
    if seed_state != "cold":
        seed.mkdir(mode=0o700)
        (seed / "old-attempt-claim.json").write_bytes(b"retained claim")
        member = seed / relative
        if seed_state == "revision_symlink":
            member.parent.mkdir(parents=True, mode=0o700)
            member.symlink_to(Path(c["resolved_config"]), target_is_directory=True)
        elif seed_state != "absent_revision":
            member.mkdir(parents=True, mode=0o700)
            (member / name).write_bytes(content)
            if seed_state == "corrupt":
                (member / name).write_bytes(b"wrong fixture")
            elif seed_state == "symlink":
                (member / name).unlink()
                (member / name).symlink_to(Path(c["resolved_config"]))
        c["model_seed_root"] = str(seed)
    calls = []

    class API:
        def __init__(self, *, endpoint, token):
            pass

        def model_info(self, repo_id, *, revision, files_metadata, token):
            assert (repo_id, revision) == (model["ref"], model["revision"])
            calls.append("metadata")
            return SimpleNamespace(
                sha=revision,
                siblings=[SimpleNamespace(
                    rfilename=name, size=len(content), lfs=None,
                    blob_id=hashlib.sha1(
                        f"blob {len(content)}\0".encode() + content
                    ).hexdigest(),
                )],
            )

    def download(**kwargs):
        calls.append("download")
        (kwargs["local_dir"] / name).write_bytes(content)
        return str(kwargs["local_dir"])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=API, snapshot_download=download,
    ))
    if seed_state == "mutated":
        original_copy = helper.copy_regular

        def mutate_after_copy(*args, **kwargs):
            result = original_copy(*args, **kwargs)
            (seed / relative / name).write_bytes(b"x" * len(content))
            return result

        monkeypatch.setattr(helper, "copy_regular", mutate_after_copy)
    if seed_state in {"corrupt", "symlink", "revision_symlink", "mutated"}:
        with pytest.raises((ValueError, ModelSnapshotPreparationError)):
            helper.prepare_model(
                c, model_preparer=prepare_model_snapshot, sdk_preflight=lambda: None,
            )
        assert "download" not in calls
        assert not Path(c["output_root"]).exists()
    else:
        receipt = helper.prepare_model(
            c, model_preparer=prepare_model_snapshot, sdk_preflight=lambda: None,
        )
        assert receipt["files"] == [{
            "path": name, "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }]
        assert ("download" in calls) == (seed_state in {"cold", "absent_revision"})
        if seed_state == "verified":
            assert (seed / relative / name).read_bytes() == content
    if seed_state != "cold":
        assert (seed / "old-attempt-claim.json").read_bytes() == b"retained claim"
    assert Path(c["model_output_root"]).exists()


def test_model_inventory_hashes_held_scan_during_transient_ancestor_swap(
    tmp_path, monkeypatch
):
    root, alternate, held = (tmp_path / name for name in ("model", "alternate", "held"))
    for directory, content in ((root, b"trusted"), (alternate, b"hostile")):
        directory.mkdir(mode=0o700)
        snapshot = directory / "snapshot"
        snapshot.mkdir(mode=0o700)
        (snapshot / "member.bin").write_bytes(content)
    original_scandir = os.scandir
    swapped = False

    @contextmanager
    def swap_while_scanning(directory_fd):
        nonlocal swapped
        with original_scandir(directory_fd) as entries:
            if not swapped:
                swapped = True
                os.rename(root, held)
                os.rename(alternate, root)
                try:
                    yield entries
                finally:
                    os.rename(root, alternate)
                    os.rename(held, root)
            else:
                yield entries

    monkeypatch.setattr(helper.docker_host.os, "scandir", swap_while_scanning)
    rows = helper.docker_host.model_inventory(root, root / "snapshot")
    assert swapped
    assert rows == [{
        "path": "member.bin", "size_bytes": len(b"trusted"),
        "sha256": hashlib.sha256(b"trusted").hexdigest(),
    }]
    assert (root / "snapshot/member.bin").read_bytes() == b"trusted"
    assert (alternate / "snapshot/member.bin").read_bytes() == b"hostile"


@pytest.mark.parametrize("kind", ["source_member", "artifact_symlink", "artifact_mode"])
def test_hostile_model_roots_refuse_before_model_calls(configured, kind):
    c, verifier, auth, clock, _ = configured
    helper.prepare(c, pushed_sources=verifier, authenticator=auth, clock=clock)
    selected = c["consumer"]["docker"]
    source, artifact = (
        Path(selected["source_host_root"]),
        Path(selected["artifact_host_root"]),
    )
    source.mkdir(mode=0o700)
    artifact.mkdir(mode=0o700)
    if kind == "source_member":
        (source / "unexpected").write_text("refuse")
    elif kind == "artifact_symlink":
        artifact.rmdir()
        artifact.symlink_to(source, target_is_directory=True)
    else:
        artifact.chmod(0o755)
    with pytest.raises(ValueError):
        helper.prepare_model(
            c, model_preparer=lambda **kwargs: pytest.fail("no model dispatch"),
            sdk_preflight=lambda: None,
        )


def test_model_sdk_failure_precedes_directories_and_claim(configured):
    c, verifier, auth, clock, _ = configured
    helper.prepare(c, pushed_sources=verifier, authenticator=auth, clock=clock)

    def missing_sdk():
        raise helper.ModelSnapshotPreparationError("SDK_ADMISSION")

    with pytest.raises(helper.ModelSnapshotPreparationError):
        helper.prepare_model(
            c, sdk_preflight=missing_sdk,
            model_preparer=lambda **kwargs: pytest.fail("no model dispatch"),
        )
    assert not (Path(c["output_root"]) / "model-preparation-attempt.json").exists()
    assert not Path(c["consumer"]["docker"]["source_host_root"]).exists()
    assert not Path(c["consumer"]["docker"]["artifact_host_root"]).exists()


def test_model_preflight_missing_sdk_is_closed(monkeypatch):
    import builtins
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ModuleNotFoundError("private path and credential text")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(helper.ModelSnapshotPreparationError) as caught:
        helper.model_preflight()
    assert caught.value.stage == "SDK_ADMISSION"
    assert str(caught.value) == "model preparation failed"
    assert caught.value.__suppress_context__ is True


def test_cli_model_preflight_checks_signatures_without_client_or_download(
    tmp_path, monkeypatch, capsys,
):
    import sys
    import types

    class HfApi:
        def __init__(self, *, endpoint, token):
            pytest.fail("client construction")

        def model_info(self, repo_id, *, revision, files_metadata, token):
            pytest.fail("metadata request")

    def snapshot_download(
        *, repo_id, revision, token, endpoint, local_dir, cache_dir,
        allow_patterns, max_workers,
    ):
        pytest.fail("download request")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(HfApi=HfApi, snapshot_download=snapshot_download),
    )
    config = tmp_path / "config.json"
    config.write_text("{}")
    assert helper.main(["model-preflight", str(config)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "model_sdk_contract": "compatible",
        "model_download": False,
        "provider_qualified": False,
    }
    assert set(tmp_path.iterdir()) == {config}


@pytest.mark.parametrize("stage", ["SDK_ADMISSION", "DOWNLOAD", "DESTINATION_VERIFICATION"])
def test_cli_retains_only_existing_closed_model_stage(tmp_path, monkeypatch, capsys, stage):
    config = tmp_path / "config.json"
    config.write_text("{}")

    def fail(_):
        raise helper.ModelSnapshotPreparationError(stage)

    monkeypatch.setattr(helper, "prepare_model", fail)
    assert helper.main(["prepare-model", str(config)]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "diagnostic_code": "docker_material_preparation_failed",
        "model_preparation_stage": stage,
    }


def test_cli_unknown_exception_does_not_leak_text_or_stage(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text("{}")

    def fail(_):
        error = RuntimeError("private path and credential text")
        error.stage = "private stage"
        raise error

    monkeypatch.setattr(helper, "prepare_model", fail)
    assert helper.main(["prepare-model", str(config)]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "diagnostic_code": "docker_material_preparation_failed",
    }


def test_actual_native_writer_cpu_filesystem_probe_and_host_read(tmp_path):
    import hashlib
    import tuner

    runtime = (
        Path(tuner.__file__).resolve().parent.parent / "Trainers/sft/runtime_v1.py"
    )
    output = tmp_path / "probe-output"
    output.mkdir(mode=0o700)
    config = tmp_path / "probe.json"
    config.write_text(
        json.dumps(
            {
                "runtime_file": str(runtime),
                "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
                "output_root": str(output),
            }
        )
    )
    result = helper.filesystem_probe(config)
    assert result["training_executed"] is result["gpu_qualified"] is False
    assert (output / "native-write-probe.json").read_bytes() == b'{"fixture":true}\n'
    assert result["sha256"] == hashlib.sha256(b'{"fixture":true}\n').hexdigest()
    with pytest.raises(ValueError):
        helper.filesystem_probe(config)


def staged_public_material(configured, *, nested=False):
    """Actual admitted material/bridge/stage, with a simulated Docker namespace."""
    import time
    import tuner
    from pathlib import PurePosixPath
    from synaptic_tuner.api.v1.planning import (
        ProviderPlanContextV1,
        ProviderPlanRef,
        TrainingPlan,
    )
    from synaptic_tuner.api.v1.results import TrainingRunRef
    from tuner.dataset_prep import LocalPreparedTrainingInputSource
    from tuner.runtime.dispatch import (
        WorkerControlLocationV1,
        CanonicalWorkloadFileLocationV1,
        build_source_worker_invocation,
        materialize_worker_bundle,
    )
    from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest
    from tuner.execution.providers.docker_provider_v1.local_source import (
        DockerLocalSourceSealsV1,
    )
    from tuner.execution.providers.docker_provider_v1.model import (
        DockerEffectIdentityV1,
        DockerSourceSealRequestV1,
        labels_for,
    )
    from tests.execution.providers.docker_provider_v1.test_local_source import sign

    c, verifier, auth, clock, _ = configured
    if nested:
        c["runtime_environment"]["SYNAPTIC_PROJECT_ROOT"] = "/source/layout/project"
    helper.prepare(c, pushed_sources=verifier, authenticator=auth, clock=clock)
    consumer = helper.launch.load_config(Path(c["output_root"]) / "consumer.json")
    p, rich, m, recipes, basis = helper.docker_host._components(consumer)
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1",
        p.provider,
        basis.basis_digest,
        p.descriptor.descriptor_digest,
        p.profile_digest,
    )
    public_plan = TrainingPlan(
        "synaptic-training-plan/v2",
        basis,
        ProviderPlanRef(context.provider_context_digest),
    )
    run = TrainingRunRef(m.run_id, m.planning_request.project_ref)
    bridge = helper.DockerTrainingPreparationBridgeV1(p)
    _, prepared = bridge.coordinator_prepared(
        plan=rich, material=m, coordinator_plan=public_plan, run=run, recipes=recipes
    )
    location = WorkerControlLocationV1(PurePosixPath("/source/control"))
    bundle = materialize_worker_bundle(
        build_source_worker_invocation(
            rich, location, CanonicalWorkloadFileLocationV1(location.control_root)
        )
    )
    closure = parse_offline_sft_worker_manifest(
        bundle.closure_manifest_bytes,
        source_ref=bundle.closure_manifest_source,
        manifest_path=Path(bundle.closure_manifest_runtime_path),
    )
    package = Path(tuner.__file__).resolve().parent.parent
    members = {
        row.path.as_posix(): (package / row.path).read_bytes()
        for row in closure.closure.members
    }
    roots = {
        k: Path(consumer["docker"][k + "_host_root"]) for k in ("source", "artifact")
    }
    for root in roots.values():
        root.mkdir(mode=0o700)
    (roots["artifact"] / "run").mkdir(mode=0o700)
    for name in ("state", "tracking", "cache", "tmp", "artifacts"):
        (roots["artifact"] / "run" / name).mkdir(mode=0o700)
    identity = helper.PreparedTrainingInputIdentity.from_dict(
        rich.resolved_config.to_dict()["dataset"]
    )
    from contextlib import closing

    with closing(
        LocalPreparedTrainingInputSource(
            Path(c["prepared_publication"]), identity
        ).open_lease()
    ) as lease:
        port = DockerLocalSourceSealsV1(
            training_plan=rich,
            prepared_plan=prepared,
            bundle=bundle,
            closure_members=members,
            prepared_lease=lease,
            source_root=roots["source"],
            source_container_root=PurePosixPath("/source"),
            artifact_root=roots["artifact"],
            artifact_container_root=PurePosixPath("/artifacts"),
            deadline=time.monotonic() + 120,
            sign_seal=sign,
            coordinator_plan=public_plan,
            coordinator_material=m,
            recipes=recipes,
        )
        request = DockerSourceSealRequestV1(
            DockerEffectIdentityV1("a" * 64, "stage-effect", "stage", prepared),
            p.roots.source_ref,
            m.planning_request.source_digest,
        )
        port.seal_read_only(request)
    labels = labels_for(
        DockerEffectIdentityV1("b" * 64, "submit-effect", "submit", prepared)
    )
    return port, labels, request, roots, bundle


def test_public_material_bridge_stage_and_actual_runtime_root_binding(configured):
    from copy import deepcopy
    from Trainers.sft import runtime_v1

    port, labels, request, roots, bundle = staged_public_material(configured)
    try:
        port.revalidate_for_submit(labels=labels, source_ref=request.source_ref)
        # Translate the actual compiled locations through declared Docker mounts.
        # This is a simulated namespace, not a rewritten signed workload receipt.
        document = deepcopy(json.loads(bundle.canonical_workload_bytes))
        original = document["execution_source"]["runtime"]["roots"]
        mapped = {}
        for name, value in original.items():
            mount = "source" if value.startswith("/source/") else "artifact"
            prefix = "/source/" if mount == "source" else "/artifacts/"
            mapped[name] = str(roots[mount] / value.removeprefix(prefix))
        document["execution_source"]["runtime"]["roots"] = mapped
        document["execution_source"]["runtime"]["capability_roots"]["writable"] = str(
            roots["artifact"]
        )
        env = {
            "SYNAPTIC_" + ("ARTIFACT" if k == "artifacts" else k.upper()) + "_ROOT": v
            for k, v in mapped.items()
        }
        actual = runtime_v1.bind_runtime_roots(
            document,
            env,
            engine_file=roots["source"] / "engine/Trainers/sft/runtime_v1.py",
        )
        assert all(Path(v).is_dir() for v in mapped.values())
        assert actual.project == roots["source"] / "project"
        assert list(actual.project.iterdir()) == []
    finally:
        port.close()


@pytest.mark.parametrize("mutation", ["mode", "replacement"])
def test_nested_project_ancestor_changes_refuse_submit(configured, mutation):
    from tuner.execution.providers.docker_provider_v1.model import DockerProviderError

    port, labels, request, roots, _ = staged_public_material(configured, nested=True)
    try:
        parent = roots["source"] / "layout"
        if mutation == "mode":
            parent.chmod(0o755)
        else:
            parent.rename(parent.with_name("old-layout"))
            parent.mkdir(mode=0o700)
            (parent / "project").mkdir(mode=0o700)
        with pytest.raises(DockerProviderError):
            port.revalidate_for_submit(labels=labels, source_ref=request.source_ref)
    finally:
        port.close()
