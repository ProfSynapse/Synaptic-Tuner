from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

import tuner.runtime as runtime_api
import tuner.runtime.dispatch as dispatch_module
from synaptic_tuner.api.v1.training import (
    ArtifactPolicy,
    CanonicalDocument,
    ResourceSpec,
    RuntimeSpec,
    TrainingPlan,
)
from tuner.cloud.runtime_layout import CloudRuntimeLayout, RuntimeMount
from tuner.runtime.dispatch import (
    CanonicalWorkloadBytesV1,
    CanonicalWorkloadFileLocationV1,
    CanonicalWorkloadFileV1,
    EngineDispatcher,
    ProcessResult,
    SubprocessRunner,
    WorkerInvocationV1,
    WorkerBundleMaterializationV1,
    WorkerControlLocationV1,
    build_dispatch_invocation,
    build_source_worker_invocation,
    build_worker_invocation,
    materialize_worker_invocation,
    materialize_worker_bundle,
)
from tuner.training.methods.sft import compile_sft_workload
from tests.training.test_sft_compilation import _config, _execution_source


def _plan() -> TrainingPlan:
    source = _execution_source()
    config = _config()
    workload = compile_sft_workload(resolved_config=config, execution_source=source)
    return TrainingPlan(
        execution_source=source,
        execution_context=CanonicalDocument.from_mapping(
            {"schema_version": "test-execution-context/v1"}
        ),
        resolved_config=config,
        workload=CanonicalDocument(workload.canonical_bytes.decode("utf-8")),
        runtime=RuntimeSpec(
            "example.invalid/trainer@sha256:" + "d" * 64,
            "e" * 64,
            "3.12.3",
        ),
        resources=ResourceSpec("gpu"),
        artifact_policy=ArtifactPolicy(),
    )


def _layout(tmp_path: Path) -> CloudRuntimeLayout:
    sources = tmp_path / "sources"
    writable = tmp_path / "writable"
    engine = sources / "engine"
    project = sources / "project"
    engine.mkdir(parents=True)
    entrypoint = engine / "Trainers" / "sft" / "runtime_v1.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# staged fixture\n", encoding="utf-8")
    project.mkdir()
    mounts = tuple(
        RuntimeMount(
            name,
            writable / name,
            PurePosixPath("/workspace/run/run-1") / name,
            False,
        )
        for name in ("artifacts", "state", "tracking", "cache", "tmp")
    )
    return CloudRuntimeLayout(
        engine=RuntimeMount("engine", engine, PurePosixPath("/workspace/engine"), True),
        project=RuntimeMount(
            "project", project, PurePosixPath("/workspace/project"), True
        ),
        writable=mounts,
    )


def _control() -> WorkerControlLocationV1:
    return WorkerControlLocationV1(PurePosixPath("/source/control"))


def test_factory_derives_closed_byte_invocation_from_the_exact_plan(
    tmp_path: Path,
) -> None:
    plan = _plan()
    layout = _layout(tmp_path)
    worker = build_worker_invocation(plan, layout, _control())
    invocation = materialize_worker_invocation(worker)

    assert worker.schema_version == "synaptic-worker-invocation/v1"
    assert worker.plan_fingerprint == plan.fingerprint
    assert worker.entrypoint == PurePosixPath("Trainers/sft/runtime_v1.py")
    assert worker.interpreter == "/usr/local/bin/python3.12"
    assert tuple(name for name, _ in worker.roots) == (
        "engine",
        "project",
        "artifacts",
        "state",
        "tracking",
        "cache",
        "tmp",
    )
    assert invocation.argv == (
        "/usr/local/bin/python3.12",
        "/workspace/engine/Trainers/sft/runtime_v1.py",
        "--canonical-workload-stdin",
    )
    assert invocation.stdin == plan.workload.canonical_json.encode("utf-8")
    assert invocation.cwd == PurePosixPath("/workspace/run/run-1/tmp")
    assert invocation.environment_map["SYNAPTIC_WORKLOAD_FINGERPRINT"] == (
        worker.workload_fingerprint
    )
    assert invocation.environment_map["HF_HUB_OFFLINE"] == "1"
    assert invocation.environment_map["TRANSFORMERS_OFFLINE"] == "1"
    assert invocation.environment_map["SYNAPTIC_MODEL_SNAPSHOT"] == (
        "/workspace/run/run-1/cache/model/"
        "models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/" + "b" * 40
    )


def test_worker_bundle_is_factory_issued_durable_and_host_path_free(
    tmp_path: Path,
) -> None:
    plan = _plan()
    worker = build_worker_invocation(plan, _layout(tmp_path), _control())
    bundle = materialize_worker_bundle(worker)
    projection = json.loads(bundle.canonical_projection_bytes)

    assert type(bundle) is WorkerBundleMaterializationV1
    assert bundle.plan_fingerprint == plan.fingerprint
    assert bundle.workload_fingerprint == worker.workload_fingerprint
    assert bundle.canonical_workload_bytes == plan.workload.canonical_json.encode(
        "utf-8"
    )
    assert bundle.workload_byte_count == len(bundle.canonical_workload_bytes)
    assert bundle.workload_sha256 == hashlib.sha256(
        bundle.canonical_workload_bytes
    ).hexdigest()
    assert bundle.projection_sha256 == hashlib.sha256(
        bundle.canonical_projection_bytes
    ).hexdigest()
    assert bundle.dispatch == materialize_worker_invocation(worker)
    assert projection["plan_fingerprint"] == plan.fingerprint
    assert base64.b64decode(projection["workload"]["payload_base64"]) == (
        bundle.canonical_workload_bytes
    )
    serialized = bundle.canonical_projection_bytes.decode("utf-8").lower()
    assert str(tmp_path).lower().replace("\\", "/") not in serialized
    assert all(term not in serialized for term in ("host_path", "docker", "provider"))


def test_dispatcher_passes_one_canonical_invocation_to_runner(
    tmp_path: Path,
) -> None:
    plan = _plan()
    layout = _layout(tmp_path)
    calls = []

    class Runner:
        def run(self, invocation):
            calls.append(invocation)
            return ProcessResult(0, "ok", "")

    result = EngineDispatcher(Runner()).dispatch(plan, layout, _control())
    assert result.exit_code == 0
    assert calls == [build_dispatch_invocation(plan, layout, _control())]


def test_factory_rejects_redirected_staged_entrypoint(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    entrypoint = layout.engine.source / "Trainers" / "sft" / "runtime_v1.py"
    target = layout.engine.source / "actual.py"
    target.write_text("# actual\n", encoding="utf-8")
    entrypoint.unlink()
    try:
        entrypoint.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(ValueError, match="redirected"):
        build_worker_invocation(_plan(), layout, _control())


def test_subprocess_runner_removes_hostile_python_environment(
    tmp_path: Path, monkeypatch
) -> None:
    invocation = build_dispatch_invocation(_plan(), _layout(tmp_path), _control())
    observed = {}

    def fake_run(*args, **kwargs):
        observed.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(dispatch_module.subprocess, "run", fake_run)
    SubprocessRunner(
        base_environment={
            "PATH": "fixture",
            "PYTHONPATH": "hostile",
            "PYTHONHOME": "hostile-home",
            "PYTHONUSERBASE": "hostile-user",
        }
    ).run(invocation)

    assert "PYTHONPATH" not in observed
    assert observed["PYTHONNOUSERSITE"] == "1"
    assert observed["PYTHONSAFEPATH"] == "1"
    assert "PYTHONHOME" not in observed
    assert "PYTHONUSERBASE" not in observed


def test_file_location_derives_all_authenticated_file_fields(
    tmp_path: Path,
) -> None:
    plan = _plan()
    worker = build_worker_invocation(
        plan,
        _layout(tmp_path),
        _control(),
        CanonicalWorkloadFileLocationV1(PurePosixPath("/control/sealed")),
    )
    invocation = materialize_worker_invocation(worker)
    bundle = materialize_worker_bundle(worker)

    assert type(worker.transport) is CanonicalWorkloadFileV1
    assert worker.transport.path == PurePosixPath("/control/sealed/workload.json")
    assert worker.transport.byte_count == len(
        plan.workload.canonical_json.encode("utf-8")
    )
    assert invocation.stdin == b""
    assert bundle.canonical_workload_bytes == plan.workload.canonical_json.encode(
        "utf-8"
    )
    assert bundle.dispatch == invocation
    assert invocation.argv[2:6] == (
        "--canonical-workload-file",
        "/control/sealed/workload.json",
        "--canonical-workload-control-root",
        "/control/sealed",
    )


def test_source_worker_preserves_byte_transport_as_an_active_contract() -> None:
    worker = build_source_worker_invocation(_plan(), _control())
    invocation = materialize_worker_invocation(worker)

    assert type(worker.transport) is CanonicalWorkloadBytesV1
    assert worker._file_location is None
    assert invocation.argv[-1] == "--canonical-workload-stdin"
    assert invocation.stdin == _plan().workload.canonical_json.encode("utf-8")


def test_source_worker_authentically_reconstructs_matching_file_transport() -> None:
    location = CanonicalWorkloadFileLocationV1(_control().control_root)
    worker = build_source_worker_invocation(_plan(), _control(), location)
    first = materialize_worker_bundle(worker)
    second = materialize_worker_bundle(worker)

    assert type(worker.transport) is CanonicalWorkloadFileV1
    assert worker._file_location == location
    assert worker.transport.path == PurePosixPath("/source/control/workload.json")
    assert first.dispatch == second.dispatch
    assert first.dispatch.stdin == b""
    assert first.dispatch.argv[2:6] == (
        "--canonical-workload-file",
        "/source/control/workload.json",
        "--canonical-workload-control-root",
        "/source/control",
    )
    assert all("/workspace/control" not in value for value in first.dispatch.argv)


def test_source_worker_rejects_wrong_or_mismatched_file_location() -> None:
    with pytest.raises(TypeError, match="exact CanonicalWorkloadFileLocationV1"):
        build_source_worker_invocation(_plan(), _control(), object())
    with pytest.raises(ValueError, match="control roots must match"):
        build_source_worker_invocation(
            _plan(),
            _control(),
            CanonicalWorkloadFileLocationV1(PurePosixPath("/other/control")),
        )


def test_source_worker_reconstruction_rejects_hidden_file_location_tamper() -> None:
    location = CanonicalWorkloadFileLocationV1(_control().control_root)
    worker = build_source_worker_invocation(_plan(), _control(), location)
    object.__setattr__(
        worker,
        "_file_location",
        CanonicalWorkloadFileLocationV1(PurePosixPath("/other/control")),
    )

    with pytest.raises(ValueError, match="control roots must match"):
        materialize_worker_bundle(worker)


@pytest.mark.parametrize("mutation", ("workload", "config", "source", "artifacts"))
def test_factory_recompilation_rejects_plan_component_swaps(
    tmp_path: Path, mutation: str
) -> None:
    plan = _plan()
    if mutation == "workload":
        document = plan.workload.to_dict()
        document["run_id"] = "foreign-run"
        hostile = replace(plan, workload=CanonicalDocument.from_mapping(document))
    elif mutation == "config":
        config = plan.resolved_config.to_dict()
        config["model"]["revision"] = "f" * 40
        hostile = replace(plan, resolved_config=CanonicalDocument.from_mapping(config))
    elif mutation == "source":
        hostile = replace(
            plan, execution_source=_execution_source(engine_commit="f" * 40)
        )
    else:
        hostile = replace(
            plan, artifact_policy=ArtifactPolicy(required_kinds=("foreign_role",))
        )

    with pytest.raises(ValueError, match="differ"):
        build_worker_invocation(hostile, _layout(tmp_path), _control())


def test_factory_rejects_ambiguous_plan_subclasses(tmp_path: Path) -> None:
    class ForeignPlan(TrainingPlan):
        pass

    plan = _plan()
    foreign = ForeignPlan(
        plan.execution_source,
        plan.execution_context,
        plan.resolved_config,
        plan.workload,
        plan.runtime,
        plan.resources,
        plan.artifact_policy,
    )
    with pytest.raises(TypeError, match="exact canonical TrainingPlan"):
        build_worker_invocation(foreign, _layout(tmp_path), _control())


def test_invocations_and_transports_reject_direct_construction() -> None:
    for constructor in (
        CanonicalWorkloadBytesV1,
        CanonicalWorkloadFileV1,
        WorkerInvocationV1,
        WorkerBundleMaterializationV1,
    ):
        with pytest.raises(TypeError, match="factory-issued"):
            constructor()


def test_factory_accepts_no_caller_authored_identity_fields(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        build_worker_invocation(_plan(), _layout(tmp_path), _control(), plan_fingerprint="f" * 64)


def test_runtime_package_exports_only_the_new_worker_contract() -> None:
    assert runtime_api.WorkerInvocationV1 is WorkerInvocationV1
    assert runtime_api.WorkerBundleMaterializationV1 is WorkerBundleMaterializationV1
    assert runtime_api.materialize_worker_bundle is materialize_worker_bundle
    assert runtime_api.CanonicalWorkloadFileLocationV1 is (
        CanonicalWorkloadFileLocationV1
    )
    with pytest.raises(AttributeError):
        runtime_api.DispatchSpec
    assert not hasattr(runtime_api, "build_worker_environment")
    assert "build_worker_environment" not in runtime_api.__all__
    assert not hasattr(dispatch_module, "build_worker_environment")
    assert "build_worker_environment" not in dispatch_module.__all__


def test_materializer_rejects_replace_and_object_tampering(tmp_path: Path) -> None:
    worker = build_worker_invocation(_plan(), _layout(tmp_path), _control())
    with pytest.raises(TypeError):
        replace(worker, plan_fingerprint="f" * 64)

    object.__setattr__(worker, "plan_fingerprint", "f" * 64)
    with pytest.raises(ValueError, match="authentic factory issuance"):
        materialize_worker_invocation(worker)


def test_materializer_rejects_transport_tampering(tmp_path: Path) -> None:
    worker = build_worker_invocation(_plan(), _layout(tmp_path), _control())
    object.__setattr__(worker.transport, "sha256", "f" * 64)
    with pytest.raises(ValueError, match="authentic factory issuance"):
        materialize_worker_invocation(worker)


def test_materializer_rejects_payload_tampering_and_fabrication(
    tmp_path: Path,
) -> None:
    worker = build_worker_invocation(_plan(), _layout(tmp_path), _control())
    object.__setattr__(worker.transport, "payload", b"{}")
    with pytest.raises(ValueError, match="authentic factory issuance"):
        materialize_worker_invocation(worker)

    fabricated = object.__new__(WorkerInvocationV1)
    object.__setattr__(fabricated, "transport", worker.transport)
    with pytest.raises(ValueError, match="authentic factory issuance"):
        materialize_worker_invocation(fabricated)


def test_factory_rejects_root_ancestry_overlap_after_layout_tamper(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    object.__setattr__(
        layout.project, "target", PurePosixPath("/workspace/engine/project")
    )
    with pytest.raises(ValueError, match="overlap"):
        build_worker_invocation(_plan(), layout, _control())


@pytest.mark.parametrize(
    "control_root",
    (
        PurePosixPath("/workspace"),
        PurePosixPath("/workspace/engine/control"),
        PurePosixPath("/workspace/run/run-1/tmp/control"),
    ),
)
def test_factory_rejects_file_control_root_ancestry_collisions(
    tmp_path: Path, control_root: PurePosixPath
) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        build_worker_invocation(
            _plan(),
            _layout(tmp_path),
            _control(),
            CanonicalWorkloadFileLocationV1(control_root),
        )


def test_factory_rejects_layout_roots_outside_the_workload_lock(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    object.__setattr__(layout.engine, "target", PurePosixPath("/other/engine"))
    with pytest.raises(ValueError, match="compiled workload roots"):
        build_worker_invocation(_plan(), layout, _control())
