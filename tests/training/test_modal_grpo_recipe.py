"""Synthetic host-owned profile/runtime, real CPU preparation/public planning only."""

from argparse import Namespace
from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import yaml

from synaptic_tuner.api.v1.training_sources import (
    LocalTrainingInputPathV1,
    TrainingPreparationConfigV1,
)
from tuner.training.grpo_preparation import (
    GRPONormalizerConfigV1,
    GRPOTrainingInputNormalizerV1,
)
from tuner.training.modal_recipe import plan_modal_recipe, resolve_modal_recipe_build
from tuner.training import modal_local_preparation as local
from tuner.training import modal_standalone_runner as runner
from tuner.training.modal_host_prepared_copy import (
    modal_copied_preparation_ports,
    stage_published_modal_dataset,
)
from tuner.training.input_preparation import (
    TrainingInputPreparationServiceV1,
    PosixPrivatePreparedRootAuthorityV1,
)
import tuner.runtime_profiles as profile_module
from tuner.runtime_profiles import load_runtime_profile
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV2
from tests.contract.test_grpo_training_input_v1 import document
from tests.runtime.test_packaged_runtime_releases import _release_values, _provider

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(profile_module.__file__).resolve().parent.parent


def setup(tmp_path):
    project, engine = tmp_path / "project", tmp_path / "engine"
    project.mkdir(mode=0o700)
    engine.mkdir(mode=0o700)
    (project / ".tracking").mkdir(mode=0o700)
    output = project / ".tracking/datasets"
    output.mkdir(mode=0o700)
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    raw = b'{"prompt":"Say OK","answer":"OK"}\n'
    (source / "rows.jsonl").write_bytes(raw)
    (source / "rows.jsonl").chmod(0o600)
    normalized = GRPOTrainingInputNormalizerV1(allowed_source_roots=(source,)).prepare(
        LocalTrainingInputPathV1(source),
        config=GRPONormalizerConfigV1("rows.jsonl", "prompt"),
        output_root=output,
    )
    path = normalized.path
    import json

    manifest = json.loads((path / "manifest.json").read_bytes())
    declaration = document()
    declaration["dataset"]["ref"] = "prepared://sha256/" + manifest["semantic_digest"]
    profiles = engine / "Trainers/runtime_profiles"
    profiles.mkdir(parents=True)
    original = yaml.safe_load(
        (ASSETS / "Trainers/runtime_profiles/qwen35-sft-v1.yaml").read_text()
    )
    original["name"] = "fixture-grpo"
    original["compatibility"] = {
        "methods": ["grpo"],
        "models": [{"name": "model/example", "revisions": ["a" * 40]}],
    }
    original["runtime"]["packaged_build_profile"] = "fixture_grpo"
    shutil.copy2(
        ASSETS / "Trainers/runtime_profiles/qwen35-sft-v1.inventory.json",
        profiles / "qwen35-sft-v1.inventory.json",
    )
    (profiles / "fixture-grpo.yaml").write_text(yaml.safe_dump(original))
    build_dir = engine / "Trainers/image_profiles/fixture_grpo"
    build_dir.mkdir(parents=True)
    build = yaml.safe_load(
        (
            ROOT
            / "Trainers/image_profiles/qwen35_4b_packaged_sft_3360351c/profile.yaml"
        ).read_text()
    )
    build["name"] = "fixture_grpo"
    build["packaged_runtime"]["bootstrap"] = [
        {
            "distribution": "fixture-bootstrap",
            "version": "1.0.0",
            "filename": "fixture_bootstrap-1.0.0-py3-none-any.whl",
            "sha256": hashlib.sha256(b"fixture-bootstrap").hexdigest(),
        }
    ]  # Synthetic bytes; never installed remotely.
    caps = build["packaged_runtime"]["capabilities"]
    caps["compatibility"] = {
        "methods": ["grpo"],
        "models": [{"ref": "model/example", "revision": "a" * 40}],
        "dataset_formats": ["synaptic-grpo-prompts/v1"],
    }
    caps["contracts"] = {
        "workload_schema": "synaptic-packaged-grpo-workload/v1",
        "prepared_input_schema": "synaptic-prepared-training-input/v1",
        "artifact_contract_schema": "synaptic-grpo-artifacts/v1",
    }
    (build_dir / "profile.yaml").write_text(yaml.safe_dump(build))
    from tuner.execution.providers.modal import runtime_build

    inspector = engine / "tuner/runtime/modal_build_inspector.py"
    inspector.parent.mkdir(parents=True)
    inspector.write_bytes(
        (
            Path(runtime_build.__file__).resolve().parents[4]
            / "tuner/runtime/modal_build_inspector.py"
        ).read_bytes()
    )
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(engine), *args],
            check=True,
            capture_output=True,
            timeout=30,
        )
    config = {
        "name": "fixture",
        "target": "cloud",
        "provider": "modal",
        "method": "grpo",
        "run": {"method": "grpo", "dry_run": False},
        "job": {
            "runtime_profile": "fixture-grpo",
            "accelerator": "L40S",
            "accelerator_count": 1,
            "timeout_seconds": 120,
            "maximum_cost_minor_units": 100,
        },
        "dataset": {
            "local_file": str((path / "dataset.jsonl").relative_to(project)),
            "schema_version": "synaptic-grpo-prompts/v1",
            "expected_digest": manifest["semantic_digest"],
            "expected_row_count": 1,
        },
        "training_input": declaration,
        "limits": {"child_max_seconds": 60},
    }
    recipe = project / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(config))
    return project, engine, profiles, path, recipe, config, raw


def test_real_recipe_core_preplan_is_honest_and_exact(tmp_path):
    project, engine, profiles, path, recipe, _, raw = setup(tmp_path)
    plan = plan_modal_recipe(
        recipe, project_root=project, profiles_root=profiles, source_root=engine
    )
    doc = plan.to_dict()
    assert doc["method"] == "grpo" and doc["row_count"] == 1
    assert not set(doc) & {
        "workload_digest",
        "configuration_digest",
        "split_counts",
        "runtime_release",
    }
    assert doc["execution_status"].startswith("unsupported")
    assert plan.prepared_identity.content_digest == hashlib.sha256(raw).hexdigest()
    assert plan.recipe.training_input(plan.prepared_identity.ref).method.value == "grpo"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["run"].update(dry_run=0),
        lambda d: d["dataset"].update(expected_row_count=True),
        lambda d: d["dataset"].update(
            expected_split_counts={"train": 1, "validation": 0}
        ),
        lambda d: d["training_input"].update(method="sft"),
        lambda d: d.update(post_training={}),
        lambda d: d["job"].update(runtime_profile="missing"),
        lambda d: d["training_input"]["model"].update(revision="b" * 40),
        lambda d: d["limits"].update(child_max_seconds=True),
        lambda d: d["training_input"]["artifacts"].update(retain_checkpoints=True),
    ],
)
def test_real_yaml_controls_reject_before_provider(tmp_path, mutate, monkeypatch):
    project, engine, profiles, _, recipe, config, _ = setup(tmp_path)
    mutate(config)
    recipe.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        runner, "open_modal_host_scope", lambda **kw: pytest.fail("no provider")
    )
    with pytest.raises((ValueError, TypeError, OSError)):
        plan_modal_recipe(
            recipe, project_root=project, profiles_root=profiles, source_root=engine
        )


def test_changed_reward_column_and_mixed_build_methods_reject(tmp_path):
    project, engine, profiles, _, recipe, config, _ = setup(tmp_path)
    config["training_input"]["hyperparameters"]["prompt_column"] = "missing"
    recipe.write_text(yaml.safe_dump(config))
    with pytest.raises((ValueError, RuntimeError)):
        plan_modal_recipe(
            recipe, project_root=project, profiles_root=profiles, source_root=engine
        )
    buildpath = engine / "Trainers/image_profiles/fixture_grpo/profile.yaml"
    doc = yaml.safe_load(buildpath.read_text())
    doc["packaged_runtime"]["capabilities"]["compatibility"]["methods"] = [
        "sft",
        "grpo",
    ]
    buildpath.write_text(yaml.safe_dump(doc))
    for args in (
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "mixed",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(engine), *args], check=True, capture_output=True
        )
    with pytest.raises(ValueError, match="build compatibility differs"):
        resolve_modal_recipe_build(
            load_runtime_profile("fixture-grpo", profiles),
            profiles,
            "model/example",
            "a" * 40,
            method="grpo",
            prepared_format="synaptic-grpo-prompts/v1",
            source_root=engine,
        )


def local_inputs(tmp_path, engine, profiles, plan, monkeypatch):
    profile = load_runtime_profile("fixture-grpo", profiles)
    buildpath = profile.modal_build_profile_path(profiles.parent / "image_profiles")
    monkeypatch.setattr(local, "validate_local_launcher", lambda: None)
    bootstrap = tmp_path / "fixture_bootstrap-1.0.0-py3-none-any.whl"
    bootstrap.write_bytes(b"fixture-bootstrap")
    bootstrap.chmod(0o600)
    monkeypatch.setattr(
        local, "bootstrap_paths", lambda *a, **kw: {bootstrap.name: bootstrap}
    )

    def built(root, staged, **kwargs):
        assert root == engine
        path = staged / "synaptic_tuner-1.1.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "tuner/runtime/modal_build_inspector.py",
                (engine / "tuner/runtime/modal_build_inspector.py").read_bytes(),
            )
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(local, "prepare_current_source_wheel", built)
    holder = local.prepare_local_inputs(
        buildpath,
        root=tmp_path / "cache",
        expected_intent=plan.runtime_material_intent_digest,
        source_root=engine,
    )
    values = _release_values(
        worker_entrypoint="tuner.runtime.packaged_training_worker:main",
        compatible_methods=("grpo",),
        compatible_models=(("model/example", "a" * 40),),
        compatible_dataset_formats=("synaptic-grpo-prompts/v1",),
        workload_schema="synaptic-packaged-grpo-workload/v1",
        artifact_contract_schema="synaptic-grpo-artifacts/v1",
    )
    values.pop("image_ref")
    values.pop("image_digest")
    values["material"] = holder.layout[5]
    values["package_digest"] = holder.wheel_sha256
    values["python_version"] = "3.11.14"
    values["python_executable"] = holder.layout[4]["python_executable"]
    return profile, holder, PackagedTrainingRuntimeReleaseV2.build(**values)


def test_private_copy_preparation_final_hook_and_public_plan_without_start(
    tmp_path, monkeypatch
):
    from synaptic_tuner.api.v1.training_facade import TrainingAPI, TrainingRequest
    from synaptic_tuner.api.v1.providers import (
        ProviderCapabilities,
        ProviderDescriptor,
        ProviderRef,
    )
    from synaptic_tuner.api.v1.results import TrainingRunRef
    from synaptic_tuner.api.v1.reference.repositories import DurablePlanningStoreV1
    from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
    from tuner.training.modal_host_requests import (
        ModalPackagedRequestsV1,
        ModalPackagedPlanPortsV1,
    )
    from tuner.project.context import ProjectContext
    from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
    from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
    from tuner.execution.foundation_v2.executors import ExecutorDescriptorV1
    from tuner.execution.foundation_v2.references import ExecutionScopeV1
    from tuner.training.coordinator_service import CoordinatorTrainingService

    project, engine, profiles, path, recipe, _, raw = setup(tmp_path)
    plan = plan_modal_recipe(
        recipe, project_root=project, profiles_root=profiles, source_root=engine
    )
    private = tmp_path / "private"
    copied = stage_published_modal_dataset(
        path,
        private_root=private,
        expected_dataset_digest=plan.prepared_identity.revision,
        expected_content_digest=plan.prepared_identity.content_digest,
        prepared_format=plan.prepared_identity.format,
    )
    normalizer, config, verifier = modal_copied_preparation_ports(
        private_root=private, identity=plan.prepared_identity
    )
    preparation = TrainingInputPreparationServiceV1(
        prepared_root=private,
        normalizers={config.normalizer_ref: normalizer},
        format_verifiers={verifier.format: verifier},
        root_authority=PosixPrivatePreparedRootAuthorityV1(),
    )
    prepared = preparation.prepare(
        LocalTrainingInputPathV1(copied),
        TrainingPreparationConfigV1(
            "request", "project", plan.recipe.declaration.canonical_json(), config
        ),
    )
    assert prepared.prepared.identity == plan.prepared_identity
    shutil.rmtree(path)
    assert copied.read_bytes() == raw
    profile, holder, release = local_inputs(
        tmp_path, engine, profiles, plan, monkeypatch
    )
    try:
        components, workload = runner.bind_modal_grpo_recipe_material(
            plan=plan,
            identity=prepared.prepared.identity,
            run_id="run-grpo",
            release=release,
            provider_binding=_provider(release),
            profile=profile,
            local_inputs=holder,
        )
        clock = SimpleNamespace(now=lambda: "2026-10-14T00:00:00Z")
        request = TrainingRequest(
            "request", "project", prepared.prepared.request.canonical_json
        )
        bridge = ModalPackagedRequestsV1(
            prepared_request=request,
            run=TrainingRunRef("run-grpo", "project"),
            components=components,
            context=ProjectContext.host(
                project_root=project, engine_root=engine, manifest_path=None
            ),
        )
        provider = ProviderRef("modal", "fixture-grpo")
        descriptor = ProviderDescriptor(
            "synaptic-provider-descriptor/v1",
            "modal",
            "Modal",
            "1.0.0",
            ProviderCapabilities(True, True, True, True, True, False),
        )
        binding = ProviderExecutionBindingV1(
            provider,
            descriptor.descriptor_digest,
            "a" * 64,
            ExecutionScopeV1("account", "namespace"),
            ExecutorDescriptorV1("modal", "packaged", "1.0.0"),
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "e" * 64,
        )
        planning = ModalPackagedPlanPortsV1(
            provider=provider,
            descriptor=descriptor,
            binding=binding,
            clock=clock,
            maximum_cost_minor_units=100,
        )
        db = tmp_path / "workflow.sqlite3"
        with SQLiteDurableStoreV1(db) as records:
            service = CoordinatorTrainingService(
                loader=bridge,
                resolver=bridge,
                planning=planning,
                planning_store=DurablePlanningStoreV1(records),
                coordinator=object.__new__(TrainingCoordinatorV1),
                clock=clock,
                input_preparation=preparation,
            )
            api = TrainingAPI(service, clock=clock)
            loaded = api.load(request.canonical_json)
            resolved = api.resolve(loaded)
            final_plan = api.plan(resolved, provider)
            assert final_plan.basis.workload_digest == workload.fingerprint
            assert api.preflight(final_plan).binds(final_plan)
            assert bridge.for_plan(final_plan).run_id == "run-grpo"
        recipe.unlink()
        shutil.rmtree(engine)
        with SQLiteDurableStoreV1(db) as records:
            assert (
                DurablePlanningStoreV1(records).get_plan(final_plan.plan_fingerprint)
                == final_plan
            )
    finally:
        holder.close()


def test_automatic_paths_reject_before_local_or_provider_bootstrap(
    tmp_path, monkeypatch
):
    project, engine, profiles, _, recipe, _, _ = setup(tmp_path)
    plan = plan_modal_recipe(
        recipe, project_root=project, profiles_root=profiles, source_root=engine
    )
    monkeypatch.setattr(
        runner,
        "prepare_modal_standalone_local",
        lambda **kw: pytest.fail("no preparation"),
    )
    for function in (
        runner.run_modal_standalone_job,
        runner.qualify_modal_standalone_job,
    ):
        with pytest.raises(
            runner.ModalStandaloneRunUnavailable,
            match="modal_grpo_dispatch_unsupported",
        ):
            function(
                plan=plan,
                context=object(),
                modal_profile="fixture",
                modal_environment="fixture",
            )


def test_final_hook_rejects_lost_custody_wrong_profile_and_changed_build_material(
    tmp_path, monkeypatch
):
    from tuner.runtime.releases import (
        _domain_digest,
        PACKAGED_RUNTIME_RELEASE_V2_SCHEMA,
    )
    from tuner.execution.foundation_v2.canonical import canonical_bytes

    project, engine, profiles, _, recipe, _, _ = setup(tmp_path)
    plan = plan_modal_recipe(
        recipe, project_root=project, profiles_root=profiles, source_root=engine
    )
    profile, holder, release = local_inputs(
        tmp_path, engine, profiles, plan, monkeypatch
    )
    args = dict(
        plan=plan,
        identity=plan.prepared_identity,
        run_id="run-grpo",
        release=release,
        provider_binding=_provider(release),
        profile=profile,
        local_inputs=holder,
    )
    try:
        with pytest.raises(ValueError, match="profile or local custody"):
            runner.bind_modal_grpo_recipe_material(
                **{**args, "profile": replace(profile, profile_sha256="f" * 64)}
            )
        with pytest.raises(ValueError, match="profile or local custody"):
            runner.bind_modal_grpo_recipe_material(**{**args, "local_inputs": object()})
        changed = release.to_dict()
        changed["material"]["build_inputs"]["wheels"][0]["sha256"] = "e" * 64
        changed["material"]["build_inputs_digest"] = hashlib.sha256(
            canonical_bytes(changed["material"]["build_inputs"])
        ).hexdigest()
        changed.pop("manifest_digest")
        changed["manifest_digest"] = _domain_digest(
            PACKAGED_RUNTIME_RELEASE_V2_SCHEMA, changed
        )
        wrong = PackagedTrainingRuntimeReleaseV2.from_dict(changed)
        with pytest.raises(ValueError, match="material differs"):
            runner.bind_modal_grpo_recipe_material(
                **{**args, "release": wrong, "provider_binding": _provider(wrong)}
            )
        holder.close()
        with pytest.raises(local.ModalLocalPreparationUnavailable):
            runner.bind_modal_grpo_recipe_material(**args)
    finally:
        holder.close()


def test_snapshot_validator_receives_original_bytes_and_cannot_relabel_manifest(
    tmp_path,
):
    from tuner.training.modal_host_prepared_copy import (
        inspect_mounted_prepared_dataset,
        ModalPreparedCopyUnavailable,
    )

    _, _, _, path, _, _, raw = setup(tmp_path)
    observed = []

    def validate(manifest, data):
        observed.append(data)
        manifest["semantic_digest"] = "f" * 64

    manifest = inspect_mounted_prepared_dataset(
        path, prepared_format="synaptic-grpo-prompts/v1", snapshot_validator=validate
    )
    assert observed == [raw] and manifest["semantic_digest"] != "f" * 64

    def change(manifest, data):
        (path / "dataset.jsonl").write_bytes(data + b" ")

    with pytest.raises(ModalPreparedCopyUnavailable):
        inspect_mounted_prepared_dataset(
            path, prepared_format="synaptic-grpo-prompts/v1", snapshot_validator=change
        )


def test_handler_core_plan_and_execution_gates_without_bootstrap(tmp_path, monkeypatch):
    from tuner.handlers.modal_job_config_handler import ModalJobConfigHandler
    from tuner.project.context import ProjectContext

    project, engine, _, _, recipe, _, _ = setup(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    context = ProjectContext.host(
        project_root=project, engine_root=engine, invocation_cwd=project
    )
    monkeypatch.setattr(
        ModalJobConfigHandler, "_execute", lambda *a, **kw: pytest.fail("no execution")
    )
    monkeypatch.setattr(
        ModalJobConfigHandler,
        "_qualify",
        lambda *a, **kw: pytest.fail("no qualification"),
    )
    monkeypatch.setattr(
        ModalJobConfigHandler,
        "_quote",
        lambda *a, **kw: pytest.fail("no provider quote"),
    )
    for flag in ("plan", "qualify", "execute", "quote"):
        args = Namespace(
            job_config=str(recipe),
            json=True,
            **({flag: True} if flag != "execute" else {}),
        )
        handler = ModalJobConfigHandler(args, context=context)
        outputs = []
        errors = []
        monkeypatch.setattr(handler, "output", lambda value: outputs.append(value))
        monkeypatch.setattr(
            handler, "output_error", lambda message, **kw: errors.append(kw)
        )
        code = handler.handle()
        if flag == "plan":
            assert code == 0 and outputs[0]["method"] == "grpo" and not errors
        else:
            assert (
                code == 2
                and errors[-1]["code"] == "MODAL_GRPO_DISPATCH_UNSUPPORTED"
                and not outputs
            )


def test_grpo_profile_load_is_lightweight_in_fresh_process(tmp_path):
    _, _, profiles, _, _, _, _ = setup(tmp_path)
    code = f"""import sys
sys.path.insert(0,{str(ROOT)!r})
from tuner.runtime_profiles import load_runtime_profile
from pathlib import Path
value=load_runtime_profile('fixture-grpo',Path({str(profiles)!r}))
assert value.methods==('grpo',)
assert not any(name.split('.')[0] in ('torch','transformers','unsloth','modal','huggingface_hub') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code], check=True, capture_output=True, timeout=15
    )


def callback_publication(tmp_path, prepared_format):
    """Real existing format parsers include immutable MappingProxyType metadata."""
    from tuner.dataset_prep import prepare_dataset_v2
    from tuner.dataset_prep.conversation_messages import publish_conversation_dataset
    from tests.dataset_prep.test_context_messages_v2 import _config
    from tests.dataset_prep.test_conversation_messages import build

    output = tmp_path / "datasets"
    if prepared_format == "syntunia-sft-row/v2":
        _, _, _, config = _config(tmp_path / "source")
        published = prepare_dataset_v2(config, output)
    else:
        output.mkdir(mode=0o700)
        published = publish_conversation_dataset(output, build())
    return published.path, published.semantic_identity


@pytest.mark.parametrize(
    "prepared_format", ["syntunia-sft-row/v2", "synaptic-conversation-sft-row/v1"]
)
@pytest.mark.parametrize("mutate", [False, True])
def test_all_supported_snapshot_callbacks_preserve_authoritative_record(
    tmp_path, prepared_format, mutate, monkeypatch
):
    from tuner.training.modal_host_prepared_copy import inspect_mounted_prepared_dataset

    path, original = callback_publication(tmp_path, prepared_format)
    raw = (path / "dataset.jsonl").read_bytes()
    import tuner.training.modal_host_prepared_copy as copying

    reads = []
    original_read = copying._read_member

    def observed_read(*args):
        reads.append(args[1])
        return original_read(*args)

    monkeypatch.setattr(copying, "_read_member", observed_read)
    received = []

    def validate(record, data):
        received.append((record, data))
        if mutate:
            object.__setattr__(record.semantic_identity, "dataset_digest", "f" * 64)

    result = inspect_mounted_prepared_dataset(
        path, prepared_format=prepared_format, snapshot_validator=validate
    )
    assert reads == ["manifest.json", "dataset.jsonl", "manifest.json", "dataset.jsonl"]
    assert received[0][1] == raw
    assert result is not received[0][0] and result.semantic_identity == original
    assert (path / "dataset.jsonl").read_bytes() == raw
    assert result.semantic_identity.dataset_digest != "f" * 64


@pytest.mark.parametrize(
    "prepared_format", ["syntunia-sft-row/v2", "synaptic-conversation-sft-row/v1"]
)
def test_all_supported_snapshot_callbacks_keep_final_read_checks(
    tmp_path, prepared_format
):
    from tuner.training.modal_host_prepared_copy import (
        inspect_mounted_prepared_dataset,
        ModalPreparedCopyUnavailable,
    )

    path, _ = callback_publication(tmp_path, prepared_format)

    def change(record, raw):
        (path / "dataset.jsonl").write_bytes(raw + b" ")

    with pytest.raises(ModalPreparedCopyUnavailable):
        inspect_mounted_prepared_dataset(
            path, prepared_format=prepared_format, snapshot_validator=change
        )
