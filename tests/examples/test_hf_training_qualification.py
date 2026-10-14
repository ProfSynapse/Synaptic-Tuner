"""Actual ordinary HF composition; scripted SDK transport, no provider effects."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tests.execution.providers.test_hf_jobs_public_family import (
    material as material,
    sdk,
    Clock,
    Grants,
    Secrets,
    _native_values,
    _complete,
)
from synaptic_tuner.api.v1.reference import ReferenceHostPortsV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunOperationError
from tuner.training import default_recipe_registry
from tuner.training.contracts import (
    ResolvedTrainingRequest,
    TrainingRequest,
    CanonicalDocument,
)
from tuner.training.coordinator_material import derive_coordinator_material

# Consumer code is test-owned input, copied separately in the installed lane.
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "hf_qualification", ROOT / "examples/training_qualification/hf_host.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def _write(path, raw):
    path.write_bytes(raw)
    return {
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@pytest.fixture
def configured(material, tmp_path):
    prep = material.preparation
    rich = prep.rich_plan
    resolved = ResolvedTrainingRequest(
        TrainingRequest(CanonicalDocument.from_mapping({})),
        rich.execution_source,
        rich.execution_context,
        rich.resolved_config,
        rich.workload,
        rich.runtime,
        rich.resources,
        rich.artifact_policy,
    )
    m = derive_coordinator_material(
        resolved,
        default_recipe_registry(),
        request_id=prep.resolved.request_id,
        project_ref=prep.resolved.project_ref,
        run_id=rich.execution_source.run_id,
    )
    profile = prep.selection.runtime_profile
    s = prep.selection.projection()
    config = {
        "provider": "hf_jobs",
        "host": {
            "authority_secret": {"provider": "env", "name": "AUTHORITY"},
            "evidence_secret": {"provider": "env", "name": "READER"},
        },
        "request_file": _write(tmp_path / "request.json", m.request_bytes),
        "hf_jobs": {
            "material_file": _write(tmp_path / "material.json", m.canonical_bytes),
            "prepared_file": _write(tmp_path / "input.jsonl", material.raw),
            "runtime_profile": {
                "name": profile.name,
                "root": str(profile.inventory_path.parent),
                "sha256": profile.profile_sha256.removeprefix("sha256:"),
            },
            "selection": {
                key: s[key]
                for key in (
                    "scope",
                    "bucket_id",
                    "source_prefix",
                    "run_prefix",
                    "flavor",
                    "timeout_seconds",
                    "host_token",
                    "job_token",
                    "quote_digest",
                )
            },
            "provider": prep.selection.provider.to_dict(),
            "reader_authority_ref": "fixture-reader",
            "reader_key_ref": "fixture-reader-key",
            "maximum_cost_minor_units": 100,
            "currency": "USD",
            "read_seconds": 30,
        },
    }
    # Profile loader's named YAML and inventory may be in different subdirectories.
    config["hf_jobs"]["runtime_profile"]["root"] = str(profile.inventory_path.parent)
    import tuner

    config["hf_jobs"]["runtime_profile"]["root"] = str(
        Path(tuner.__file__).resolve().parent.parent / "Trainers/runtime_profiles"
    )
    return config, material


def _ports(store):
    return ReferenceHostPortsV1(store, store, Clock(), Grants(), Secrets())


def test_preview_is_pure_and_matches_actual_public_plan(configured, tmp_path):
    config, material = configured
    preview = adapter.preview(config)
    assert preview["public_preflight"] == "deferred_until_host_composition"
    assert material.provider.calls == []
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    (root / "scratch").mkdir(mode=0o700)
    with SQLiteDurableStoreV1(root / "workflow.sqlite3") as store:
        with adapter.open_host(
            config,
            state_root=root,
            mode="start",
            ports=_ports(store),
            sdk=sdk,
            api=material.provider,
        ) as (training, runs, provider, run):
            request = training.load(Path(config["request_file"]["path"]).read_text())
            plan = training.plan(training.resolve(request), provider)
            assert plan.basis.to_dict() == preview["basis"]
            assert run.to_dict() == preview["run"]
            assert training.preflight(plan).ready
            assert material.provider.calls == []


def test_public_start_native_five_and_source_free_cold_reads(configured, tmp_path):
    config, material = configured
    preview = adapter.preview(config)
    config["run"] = preview["run"]
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    (root / "scratch").mkdir(mode=0o700)
    with SQLiteDurableStoreV1(root / "workflow.sqlite3") as store:
        with adapter.open_host(
            config,
            state_root=root,
            mode="start",
            ports=_ports(store),
            sdk=sdk,
            api=material.provider,
        ) as (training, runs, provider, expected):
            request = training.load(Path(config["request_file"]["path"]).read_text())
            plan = training.plan(training.resolve(request), provider)
            start = training.start(plan, training.preflight(plan))
            assert start.run == expected
            assert runs.outcome(expected).state.value == "running"
            # Actual admitted preparation from the adapter has identical original bytes.
            with adapter._preparation(config, Clock()) as (prep, _, _):
                values = _native_values(prep)
                material.preparation = prep
                _complete(material, values)
    for spec in (
        config["request_file"],
        config["hf_jobs"]["material_file"],
        config["hf_jobs"]["prepared_file"],
    ):
        Path(spec["path"]).unlink()
    config["hf_jobs"]["runtime_profile"]["root"] = "/missing-profile"
    calls = len([item for item in material.provider.calls if item[0] == "submit"])
    with SQLiteDurableStoreV1(root / "workflow.sqlite3", create=False) as store:
        ports = _ports(store)
        with adapter.open_host(
            config,
            state_root=root,
            mode="read",
            ports=ports,
            sdk=sdk,
            api=material.provider,
        ) as (training, runs, provider, run):
            assert training is None
            assert runs.outcome(run).state.value == "succeeded"
            assert runs.verify(run).verified
            for role, (_, raw) in values.items():
                assert (
                    b"".join(
                        runs.artifacts(
                            RunArtifactRequest(run, role, len(raw))
                        ).iter_bytes()
                    )
                    == raw
                )
            with pytest.raises(RunOperationError):
                runs.cancel(run, "not restored")
            with pytest.raises(RunOperationError):
                runs.reconcile(run)
        assert ports.grants.calls == []
    assert (
        len([item for item in material.provider.calls if item[0] == "submit"])
        == calls
        == 1
    )


@pytest.mark.parametrize("changed", ["request", "input", "profile", "run"])
def test_changed_inputs_refuse_before_secrets_or_sdk(configured, tmp_path, changed):
    config, material = configured
    if changed == "request":
        Path(config["request_file"]["path"]).write_bytes(b'{"changed":true}')
    elif changed == "input":
        Path(config["hf_jobs"]["prepared_file"]["path"]).write_bytes(b"{}\n")
    elif changed == "profile":
        config["hf_jobs"]["runtime_profile"]["sha256"] = "f" * 64
    else:
        config["run"] = {"run_id": "another", "project_ref": "fixture-project"}
    with pytest.raises(ValueError):
        adapter.preview(config)
    assert material.provider.calls == []


@pytest.mark.parametrize("cancel_enabled", [False, True])
def test_unknown_submit_never_adopted_by_cold_reader(configured, tmp_path, cancel_enabled):
    config, material = configured
    if cancel_enabled:
        config["hf_jobs"]["allow_cancel_reconcile"] = True
    config["run"] = adapter.preview(config)["run"]
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    (root / "scratch").mkdir(mode=0o700)
    original = material.provider.run_job

    def lost_reply(**kwargs):
        original(**kwargs)
        raise TimeoutError("private transport failure")

    material.provider.run_job = lost_reply
    with SQLiteDurableStoreV1(root / "workflow.sqlite3") as store:
        with adapter.open_host(
            config,
            state_root=root,
            mode="start",
            ports=_ports(store),
            sdk=sdk,
            api=material.provider,
        ) as (training, runs, provider, expected):
            request = training.load(Path(config["request_file"]["path"]).read_text())
            plan = training.plan(training.resolve(request), provider)
            preflight = training.preflight(plan)
            run = training.start(plan, preflight).run
            assert runs.show(run).state.value == "reconcile_required"
            training.start(plan, preflight)
    calls = len(material.provider.calls)
    with SQLiteDurableStoreV1(root / "workflow.sqlite3", create=False) as store:
        with pytest.raises(ValueError, match="read_unavailable"):
            with adapter.open_host(
                config,
                state_root=root,
                mode="read",
                ports=_ports(store),
                sdk=sdk,
                api=material.provider,
            ):
                pytest.fail("unknown submission is not a known-job reader")
    assert len(material.provider.calls) == calls
    assert len([item for item in material.provider.calls if item[0] == "submit"]) == 1


@pytest.mark.parametrize("cancel_enabled", [False, True])
def test_cold_wrong_original_key_rejects_without_provider(configured, tmp_path, cancel_enabled):
    config, material = configured
    if cancel_enabled:
        config["hf_jobs"]["allow_cancel_reconcile"] = True
    config["run"] = adapter.preview(config)["run"]
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    (root / "scratch").mkdir(mode=0o700)
    with SQLiteDurableStoreV1(root / "workflow.sqlite3") as store:
        with adapter.open_host(
            config,
            state_root=root,
            mode="start",
            ports=_ports(store),
            sdk=sdk,
            api=material.provider,
        ) as (training, _, provider, _):
            request = training.load(Path(config["request_file"]["path"]).read_text())
            plan = training.plan(training.resolve(request), provider)
            training.start(plan, training.preflight(plan))
    calls = len(material.provider.calls)
    config["host"]["authority_secret"]["name"] = "WRONG_ORIGINAL_KEY"
    with SQLiteDurableStoreV1(root / "workflow.sqlite3", create=False) as store:
        with pytest.raises(ValueError, match="read_unavailable"):
            with adapter.open_host(
                config,
                state_root=root,
                mode="read",
                ports=_ports(store),
                sdk=sdk,
                api=material.provider,
            ):
                pytest.fail("changed original authority cannot read")
    assert len(material.provider.calls) == calls


@pytest.mark.parametrize(
    "field,value", [("read_seconds", True), ("reader_key_ref", ""),
                    ("allow_cancel_reconcile", 1)]
)
def test_metadata_refuses_before_secret_resolution(configured, tmp_path, field, value):
    config, material = configured
    config["hf_jobs"][field] = value

    class ForbiddenSecrets:
        def resolve(self, ref):
            raise AssertionError("malformed config reached secret resolution")

    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    with SQLiteDurableStoreV1(root / "workflow.sqlite3") as store:
        ports = ReferenceHostPortsV1(
            store, store, Clock(), Grants(), ForbiddenSecrets()
        )
        with pytest.raises(ValueError, match="hf_configuration_invalid"):
            with adapter.open_host(
                config,
                state_root=root,
                mode="start",
                ports=ports,
                sdk=sdk,
                api=material.provider,
            ):
                pytest.fail("malformed config admitted")
    assert material.provider.calls == []


@pytest.mark.parametrize("cancel_enabled,cancel_permission", [(False, False), (True, True), (True, False)])
def test_shared_launch_real_host_ports_and_cold_known_native_five(
    configured, tmp_path, monkeypatch, cancel_enabled, cancel_permission
):
    """Real consumer retention/grants/secrets/SQLite; only SDK boundary scripted."""
    import os

    config, material = configured
    if cancel_enabled:
        config["hf_jobs"]["allow_cancel_reconcile"] = True
    shared_path = Path(
        os.environ.get(
            "SYNAPTIC_TEST_TRAINING_QUALIFICATION_LAUNCH",
            str(ROOT / "examples/training_qualification/launch.py"),
        )
    )
    shared_spec = importlib.util.spec_from_file_location(
        "qualification_launch", shared_path
    )
    shared = importlib.util.module_from_spec(shared_spec)
    shared_spec.loader.exec_module(shared)
    root = tmp_path / "one-shot-consumer"
    config["state_root"] = str(root)
    config["host"].update(
        grant_ref="fixture-consumer-grant",
        allowed_requirements=[
            {
                "operation": "training.start",
                "paid_effect": True,
                "maximum_cost_minor_units": 100,
                "currency": "USD",
            }
        ],
    )
    from synaptic_tuner.api.v1.secrets import SecretRef
    if cancel_permission:
        from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement
        config["host"]["allowed_requirements"].append(
            AuthorizationRequirement("training.cancel", False).to_dict())

    for name in ("AUTHORITY", "READER", "HOST_TOKEN", "JOB_TOKEN"):
        monkeypatch.setenv(name, Secrets().resolve(SecretRef("env", name)))
    options = {"sdk": sdk, "api": material.provider}
    result = shared.launch(config, adapter=adapter, adapter_options=options)
    assert result["start"]["accepted"] is True
    assert set(result["start"]) == {"run", "accepted"}
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    calls = len(material.provider.calls)
    with pytest.raises(FileExistsError):
        shared.launch(config, adapter=adapter, adapter_options=options)
    assert len(material.provider.calls) == calls
    retained = shared.load_config(root / "config.json")
    assert retained["run"] == result["start"]["run"]
    assert (
        shared.known(retained, "show", adapter=adapter, adapter_options=options)[
            "state"
        ]
        == "queued"
    )
    with adapter._preparation(retained, Clock()) as (prep, _, _):
        values = _native_values(prep)
        material.preparation = prep
    for item in (
        config["request_file"],
        config["hf_jobs"]["material_file"],
        config["hf_jobs"]["prepared_file"],
    ):
        Path(item["path"]).unlink()
    if cancel_enabled:
        from tests.execution.providers.test_hf_jobs_public_lifecycle import cancellation
        cancellation(material)
        if not cancel_permission:
            from synaptic_tuner.api.v1.runs_facade import RunOperationError
            with pytest.raises(RunOperationError):
                shared.known(retained, "cancel", reason="current permission missing",
                             adapter=adapter, adapter_options=options)
            assert not [item for item in material.provider.calls if item[0] == "cancel"]
            assert len([item for item in material.provider.calls if item[0] == "submit"]) == 1
            return
        canceled = shared.known(
            retained, "cancel", reason="bounded consumer cancellation",
            adapter=adapter, adapter_options=options)
        assert canceled["state"] == "cancel_requested"
        shared.known(retained, "reconcile", adapter=adapter, adapter_options=options)
        assert len([item for item in material.provider.calls if item[0] == "cancel"]) == 1
    _complete(material, values)
    assert (
        shared.known(retained, "observe", adapter=adapter, adapter_options=options)[
            "state"
        ]
        == "succeeded"
    )
    verified = shared.known(
        retained, "verify", adapter=adapter, adapter_options=options
    )
    assert verified["verified"] is True
    assert shared.known(retained, "show", adapter=adapter, adapter_options=options)["state"] == "succeeded"
    assert set(verified) == {"run", "verified", "checked_at"}
    assert json.loads(json.dumps(verified, allow_nan=False)) == verified
    destination = tmp_path / "downloads"
    destination.mkdir(mode=0o700)
    paths = shared.known(
        retained,
        "download",
        destination=destination,
        adapter=adapter,
        adapter_options=options,
    )["paths"]
    assert len(paths) == 5
    assert {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths} == {
        hashlib.sha256(raw).hexdigest() for _, raw in values.values()
    }
    assert len([item for item in material.provider.calls if item[0] == "submit"]) == 1


def test_shared_show_authenticates_stage_failure_without_sdk_or_source(configured, tmp_path, monkeypatch):
    import os
    from synaptic_tuner.api.v1.secrets import SecretRef

    config, material = configured
    path = Path(os.environ.get("SYNAPTIC_TEST_TRAINING_QUALIFICATION_LAUNCH",
        str(ROOT / "examples/training_qualification/launch.py")))
    spec = importlib.util.spec_from_file_location("qualification_stage_launch", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    config["state_root"] = str(tmp_path / "stage-attempt")
    config["host"].update(grant_ref="fixture-stage-grant", allowed_requirements=[{
        "operation": "training.start", "paid_effect": True,
        "maximum_cost_minor_units": 100, "currency": "USD"}])
    for name in ("AUTHORITY", "READER", "HOST_TOKEN", "JOB_TOKEN"):
        monkeypatch.setenv(name, Secrets().resolve(SecretRef("env", name)))
    def failure(*args, **kwargs):
        raise RuntimeError("secret-provider-message")
    material.provider.batch_bucket_files = failure
    shared.launch(config, adapter=adapter, adapter_options={"sdk": sdk, "api": material.provider})
    retained = shared.load_config(Path(config["state_root"]) / "config.json")
    before = list(material.provider.calls)
    def forbidden(*args, **kwargs):
        raise AssertionError("local show constructed provider/source authority")
    monkeypatch.setattr(adapter, "_preparation", forbidden)
    monkeypatch.setattr(sdk, "HfApi", forbidden)
    shown = shared.known(retained, "show", adapter=adapter)
    assert shown["local_workflow_phase"] == "stage_reconcile_required"
    assert shown["local_stage_diagnostic"]["phase"] == "upload"
    assert shown["local_stage_diagnostic"]["failure_class"] == "sdk"
    assert shown["replay_authorized"] is shown["provider_evidence"] is False
    assert material.provider.calls == before
    monkeypatch.setenv("AUTHORITY", "different-original-key" * 4)
    with pytest.raises(Exception):
        shared.known(retained, "show", adapter=adapter)
    assert material.provider.calls == before
