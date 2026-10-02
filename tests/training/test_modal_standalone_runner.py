"""Provider-free V2 transcript through the actual config-first CLI host."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import os
import sys
import asyncio

import pytest

from synaptic_tuner.api.v1.training_facade import TrainingAPI
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tests.dataset_prep.test_context_messages_v2 import _config
from tests.execution.providers.test_modal_runtime_build import _candidate
from tuner.dataset_prep import prepare_dataset_v2
from tuner.execution.providers.modal.contracts import provider_entry_identity
from tuner.execution.providers.modal.contracts import operation_path
from tuner.execution.providers.modal.coordinator_producer import MODAL_TRAINING_ARTIFACT_BOUNDS_V1
from tuner.execution.providers.modal.bounded_volume_read import BoundedModalVolumeReader
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.coordinator_v1.model import WorkflowPhaseV1
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.packaged_worker import PACKAGED_WORKER_FAILURE_STAGES
from tuner.execution.providers.modal.packaged_binding import (
    MODAL_PACKAGED_RUNTIME_FACTS_V2_SCHEMA, ModalPackagedRuntimeFactsV1,
)
from tuner.execution.providers.modal.packaged_reader import (
    ModalPackagedArtifactMember, ModalPackagedCompletionObservation, ModalPackagedReader,
)
from tuner.execution.providers.modal.packaged_staging import (
    ModalPackagedInputStager, ModalPackagedStageReceipt,
)
from tuner.execution.providers.modal.runtime_build import build_modal_runtime_release_v2
from tuner.execution.providers.modal.runtime_release_qualification import ModalRuntimeReleaseQualificationReceiptV1
from tuner.project.context import ProjectContext
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV2
from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1, ModalPackagedReadUnavailable
from tuner.training.modal_host_requests import ModalPackagedResolutionUnavailable
from tuner.training.modal_host_runtime import ModalHostRuntimeV1
from tuner.training.modal_host_runtime import ModalHostBootstrapUnavailable
from tuner.training.modal_host_qualification import (
    ModalHostCPUQualificationV1, ModalHostQualificationUnavailable,
)
from tuner.training.modal_recipe import ModalSFTRecipePlanV1, load_modal_sft_recipe
from tuner.training.packaged_compilation import (
    PACKAGED_SFT_WORKLOAD_SCHEMA, compile_packaged_sft_workload,
    packaged_configuration_digest,
)
from tuner.cloud.derived_training_image import load_profile
from tuner.handlers.modal_job_config_handler import ModalJobConfigHandler
from tuner.cli.router import route_command
import tuner.training.modal_standalone_runner as runner


pytestmark = pytest.mark.skipif(os.name != "posix", reason="first live host is POSIX-only")
ROOT = Path(__file__).resolve().parents[2]


def test_real_modal_image_profile_admits_current_packaged_workload():
    profile = load_profile(
        ROOT / "Trainers" / "image_profiles"
        / "qwen35_4b_packaged_sft_3360351c" / "profile.yaml"
    )
    assert profile.packaged_runtime is not None
    assert (
        profile.packaged_runtime["capabilities"]["contracts"]["workload_schema"]
        == PACKAGED_SFT_WORKLOAD_SCHEMA
    )


def _release(recipe):
    original = build_modal_runtime_release_v2(_candidate(), release_ref="runtime:fake-v2")
    names = (
        "release_ref", "package_name", "package_version", "package_digest",
        "source_provenance_digest", "worker_entrypoint", "worker_closure_digest",
        "python_implementation", "python_version", "python_executable",
        "python_executable_digest", "installed_distributions_digest",
        "installed_distribution_count", "platform_system", "platform_machine",
        "cuda_version", "runtime_facts", "compatible_methods", "compatible_models",
        "compatible_dataset_formats", "workload_schema", "prepared_input_schema",
        "artifact_contract_schema",
    )
    values = {name: getattr(original, name) for name in names}
    values["compatible_models"] = ((recipe.model.ref, recipe.model.revision),)
    values["workload_schema"] = PACKAGED_SFT_WORKLOAD_SCHEMA
    return PackagedTrainingRuntimeReleaseV2.build(**values, material=original.material)


def _facts(release):
    return ModalPackagedRuntimeFactsV1(
        account_ref="workspace", workspace_ref="workspace",
        environment_ref="main", client_ref="train-config-host", sdk_version="1.5.4",
        app_name="owned-app", app_id="ap-owned", deployment_generation=1,
        function_name="packaged-training", function_id="fu-training",
        self_check_function_name="packaged-self-check", self_check_function_id="fu-check",
        deployment_spec_digest="d" * 64, image_id="im-exact", image_digest=None,
        package_digest=release.package_digest,
        installed_distributions_digest=release.installed_distributions_digest,
        worker_entrypoint=release.worker_entrypoint,
        worker_closure_digest=release.worker_closure_digest,
        control_volume_id="vo-control", artifact_volume_id="vo-artifacts",
        material_digest=release.material_digest, model_cache_volume_id="vo-cache",
        schema_version=MODAL_PACKAGED_RUNTIME_FACTS_V2_SCHEMA,
    )


def _setup(tmp_path, monkeypatch, *, failed=False):
    monkeypatch.setattr(runner.platform, "python_version", lambda: "3.11.14")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    _, _, _, dataset_config = _config(tmp_path / "source")
    root = tmp_path / ".tracking" / "datasets"
    publication = prepare_dataset_v2(dataset_config, root)
    semantic = publication.semantic_identity
    original = load_modal_sft_recipe(
        ROOT / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml",
        profiles_root=ROOT / "Trainers/runtime_profiles",
    )
    recipe = replace(
        original, dataset_locator=str(publication.path.relative_to(tmp_path) / "dataset.jsonl"),
        dataset_digest=semantic.dataset_digest,
        train_rows=semantic.split_counts["train"],
        validation_rows=semantic.split_counts["validation"],
    )
    from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity

    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + semantic.dataset_digest,
        semantic.dataset_digest, semantic.dataset_sha256, semantic.dataset_bytes,
        "syntunia-sft-row/v2",
    )
    config = recipe.packaged_config(identity)
    plan = ModalSFTRecipePlanV1(
        recipe, identity, "sha256:" + "a" * 64, "fixture/base@sha256:" + "b" * 64,
        "c" * 64, compile_packaged_sft_workload(resolved_config=config).fingerprint,
        packaged_configuration_digest(config),
    )
    release = _release(recipe)
    facts = _facts(release)
    facts.validate_release(release)
    client = SimpleNamespace(stub=object())
    events = []

    class _Volume:
        ids = {"control": "vo-control", "artifacts": "vo-artifacts", "cache": "vo-cache"}
        published = {}
        publications = []
        readbacks = []

        def __init__(self, volume_id):
            self.object_id = volume_id
            self.is_hydrated = False
            self.pending = {}

        @classmethod
        def from_name(cls, name, *, environment_name, create_if_missing, version, client):
            assert environment_name == "main"
            assert create_if_missing is False
            assert version == 1
            assert client is modal_client
            return cls(cls.ids[name])

        def hydrate(self, client):
            assert client is modal_client
            self.is_hydrated = True

        def batch_upload(self, *, force):
            assert force is False
            volume = self

            class _Batch:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    if exc_type is None:
                        assert len(volume.pending) == 1
                        for path, value in volume.pending.items():
                            key = (volume.object_id, path)
                            assert key not in _Volume.published
                            _Volume.published[key] = value
                            _Volume.publications.append(key)
                        volume.pending.clear()
                    return False

                def put_file(self, source, path):
                    assert path not in volume.pending
                    volume.pending[path] = source.read()

            return _Batch()

        def commit(self):
            pytest.fail("host commit is redundant")

    modal_client = client

    async def read_marker(_reader, *, volume_id, path, expected_size, expected_sha256, max_bytes):
        assert volume_id in _Volume.ids.values()
        assert path.startswith(".synaptic-volume-marker-")
        assert expected_size == max_bytes == 32
        key = (volume_id, path)
        value = _Volume.published[key]
        assert sha256(value).hexdigest() == expected_sha256
        _Volume.readbacks.append(key)
        return value

    monkeypatch.setattr(BoundedModalVolumeReader, "read_exact", read_marker)

    class _Function:
        @staticmethod
        def from_name(*_args, **_kwargs):
            class _Call:
                object_id = "fc-owned"

            return SimpleNamespace(
                hydrate=lambda bound_client: None,
                is_hydrated=True,
                spawn=lambda payload: events.append("spawn") or _Call(),
            )

    class _SDK:
        __version__ = "1.5.4"
        Function = _Function
        Volume = _Volume

    def init_fake_reader(_reader, *, sdk, client, http_session_factory=None):
        assert sdk is _SDK and client is modal_client
        assert http_session_factory is None

    monkeypatch.setattr(BoundedModalVolumeReader, "__init__", init_fake_reader)

    monkeypatch.setitem(sys.modules, "modal", _SDK)
    async_utils = ModuleType("modal._utils.async_utils")
    def create_blocking(operation):
        assert getattr(operation, "__self__", None) is None
        return lambda *args, **kwargs: asyncio.run(operation(*args, **kwargs))
    async_utils.synchronizer = SimpleNamespace(create_blocking=create_blocking)
    monkeypatch.setitem(sys.modules, "modal._utils", ModuleType("modal._utils"))
    monkeypatch.setitem(sys.modules, "modal._utils.async_utils", async_utils)
    monkeypatch.setattr(runner, "open_modal_host_scope", lambda **kw: (client, facts.client_binding))
    monkeypatch.setattr(runner, "observe_modal_host_scope", lambda **kw: (
        facts.account_ref, facts.workspace_ref, facts.environment_ref, facts.client_ref,
    ))
    deployment_facts = SimpleNamespace(facts_digest="a" * 64)
    runtime = ModalHostRuntimeV1(
        release, deployment_facts, facts,
        SimpleNamespace(gpu_only_timeout_estimate_minor_units=125, quote_digest="e" * 64),
        (("vo-control", "control"), ("vo-artifacts", "artifacts"), ("vo-cache", "cache")),
        b"k" * 32,
        SimpleNamespace(observe=lambda **kw: facts),
    )
    monkeypatch.setattr(runner, "prepare_modal_runtime_for_host", lambda **kw: events.append("bootstrap") or runtime)

    def qualify(**kwargs):
        events.append("cpu")
        effect_id = kwargs["effect_id"]
        receipt = ModalRuntimeReleaseQualificationReceiptV1(
            effect_id, "b" * 64, "fc-cpu", deployment_facts.facts_digest,
            "c" * 64,
            operation_path(effect_id, "runtime-release-qualification", "output", "evidence.json"),
            4, "d" * 64, "entry-cpu",
        )
        return ModalHostCPUQualificationV1(
            receipt, receipt.output_sha256, "fc-cpu",
            release.manifest_digest, deployment_facts.facts_digest,
        )

    monkeypatch.setattr(runner, "qualify_modal_runtime_for_host", qualify)

    def stage(_stager, material):
        descriptor = material.descriptor
        events.append("stage")
        return ModalPackagedStageReceipt(
            descriptor.stage_effect_id, material.execution_binding_digest,
            descriptor.artifact_volume_id, descriptor.relative_path,
            descriptor.identity.size_bytes, descriptor.identity.content_digest,
            provider_entry_identity(
                descriptor.artifact_volume_id, descriptor.relative_path,
                descriptor.identity.size_bytes,
            ),
        )

    monkeypatch.setattr(ModalPackagedInputStager, "stage_once", stage)
    completion_digest = "f" * 64

    def completion(_reader, binding, *, provider_job_ref):
        effect_id = binding.command.operation.effect.effect_id
        members = tuple(ModalPackagedArtifactMember(
            role, f"operations/{effect_id}/output/{role}", 4,
            sha256(b"data").hexdigest(), f"entry-{role}",
        ) for role in sorted((
            "workload_record", "training_lineage", "training_metrics",
            "final_model", "tokenizer",
        )))
        return ModalPackagedCompletionObservation(
            effect_id, binding.command_digest, provider_job_ref,
            release.manifest_digest, binding.provider_binding.binding_digest,
            binding.execution_binding.binding_digest, completion_digest, members,
        )

    monkeypatch.setattr(ModalPackagedReader, "observe_completion", completion)
    def stream_artifact(_reader, *_args, maximum_bytes, **_kwargs):
        assert maximum_bytes <= MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes
        return iter((b"data",))

    monkeypatch.setattr(ModalPackagedReader, "iter_artifact", stream_artifact)
    if failed:
        def poll(_self, _binding, _ref, **_kwargs):
            raise ModalPackagedReadUnavailable("modal_packaged_call_failed")
    else:
        def poll(_self, _binding, _ref, **_kwargs):
            return ModalFunctionCallState.RETURNED, completion_digest
    monkeypatch.setattr(ModalPackagedCoordinatorReaderV1, "_poll_packaged_call", poll)
    return plan, ProjectContext.standalone(engine_root=tmp_path), events


def test_cli_v2_fake_transcript_prepares_submits_verifies_and_downloads(tmp_path, monkeypatch, capsys):
    plan, context, events = _setup(tmp_path, monkeypatch)
    download_maxima = []
    original_download = runner.download_verified_modal_artifact

    def observe_download(*args, **kwargs):
        download_maxima.append(kwargs["maximum_bytes"])
        return original_download(*args, **kwargs)

    monkeypatch.setattr(runner, "download_verified_modal_artifact", observe_download)
    monkeypatch.setattr(
        "tuner.training.modal_recipe.plan_modal_sft_recipe",
        lambda *_args, **_kwargs: plan,
    )
    assert route_command(Namespace(
        command="train", job_config=str(ROOT / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml"),
        plan=False, quote=False, modal_profile="explicit", modal_environment="main", json=True,
    ), context=context) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["gpu_only_timeout_estimate_minor_units"] == 125
    assert payload["data"]["operator_maximum_cost_minor_units"] == 200
    assert len(payload["data"]["verified_artifacts"]) == 5
    assert all(Path(path).read_bytes() == b"data" for path in payload["data"]["verified_artifacts"])
    assert download_maxima == [MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes] * 5
    assert events == ["bootstrap", "cpu", "stage", "spawn"]
    volume = sys.modules["modal"].Volume
    assert len(volume.publications) == 3
    assert volume.readbacks == volume.publications
    assert {volume_id for volume_id, _ in volume.publications} == {
        "vo-control", "vo-artifacts", "vo-cache",
    }
    assert all(len(value) == 32 for value in volume.published.values())


def test_cli_v2_marker_readback_mismatch_keeps_submit_claim_without_spawning(
        tmp_path, monkeypatch):
    plan, context, events = _setup(tmp_path, monkeypatch)

    async def wrong_readback(_reader, **_kwargs):
        return b"x" * 32

    monkeypatch.setattr(BoundedModalVolumeReader, "read_exact", wrong_readback)
    with pytest.raises(runner.ModalStandalonePhaseUnavailable) as caught:
        runner.run_modal_standalone_job(
            plan=plan, context=context, modal_profile="explicit",
            modal_environment="main",
        )
    assert caught.value.phase == "RUN_SUBMIT_RECONCILE_REQUIRED"
    assert caught.value.retry_authorized is False
    assert events == ["bootstrap", "cpu", "stage"]
    volume = sys.modules["modal"].Volume
    assert len(volume.publications) == 1
    assert len(volume.published) == 1


def test_cli_v2_failed_returned_job_is_closed_without_artifact_publication(tmp_path, monkeypatch, capsys):
    plan, context, events = _setup(tmp_path, monkeypatch, failed=True)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ), context)
    assert handler._execute(plan) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "MODAL_TRAINING_UNAVAILABLE"
    assert payload["error"]["details"] == {
        "phase": "RUN_WORKER_FAILED",
        "failure_class": "UNAVAILABLE",
        "location": "modal_standalone_runner.worker_result",
        "retry_authorized": False,
    }
    assert events == ["bootstrap", "cpu", "stage", "spawn"]
    assert not list((tmp_path / "state" / "synaptic-training").glob("*-artifacts"))


@pytest.mark.parametrize("stage", sorted(PACKAGED_WORKER_FAILURE_STAGES))
def test_exact_worker_failure_stage_projects_closed_public_code(
        tmp_path, monkeypatch, capsys, stage):
    plan, context, events = _setup(tmp_path, monkeypatch)
    frozen_now = runner.UTCClock().now()
    frozen_epoch = int(datetime.fromisoformat(frozen_now.replace("Z", "+00:00")).timestamp())

    class FrozenUTCClock:
        def now(self):
            return frozen_now

        def now_iso(self):
            return frozen_now

        def now_epoch(self):
            return frozen_epoch

    monkeypatch.setattr(runner, "UTCClock", FrozenUTCClock)

    def failed(*_args, **_kwargs):
        raise ModalPackagedReadUnavailable("modal_packaged_call_failed_" + stage)

    monkeypatch.setattr(ModalPackagedCoordinatorReaderV1, "_poll_packaged_call", failed)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ), context)
    assert handler._execute(plan) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["details"] == {
        "phase": "RUN_WORKER_" + stage,
        "failure_class": "UNAVAILABLE",
        "location": "modal_standalone_runner.worker_result",
        "retry_authorized": False,
    }
    assert events == ["bootstrap", "cpu", "stage", "spawn"]
    assert not list((tmp_path / "state" / "synaptic-training").glob("*-artifacts"))


def test_cli_v2_hostile_provider_error_is_never_serialized(tmp_path, monkeypatch, capsys):
    plan, context, events = _setup(tmp_path, monkeypatch)
    secret = "HF_TOKEN=private-and-absolute-/home/private/customer"

    def fail(*_args, **_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(runner, "prepare_modal_runtime_for_host", fail)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ), context)
    assert handler._execute(plan) == 2
    serialized = capsys.readouterr().out
    assert secret not in serialized
    assert "/home/private/customer" not in serialized
    assert json.loads(serialized)["error"]["code"] == "MODAL_TRAINING_UNAVAILABLE"
    assert events == []


def test_cli_cpu_qualification_only_never_stages_training_or_starts_gpu(tmp_path, monkeypatch, capsys):
    plan, context, events = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "tuner.training.modal_recipe.plan_modal_sft_recipe",
        lambda *_args, **_kwargs: plan,
    )
    assert route_command(Namespace(
        command="train", job_config=str(ROOT / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml"),
        plan=False, quote=False, qualify=True,
        modal_profile="explicit", modal_environment="main", json=True,
    ), context=context) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["training_executed"] is False
    assert payload["data"]["gpu_qualified"] is False
    assert events == ["bootstrap", "cpu"]


def test_cpu_qualification_failure_blocks_gpu_start(tmp_path, monkeypatch, capsys):
    plan, context, events = _setup(tmp_path, monkeypatch)
    def fail(**_kwargs):
        events.append("cpu")
        raise RuntimeError("private-token-/home/owner/path")
    monkeypatch.setattr(runner, "qualify_modal_runtime_for_host", fail)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ), context)
    assert handler._execute(plan) == 2
    output = capsys.readouterr().out
    assert "private-token" not in output and "/home/owner/path" not in output
    assert events == ["bootstrap", "cpu"]


def test_ambiguous_cpu_submit_keeps_claim_and_never_replays_or_starts_gpu(
        tmp_path, monkeypatch, capsys):
    plan, context, events = _setup(tmp_path, monkeypatch)

    def ambiguous(**kwargs):
        events.append("cpu")
        kwargs["private_storage"].attempts.claim(
            "cpu-ambiguous-fixture", canonical_bytes({"attempt": "claimed-once"}),
        )
        raise TimeoutError("private-provider-response-missing")

    monkeypatch.setattr(runner, "qualify_modal_runtime_for_host", ambiguous)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ), context)
    assert handler._execute(plan) == 2
    assert handler._execute(plan) == 2
    serialized = capsys.readouterr().out
    assert "private-provider-response-missing" not in serialized
    assert events == ["bootstrap", "cpu", "bootstrap", "cpu"]
    assert not list((tmp_path / "state" / "synaptic-training").glob("*-artifacts"))


def test_explicit_fresh_attempt_uses_private_distinct_journal_and_preserves_default_claim(
        tmp_path, monkeypatch):
    plan, context, events = _setup(tmp_path, monkeypatch)
    original = runner.prepare_modal_runtime_for_host

    def one_shot_bootstrap(**kwargs):
        kwargs["private_storage"].attempts.claim(
            "build-same-intent", canonical_bytes({"attempt": "one-shot"}),
        )
        return original(**kwargs)

    monkeypatch.setattr(runner, "prepare_modal_runtime_for_host", one_shot_bootstrap)
    first = runner.qualify_modal_standalone_job(
        plan=plan, context=context, modal_profile="explicit", modal_environment="main",
    )
    with pytest.raises(runner.ModalStandaloneRunUnavailable):
        runner.qualify_modal_standalone_job(
            plan=plan, context=context, modal_profile="explicit", modal_environment="main",
        )
    second = runner.qualify_modal_standalone_job(
        plan=plan, context=context, modal_profile="explicit", modal_environment="main",
        fresh_attempt=True,
    )
    assert first.training_executed is False and second.training_executed is False
    state = tmp_path / "state" / "synaptic-training"
    assert (state / "modal-host.sqlite3").is_file()
    attempts = list(state.glob("attempt-cpu-qual-*"))
    assert len(attempts) == 1
    assert attempts[0].stat().st_mode & 0o777 == 0o700
    assert (attempts[0] / "modal-host.sqlite3").is_file()
    assert "stage" not in events and "spawn" not in events


@pytest.mark.parametrize("qualify_only", [True, False])
@pytest.mark.parametrize("diagnosis", (
    "SOURCE_ARCHIVE_INVALID", "BUILD_INPUTS_INVALID", "APP_START_TIMEOUT",
    "CAPTURE_OUTPUT_INSPECTOR_REJECTED",
    "RELEASE_OBSERVE_SCOPE_UNAVAILABLE",
    "RELEASE_OBSERVE_OBSERVATION_UNAVAILABLE",
    "RELEASE_OBSERVE_OBSERVATION_INVALID",
    "RELEASE_ATTEMPT_BOUNDED_DEPLOY_UNAVAILABLE",
    "RELEASE_ATTEMPT_SCOPE_UNAVAILABLE",
    "RELEASE_ATTEMPT_OBSERVATION_UNAVAILABLE",
    "RELEASE_ATTEMPT_OBSERVATION_INVALID",
    "RELEASE_ATTEMPT_RESOURCE_UNAVAILABLE",
    "RELEASE_ATTEMPT_CONSTRUCTION_FAILED",
    "RELEASE_ATTEMPT_DEPLOYMENT_INDETERMINATE",
    "RELEASE_ATTEMPT_ACKNOWLEDGEMENT_INVALID",
))
def test_known_bootstrap_failure_surfaces_only_closed_diagnostic(
        tmp_path, monkeypatch, capsys, qualify_only, diagnosis):
    plan, context, events = _setup(tmp_path, monkeypatch)

    def source_archive_failure(**_kwargs):
        events.append("bootstrap")
        try:
            raise RuntimeError("HF_TOKEN=private /home/owner/dataset.jsonl")
        except RuntimeError:
            raise ModalHostBootstrapUnavailable(diagnosis) from None

    monkeypatch.setattr(runner, "prepare_modal_runtime_for_host", source_archive_failure)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ), context)
    assert (handler._qualify(plan) if qualify_only else handler._execute(plan)) == 2
    output = capsys.readouterr().out
    assert "HF_TOKEN" not in output and "/home/owner" not in output
    payload = json.loads(output)
    assert payload["error"]["code"] == (
        "MODAL_QUALIFICATION_UNAVAILABLE" if qualify_only else "MODAL_TRAINING_UNAVAILABLE"
    )
    assert payload["error"]["details"] == {
        "phase": ModalHostBootstrapUnavailable(diagnosis).phase,
        "failure_class": ModalHostBootstrapUnavailable(diagnosis).failure_class,
        "location": ModalHostBootstrapUnavailable(diagnosis).location,
        "retry_authorized": False,
    }
    assert events == ["bootstrap"]


def test_unknown_bootstrap_exception_remains_generic(tmp_path, monkeypatch, capsys):
    plan, context, _events = _setup(tmp_path, monkeypatch)

    def hostile(**_kwargs):
        raise RuntimeError("HF_TOKEN=private /home/owner/dataset.jsonl")

    monkeypatch.setattr(runner, "prepare_modal_runtime_for_host", hostile)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ), context)
    assert handler._qualify(plan) == 2
    output = capsys.readouterr().out
    assert "HF_TOKEN" not in output and "/home/owner" not in output
    assert "details" not in json.loads(output)["error"]


@pytest.mark.parametrize("phase", (
    "DISPATCH_FUNCTION_IDENTITY", "DISPATCH_SPAWN_INDETERMINATE",
    "DISPATCH_CATALOG_INDETERMINATE", "CALL_PARENT_SETUP",
    "CALL_INSTALLED_CHILD", "CALL_PARENT_RELEASE", "CALL_CHILD_RESULT",
))
@pytest.mark.parametrize("qualify_only", [True, False])
def test_cpu_failure_surfaces_closed_stage_in_both_cli_modes(
        tmp_path, monkeypatch, capsys, qualify_only, phase):
    plan, context, events = _setup(tmp_path, monkeypatch)

    def fail_cpu(**_kwargs):
        events.append("cpu")
        raise ModalHostQualificationUnavailable(phase) from None

    monkeypatch.setattr(runner, "qualify_modal_runtime_for_host", fail_cpu)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ), context)
    assert (handler._qualify(plan) if qualify_only else handler._execute(plan)) == 2
    output = capsys.readouterr().out
    assert "HF_TOKEN" not in output
    payload = json.loads(output)
    assert payload["error"]["details"] == {
        "phase": phase,
        "failure_class": ModalHostQualificationUnavailable(phase).failure_class,
        "location": ModalHostQualificationUnavailable(phase).location,
        "retry_authorized": False,
    }
    assert events == ["bootstrap", "cpu"]


@pytest.mark.parametrize("phase, failure_class, location, method", (
    ("RUN_HOST_ASSEMBLY", "UNAVAILABLE", "modal_standalone_runner.compose_host", None),
    ("RUN_PUBLIC_PREPARE", "UNAVAILABLE", "modal_standalone_runner.public_prepare", "prepare"),
    ("RUN_PUBLIC_LOAD", "UNAVAILABLE", "modal_standalone_runner.public_load", "load"),
    ("RUN_PUBLIC_RESOLVE", "UNAVAILABLE", "modal_standalone_runner.public_resolve", "resolve"),
    ("RUN_PUBLIC_PLAN", "UNAVAILABLE", "modal_standalone_runner.public_plan", "plan"),
    ("RUN_PUBLIC_PREFLIGHT", "UNAVAILABLE", "modal_standalone_runner.public_preflight", "preflight"),
    ("RUN_START_INDETERMINATE", "INDETERMINATE", "modal_standalone_runner.training_start", "start"),
))
def test_post_cpu_host_failures_have_only_closed_non_retryable_diagnostics(
        tmp_path, monkeypatch, phase, failure_class, location, method):
    plan, context, events = _setup(tmp_path, monkeypatch)
    secret = "HF_TOKEN=private-and-absolute-/home/private/customer"

    def fail(*_args, **_kwargs):
        raise RuntimeError(secret)

    if phase == "RUN_HOST_ASSEMBLY":
        monkeypatch.setattr(runner, "compose_modal_packaged_reference_host", fail)
    else:
        monkeypatch.setattr(TrainingAPI, method, fail)

    with pytest.raises(runner.ModalStandalonePhaseUnavailable) as caught:
        runner.run_modal_standalone_job(
            plan=plan, context=context, modal_profile="explicit",
            modal_environment="main",
        )
    diagnosis = caught.value
    assert type(diagnosis) is runner.ModalStandalonePhaseUnavailable
    assert (diagnosis.phase, diagnosis.failure_class, diagnosis.location,
            diagnosis.retry_authorized) == (phase, failure_class, location, False)
    assert diagnosis.args == ("modal_standalone_phase_unavailable",)
    assert secret not in str(diagnosis)
    assert "/home/private/customer" not in repr(diagnosis)
    assert events == ["bootstrap", "cpu"]


def test_post_start_workflow_lookup_failure_is_closed_without_private_details(tmp_path, monkeypatch):
    plan, context, events = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        TrainingAPI, "start", lambda *_args, **_kwargs: SimpleNamespace(
            accepted=True, run=SimpleNamespace(run_id="no-workflow"),
        ),
    )
    with pytest.raises(runner.ModalStandalonePhaseUnavailable) as caught:
        runner.run_modal_standalone_job(
            plan=plan, context=context, modal_profile="explicit",
            modal_environment="main",
        )
    assert (caught.value.phase, caught.value.failure_class,
            caught.value.location, caught.value.retry_authorized) == (
                "RUN_WORKFLOW_STATE", "UNAVAILABLE",
                "modal_standalone_runner.workflow_state", False,
            )
    assert events == ["bootstrap", "cpu"]


@pytest.mark.parametrize("phase, failure_class, location", (
    ("RUN_READ_BINDING", "UNAVAILABLE", "modal_standalone_runner.read_binding"),
    ("RUN_OUTCOME", "UNAVAILABLE", "modal_standalone_runner.outcome"),
    ("RUN_VERIFY", "UNAVAILABLE", "modal_standalone_runner.verify"),
    ("RUN_ARTIFACT_DOWNLOAD", "UNAVAILABLE", "modal_standalone_runner.artifact_download"),
))
def test_post_submit_host_boundaries_are_closed_without_private_details(
        tmp_path, monkeypatch, phase, failure_class, location):
    plan, context, events = _setup(tmp_path, monkeypatch)
    secret = "HF_TOKEN=private-and-absolute-/home/private/customer"

    def fail(*_args, **_kwargs):
        raise RuntimeError(secret)

    if phase == "RUN_READ_BINDING":
        monkeypatch.setattr(ModalPackagedCoordinatorReaderV1, "_binding", fail)
    elif phase == "RUN_OUTCOME":
        monkeypatch.setattr(RunsAPI, "outcome", fail)
    elif phase == "RUN_VERIFY":
        monkeypatch.setattr(RunsAPI, "verify", fail)
    else:
        monkeypatch.setattr(runner, "download_verified_modal_artifact", fail)
    with pytest.raises(runner.ModalStandalonePhaseUnavailable) as caught:
        runner.run_modal_standalone_job(
            plan=plan, context=context, modal_profile="explicit",
            modal_environment="main",
        )
    assert (caught.value.phase, caught.value.failure_class,
            caught.value.location, caught.value.retry_authorized) == (
                phase, failure_class, location, False,
            )
    assert secret not in str(caught.value)
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_stage_failure_is_closed_and_never_submits(
        tmp_path, monkeypatch, capsys):
    plan, context, events = _setup(tmp_path, monkeypatch)
    frozen_now = runner.UTCClock().now()
    frozen_epoch = int(datetime.fromisoformat(frozen_now.replace("Z", "+00:00")).timestamp())

    class FrozenUTCClock:
        def now(self):
            return frozen_now

        def now_iso(self):
            return frozen_now

        def now_epoch(self):
            return frozen_epoch

    monkeypatch.setattr(runner, "UTCClock", FrozenUTCClock)
    secret = "HF_TOKEN=private-and-absolute-/home/private/customer"

    def fail_stage(*_args, **_kwargs):
        events.append("stage")
        raise RuntimeError(secret)

    monkeypatch.setattr(ModalPackagedInputStager, "stage_once", fail_stage)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ), context)
    assert handler._execute(plan) == 2
    payload = json.loads(capsys.readouterr().out)
    details = payload["error"]["details"]
    assert details["phase"] in {
        "RUN_START_INDETERMINATE", "RUN_STAGE_RECONCILE_REQUIRED",
    }
    assert details == {
        "phase": details["phase"],
        "failure_class": "INDETERMINATE",
        "location": (
            "modal_standalone_runner.training_start"
            if details["phase"] == "RUN_START_INDETERMINATE"
            else "modal_standalone_runner.stage_reconcile"
        ),
        "retry_authorized": False,
    }
    assert secret not in json.dumps(payload)
    assert events == ["bootstrap", "cpu", "stage"]


@pytest.mark.parametrize("workflow_phase, expected", (
    (WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED, "RUN_STAGE_RECONCILE_REQUIRED"),
    (WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED, "RUN_SUBMIT_RECONCILE_REQUIRED"),
    (WorkflowPhaseV1.FAILED, "RUN_WORKFLOW_FAILED"),
    (WorkflowPhaseV1.CONTRADICTED, "RUN_WORKFLOW_CONTRADICTED"),
    (WorkflowPhaseV1.QUEUED, None),
))
def test_accepted_start_durable_phase_mapping_is_closed(workflow_phase, expected):
    assert runner._post_start_phase(workflow_phase) == expected


def test_accepted_start_phase_mapping_rejects_untyped_state():
    with pytest.raises(ValueError, match="invalid_modal_workflow_phase"):
        runner._post_start_phase("stage_reconcile_required")


@pytest.mark.parametrize("stage, phase, location", (
    ("RICH", "RUN_RESOLVE_RICH", "modal_standalone_runner.resolve_rich"),
    ("DERIVE", "RUN_RESOLVE_DERIVE", "modal_standalone_runner.resolve_derive"),
    ("REPARSE", "RUN_RESOLVE_REPARSE", "modal_standalone_runner.resolve_reparse"),
))
def test_public_resolve_projects_only_closed_inner_stage(
        tmp_path, monkeypatch, stage, phase, location):
    plan, context, events = _setup(tmp_path, monkeypatch)

    def fail(*_args, **_kwargs):
        raise ModalPackagedResolutionUnavailable(stage)

    monkeypatch.setattr(TrainingAPI, "resolve", fail)
    with pytest.raises(runner.ModalStandalonePhaseUnavailable) as caught:
        runner.run_modal_standalone_job(
            plan=plan, context=context, modal_profile="explicit",
            modal_environment="main",
        )
    assert (caught.value.phase, caught.value.failure_class,
            caught.value.location, caught.value.retry_authorized) == (
                phase, "UNAVAILABLE", location, False,
            )
    assert caught.value.args == ("modal_standalone_phase_unavailable",)
    assert events == ["bootstrap", "cpu"]


def test_closed_resolve_error_during_start_keeps_start_ambiguity(tmp_path, monkeypatch):
    plan, context, events = _setup(tmp_path, monkeypatch)

    def fail(*_args, **_kwargs):
        raise ModalPackagedResolutionUnavailable("RICH")

    monkeypatch.setattr(TrainingAPI, "start", fail)
    with pytest.raises(runner.ModalStandalonePhaseUnavailable) as caught:
        runner.run_modal_standalone_job(
            plan=plan, context=context, modal_profile="explicit",
            modal_environment="main",
        )
    assert caught.value.phase == "RUN_START_INDETERMINATE"
    assert caught.value.retry_authorized is False
    assert events == ["bootstrap", "cpu"]


def test_post_cpu_start_rejection_remains_indeterminate(tmp_path, monkeypatch):
    plan, context, events = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        TrainingAPI, "start", lambda *_args, **_kwargs: SimpleNamespace(accepted=False),
    )
    with pytest.raises(runner.ModalStandalonePhaseUnavailable) as caught:
        runner.run_modal_standalone_job(
            plan=plan, context=context, modal_profile="explicit",
            modal_environment="main",
        )
    assert caught.value.phase == "RUN_START_INDETERMINATE"
    assert caught.value.retry_authorized is False
    assert events == ["bootstrap", "cpu"]
