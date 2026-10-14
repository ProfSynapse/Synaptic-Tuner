"""Real public Data/fixture/preparation handoff; provider facts are CPU fixtures."""

from dataclasses import replace, fields
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import shutil

import pytest
import yaml
import SynthChat

import test_environment_admission_v1 as admission
import test_environment_sessions_v1 as sessions
from synaptic_tuner.api.v1.data_facade import (
    DataAPI,
    DataMode,
    DataRunState,
    DataScenarioTarget,
    DatasetValidateRequest,
)
from synaptic_tuner.api.v1.training_facade import TrainingAPI
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import (
    LocalTrainingInputPathV1,
    TrainingPreparationConfigV1,
)
from tuner.training.input_preparation import (
    TrainingInputPreparationServiceV1,
    PosixPrivatePreparedRootAuthorityV1,
)
from tuner.training.conversation_preparation import (
    ConversationTrainingInputNormalizerV1,
    ConversationPreparedInputFormatVerifierV1,
)
from tuner.dataset_prep.conversation_messages import (
    NORMALIZER_REF,
    ROW_SCHEMA,
    ConversationNormalizerConfigV1,
)
from test_public_training_input_v1 import _document


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def data_workflow(tmp_path, monkeypatch):
    # Helper source/config locations derive the actual installed package.
    monkeypatch.setattr(
        admission, "ROOT", Path(SynthChat.__file__).resolve().parent.parent
    )
    base = admission.harness.__wrapped__(tmp_path)
    selected = sessions.selected.__wrapped__(
        base, tmp_path, monkeypatch, SimpleNamespace(param="bounded_local")
    )
    operations, request, ports, host, _, scenario, _, scratch, runtimes, _ = selected
    fixture_reads = []
    original_read = sessions.BoundedLocalEnvironmentRuntime.read_text

    def real_read(runtime, path):
        result = original_read(runtime, path)
        fixture_reads.append((path, result))
        return result

    monkeypatch.setattr(sessions.BoundedLocalEnvironmentRuntime, "read_text", real_read)
    environment = {
        "fixture": {"files": {"declared.txt": "bound fixture"}},
        "assertions": [
            {"type": "file_contains", "path": "declared.txt", "text": "bound fixture"}
        ],
    }
    definitions = {
        name: {
            "type": "behavioral",
            "environment_execution": "required",
            "environment": environment,
            "prompts": {
                "system": "system " + name,
                "user": "user " + name,
                "assistant": "reply",
            },
            "rubrics": {"response": ["repair"]},
        }
        for name in ("group_a", "group_b")
    }
    scenario.write_text(yaml.safe_dump({"scenarios": definitions}))
    sessions.update_host(
        host,
        execution={
            "tool_action_hints": {},
            "candidate_projection": {
                "response": ["conversations", -1],
                "system": ["conversations", 0, "content"],
                "environment": ["metadata", "resolved_environment_config"],
            },
        },
    )
    clients = [
        sessions.ScriptedClient(
            [
                "system a",
                "user a",
                "initial a",
                json.dumps({"role": "assistant", "content": "generated revised a"}),
                "system b",
                "user b",
                "initial b",
                json.dumps({"role": "assistant", "content": "generated revised b"}),
            ],
            [0.1, 0.95, 0.95] * 2,
        ),
        sessions.ScriptedClient(
            [
                json.dumps({"role": "assistant", "content": "improved revised a"}),
                json.dumps({"role": "assistant", "content": "improved revised b"}),
            ],
            [0.1, 0.95, 0.95] * 2,
        ),
    ]

    class Factory:
        def __init__(self):
            self.created = []
            self.fixture_reads = fixture_reads

        def create(self, **kwargs):
            client = clients.pop(0)
            self.created.append(client)
            return client

    operations._ports = replace(ports, llm_clients=Factory())
    api = DataAPI(operations)
    generate = replace(
        request,
        request_id="joined-generate",
        scenarios=(DataScenarioTarget("group_a", 1), DataScenarioTarget("group_b", 1)),
    )
    started = api.start(api.plan(generate))
    generated = api.show(started.run)
    assert generated.state is DataRunState.SUCCEEDED and generated.rows_written == 2
    improve = replace(
        request,
        request_id="joined-improve",
        mode=DataMode.IMPROVE,
        scenarios=(),
        input_dataset_ref=generate.output_ref,
        output_ref=str(tmp_path / "improved.jsonl"),
        rubric_refs=("repair",),
    )
    started = api.start(api.plan(improve))
    improved = api.show(started.run)
    assert improved.state is DataRunState.SUCCEEDED and improved.rows_written == 2
    report = api.validate(
        DatasetValidateRequest(request.project_ref, improve.output_ref)
    )
    assert not report.findings and (report.rows_checked, report.rows_valid) == (2, 2)
    assert not list(scratch.iterdir()) and runtimes and not clients
    assert fixture_reads and all(
        path == "declared.txt" and content == "bound fixture"
        for path, content in fixture_reads
    )
    return (
        api,
        generated,
        improved,
        Path(improve.output_ref),
        operations._ports,
        scenario,
    )


@pytest.mark.parametrize("bind_outcome", [True, False])
def test_real_data_fixture_improvement_prepare_and_retained_source(
    tmp_path, monkeypatch, bind_outcome
):
    api, generated, outcome, dataset, ports, scenario = data_workflow(
        tmp_path, monkeypatch
    )
    from SynthChat.result_writer import metadata_path

    metadata = metadata_path(dataset)
    rows = [json.loads(line) for line in dataset.read_bytes().splitlines()]
    assert {row["metadata"]["scenario"] for row in rows} == {"group_a", "group_b"}
    assert [row["conversations"][-1]["content"] for row in rows] == [
        "improved revised a",
        "improved revised b",
    ]
    source = dataset.parent
    config_document = {
        "schema_version": "synaptic-conversation-preparation/v1",
        "source": {
            "dataset": {
                "member": dataset.name,
                "sha256": sha(dataset.read_bytes()),
                "size_bytes": dataset.stat().st_size,
            },
            "metadata": {
                "member": metadata.name,
                "sha256": sha(metadata.read_bytes()),
                "size_bytes": metadata.stat().st_size,
                "row_count_path": ["rows_written"],
            },
            "rows": outcome.rows_written,
            "provenance": {"origin": "public DataAPI"},
            "data_outcome": outcome.to_dict() if bind_outcome else None,
        },
        "projection": {
            "messages_path": ["conversations"],
            "role_field": "role",
            "content_field": "content",
            "role_map": {"system": "system", "user": "user", "assistant": "assistant"},
        },
        "selection": {"rows": None, "target_index": None},
        "grouping": {"path": ["metadata", "scenario"]},
        "split": {"kind": "heldout_groups", "validation_groups": ["group_b"]},
    }
    if bind_outcome:
        wrong = deepcopy(config_document)
        wrong["source"]["data_outcome"] = generated.to_dict()
        with pytest.raises(ValueError):
            ConversationNormalizerConfigV1.from_dict(wrong)
    config_path = tmp_path / "conversation-config.json"
    config_path.write_text(json.dumps(config_document))
    config = ConversationNormalizerConfigV1.from_dict(
        json.loads(config_path.read_text())
    )
    prepared_root = tmp_path / "prepared"
    service = TrainingInputPreparationServiceV1(
        prepared_root=prepared_root,
        normalizers={
            NORMALIZER_REF: ConversationTrainingInputNormalizerV1(
                allowed_source_roots=(source,)
            )
        },
        format_verifiers={ROW_SCHEMA: ConversationPreparedInputFormatVerifierV1()},
        root_authority=PosixPrivatePreparedRootAuthorityV1(),
    )
    training = TrainingAPI(
        service, clock=SimpleNamespace(now=lambda: "2026-10-14T00:00:00Z")
    )
    public = _document()
    public["model"] = {
        "ref": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
    public["hyperparameters"].update(
        dataset_format="messages",
        completion_only_loss=True,
        assistant_only_loss=False,
        use_preassigned_splits=True,
        split_dataset=False,
        prompt_render="prompt_completion",
        packing=False,
        require_memory_efficient_loss=True,
    )
    public["artifacts"].update(
        required_kinds=[
            "final_model",
            "tokenizer",
            "training_lineage",
            "training_metrics",
            "workload_record",
        ],
        retain_checkpoints=False,
    )
    prepared = training.prepare(
        LocalTrainingInputPathV1(source),
        TrainingPreparationConfigV1(
            "joined-prepare",
            outcome.run.project_ref,
            TrainingInputV1.from_dict(public).canonical_json(),
            config,
        ),
    )
    candidate_api, provider, compiled, components, owned_build_path = (
        candidate_training_api(prepared, tmp_path, monkeypatch)
    )
    release = json.loads(components.execution_context.canonical_json)["runtime_release"]
    from tuner.runtime.releases import (
        PackagedTrainingRuntimeReleaseV1,
        PackagedExecutionBindingV1,
        ProviderRuntimeBindingV1,
    )

    candidate = PackagedTrainingRuntimeReleaseV1.from_dict(release)
    original = {
        field.name: getattr(candidate, field.name)
        for field in fields(candidate)
        if field.name not in {"schema_version", "manifest_digest"}
    }
    original["compatible_dataset_formats"] = ("syntunia-sft-row/v2",)
    historical = PackagedTrainingRuntimeReleaseV1.build(**original)
    historical_provider = ProviderRuntimeBindingV1.build(
        provider_ref="provider:cpu-fixture",
        runtime_release=historical,
        provider_facts_schema="fixture-runtime-facts/v1",
        provider_facts_digest=sha(b"synthetic CPU provider facts"),
    )
    historical_binding = PackagedExecutionBindingV1.build(
        run_ref="historical-refusal",
        runtime_release=historical,
        provider_runtime_binding=historical_provider,
        **prepared.prepared.identity.execution_binding_fields(),
        workload_digest=compiled.fingerprint,
        configuration_digest=components.execution_source.configuration_digest,
        artifact_policy_digest=components.execution_source.artifact_policy_digest,
    )
    from tuner.training.packaged_boundary import validate_packaged_material
    from tuner.training.contracts import CanonicalDocument, TrainingRequest

    historical_components = replace(
        components,
        execution_source=historical_binding,
        execution_context=CanonicalDocument.from_mapping(
            {
                "schema_version": "synaptic-packaged-execution-context/v1",
                "runtime_release": historical.to_dict(),
            }
        ),
    )
    with pytest.raises(ValueError, match="runtime release does not admit"):
        validate_packaged_material(
            request=TrainingRequest(
                CanonicalDocument(prepared.prepared.request.canonical_json)
            ),
            material=historical_components,
        )
    resolved = candidate_api.resolve(prepared.prepared.request)
    plan = candidate_api.plan(resolved, provider)
    llm_calls = [
        (client.chat_calls, client.judge_calls) for client in ports.llm_clients.created
    ]
    assert len(llm_calls) == 2 and all(
        chat > 0 and judge > 0 for chat, judge in llm_calls
    )
    before = prepared.retained_source.open_lease()
    raw = before.take_stream().read(prepared.prepared.identity.size_bytes)
    before.close()
    dataset.unlink()
    metadata.unlink()
    scenario.unlink()
    config_path.unlink()
    owned_build_path.unlink()
    generated_path = tmp_path / "host-output.jsonl"
    generated_path.unlink()
    metadata_path(generated_path).unlink()
    shutil.rmtree(ports.config_dir)
    shutil.rmtree(ports.rubrics_dir)
    lease = prepared.retained_source.open_lease()
    assert (
        lease.identity == prepared.prepared.identity
        and lease.take_stream().read(len(raw)) == raw
    )
    assert sha(raw) == prepared.prepared.identity.content_digest
    assert (
        json.loads(prepared.prepared.request.canonical_json)["model"] == public["model"]
    )
    lease.close()
    assert candidate_api.resolve(prepared.prepared.request) == resolved
    assert candidate_api.plan(resolved, provider) == plan
    assert plan.basis.workload_digest == compiled.fingerprint
    assert [
        (client.chat_calls, client.judge_calls) for client in ports.llm_clients.created
    ] == llm_calls


def candidate_training_api(prepared, tmp_path, monkeypatch):
    """Real planning ports with synthetic CPU release facts, never observations."""
    import os
    import tuner
    from tuner.runtime_profiles import load_runtime_profile
    from tuner.cloud.derived_training_image import load_profile
    from tuner.execution.providers.modal.runtime_build import plan_modal_build_material
    from tuner.runtime.releases import (
        PackagedTrainingRuntimeReleaseV1,
        ProviderRuntimeBindingV1,
        PackagedExecutionBindingV1,
    )
    from tuner.training.packaged_compilation import (
        compile_packaged_sft_workload,
        packaged_configuration_digest,
        packaged_artifact_policy_digest,
        PACKAGED_ENTRYPOINT,
    )
    from tuner.training.contracts import (
        ArtifactPolicy,
        CanonicalDocument,
        ResolvedTrainingComponents,
        RuntimeSpec,
        ResourceSpec,
    )
    from tuner.training.modal_host_requests import (
        ModalPackagedRequestsV1,
        ModalPackagedPlanPortsV1,
    )
    from tuner.project.context import ProjectContext
    from synaptic_tuner.api.v1.results import TrainingRunRef
    from synaptic_tuner.api.v1.providers import (
        ProviderRef,
        ProviderDescriptor,
        ProviderCapabilities,
    )
    from synaptic_tuner.api.v1.reference.repositories import DurablePlanningStoreV1
    from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1
    from tuner.training.coordinator_service import CoordinatorTrainingService
    from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
    from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
    from tuner.execution.foundation_v2.references import ExecutionScopeV1
    from tuner.execution.foundation_v2.executors import (
        ExecutorDescriptorV1,
        AdapterDescriptorV1,
    )

    package_root = Path(tuner.__file__).resolve().parent.parent
    profiles = package_root / "Trainers/runtime_profiles"
    profile = load_runtime_profile("qwen35-conversation-sft-candidate-v1", profiles)
    public = TrainingInputV1.from_json(prepared.prepared.request.canonical_json)
    profile.resolve(
        model=public.model.ref, model_revision=public.model.revision, method="sft"
    )
    build_path = Path(
        os.environ.get(
            "SYNAPTIC_TEST_CONVERSATION_BUILD_PROFILE",
            str(
                package_root
                / "Trainers/image_profiles/qwen35_4b_conversation_sft_candidate_v1/profile.yaml"
            ),
        )
    )
    owned_build_path = tmp_path / "host-candidate-build.yaml"
    shutil.copy2(build_path, owned_build_path)
    build_path = owned_build_path
    build = load_profile(build_path)
    from tuner.execution.providers.modal import runtime_build
    import subprocess

    expected_root = Path(runtime_build.__file__).resolve().parents[4]
    git_calls = []
    # Explicit synthetic CPU Git observation: installed wheels contain no .git.
    # This is NOT source/build/provider admission or an observed checkout.
    fixture_commit = "d" * 40

    def observe_fixture_git(argv, **kwargs):
        assert argv == ["git", "-C", str(expected_root), "rev-parse", "HEAD"]
        assert kwargs == {"capture_output": True, "timeout": 30, "check": False}
        git_calls.append(tuple(argv))
        assert len(git_calls) == 1
        return subprocess.CompletedProcess(
            argv, 0, (fixture_commit + "\n").encode(), b""
        )

    with monkeypatch.context() as scope:
        scope.setattr(
            runtime_build, "subprocess", SimpleNamespace(run=observe_fixture_git)
        )
        intent = plan_modal_build_material(build_path)
    assert intent["engine_source_commit"] == fixture_commit and len(git_calls) == 1
    assert intent["profile_digest"] == build.canonical_sha256
    compatibility = build.packaged_runtime["capabilities"]["compatibility"]
    compatibility_document = json.loads(
        (
            profiles / "qwen35-conversation-sft-candidate-v1.compatibility.json"
        ).read_bytes()
    )
    assert compatibility_document == {
        "release_ref": "runtime:" + profile.name,
        "compatibility": compatibility,
    }
    assert compatibility["dataset_formats"] == [ROW_SCHEMA]
    assert (
        build.packaged_runtime["wheel"]["sha256"]
        == "13aff34e7e0a0414a46770637b54d2393320097a02fc5bc0a42fe1e7d48f8199"
    )
    config = CanonicalDocument.from_mapping(
        {
            "schema_version": "synaptic-packaged-sft-config/v1",
            "method": "sft",
            "execution": {"mode": "packaged_runtime"},
            "model": {**public.model.to_dict(), "load_in_4bit": False},
            "dataset": prepared.prepared.identity.to_dict(),
            "sft": public.hyperparameters.to_dict(),
        }
    )
    compiled = compile_packaged_sft_workload(resolved_config=config)
    policy = ArtifactPolicy(
        public.artifacts.required_kinds, public.artifacts.retain_checkpoints
    )
    closure = json.loads(
        (
            package_root / "tuner/runtime/manifests/packaged-training-worker-v1.json"
        ).read_bytes()
    )
    # Synthetic contract metadata cannot establish installed/deployed image facts.
    release = PackagedTrainingRuntimeReleaseV1.build(
        release_ref="runtime:cpu-fixture-conversation-candidate",
        package_name="synaptic-tuner",
        package_version="1.1.0",
        package_digest=build.packaged_runtime["wheel"]["sha256"],
        source_provenance_digest=sha(b"synthetic CPU contract provenance"),
        worker_entrypoint=PACKAGED_ENTRYPOINT,
        worker_closure_digest=closure["closure_digest"],
        image_ref=profile.image,
        image_digest=profile.image.split(":")[-1],
        python_implementation="cpython",
        python_version=profile.runtime_facts["python_version"],
        python_executable=build.packaged_runtime["python"]["executable"],
        python_executable_digest=build.packaged_runtime["python"]["executable_digest"],
        installed_distributions_digest=sha(
            json.dumps(
                sorted(
                    (item["name"], item["version"])
                    for item in json.loads(profile.inventory_path.read_bytes())[
                        "distributions"
                    ]
                ),
                separators=(",", ":"),
            ).encode()
        ),
        installed_distribution_count=profile.distribution_count,
        platform_system="linux",
        platform_machine="x86_64",
        cuda_version=None,
        runtime_facts={"synthetic_cpu_contract": True},
        compatible_methods=tuple(compatibility["methods"]),
        compatible_models=tuple(
            (item["ref"], item["revision"]) for item in compatibility["models"]
        ),
        compatible_dataset_formats=tuple(compatibility["dataset_formats"]),
        workload_schema=compiled.schema_version,
        prepared_input_schema="synaptic-prepared-training-input/v1",
        artifact_contract_schema="synaptic-sft-artifacts/v1",
    )
    provider_runtime = ProviderRuntimeBindingV1.build(
        provider_ref="provider:cpu-fixture",
        runtime_release=release,
        provider_facts_schema="fixture-runtime-facts/v1",
        provider_facts_digest=sha(b"synthetic CPU provider facts"),
    )
    execution = PackagedExecutionBindingV1.build(
        run_ref="joined-candidate",
        runtime_release=release,
        provider_runtime_binding=provider_runtime,
        **prepared.prepared.identity.execution_binding_fields(),
        workload_digest=compiled.fingerprint,
        configuration_digest=packaged_configuration_digest(config),
        artifact_policy_digest=packaged_artifact_policy_digest(policy),
    )
    components = ResolvedTrainingComponents(
        execution,
        CanonicalDocument.from_mapping(
            {
                "schema_version": "synaptic-packaged-execution-context/v1",
                "runtime_release": release.to_dict(),
            }
        ),
        config,
        RuntimeSpec(
            release.image_ref,
            release.installed_distributions_digest,
            release.python_version,
        ),
        ResourceSpec("cpu"),
        policy,
    )
    run = TrainingRunRef("joined-candidate", prepared.prepared.request.project_ref)
    requests = ModalPackagedRequestsV1(
        prepared_request=prepared.prepared.request,
        run=run,
        components=components,
        context=ProjectContext.standalone(engine_root=tmp_path),
    )
    provider = ProviderRef("modal", profile.name)
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1",
        "modal",
        "Synthetic CPU planning",
        "1.0.0",
        ProviderCapabilities(False, False, False, False, False, False),
    )
    binding = ProviderExecutionBindingV1(
        provider,
        descriptor.descriptor_digest,
        profile.profile_sha256.removeprefix("sha256:"),
        ExecutionScopeV1("cpu-fixture", "cpu-fixture"),
        ExecutorDescriptorV1("modal", "unavailable", "1.0.0"),
        AdapterDescriptorV1("modal", "unavailable", "1.0.0").digest,
        *[sha(value) for value in (b"resources", b"quote", b"secrets")],
    )
    clock = SimpleNamespace(now=lambda: "2026-10-14T00:00:00Z")
    planning = ModalPackagedPlanPortsV1(
        provider=provider,
        descriptor=descriptor,
        binding=binding,
        clock=clock,
        maximum_cost_minor_units=1,
    )
    store = DurablePlanningStoreV1(InMemoryDurableRecordStoreV1())

    class NoEffects:
        def __getattr__(self, name):
            raise AssertionError("Planning must not execute " + name)

    denied = NoEffects()
    coordinator = TrainingCoordinatorV1(planning, store, *[denied] * 9, clock, requests)
    operations = CoordinatorTrainingService(
        loader=requests,
        resolver=requests,
        planning=planning,
        planning_store=store,
        coordinator=coordinator,
        clock=clock,
    )
    return (
        TrainingAPI(operations, clock=clock),
        provider,
        compiled,
        components,
        owned_build_path,
    )
