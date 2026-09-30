from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from io import BytesIO
from hashlib import sha256
from types import ModuleType, SimpleNamespace

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_sources import (
    LocalTrainingInputPathV1, TrainingPreparationConfigV1,
)
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from tuner.execution.providers.modal.packaged_composition import compose_modal_packaged_adapter
from tuner.execution.providers.modal.packaged_composition import (
    ModalPackagedAuthorityPorts, ModalPackagedCatalogPorts, ModalPackagedSourcePorts,
)
from tuner.execution.providers.modal.packaged_staging import (
    ModalPackagedInputStager, ModalPackagedStageReceipt,
)
from tuner.execution.providers.modal.packaged_reader import (
    ModalPackagedArtifactMember, ModalPackagedCompletionObservation,
    ModalPackagedReader,
)
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.contracts import provider_entry_identity
from tuner.training.contracts import (
    CanonicalDocument, ResourceSpec, RetainedTrainingInputStreamLease, RuntimeSpec,
)
from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1
from tuner.training.input_preparation import (
    PUBLISHED_PREPARED_DATASET_NORMALIZER_V1,
    PublishedPreparedDatasetNormalizerConfigV1,
    PublishedPreparedDatasetNormalizerV1,
    TrainingInputPreparationServiceV1, default_dataset_format_verifiers_v1,
)
from tuner.training.packaged_compilation import PACKAGED_CONTEXT_SCHEMA, compile_packaged_sft_workload
from tuner.project.context import ProjectContext
from tuner.training.modal_host_composition import compose_modal_packaged_reference_host

from examples.modal_chat.authority import UTCClock
from tests.execution.providers.test_modal_packaged_composition import _components
from tests.execution.providers.test_modal_packaged_binding import _binding
from tests.execution.providers.test_modal_packaged_dispatch import Auth
from tests.execution.providers.test_modal_packaged_deployment import Reader
from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume
from tests.training.test_modal_host_effects import _Storage
from tests.training.test_packaged_execution_material import packaged_fixture
from tests.training.test_input_preparation import _ExplicitTestRootAuthority
from tests.dataset_prep.test_context_messages_v2 import _config
from tuner.dataset_prep import DatasetPublicationUncertainV1, prepare_dataset_v2


class _Secrets:
    def resolve(self, _ref):
        return "host-only-test-authority-value-" + "x" * 40


class _Effects:
    bindings = object()

    def authorize(self, requirements):
        raise AssertionError("composition must not authorize paid effects")

    def bind(self, grant, *, operation, requirements):
        raise AssertionError("composition must not bind paid effects")

    def authenticate(self, binding):
        raise AssertionError("composition must not read retained bindings")


def test_reference_host_composes_public_api_without_provider_or_catalog_reads(tmp_path):
    _, arguments, events, _ = _components()
    adapter = compose_modal_packaged_adapter(**arguments)
    public, components = packaged_fixture()
    host = compose_modal_packaged_reference_host(
        adapter=adapter, effects=_Effects(),
        prepared_request=TrainingRequest("request", "project", public.canonical_json()),
        run=TrainingRunRef("run-packaged", "project"),
        components=components,
        context=ProjectContext.standalone(engine_root=tmp_path),
        input_preparation=None, clock=UTCClock(), secrets=_Secrets(),
        authority_secret=SecretRef("env", "SYNAPTIC_TEST_AUTHORITY"),
        reader_secret=SecretRef("env", "SYNAPTIC_TEST_READER"),
        profile_digest="a" * 64, quote_digest="b" * 64,
        maximum_cost_minor_units=100,
    )
    assert host.api.training is not None
    assert host.api.runs is not None
    assert host.api.artifacts is not None
    loaded = host.api.training.load(public.canonical_json())
    resolved = host.api.training.resolve(loaded)
    plan = host.api.training.plan(resolved, ProviderRef("modal", adapter.effect_executor.profile_ref))
    preflight = host.api.training.preflight(plan)
    assert preflight.authorization[0].maximum_cost_minor_units == 100
    assert events == []


def test_public_training_api_prepares_existing_publication_without_provider(tmp_path):
    _, _, _, dataset_config = _config(tmp_path / "source")
    root = (tmp_path / "prepared").resolve()
    try:
        publication = prepare_dataset_v2(dataset_config, root)
    except DatasetPublicationUncertainV1:
        publication = prepare_dataset_v2(dataset_config, root)
    semantic = publication.semantic_identity
    service = TrainingInputPreparationServiceV1(
        prepared_root=root,
        normalizers={PUBLISHED_PREPARED_DATASET_NORMALIZER_V1:
            PublishedPreparedDatasetNormalizerV1(authorized_root=root)},
        format_verifiers=default_dataset_format_verifiers_v1(),
        root_authority=_ExplicitTestRootAuthority(),
    )
    _, arguments, events, _ = _components()
    adapter = compose_modal_packaged_adapter(**arguments)
    public, components = packaged_fixture()
    host = compose_modal_packaged_reference_host(
        adapter=adapter, effects=_Effects(),
        prepared_request=TrainingRequest("request", "project", public.canonical_json()),
        run=TrainingRunRef("run-packaged", "project"), components=components,
        context=ProjectContext.standalone(engine_root=tmp_path),
        input_preparation=service, clock=UTCClock(), secrets=_Secrets(),
        authority_secret=SecretRef("env", "SYNAPTIC_TEST_AUTHORITY"),
        reader_secret=SecretRef("env", "SYNAPTIC_TEST_READER"),
        profile_digest="a" * 64, quote_digest="b" * 64,
        maximum_cost_minor_units=100,
    )
    prepared = host.api.training.prepare(
        LocalTrainingInputPathV1(publication.path / "dataset.jsonl"),
        TrainingPreparationConfigV1(
            "request", "project", public.canonical_json(),
            PublishedPreparedDatasetNormalizerConfigV1(
                semantic.dataset_digest, semantic.split_counts["train"],
                semantic.split_counts["validation"],
            ),
        ),
    )
    assert prepared.prepared.identity.revision == semantic.dataset_digest
    assert events == []


def test_public_training_api_stages_and_submits_once_with_fake_modal(tmp_path, monkeypatch):
    bound, arguments, events, _ = _components()
    effects_binding = _binding()
    assert bound == effects_binding
    arguments["deployment_observer"]._reader = Reader(effects_binding.provider_facts)
    public, original = packaged_fixture()
    release = effects_binding.runtime_release
    execution = effects_binding.execution_binding
    components = replace(
        original, execution_source=execution,
        execution_context=CanonicalDocument.from_mapping({
            "schema_version": PACKAGED_CONTEXT_SCHEMA,
            "runtime_release": release.to_dict(),
        }),
        runtime=RuntimeSpec(
            release.image_ref, release.installed_distributions_digest,
            release.python_version,
        ),
        resources=ResourceSpec("A100-80GB", 1, 1800),
    )

    class Source:
        def open_lease(self):
            from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
            identity = PreparedTrainingInputIdentity.from_dict(
                execution.to_dict()["prepared_input"],
            )
            return RetainedTrainingInputStreamLease(identity, BytesIO(b"unused"))

    storage = _Storage()
    signer = Auth()
    effects = ModalPackagedHostEffectsV1(
        storage=storage, runtime_release=release,
        provider_binding=effects_binding.provider_binding,
        provider_facts=effects_binding.provider_facts,
        execution_binding=execution, retained_source=Source(),
        workload_bytes=compile_packaged_sft_workload(
            resolved_config=components.resolved_config,
        ).canonical_bytes,
        artifact_policy=components.artifact_policy,
        signer=signer, key_ref="dispatch-key",
        maximum_cost_minor_units=100,
    )

    stage_calls = []

    def stage_once(_stager, material):
        stage_calls.append(material.descriptor.stage_effect_id)
        descriptor = material.descriptor
        return ModalPackagedStageReceipt(
            descriptor.stage_effect_id, material.execution_binding_digest,
            descriptor.artifact_volume_id, descriptor.relative_path,
            descriptor.identity.size_bytes, descriptor.identity.content_digest,
            provider_entry_identity(
                descriptor.artifact_volume_id, descriptor.relative_path,
                descriptor.identity.size_bytes,
            ),
        )

    monkeypatch.setattr(ModalPackagedInputStager, "stage_once", stage_once)
    FakeVolume.registry = {
        "control-name": FakeVolume(effects_binding.provider_facts.control_volume_id),
        "artifact-name": FakeVolume(effects_binding.provider_facts.artifact_volume_id),
    }
    def unexpected_commit(_volume):
        raise AssertionError("host commit is redundant")
    monkeypatch.setattr(FakeVolume, "commit", unexpected_commit, raising=False)
    async_utils = ModuleType("modal._utils.async_utils")
    def create_blocking(operation):
        assert getattr(operation, "__self__", None) is None
        return lambda *args, **kwargs: asyncio.run(operation(*args, **kwargs))
    async_utils.synchronizer = SimpleNamespace(create_blocking=create_blocking)
    monkeypatch.setitem(sys.modules, "modal", ModuleType("modal"))
    monkeypatch.setitem(sys.modules, "modal._utils", ModuleType("modal._utils"))
    monkeypatch.setitem(sys.modules, "modal._utils.async_utils", async_utils)
    class ExactReader:
        def __init__(self, *, sdk, client):
            assert sdk is arguments["sdk"] and client is arguments["client"]
        async def read_exact(self, *, volume_id, path, expected_size, expected_sha256, max_bytes):
            assert expected_size == max_bytes == 32
            value = next(volume.files[path] for volume in FakeVolume.registry.values()
                         if volume.object_id == volume_id)
            assert len(value) == 32 and sha256(value).hexdigest() == expected_sha256
            return value
    monkeypatch.setattr(
        "tuner.execution.providers.modal.packaged_transport.BoundedModalVolumeReader",
        ExactReader,
    )
    arguments["catalogs"] = ModalPackagedCatalogPorts(
        effects.bindings, effects.stage_receipts, effects.calls,
    )
    arguments["sources"] = ModalPackagedSourcePorts(
        effects.stages, effects.dispatches, effects.marker_materials,
    )
    arguments["authorities"] = ModalPackagedAuthorityPorts(effects, signer)
    adapter = compose_modal_packaged_adapter(**arguments)
    host = compose_modal_packaged_reference_host(
        adapter=adapter, effects=effects,
        prepared_request=TrainingRequest("request", "project", public.canonical_json()),
        run=TrainingRunRef("run-packaged", "project"),
        components=components,
        context=ProjectContext.standalone(engine_root=tmp_path),
        input_preparation=None, clock=UTCClock(), secrets=_Secrets(),
        authority_secret=SecretRef("env", "SYNAPTIC_TEST_AUTHORITY"),
        reader_secret=SecretRef("env", "SYNAPTIC_TEST_READER"),
        profile_digest="a" * 64, quote_digest="b" * 64,
        maximum_cost_minor_units=100,
    )
    api = host.api
    request = api.training.load(public.canonical_json())
    resolved = api.training.resolve(request)
    plan = api.training.plan(resolved, ProviderRef("modal", adapter.effect_executor.profile_ref))
    preflight = api.training.preflight(plan)
    assert events == []
    try:
        started = api.training.start(plan, preflight)
    except Exception as error:
        raise AssertionError((str(error), events, storage.attempts.claimed,
                              storage.catalogs["packaged-bindings-v1"].values.keys(),
                              storage.catalogs["packaged-stage-receipts-v1"].values.keys(),
                              storage.catalogs["packaged-calls-v1"].values.keys(),
                              stage_calls)) from None
    assert started.accepted is True
    assert len(storage.attempts.claimed) == 2

    completion_digest = "e" * 64

    def observe_completion(_reader, submit_binding, *, provider_job_ref):
        assert provider_job_ref == "fc-1"
        effect_id = submit_binding.command.operation.effect.effect_id
        members = tuple(ModalPackagedArtifactMember(
            role, f"operations/{effect_id}/output/{role}", 4,
            sha256(b"data").hexdigest(), f"entry-{role}",
        ) for role in sorted((
            "workload_record", "training_lineage", "training_metrics",
            "final_model", "tokenizer",
        )))
        return ModalPackagedCompletionObservation(
            effect_id, submit_binding.command_digest, provider_job_ref,
            submit_binding.runtime_release.manifest_digest,
            submit_binding.provider_binding.binding_digest,
            submit_binding.execution_binding.binding_digest,
            completion_digest, members,
        )

    def iter_artifact(_reader, _binding, _observation, *, role, maximum_bytes):
        assert maximum_bytes >= 4
        assert role in {"workload_record", "training_lineage", "training_metrics", "final_model", "tokenizer"}
        yield b"data"

    monkeypatch.setattr(ModalPackagedReader, "observe_completion", observe_completion)
    monkeypatch.setattr(ModalPackagedReader, "iter_artifact", iter_artifact)
    monkeypatch.setattr(
        ModalPackagedCoordinatorReaderV1, "_poll_packaged_call",
        lambda _self, _binding, _ref: (ModalFunctionCallState.RETURNED, completion_digest),
    )
    assert api.runs.outcome(started.run).state.value == "succeeded"
    assert api.runs.verify(started.run).verified is True
    stream = api.runs.artifacts(RunArtifactRequest(started.run, "final_model", 4))
    assert b"".join(stream.iter_bytes()) == b"data"
