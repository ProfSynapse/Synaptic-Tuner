"""Cold host storage proof with explicitly retained fake-provider authority.

Reuses the packaged synthetic fixture, not a live service. Fresh connections and
composition recover durable workflow/catalog values; secret keys, admitted runtime
material and the same provider double are supplied explicitly. This does not
restore consumed source leases, one-use payloads, or production log proof.
"""

from __future__ import annotations

import asyncio
import sys
import pytest
from dataclasses import replace
from io import BytesIO
from hashlib import sha256
from types import ModuleType, SimpleNamespace

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import (
    RunArtifactRequest,
    RunListRequest,
    RunLogsRequest,
    RunOperationError,
)
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from tuner.execution.providers.modal.packaged_composition import (
    compose_modal_packaged_adapter,
)
from tuner.execution.providers.modal.packaged_composition import (
    ModalPackagedAuthorityPorts,
    ModalPackagedCatalogPorts,
    ModalPackagedSourcePorts,
)
from tuner.execution.providers.modal.packaged_staging import (
    ModalPackagedInputStager,
    ModalPackagedStageReceipt,
)
from tuner.execution.providers.modal.packaged_reader import (
    ModalPackagedArtifactMember,
    ModalPackagedCompletionObservation,
    ModalPackagedReader,
)
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.contracts import provider_entry_identity
from tuner.training.contracts import (
    CanonicalDocument,
    ResourceSpec,
    RetainedTrainingInputStreamLease,
    RuntimeSpec,
)
from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1
from tuner.training.packaged_compilation import (
    PACKAGED_CONTEXT_SCHEMA,
    compile_packaged_sft_workload,
)
from tuner.project.context import ProjectContext
from tuner.training.modal_host_composition import compose_modal_packaged_reference_host

from tests.execution.providers.test_modal_packaged_composition import _components
from tests.execution.providers.test_modal_packaged_binding import _binding
from tests.execution.providers.test_modal_packaged_dispatch import Auth
from tests.execution.providers.test_modal_packaged_deployment import Reader
from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume
from tests.training.test_packaged_execution_material import packaged_fixture


from datetime import datetime, timezone
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from tuner.training.modal_host_storage import ModalHostStorageV1


class UTCClock:
    def __init__(self, offset=0):
        self.offset = offset

    def now_epoch(self):
        import time

        return int(time.time()) + self.offset

    def now(self):
        return (
            datetime.fromtimestamp(self.now_epoch(), timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    now_iso = now


class _Secrets:
    def resolve(self, ref):
        return "host-only-test-authority-value-" + "x" * 40


class Storage:
    def __init__(self, journal):
        self.journal = journal
        self.attempts = self
        self.claimed = []

    def claim(self, key, evidence):
        result = self.journal.attempts.claim(key, evidence)
        self.claimed.append(key)
        return result

    def resolve(self, key):
        return self.journal.attempts.resolve(key)

    def catalog(self, *args, **kwargs):
        return self.journal.catalog(*args, **kwargs)


def test_cold_modal_host_reads_retained_state_without_replaying_effects(
    tmp_path, monkeypatch
):
    bound, arguments, events, _ = _components()
    effects_binding = _binding()
    assert bound == effects_binding
    arguments["deployment_observer"]._reader = Reader(effects_binding.provider_facts)
    public, original = packaged_fixture()
    release = effects_binding.runtime_release
    execution = effects_binding.execution_binding
    configuration = original.resolved_config
    components = replace(
        original,
        execution_source=execution,
        resolved_config=configuration,
        execution_context=CanonicalDocument.from_mapping(
            {
                "schema_version": PACKAGED_CONTEXT_SCHEMA,
                "runtime_release": release.to_dict(),
            }
        ),
        runtime=RuntimeSpec(
            release.image_ref,
            release.installed_distributions_digest,
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

    tmp_path.chmod(0o700)
    records = SQLiteDurableStoreV1(tmp_path / "workflow.sqlite")
    journal = ModalHostStorageV1(tmp_path / "attempts.sqlite", "test")
    storage = Storage(journal)
    signer = Auth()
    effects = ModalPackagedHostEffectsV1(
        storage=storage,
        runtime_release=release,
        provider_binding=effects_binding.provider_binding,
        provider_facts=effects_binding.provider_facts,
        execution_binding=execution,
        retained_source=Source(),
        workload_bytes=compile_packaged_sft_workload(
            resolved_config=components.resolved_config,
        ).canonical_bytes,
        artifact_policy=components.artifact_policy,
        signer=signer,
        key_ref="dispatch-key",
        maximum_cost_minor_units=100,
    )

    stage_calls = []
    boundaries = []
    original_bind = effects.bind

    def bind(*args, **kwargs):
        boundaries.append("BIND_ENTER")
        result = original_bind(*args, **kwargs)
        boundaries.append("BIND_RETURN")
        return result

    monkeypatch.setattr(effects, "bind", bind)

    def stage_once(_stager, material):
        boundaries.append("STAGE_TRANSPORT")
        stage_calls.append(material.descriptor.stage_effect_id)
        descriptor = material.descriptor
        return ModalPackagedStageReceipt(
            descriptor.stage_effect_id,
            material.execution_binding_digest,
            descriptor.artifact_volume_id,
            descriptor.relative_path,
            descriptor.identity.size_bytes,
            descriptor.identity.content_digest,
            provider_entry_identity(
                descriptor.artifact_volume_id,
                descriptor.relative_path,
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

        async def read_exact(
            self, *, volume_id, path, expected_size, expected_sha256, max_bytes
        ):
            assert expected_size == max_bytes == 32
            value = next(
                volume.files[path]
                for volume in FakeVolume.registry.values()
                if volume.object_id == volume_id
            )
            assert len(value) == 32 and sha256(value).hexdigest() == expected_sha256
            return value

    monkeypatch.setattr(
        "tuner.execution.providers.modal.packaged_transport.BoundedModalVolumeReader",
        ExactReader,
    )
    arguments["catalogs"] = ModalPackagedCatalogPorts(
        effects.bindings,
        effects.stage_receipts,
        effects.calls,
    )
    arguments["sources"] = ModalPackagedSourcePorts(
        effects.stages,
        effects.dispatches,
        effects.marker_materials,
    )
    arguments["authorities"] = ModalPackagedAuthorityPorts(effects, signer)
    adapter = compose_modal_packaged_adapter(**arguments)
    host_options = dict(
        adapter=adapter,
        effects=effects,
        records=records,
        streams=records,
        prepared_request=TrainingRequest("request", "project", public.canonical_json()),
        run=TrainingRunRef("run-packaged", "project"),
        components=components,
        context=ProjectContext.standalone(engine_root=tmp_path),
        input_preparation=None,
        clock=UTCClock(),
        secrets=_Secrets(),
        authority_secret=SecretRef("env", "SYNAPTIC_TEST_AUTHORITY"),
        reader_secret=SecretRef("env", "SYNAPTIC_TEST_READER"),
        profile_digest="a" * 64,
        quote_digest="b" * 64,
        maximum_cost_minor_units=100,
    )
    host = compose_modal_packaged_reference_host(**host_options)
    api = host.api
    request = api.training.load(public.canonical_json())
    resolved = api.training.resolve(request)
    plan = api.training.plan(
        resolved, ProviderRef("modal", adapter.effect_executor.profile_ref)
    )
    preflight = api.training.preflight(plan)
    assert events == []
    foundation_type = type(host.composition.foundation)
    original_execute = foundation_type.execute

    def execute(self, *args, **kwargs):
        boundaries.append("FOUNDATION_ENTER")
        result = original_execute(self, *args, **kwargs)
        boundaries.append("FOUNDATION_RETURN")
        return result

    monkeypatch.setattr(foundation_type, "execute", execute)
    try:
        started = api.training.start(plan, preflight)
    except Exception as error:
        code = getattr(error, "code", None)
        raise AssertionError(
            (type(error).__name__, getattr(code, "value", None), boundaries)
        ) from None
    assert started.accepted is True
    assert len(storage.attempts.claimed) == 2
    assert boundaries == [
        "BIND_ENTER",
        "BIND_RETURN",
        "FOUNDATION_ENTER",
        "STAGE_TRANSPORT",
        "FOUNDATION_RETURN",
        "BIND_ENTER",
        "BIND_RETURN",
        "FOUNDATION_ENTER",
        "FOUNDATION_RETURN",
    ]

    completion_digest = "e" * 64

    def observe_completion(_reader, submit_binding, *, provider_job_ref):
        assert provider_job_ref == "fc-1"
        effect_id = submit_binding.command.operation.effect.effect_id
        members = tuple(
            ModalPackagedArtifactMember(
                role,
                f"operations/{effect_id}/output/{role}",
                4,
                sha256(b"data").hexdigest(),
                f"entry-{role}",
            )
            for role in sorted(
                (
                    "workload_record",
                    "training_lineage",
                    "training_metrics",
                    "final_model",
                    "tokenizer",
                )
            )
        )
        return ModalPackagedCompletionObservation(
            effect_id,
            submit_binding.command_digest,
            provider_job_ref,
            submit_binding.runtime_release.manifest_digest,
            submit_binding.provider_binding.binding_digest,
            submit_binding.execution_binding.binding_digest,
            completion_digest,
            members,
        )

    def iter_artifact(_reader, _binding, _observation, *, role, maximum_bytes):
        assert maximum_bytes >= 4
        assert role in {
            "workload_record",
            "training_lineage",
            "training_metrics",
            "final_model",
            "tokenizer",
        }
        yield b"data"

    monkeypatch.setattr(ModalPackagedReader, "observe_completion", observe_completion)
    monkeypatch.setattr(ModalPackagedReader, "iter_artifact", iter_artifact)
    monkeypatch.setattr(
        ModalPackagedCoordinatorReaderV1,
        "_poll_packaged_call",
        lambda _self, _binding, _ref: (
            ModalFunctionCallState.RETURNED,
            completion_digest,
        ),
    )
    submit_binding = effects.bindings.resolve(next(reversed(storage.claimed)))
    assert api.runs.outcome(started.run).state.value == "succeeded"
    assert api.runs.verify(started.run).verified is True
    stream = api.runs.artifacts(RunArtifactRequest(started.run, "final_model", 4))
    assert b"".join(stream.iter_bytes()) == b"data"

    # The provider double and signing material remain explicit host inputs.
    # SQLite restores records/catalogs, not source leases or remote authority.
    before_events = list(events)
    prior_binding = effects.bindings.resolve(submit_binding.command_digest)
    prior_outcome = api.runs.show(started.run)
    records.close()
    journal.close()
    records = SQLiteDurableStoreV1(tmp_path / "workflow.sqlite")
    journal = ModalHostStorageV1(tmp_path / "attempts.sqlite", "test")

    class NoSource:
        def open_lease(self):
            pytest.fail("cold read reopened a consumed source")

    cold_effects = ModalPackagedHostEffectsV1(
        storage=journal,
        runtime_release=effects._release,
        provider_binding=effects._provider,
        provider_facts=effects._facts,
        execution_binding=effects._execution,
        retained_source=NoSource(),
        workload_bytes=effects._workload,
        artifact_policy=effects._policy,
        signer=signer,
        key_ref=effects._key_ref,
        maximum_cost_minor_units=effects._cost,
    )

    def no_effect(*args, **kwargs):
        pytest.fail("cold read attempted new effect authority")

    cold_effects.authorize = no_effect
    cold_effects.bind = no_effect
    arguments["catalogs"] = ModalPackagedCatalogPorts(
        cold_effects.bindings, cold_effects.stage_receipts, cold_effects.calls
    )
    arguments["sources"] = ModalPackagedSourcePorts(
        cold_effects.stages, cold_effects.dispatches, cold_effects.marker_materials
    )
    arguments["authorities"] = ModalPackagedAuthorityPorts(cold_effects, signer)
    cold_adapter = compose_modal_packaged_adapter(**arguments)
    host_options.update(
        adapter=cold_adapter,
        effects=cold_effects,
        records=records,
        streams=records,
        clock=UTCClock(60),
    )
    cold = compose_modal_packaged_reference_host(**host_options)
    try:
        assert cold.composition.stores.workflow_store._records is records
        assert cold.composition.stores.workflow_store._streams is records
        assert (
            cold_effects.bindings.resolve(submit_binding.command_digest)
            == prior_binding
        )
        assert cold.api.runs.show(started.run).state == prior_outcome.state
        page = cold.api.runs.list(RunListRequest("project"))
        assert tuple(item.run for item in page.outcomes) == (started.run,)
        assert cold.api.runs.outcome(started.run).state.value == "succeeded"
        for operation in (
            lambda: cold.api.runs.logs(RunLogsRequest(started.run)),
            lambda: cold.api.runs.cancel(started.run, "test cancellation"),
            lambda: cold.api.runs.reconcile(started.run),
        ):
            with pytest.raises(RunOperationError) as error:
                operation()
            assert error.value.code.value == "capability_unavailable"
        assert (
            b"".join(
                cold.api.runs.artifacts(
                    RunArtifactRequest(started.run, "final_model", 4)
                ).iter_bytes()
            )
            == b"data"
        )
        assert len(stage_calls) == 1
        assert events == before_events
        assert cold_effects._stage_effects == {}
        assert cold_effects.stages._items == cold_effects.dispatches._items == {}
    finally:
        records.close()
        journal.close()
