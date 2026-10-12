"""Optional transport failures cannot replace the mandatory CPU proof."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from tests.training import test_modal_host_qualification as helpers
from tuner.training import modal_host_qualification as host
from tuner.cloud.modal_runtime_qualification_operator import ModalRuntimeQualificationOutcome
from tuner.execution.providers.modal.runtime_release_qualification import (
    ModalRuntimeReleaseFixtureReceiptV1, ModalRuntimeReleaseQualificationReceiptV1,
)
from tuner.execution.providers.modal.runtime_release_qualification_reader import ModalRuntimeReleaseQualificationObservation
from tuner.execution.providers.modal.runtime_release_deployment import (
    MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA,
    ModalRuntimeReleaseVolumeSpecV1, ModalRuntimeReleaseSecretSpecV1,
)
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV2
from tuner.execution.providers.modal.contracts import operation_path


def bound_runtime(monkeypatch):
    original = helpers._plan()
    source = original.release
    names = (
        "release_ref package_name package_version package_digest source_provenance_digest "
        "worker_entrypoint worker_closure_digest python_implementation python_version "
        "python_executable python_executable_digest installed_distributions_digest "
        "installed_distribution_count platform_system platform_machine cuda_version runtime_facts "
        "compatible_methods compatible_models compatible_dataset_formats workload_schema "
        "prepared_input_schema artifact_contract_schema"
    ).split()
    release = PackagedTrainingRuntimeReleaseV2.build(**{name: getattr(source, name) for name in names},
        material={"kind": "published_oci", "image": {"reference": source.image_ref, "digest": source.image_digest}})
    training = replace(original.functions[0], volume_roles=("control", "artifacts", "model_cache"),
        module="tuner.runtime.runtime_release_modal_training", qualname="run_modal_packaged_training",
        secret_names=(original.secrets[0].name, "hf-token"), restrict_modal_access=False)
    plan = replace(original, release=release, functions=(training, original.functions[1]),
        volumes=(*original.volumes, ModalRuntimeReleaseVolumeSpecV1("model_cache", "cache-volume", "/mnt/model-cache")),
        secrets=(*original.secrets, ModalRuntimeReleaseSecretSpecV1("hf-token", ("HF_TOKEN",))),
        schema_version=MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA)
    monkeypatch.setattr(helpers, "_plan", lambda: plan)
    from tuner.runtime.runtime_release_modal_training import run_modal_packaged_training
    monkeypatch.setattr(helpers, "packaged_training_entry", run_modal_packaged_training)
    return helpers._runtime()


@pytest.mark.parametrize("marker_fails,optional_result", [(True, False), (False, False), (False, True)])
def test_optional_stall_reserves_primary_budget_and_marker_failure_never_submits(monkeypatch, tmp_path,
        marker_fails, optional_result):
    runtime, client = bound_runtime(monkeypatch)
    tmp_path.chmod(0o700)
    events = []
    clock = [1000.0]
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    # Synchronous bounded fake preserves deadlines while avoiding wall-clock sleeps.
    def bounded(operation, *, deadline, code):
        assert deadline > clock[0]
        return operation()
    monkeypatch.setattr(host, "_bounded", bounded)
    monkeypatch.setattr(host, "_CurrentQualificationObserver", lambda **kwargs: SimpleNamespace(observe=lambda facts: facts))

    class Operator:
        def __init__(self, **kwargs): pass
        def stage_fixture_once(self, *, effect_id, deployment_facts):
            events.append("fixture")
            return ModalRuntimeReleaseFixtureReceiptV1.create(effect_id=effect_id,
                artifact_volume_id=next(item.volume_id for item in deployment_facts.volumes if item.spec.role == "artifacts"))
        def stage_transport_marker(self, **kwargs):
            events.append("marker")
            if marker_fails: raise ValueError("private provider failure")
        def submit_once(self, *args, **kwargs):
            events.append("spawn")
            return ModalRuntimeQualificationOutcome("found", "fc-cpu123")

    class Reader:
        def __init__(self, **kwargs): pass
        def capture_transport_probe(self, dispatch, *, provider_call_id, host_deadline):
            assert host_deadline == 1120
            clock[0] = host_deadline
            events.append("capture_stalled")
            raise TimeoutError
        def observe(self, dispatch, *, provider_call_id):
            events.append("core_verified")
            output = b"core-proof"
            receipt = ModalRuntimeReleaseQualificationReceiptV1(
                effect_id=dispatch.effect_id, dispatch_digest=dispatch.dispatch_digest,
                provider_call_id=provider_call_id, deployment_facts_digest=dispatch.deployment_facts.facts_digest,
                current_observation_digest="a" * 64,
                output_path=operation_path(dispatch.effect_id, "runtime-release-qualification", "output", "evidence.json"),
                output_size=len(output), output_sha256=hashlib.sha256(output).hexdigest(), output_provider_entry_id="entry")
            return ModalRuntimeReleaseQualificationObservation(receipt, output)

    class Call(helpers._Call):
        def get(self, timeout):
            result = super().get(timeout)
            return {**result, "transport_probe_status": "unavailable"} if optional_result else result
    monkeypatch.setattr(helpers.SDK, "FunctionCall", SimpleNamespace(from_id=lambda *args, **kwargs: Call()), raising=False)
    monkeypatch.setattr(host, "ModalRuntimeQualificationOperator", Operator)
    monkeypatch.setattr(host, "ModalRuntimeReleaseQualificationReader", Reader)
    kwargs = dict(sdk=helpers.SDK, client=client, client_binding=runtime.facts.client_binding,
        runtime=runtime, private_storage=helpers._Journal(events), effect_id="cpu-transport-test",
        environment_name="production", capture_progress=True, transport_root=tmp_path)
    if marker_fails:
        with pytest.raises(host.ModalHostQualificationUnavailable) as error:
            host.qualify_modal_runtime_for_host(**kwargs)
        assert error.value.phase == "TRANSPORT_MARKER_STAGE"
        assert events == ["claim", "fixture", "marker"]
    else:
        result = host.qualify_modal_runtime_for_host(**kwargs)
        assert result.transport_status == "capture_unavailable"
        assert result.transport_verification is None
        assert result.transport_probe is not None
        assert events == ["claim", "fixture", "marker", "spawn", "capture_stalled", "core_verified"]
        assert clock[0] == 1120
