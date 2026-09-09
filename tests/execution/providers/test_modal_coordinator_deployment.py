from dataclasses import replace

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.coordinator_deployment import build_modal_coordinator_deployment
from tuner.execution.providers.modal.config import ModalRuntimeLockV1
from tuner.execution.providers.modal.deployment_v1 import ModalDeploymentSpecV1
from tuner.execution.providers.modal.resolution import ModalDeploymentSelectionV1
from tests.execution.providers.test_modal_coordinator_adapter import inputs
from tests.execution.providers.test_modal_deployment_v1 import SDK, App, Image, Secret, Volume


def values():
    configured = inputs(); profile, binding = configured["profile"], configured["binding"]
    selection = ModalDeploymentSelectionV1.from_profile(
        profile, binding=binding, runtime_environment=configured["runtime_environment"],
        timeout_seconds=configured["timeout_seconds"],
    )
    lock = ModalRuntimeLockV1.packaged()
    secret = profile.secrets[0]
    spec = ModalDeploymentSpecV1(
        profile.deployment_ref, profile.function_name, lock.registry_reference,
        profile.control_volume_ref, profile.artifact_volume_ref, secret.name,
        secret.required_keys, configured["runtime_environment"], configured["timeout_seconds"],
    )
    return profile, selection, spec


def build(**changes):
    profile, selection, spec = values()
    arguments = dict(
        sdk=SDK, client=object(), environment_name=selection.environment_ref, spec=spec,
        profile=profile,
        deployment_selection_bytes=canonical_bytes(selection.to_dict()), provider_id="modal",
        profile_ref="modal-a10-v1", executor_id="modal-coordinator-executor",
        executor_implementation_version="0.1.0", control_volume_id="vo-control",
        artifact_volume_id="vo-artifact", evidence_environment_key="SYNAPTIC_EVIDENCE_MAC_KEY",
        evidence_key_ref="stage-key", model_token_key="HF_TOKEN",
    )
    arguments.update(changes)
    return build_modal_coordinator_deployment(**arguments)


def sdk_counts():
    return (len(App.calls), len(Image.calls), len(Volume.calls), len(Secret.calls))


@pytest.mark.parametrize("mutation", ["extra", "modal_id", "modal_secret"])
def test_consistently_rebound_secret_cannot_declare_host_credentials(mutation):
    profile, _, spec = values()
    evidence_key, model_key = "SYNAPTIC_EVIDENCE_MAC_KEY", "HF_TOKEN"
    if mutation == "modal_id":
        model_key = "MODAL_TOKEN_ID"
    elif mutation == "modal_secret":
        evidence_key = "MODAL_TOKEN_SECRET"
    keys = (evidence_key, model_key) + (("EXTRA_TOKEN",) if mutation == "extra" else ())
    profile = replace(profile, secrets=(replace(profile.secrets[0], required_keys=keys),))
    spec = replace(spec, runtime_secret_keys=keys)
    selection = ModalDeploymentSelectionV1.from_profile(
        profile, binding=inputs()["binding"], runtime_environment=spec.environment,
        timeout_seconds=spec.timeout_seconds,
    )
    before = sdk_counts()
    with pytest.raises(ValueError, match="only model and evidence"):
        build(
            profile=profile, spec=spec,
            deployment_selection_bytes=canonical_bytes(selection.to_dict()),
            evidence_environment_key=evidence_key, model_token_key=model_key,
        )
    assert sdk_counts() == before


def test_candidate_build_is_lazy_and_keeps_one_bytes_argument():
    before = (len(App.calls), len(Image.calls), len(Volume.calls), len(Secret.calls))
    built = build()
    after = (len(App.calls), len(Image.calls), len(Volume.calls), len(Secret.calls))
    assert tuple(right-left for left, right in zip(before, after)) == (1, 1, 2, 1)
    assert built.app.function_kwargs["name"] == values()[2].function_name
    assert built.app.function_kwargs["retries"] == 0
    with pytest.raises(ValueError, match="dispatch"):
        built.function("not-bytes")
    assert built.artifact_volume.commits == built.control_volume.commits == 0


def test_preadmission_failure_has_no_commit_or_completion_side_effect():
    built = build()
    with pytest.raises(ValueError):
        built.function(b"not-a-canonical-dispatch")
    assert built.artifact_volume.commits == 0
    assert built.control_volume.commits == 0


def test_preadmission_failure_uses_no_source_process_or_completion(monkeypatch):
    calls = []

    class Unused:
        def materialize(self, *args, **kwargs): calls.append("materialize")
        def run(self, *args, **kwargs): calls.append("run")
        def finalize(self, *args, **kwargs): calls.append("finalize")

    for symbol in ("GitDualCloneMaterializer", "SubprocessSftRunner", "MountedModalCoordinatorProducer"):
        monkeypatch.setattr(
            f"tuner.execution.providers.modal.coordinator_deployment.{symbol}",
            lambda *args, **kwargs: Unused(),
        )
    built = build()
    with pytest.raises(ValueError):
        built.function(b"not-a-canonical-dispatch")
    assert calls == []


def test_successful_wrapper_orders_prepared_artifact_then_artifact_then_control(monkeypatch):
    events = []

    class Worker:
        def __init__(self, **kwargs): events.append("compose")
        def __call__(self, dispatch, job, commit_prepared):
            events.append("worker"); commit_prepared(); return {"status_code": "completed"}

    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_deployment.MountedModalCoordinatorWorker",
        Worker,
    )
    built = build()
    built.artifact_volume.commit = lambda: events.append("artifact")
    built.control_volume.commit = lambda: events.append("control")
    assert built.function(b"dispatch") == {"status_code": "completed"}
    assert events == ["compose", "worker", "artifact", "artifact", "control"]


@pytest.mark.parametrize("change", [
    {"provider_id": "alien"}, {"executor_id": "alien"},
    {"executor_implementation_version": "9.9.9"},
    {"control_volume_id": "same", "artifact_volume_id": "same"},
    {"evidence_environment_key": "UNDECLARED"}, {"model_token_key": "UNDECLARED"},
])
def test_static_substitution_is_rejected_before_app_construction(change):
    before = sdk_counts()
    with pytest.raises(ValueError): build(**change)
    assert sdk_counts() == before


def test_selection_substitution_is_rejected_before_app_construction():
    _, selection, _ = values()
    poisoned = replace(selection, timeout_seconds=selection.timeout_seconds + 1)
    before = sdk_counts()
    with pytest.raises(ValueError):
        build(deployment_selection_bytes=canonical_bytes(poisoned.to_dict()))
    assert sdk_counts() == before


@pytest.mark.parametrize("mutation", [
    "environment", "control", "artifact", "secret_name", "secret_keys",
    "profile", "same_keys", "bad_key", "registry",
])
def test_profile_spec_and_key_substitutions_make_no_sdk_objects(mutation):
    profile, selection, spec = values()
    changes = {}
    if mutation == "environment":
        changes["environment_name"] = "alien-environment"
    elif mutation == "control":
        changes["spec"] = replace(spec, control_volume_name="alien-control")
    elif mutation == "artifact":
        changes["spec"] = replace(spec, artifact_volume_name="alien-artifact")
    elif mutation == "secret_name":
        changes["spec"] = replace(spec, runtime_secret_name="alien-secret")
    elif mutation == "secret_keys":
        changes["spec"] = replace(spec, runtime_secret_keys=spec.runtime_secret_keys + ("EXTRA_TOKEN",))
    elif mutation == "profile":
        changes["profile"] = replace(profile, profile="alien-profile")
    elif mutation == "same_keys":
        changes["model_token_key"] = "SYNAPTIC_EVIDENCE_MAC_KEY"
    elif mutation == "bad_key":
        changes["evidence_environment_key"] = "not-an-environment-symbol"
    else:
        changes["spec"] = replace(
            spec, registry_reference="alien.example/runtime@sha256:" + "0" * 64,
        )
    before = sdk_counts()
    with pytest.raises(ValueError):
        build(**changes)
    assert sdk_counts() == before
