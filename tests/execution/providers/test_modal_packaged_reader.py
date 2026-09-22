"""Provider-free completion and artifact readback qualification."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.contracts import operation_path, provider_entry_identity
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.packaged_deployment import ModalPackagedDeploymentObserver
from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader

from tests.execution.providers.test_modal_packaged_deployment import Reader
from tests.execution.providers.test_modal_packaged_dispatch import _case
from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume, SDK


ROLES = (
    "final_model", "tokenizer", "training_lineage", "training_metrics",
    "workload_record",
)


class Verifier:
    def __init__(self, valid=True):
        self.valid = valid
        self.calls = []
    def verify(self, purpose, payload, tag, key_ref):
        self.calls.append((purpose, payload, tag, key_ref))
        return self.valid and tag == b"tag"


def _reader(*, roles=ROLES):
    binding = _case()[0]
    facts = binding.provider_facts
    client = object()
    control = FakeVolume(facts.control_volume_id)
    artifacts = FakeVolume(facts.artifact_volume_id)
    FakeVolume.calls = []
    FakeVolume.registry = {"control-name": control, "artifact-name": artifacts}
    facade = ExplicitModal154ReadFacade(
        facts.client_binding, sdk=SDK, client=client,
        scope_observer=lambda supplied: (
            facts.account_ref, facts.workspace_ref, facts.environment_ref,
            facts.client_ref,
        ) if supplied is client else (),
        deployment_observer=lambda **_: None,
        volume_names={
            facts.control_volume_id: "control-name",
            facts.artifact_volume_id: "artifact-name",
        },
    )
    deployment = ModalPackagedDeploymentObserver(
        sdk=SDK, client=client, client_binding=facts.client_binding,
        reader=Reader(facts),
    )
    verifier = Verifier()
    reader = ModalPackagedReader(
        facade=facade, deployment_observer=deployment,
        verifier=verifier, key_ref="evidence-key",
    )
    effect_id = binding.command.operation.effect.effect_id
    members = []
    for index, role in enumerate(roles):
        payload = f"artifact-{index}".encode()
        path = operation_path(effect_id, "output", f"artifact-{index}.bin")
        artifacts.files[path] = payload
        members.append({
            "role": role, "path": path, "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "provider_entry_id": provider_entry_identity(
                facts.artifact_volume_id, path, len(payload),
            ),
        })
    completion = canonical_bytes({
        "schema_version": "synaptic-modal-packaged-completion/v1",
        "effect_id": effect_id,
        "command_digest": binding.command_digest,
        "provider_job_ref": "fc-1",
        "runtime_release_digest": binding.runtime_release.manifest_digest,
        "provider_runtime_binding_digest": binding.provider_binding.binding_digest,
        "execution_binding_digest": binding.execution_binding.binding_digest,
        "stage_receipt_sha256": "1" * 64,
        "inventory_sha256": "2" * 64,
        "terminal_sha256": "3" * 64,
        "members": members,
    })
    evidence = operation_path(effect_id, "evidence")
    control.files[evidence + "/packaged-completion.json"] = completion
    control.files[evidence + "/packaged-completion.mac"] = b"tag"
    return binding, reader, verifier, control, artifacts, members


def test_observes_exact_authenticated_completion_and_inventory() -> None:
    binding, reader, verifier, _, _, members = _reader()
    observed = reader.observe_completion(binding, provider_job_ref="fc-1")
    assert observed.effect_id == binding.command.operation.effect.effect_id
    assert observed.command_digest == binding.command_digest
    assert tuple(item.role for item in observed.members) == ROLES
    assert observed.completion_digest == hashlib.sha256(
        verifier.calls[0][1]
    ).hexdigest()
    assert [item.provider_entry_id for item in observed.members] == [
        item["provider_entry_id"] for item in members
    ]


def test_streams_bounded_artifact_then_rechecks_inventory() -> None:
    binding, reader, _, _, artifacts, _ = _reader()
    observed = reader.observe_completion(binding, provider_job_ref="fc-1")
    expected = artifacts.files[observed.members[0].path]
    chunks = list(reader.iter_artifact(
        binding, observed, role=observed.members[0].role,
        maximum_bytes=len(expected),
    ))
    assert b"".join(chunks) == expected


@pytest.mark.parametrize("fault", ("job", "command", "binding", "mac"))
def test_completion_substitution_or_bad_authentication_is_rejected(fault: str) -> None:
    binding, reader, verifier, control, _, _ = _reader()
    evidence = operation_path(binding.command.operation.effect.effect_id, "evidence")
    path = evidence + "/packaged-completion.json"
    document = __import__("json").loads(control.files[path])
    if fault == "job":
        document["provider_job_ref"] = "fc-other"
    elif fault == "command":
        document["command_digest"] = "0" * 64
    elif fault == "binding":
        document["execution_binding_digest"] = "0" * 64
    else:
        control.files[evidence + "/packaged-completion.mac"] = b"wrong"
    if fault != "mac":
        control.files[path] = canonical_bytes(document)
    with pytest.raises(ValueError):
        reader.observe_completion(binding, provider_job_ref="fc-1")


def test_inventory_requires_the_exact_five_training_artifact_roles() -> None:
    binding, reader, _, _, _, _ = _reader(
        roles=("one", "two", "three", "four", "five"),
    )
    with pytest.raises(ValueError, match="artifact|inventory"):
        reader.observe_completion(binding, provider_job_ref="fc-1")


def test_truncated_or_changed_artifact_fails_stream_completion() -> None:
    binding, reader, _, _, artifacts, _ = _reader()
    observed = reader.observe_completion(binding, provider_job_ref="fc-1")
    member = observed.members[0]
    artifacts.files[member.path] = artifacts.files[member.path][:-1]
    with pytest.raises(ValueError):
        list(reader.iter_artifact(
            binding, observed, role=member.role, maximum_bytes=member.size,
        ))


@pytest.mark.parametrize("maximum", (True, 0, 193 * 1024 * 1024))
def test_invalid_stream_bounds_fail_before_body_read(maximum) -> None:
    binding, reader, _, _, artifacts, _ = _reader()
    observed = reader.observe_completion(binding, provider_job_ref="fc-1")
    before = dict(artifacts.files)
    with pytest.raises(ValueError, match="bound"):
        list(reader.iter_artifact(
            binding, observed, role=observed.members[0].role,
            maximum_bytes=maximum,
        ))
    assert artifacts.files == before
