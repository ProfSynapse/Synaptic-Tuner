"""Native source admission through real Foundation and signed Modal evidence.

Only consumer storage/verifier ports and the SDK's read facade are test doubles.
No provider client, artifact-body read or resource mutation is allowed here.
"""

import traceback
from pathlib import Path
import subprocess
import sys

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunsAPI, RunVerification
from tests.execution.coordinator_v1 import test_state_machine as state_cases
from tests.execution.providers import (
    test_modal_coordinator_read_transport as transport_cases,
)
from tests.execution.providers.test_modal_coordinator_reader import Authority, Catalog
from tuner.execution.coordinator_v1.model import (
    ProviderReadPurposeV1,
    VerificationVerdictV1,
)
from tuner.execution.coordinator_v1.state_machine import (
    apply_artifact_verification,
    apply_provider_observation,
    apply_reverification,
    project_run_outcome,
    provider_run_read_request,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
)
from tuner.execution.providers.modal.coordinator_reader import ModalCoordinatorRunReader
from tuner.execution.providers.modal.coordinator_producer import (
    MountedModalCoordinatorProducer,
)
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.inference_binding import (
    ModalInferenceBindingError,
    ModalInferenceSourceBinder,
)


def test_source_binding_import_is_provider_and_ml_free():
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            f"import sys; sys.path.insert(0, {str(root)!r}); "
            "import tuner.execution.providers.modal.inference_binding; "
            "assert not any(name.split('.')[0] in ('modal', 'torch', 'transformers') "
            "for name in sys.modules)",
        ],
        cwd="/tmp",
        env={},
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode()


class _Store:
    def __init__(self, value):
        self.value = value

    def get(self, run):
        assert run == self.value.run
        return self.value


class _Runs:
    def __init__(self, store):
        self.store = store
        self.calls = []

    def reverify(self, run):
        self.calls.append("reverify")
        current = self.store.get(run)
        receipt = state_cases.verification_receipt(
            current, current.artifact_manifest, VerificationVerdictV1.VERIFIED, "b"
        )
        self.store.value = apply_reverification(
            current, current.artifact_manifest, receipt, state_cases.Verifier()
        )
        return RunVerification(run, True, "2026-09-10T00:00:00Z")

    def outcome(self, run):
        self.calls.append("outcome")
        return project_run_outcome(self.store.get(run))

    def artifacts(self, request):
        pytest.fail("source binding must not read public artifact bodies")


def _case(monkeypatch):
    retained = {}
    real_case = transport_cases.real_launch_bundle_case

    def capture(patch):
        value = real_case(patch)
        retained.update(value)
        return value

    monkeypatch.setattr(transport_cases, "real_launch_bundle_case", capture)
    original_finalize = MountedModalCoordinatorProducer.finalize

    def finalize_for_retained_job(self, invocation, result, *, job_ref):
        submit = parse_exact_command(retained["envelope"].submit_binding.command_bytes)
        selected_run = TrainingRunRef(
            submit.preparation.run_id, submit.preparation.project_ref
        )
        selected = retained["harness"].workflows.get(selected_run)
        return original_finalize(
            self,
            invocation,
            result,
            job_ref=selected.provider_run_ref.reference.provider_job_ref,
        )

    monkeypatch.setattr(
        MountedModalCoordinatorProducer, "finalize", finalize_for_retained_job
    )
    transport, binding, stored, body_reads, signer, key_ref = (
        transport_cases._completed_transport(monkeypatch)
    )
    harness = retained["harness"]
    command = parse_exact_command(binding.command_bytes)
    run = TrainingRunRef(command.preparation.run_id, command.preparation.project_ref)
    workflow = harness.workflows.get(run)
    foundation_record = harness.foundation.get(workflow.submit.effect_id)
    assessment = harness.foundation.assess(foundation_record)
    authority = Authority()
    reader = ModalCoordinatorRunReader(
        catalog=Catalog(binding),
        binding_authority=retained["authority"],
        foundation_authenticator=harness.authenticator,
        assessment_authenticator=harness.foundation,
        evidence_authority=authority,
        transport=transport,
        observed_at="2026-09-10T00:00:00Z",
    )

    def request(current, purpose):
        return provider_run_read_request(
            current,
            foundation_record,
            assessment,
            harness.authenticator,
            harness.foundation,
            purpose=purpose,
        )

    observation_request = request(workflow, ProviderReadPurposeV1.OBSERVE)
    done = apply_provider_observation(
        workflow, observation_request, reader.observe(observation_request), authority
    )
    manifest = reader.artifacts(request(done, ProviderReadPurposeV1.ARTIFACTS))
    receipt = state_cases.verification_receipt(
        done, manifest, VerificationVerdictV1.VERIFIED
    )
    verified = apply_artifact_verification(
        done, manifest, receipt, state_cases.Verifier()
    )
    store = _Store(verified)
    operations = _Runs(store)
    runs = RunsAPI(operations)
    binder = ModalInferenceSourceBinder(
        runs=runs,
        workflows=store,
        foundation=harness.foundation,
        foundation_authenticator=harness.authenticator,
        assessment_authenticator=harness.foundation,
        reader=reader,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("native source admission streamed an artifact body")

    monkeypatch.setattr(ExplicitModal154ReadFacade, "iter_complete", forbidden)
    assert body_reads == []
    return (
        binder,
        runs,
        run,
        operations,
        store,
        stored,
        body_reads,
        signer,
        key_ref,
        command,
    )


def test_source_binding_reuses_full_signed_transport_without_body_reads(monkeypatch):
    binder, runs, run, operations, store, _, bodies, _, _, command = _case(monkeypatch)
    source = binder.bind(runs, run)
    assert operations.calls == ["reverify", "outcome"]
    assert source.run == run
    assert source.manifest_digest == store.value.artifact_manifest.manifest_digest
    assert (
        source.artifact_source_digest
        == store.value.artifact_manifest.artifact_source_digest
    )
    assert source.command_binding.command_bytes == command.canonical_bytes
    assert source.artifact_volume_id == "artifact-id"
    assert len(source.native_members) == 5
    assert bodies == []
    assert not hasattr(source, "local_model")
    assert not hasattr(source, "model_kind")


@pytest.mark.parametrize("record_name", ["terminal-evidence", "completion-manifest"])
def test_source_binding_rejects_changed_signed_control_evidence(
    monkeypatch, record_name
):
    binder, runs, run, _, _, stored, bodies, _, _, command = _case(monkeypatch)
    effect = command.operation.effect.effect_id
    stored[f"operations/{effect}/evidence/{record_name}.v1.mac"] = b"invalid"
    with pytest.raises(ModalInferenceBindingError) as caught:
        binder.bind(runs, run)
    assert str(caught.value) == "modal_inference_source_invalid"
    assert bodies == []


def test_source_binding_redacts_provider_exception(monkeypatch):
    binder, runs, run, _, _, _, bodies, _, _, _ = _case(monkeypatch)

    def unavailable(*args, **kwargs):
        raise RuntimeError("private-provider-response-do-not-expose")

    monkeypatch.setattr(ExplicitModal154ReadFacade, "inspect_deployment", unavailable)
    with pytest.raises(ModalInferenceBindingError) as caught:
        binder.bind(runs, run)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "private-provider-response-do-not-expose" not in rendered
    assert bodies == []


@pytest.mark.parametrize("field", ["artifact_volume_id", "job_ref", "effect_id"])
def test_source_binding_rejects_valid_tag_for_other_native_identity(monkeypatch, field):
    binder, runs, run, _, _, stored, bodies, signer, key_ref, command = _case(
        monkeypatch
    )
    effect = command.operation.effect.effect_id
    path = f"operations/{effect}/evidence/completion-manifest.v1.json"
    document = parse_canonical_object(stored[path], name="test completion")
    document[field] = "other-identity"
    stored[path] = canonical_bytes(document)
    tag = signer.sign("modal-completion/v1", stored[path], key_ref)
    assert signer.verify("modal-completion/v1", stored[path], tag, key_ref) is True
    stored[path.removesuffix(".json") + ".mac"] = tag
    with pytest.raises(ModalInferenceBindingError):
        binder.bind(runs, run)
    assert bodies == []
