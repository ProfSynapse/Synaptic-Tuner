"""Original-command authenticated public readback and native SFT verification."""

import base64
import hashlib
import hmac
import json

from synaptic_tuner.api.v1.results import VerifiedArtifact
from tuner.execution.coordinator_v1.model import (
    ProviderRunReadRequestV1, ProviderReadPurposeV1, ProviderRunPhaseV1,
    ProviderRunObservationContentV1, ArtifactManifestV1,
    ArtifactVerificationContentV1, AuthenticatedArtifactVerificationReceiptV1,
    AuthenticatedFoundationRecordAssessmentV1, VerificationVerdictV1, WorkflowPhaseV1,
)
from tuner.execution.coordinator_v1.state_machine import provider_run_read_request
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.runtime.artifacts import ArtifactInventory, ArtifactEntry
from tuner.runtime.dispatch import ProcessResult
from tuner.runtime.verification import VerificationService, WorkloadBindingVerifier, VerificationStatus
from tuner.training.methods.sft import compile_sft_workload, SFT_ARTIFACT_CONTRACT
from .model import HFJobsError


class HFJobsReader:
    def __init__(self, *, preparation, transport, foundation_authenticator,
                 assessment_authenticator, evidence_authority, clock):
        self.preparation, self.transport = preparation, transport
        self.foundation, self.assessments = foundation_authenticator, assessment_authenticator
        self.evidence_authority, self.clock = evidence_authority, clock

    def _validate(self, request, purpose):
        if (
            type(request) is not ProviderRunReadRequestV1
            or request.purpose is not purpose
        ):
            raise HFJobsError("binding_mismatch")
        ProviderRunReadRequestV1(
            **{name: getattr(request, name) for name in request.__dataclass_fields__}
        )
        record = request.foundation_record
        if (
            self.foundation.authenticate_grant(record.grant, record.command_bytes)
            is not True
            or any(
                self.foundation.authenticate_receipt(value) is not True
                for value in record.results
            )
            or any(
                self.foundation.authenticate_invalid_evidence(value) is not True
                for value in record.invalid_evidence
            )
        ):
            raise HFJobsError("binding_mismatch")
        assessment = type(request.assessment).parse(request.assessment.canonical_bytes)
        if (
            assessment != request.assessment
            or self.assessments.authenticate(assessment) is not True
            or assessment.content.foundation_record_digest != record.record_digest
            or assessment.content.authenticated_receipt_digests
            != tuple(value.authenticated_receipt_digest for value in record.results)
        ):
            raise HFJobsError("binding_mismatch")
        command = parse_exact_command(request.submit_command_bytes)
        retained = self.transport.records.read(
            partition="effects", key=self.transport._key("command", command.digest)
        )
        if (
            retained is None
            or base64.b64decode(
                json.loads(retained.canonical)["command_base64"], validate=True
            )
            != command.canonical_bytes
        ):
            raise HFJobsError("binding_mismatch")
        ref = request.provider_run.reference
        if self.preparation is None:
            self.transport._validate(command, command.operation.effect.kind)
            selected = self.transport._read_selection()
            expected_scope = (
                selected["provider"]["provider_id"],
                selected["provider"]["profile_ref"],
                selected["scope"]["account_ref"],
                selected["scope"]["namespace_ref"],
            )
        else:
            selection = self.preparation.selection
            expected_scope = (
                selection.provider.provider_id,
                selection.provider.profile_ref,
                selection.scope.account_ref,
                selection.scope.namespace_ref,
            )
            if (
                command.preparation.plan_fingerprint
                != self.preparation.plan.plan_fingerprint
                or command.preparation.execution_binding_digest
                != self.preparation.binding.binding_digest
            ):
                raise HFJobsError("binding_mismatch")
        if (
            ref.provider_id,
            ref.profile_ref,
            ref.account_ref,
            ref.namespace_ref,
        ) != expected_scope or command.canonical_bytes != record.command_bytes:
            raise HFJobsError("binding_mismatch")
        receipt = self.transport.retained("submit", command.digest)
        if receipt["job_id"] != ref.provider_job_ref:
            raise HFJobsError("binding_mismatch")
        return command, ref

    def observe(self, request):
        command, ref = self._validate(request, ProviderReadPurposeV1.OBSERVE)
        phase, _ = self.transport.observe(command.digest)
        phases = {"SCHEDULING": ProviderRunPhaseV1.QUEUED, "RUNNING": ProviderRunPhaseV1.RUNNING,
            "COMPLETED": ProviderRunPhaseV1.SUCCEEDED, "ERROR": ProviderRunPhaseV1.FAILED,
            "CANCELED": ProviderRunPhaseV1.CANCELLED}
        if phase not in phases:
            raise HFJobsError("read_unavailable")
        content = ProviderRunObservationContentV1("synaptic-provider-run-observation-content/v1",
            request.request_digest, request.source_workflow_record_digest, request.source_revision,
            request.run, request.provider_run.binding_digest, ref.provider_id, ref.profile_ref,
            ref.account_ref, ref.namespace_ref, ref.provider_job_ref, phases[phase],
            canonical_bytes({"schema_version": "synaptic-hf-job-observation/v1",
                             "job_id": ref.provider_job_ref, "command_digest": command.digest, "status": phase}),
            "provider_failed" if phase == "ERROR" else None,
            "hf-jobs-reader", "0.1.0", self.clock.now())
        return self.evidence_authority.observation(content)

    def logs(self, request, query):
        raise HFJobsError("capability_unavailable")

    def artifacts(self, request):
        command, ref = self._validate(request, ProviderReadPurposeV1.ARTIFACTS)
        completion, _ = self.transport.native(command.digest)
        artifacts = tuple(VerifiedArtifact(row["role"], row["sha256"], row["size_bytes"])
                          for row in completion["artifacts"])
        evidence = canonical_bytes(completion)
        return ArtifactManifestV1.build(run=request.run, provider_run=ref, artifacts=artifacts,
            artifact_source_digest=domain_digest("synaptic-hf-native-artifacts/v1", evidence),
            canonical_evidence=evidence)

    def iter_artifact_bytes(self, request, manifest, role, *, maximum_bytes):
        command, _ = self._validate(request, ProviderReadPurposeV1.ARTIFACTS)
        if manifest != self.artifacts(request):
            raise HFJobsError("artifact_invalid")
        completion, _ = self.transport.native(command.digest)
        matches = [row for row in completion["artifacts"] if row["role"] == role]
        if len(matches) != 1 or type(maximum_bytes) is not int or matches[0]["size_bytes"] > maximum_bytes:
            raise HFJobsError("artifact_invalid")
        yield from self.transport.iter_artifact(command.digest, matches[0])


class HFJobsArtifactVerifier:
    def __init__(self, *, reader, authority_ref, key_ref, key, clock):
        if type(key) is not bytes or len(key) < 32:
            raise HFJobsError("input_invalid")
        self.reader, self.authority_ref, self.key_ref, self.key, self.clock = reader, authority_ref, key_ref, key, clock
        self.foundation = None

    def bind_foundation(self, foundation):
        if self.foundation is not None:
            raise HFJobsError("binding_mismatch")
        self.foundation = foundation

    def _tag(self, content):
        return hmac.new(self.key, b"hf-jobs-native-verification/v1\0" + canonical_bytes({
            "authority_ref": self.authority_ref, "key_ref": self.key_ref,
            "content_digest": content.content_digest}), hashlib.sha256).hexdigest()

    def _issue(self, workflow, manifest):
        if self.foundation is None or workflow.submit is None:
            raise HFJobsError("artifact_invalid")
        record = self.foundation.get(workflow.submit.effect_id)
        assessment = AuthenticatedFoundationRecordAssessmentV1.parse(
            workflow.submit.foundation_bindings[-1].canonical_assessment_bytes
        )
        request = provider_run_read_request(
            workflow,
            record,
            assessment,
            self.reader.foundation,
            self.reader.assessments,
            purpose=ProviderReadPurposeV1.ARTIFACTS,
        )
        if self.reader.artifacts(request) != manifest:
            raise HFJobsError("artifact_invalid")
        workload = (
            self.reader.transport._read_material["workload"]
            if self.reader.preparation is None
            else compile_sft_workload(
                resolved_config=self.reader.preparation.rich_plan.resolved_config,
                execution_source=self.reader.preparation.rich_plan.execution_source,
            )
        )
        completion = json.loads(manifest.canonical_evidence)
        inventory = ArtifactInventory(
            tuple(
                ArtifactEntry(
                    row["role"], row["path"], row["sha256"], row["size_bytes"]
                )
                for row in completion["artifacts"]
            )
        )
        transport, command_digest = (
            self.reader.transport,
            parse_exact_command(request.submit_command_bytes).digest,
        )

        class NativeReader:
            def read_bytes(self, artifact, *, maximum):
                from tuner.runtime.verification import ArtifactReadLimitExceeded

                if artifact.size > maximum:
                    raise ArtifactReadLimitExceeded
                member = next(
                    row
                    for row in completion["artifacts"]
                    if row["path"] == artifact.path
                )
                return b"".join(transport.iter_artifact(command_digest, member))

        if self.reader.preparation is None:
            closure = self.reader.transport._read_material["projection"][
                "closure_manifest"
            ]
            closure_digest, closure_path = (
                closure["closure_digest"],
                closure["runtime_path"],
            )
        else:
            closure_digest = self.reader.preparation.bundle.closure_digest
            closure_path = str(
                self.reader.preparation.bundle.closure_manifest_runtime_path
            )
        report = VerificationService(
            WorkloadBindingVerifier(
                closure_digest=closure_digest, closure_manifest_path=closure_path
            )
        ).verify(
            provider_completed=True,
            process=ProcessResult(0),
            workload=workload,
            contract=SFT_ARTIFACT_CONTRACT,
            inventory=inventory,
            reader=NativeReader(),
        )
        if report.status is not VerificationStatus.VERIFIED:
            raise HFJobsError("artifact_invalid")
        evidence = canonical_bytes(
            {
                "schema_version": "synaptic-hf-native-verification/v1",
                "read_request_digest": request.request_digest,
                "semantic_checks": [
                    {"code": row.code, "passed": row.passed}
                    for row in report.semantic_checks
                ],
            }
        )
        content = ArtifactVerificationContentV1(
            "synaptic-artifact-verification-content/v1",
            workflow.record_digest,
            workflow.revision,
            workflow.run,
            workflow.provider_run_ref.binding_digest,
            manifest.manifest_digest,
            manifest.artifact_source_digest,
            manifest.artifacts,
            manifest.artifacts,
            VerificationVerdictV1.VERIFIED,
            None,
            "hf-jobs-native-verifier",
            "0.1.0",
            evidence,
            self.clock.now(),
        )
        return AuthenticatedArtifactVerificationReceiptV1(
            content, self.authority_ref, self.key_ref, self._tag(content)
        )

    def verify(self, workflow, manifest):
        if workflow.phase not in {WorkflowPhaseV1.SUCCEEDED_UNVERIFIED, WorkflowPhaseV1.VERIFICATION_FAILED}:
            raise HFJobsError("artifact_invalid")
        return self._issue(workflow, manifest)

    def replay(self, workflow, manifest, prior_receipt):
        if (workflow.phase is not WorkflowPhaseV1.VERIFIED or not workflow.verification_receipts
                or workflow.verification_receipts[-1] != prior_receipt or self.authenticate(prior_receipt) is not True):
            raise HFJobsError("artifact_invalid")
        return self._issue(workflow, manifest)

    def authenticate(self, receipt):
        try:
            parsed = AuthenticatedArtifactVerificationReceiptV1.parse(receipt.canonical_bytes)
            return (parsed == receipt and parsed.authority_ref == self.authority_ref
                    and parsed.key_ref == self.key_ref and hmac.compare_digest(parsed.tag, self._tag(parsed.content)))
        except Exception:
            return False
