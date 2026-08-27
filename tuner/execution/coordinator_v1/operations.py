"""Internal provider-neutral run operations; deliberately not package-exported."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1
from synaptic_tuner.api.v1.providers import ProviderDescriptor
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import (
    RunArtifactRequest,
    RunArtifactStream,
    RunListRequest,
    RunLogPage,
    RunLogsRequest,
    RunOperationCode,
    RunOperationError,
    RunOutcome,
    RunPage,
    RunVerification,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest

from .coordinator import (
    ApplyArtifactVerificationTransitionV1,
    ApplyProviderObservationTransitionV1,
    ApplyReverificationTransitionV1,
    CoordinatorCodeV1,
    CoordinatorErrorV1,
)
from .cursors import (
    AuthenticatedCursorV1,
    CursorContentV1,
    CursorKindV1,
    decode_cursor,
    encode_cursor,
)
from .model import (
    ArtifactManifestV1,
    AuthenticatedProviderLogPageV1,
    ProviderLogQueryV1,
    ProviderReadPurposeV1,
    WorkflowPhaseV1,
    WorkflowRecordV1,
)
from .state_machine import (
    apply_artifact_verification,
    apply_provider_observation,
    apply_reverification,
    project_run_outcome,
    provider_run_read_request,
)


def _closed(code: RunOperationCode) -> RunOperationError:
    return RunOperationError(code)


def _invoke(operation, code: RunOperationCode):
    failed = False
    value = None
    try:
        value = operation()
    except Exception:
        failed = True
    if failed:
        raise _closed(code)
    return value


def _project_digest(project_ref: str) -> bytes:
    return bytes.fromhex(
        domain_digest(
            "synaptic-runs-list-project/v1",
            canonical_bytes({"project_ref": project_ref}),
        )
    )


def _run_key(run: TrainingRunRef) -> str:
    return domain_digest(
        "synaptic-coordinator-run-key/v1", canonical_bytes(run.to_dict())
    )


def _log_run_digest(run: TrainingRunRef) -> bytes:
    return bytes.fromhex(
        domain_digest("synaptic-runs-log-run/v1", canonical_bytes(run.to_dict()))
    )


def _decode_list_cursor(request: RunListRequest, authority) -> str | None:
    if request.cursor is None:
        return None
    failed = False
    cursor = None
    try:
        cursor = decode_cursor(request.cursor)
    except Exception:
        failed = True
    if failed or type(cursor) is not AuthenticatedCursorV1:
        raise _closed(RunOperationCode.CURSOR_INVALID)
    if (
        cursor.content.kind is not CursorKindV1.RUN_LIST
        or cursor.content.query_digest != _project_digest(request.project_ref)
        or cursor.content.after_run_key is None
    ):
        raise _closed(RunOperationCode.CURSOR_INVALID)
    verified = _invoke(
        lambda: authority.verify(cursor), RunOperationCode.CURSOR_INVALID
    )
    if verified is not True:
        raise _closed(RunOperationCode.CURSOR_INVALID)
    return cursor.content.after_run_key.hex()


def _decode_log_cursor(request: RunLogsRequest, authority) -> int | None:
    if request.cursor is None:
        return None
    failed = False
    cursor = None
    try:
        cursor = decode_cursor(request.cursor)
    except Exception:
        failed = True
    if failed or type(cursor) is not AuthenticatedCursorV1:
        raise _closed(RunOperationCode.CURSOR_INVALID)
    if (
        cursor.content.kind is not CursorKindV1.RUN_LOGS
        or cursor.content.query_digest != _log_run_digest(request.run)
        or cursor.content.after_sequence is None
    ):
        raise _closed(RunOperationCode.CURSOR_INVALID)
    verified = _invoke(
        lambda: authority.verify(cursor), RunOperationCode.CURSOR_INVALID
    )
    if verified is not True or cursor.content.after_sequence > 2**63 - 1:
        raise _closed(RunOperationCode.CURSOR_INVALID)
    return cursor.content.after_sequence


@dataclass(slots=True)
class _ArtifactStreamV1:
    run: TrainingRunRef
    artifact: VerifiedArtifact
    maximum_bytes: int
    _reader: object
    _request: object
    _manifest: ArtifactManifestV1
    _used: bool = False

    def iter_bytes(self):
        if self._used:
            raise _closed(RunOperationCode.ARTIFACT_CONTENT_INVALID)
        self._used = True
        if self.artifact.size_bytes == 0:
            return iter(())
        return self._consume()

    def _consume(self):
        failed = False
        iterator = None
        try:
            iterator = iter(
                self._reader.iter_artifact_bytes(
                    self._request,
                    self._manifest,
                    self.artifact.role,
                    maximum_bytes=self.maximum_bytes,
                )
            )
        except Exception:
            failed = True
        if failed:
            raise _closed(RunOperationCode.ARTIFACT_CONTENT_INVALID)
        total = 0
        digest = hashlib.sha256()
        while True:
            failed = False
            done = False
            chunk = None
            try:
                chunk = next(iterator)
            except StopIteration:
                done = True
            except Exception:
                failed = True
            if failed:
                raise _closed(RunOperationCode.ARTIFACT_CONTENT_INVALID)
            if done:
                break
            if type(chunk) is not bytes or not chunk or len(chunk) > 1_048_576:
                raise _closed(RunOperationCode.ARTIFACT_CONTENT_INVALID)
            total += len(chunk)
            if total > self.artifact.size_bytes or total > self.maximum_bytes:
                raise _closed(RunOperationCode.ARTIFACT_CONTENT_INVALID)
            digest.update(chunk)
            yield chunk
        if total != self.artifact.size_bytes or digest.hexdigest() != self.artifact.sha256:
            raise _closed(RunOperationCode.ARTIFACT_CONTENT_INVALID)


class TrainingOperationsV1:
    _ACTIVE_OBSERVATION = frozenset(
        {
            WorkflowPhaseV1.QUEUED,
            WorkflowPhaseV1.RUNNING,
            WorkflowPhaseV1.CANCEL_REQUESTED,
        }
    )

    def __init__(
        self,
        planning,
        planning_store,
        workflow_store,
        coordinator,
        foundation,
        foundation_authenticator,
        assessment_authenticator,
        reader,
        observation_authenticator,
        log_authenticator,
        artifact_verifier,
        cursor_authority,
        clock,
    ):
        self._planning = planning
        self._plans = planning_store
        self._workflows = workflow_store
        self._coordinator = coordinator
        self._foundation = foundation
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._reader = reader
        self._observation_authenticator = observation_authenticator
        self._log_authenticator = log_authenticator
        self._artifact_verifier = artifact_verifier
        self._cursor_authority = cursor_authority
        self._clock = clock

    def _issue_cursor(self, content: CursorContentV1) -> str:
        issued = _invoke(
            lambda: self._cursor_authority.issue(content),
            RunOperationCode.INTEGRITY_ERROR,
        )
        if type(issued) is not AuthenticatedCursorV1:
            raise _closed(RunOperationCode.INTEGRITY_ERROR)
        failed = False
        token = None
        decoded = None
        try:
            token = encode_cursor(issued)
            decoded = decode_cursor(token)
        except Exception:
            failed = True
        if failed or decoded != issued or decoded.content != content:
            raise _closed(RunOperationCode.INTEGRITY_ERROR)
        verified = _invoke(
            lambda: self._cursor_authority.verify(decoded),
            RunOperationCode.INTEGRITY_ERROR,
        )
        if verified is not True:
            raise _closed(RunOperationCode.INTEGRITY_ERROR)
        return token

    def _workflow(self, run: TrainingRunRef) -> WorkflowRecordV1:
        if type(run) is not TrainingRunRef:
            raise _closed(RunOperationCode.RUN_MISSING)
        retained = _invoke(
            lambda: self._workflows.get(run), RunOperationCode.INTEGRITY_ERROR
        )
        if type(retained) is not WorkflowRecordV1 or retained.run != run:
            raise _closed(RunOperationCode.RUN_MISSING)
        return retained

    def _descriptor(self, workflow: WorkflowRecordV1) -> ProviderDescriptor:
        context = _invoke(
            lambda: self._plans.get_context(workflow.provider_context_digest),
            RunOperationCode.INTEGRITY_ERROR,
        )
        if type(context) is not ProviderPlanContextV1 or (
            context.provider_context_digest != workflow.provider_context_digest
            or context.provider != workflow.provider
            or context.descriptor_digest != workflow.provider_descriptor_digest
        ):
            raise _closed(RunOperationCode.INTEGRITY_ERROR)
        descriptor = _invoke(
            lambda: self._planning.describe(context.provider),
            RunOperationCode.INTEGRITY_ERROR,
        )
        if type(descriptor) is not ProviderDescriptor or (
            descriptor.provider_id != workflow.provider.provider_id
            or descriptor.descriptor_digest != workflow.provider_descriptor_digest
        ):
            raise _closed(RunOperationCode.INTEGRITY_ERROR)
        return descriptor

    @staticmethod
    def _capability(descriptor: ProviderDescriptor, name: str) -> None:
        if getattr(descriptor.capabilities, name, None) is not True:
            raise _closed(RunOperationCode.CAPABILITY_UNAVAILABLE)

    def _request(self, workflow, purpose):
        if workflow.submit is None:
            raise _closed(RunOperationCode.READ_INELIGIBLE)
        record = _invoke(
            lambda: self._foundation.get(workflow.submit.effect_id),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        if record is None:
            raise _closed(RunOperationCode.PROVIDER_READ_INVALID)
        assessment = _invoke(
            lambda: self._foundation.assess(record),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        return _invoke(
            lambda: provider_run_read_request(
                workflow,
                record,
                assessment,
                self._foundation_authenticator,
                self._assessment_authenticator,
                purpose=purpose,
            ),
            RunOperationCode.PROVIDER_READ_INVALID,
        )

    def _cas(self, current, replacement, transition):
        if replacement is current:
            return current
        for _ in range(3):
            swapped = _invoke(
                lambda: self._workflows.compare_and_swap(
                    current, replacement, transition=transition
                ),
                RunOperationCode.INTEGRITY_ERROR,
            )
            if swapped is True:
                return replacement
            retained = self._workflow(current.run)
            if retained == replacement:
                return retained
            if retained == current:
                continue
            descendant = _invoke(
                lambda: self._workflows.is_descendant(replacement, retained),
                RunOperationCode.INTEGRITY_ERROR,
            )
            if descendant is True:
                return retained
            raise _closed(RunOperationCode.STATE_CONFLICT)
        raise _closed(RunOperationCode.STATE_CONFLICT)

    def list(self, request: RunListRequest) -> RunPage:
        if type(request) is not RunListRequest:
            raise _closed(RunOperationCode.CURSOR_INVALID)
        after = _decode_list_cursor(request, self._cursor_authority)
        page = _invoke(
            lambda: self._workflows.list_page(
                request.project_ref, after_run_key=after, limit=request.limit
            ),
            RunOperationCode.CURSOR_INVALID if after is not None else RunOperationCode.INTEGRITY_ERROR,
        )
        outcomes = tuple(project_run_outcome(record) for record in page.records)
        next_cursor = None
        if page.has_more:
            if not page.records:
                raise _closed(RunOperationCode.INTEGRITY_ERROR)
            next_cursor = self._issue_cursor(
                CursorContentV1(
                    CursorKindV1.RUN_LIST,
                    _project_digest(request.project_ref),
                    after_run_key=bytes.fromhex(_run_key(page.records[-1].run)),
                )
            )
        return RunPage(request, outcomes, next_cursor, page.has_more)

    def show(self, run: TrainingRunRef) -> RunOutcome:
        return project_run_outcome(self._workflow(run))

    def outcome(self, run: TrainingRunRef) -> RunOutcome:
        workflow = self._workflow(run)
        if workflow.phase not in self._ACTIVE_OBSERVATION:
            return project_run_outcome(workflow)
        descriptor = self._descriptor(workflow)
        self._capability(descriptor, "observe")
        request = self._request(workflow, ProviderReadPurposeV1.OBSERVE)
        observation = _invoke(
            lambda: self._reader.observe(request), RunOperationCode.PROVIDER_READ_INVALID
        )
        replacement = _invoke(
            lambda: apply_provider_observation(
                workflow, request, observation, self._observation_authenticator
            ),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        retained = self._cas(
            workflow,
            replacement,
            ApplyProviderObservationTransitionV1(request, observation),
        )
        return project_run_outcome(retained)

    def logs(self, public_request: RunLogsRequest) -> RunLogPage:
        if type(public_request) is not RunLogsRequest:
            raise _closed(RunOperationCode.LOG_BOUNDS_INVALID)
        after = _decode_log_cursor(public_request, self._cursor_authority)
        workflow = self._workflow(public_request.run)
        descriptor = self._descriptor(workflow)
        self._capability(descriptor, "logs")
        query = ProviderLogQueryV1(after, public_request.limit, public_request.maximum_bytes)
        request = self._request(workflow, ProviderReadPurposeV1.LOGS)
        supplied = _invoke(
            lambda: self._reader.logs(request, query),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        page = _invoke(
            lambda: AuthenticatedProviderLogPageV1.parse(supplied.canonical_bytes),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        authenticated = _invoke(
            lambda: self._log_authenticator.authenticate(page),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        if authenticated is not True:
            raise _closed(RunOperationCode.PROVIDER_READ_INVALID)
        content = page.content
        ref = request.provider_run.reference
        expected = (
            request.request_digest,
            query.log_query_digest,
            workflow.record_digest,
            workflow.revision,
            workflow.run,
            request.provider_run.binding_digest,
            ref.provider_id,
            ref.profile_ref,
            ref.account_ref,
            ref.namespace_ref,
            ref.provider_job_ref,
            after,
        )
        actual = (
            content.read_request_digest,
            content.log_query_digest,
            content.source_workflow_record_digest,
            content.source_revision,
            content.run,
            content.provider_run_binding_digest,
            content.provider_id,
            content.profile_ref,
            content.account_ref,
            content.namespace_ref,
            content.provider_job_ref,
            content.after_sequence,
        )
        if actual != expected or len(content.entries) > query.limit or content.total_bytes > query.maximum_bytes:
            raise _closed(RunOperationCode.PROVIDER_READ_INVALID)
        next_cursor = None
        if content.truncated:
            if not content.entries:
                raise _closed(RunOperationCode.PROVIDER_READ_INVALID)
            next_cursor = self._issue_cursor(
                CursorContentV1(
                    CursorKindV1.RUN_LOGS,
                    _log_run_digest(workflow.run),
                    after_sequence=content.entries[-1].sequence,
                )
            )
        return RunLogPage(
            public_request,
            content.entries,
            content.total_bytes,
            next_cursor,
            content.truncated,
        )

    def cancel(self, run: TrainingRunRef, reason: str) -> RunOutcome:
        workflow = self._workflow(run)
        descriptor = self._descriptor(workflow)
        self._capability(descriptor, "cancel")
        self._capability(descriptor, "reconcile")
        failure = None
        retained = None
        try:
            retained = self._coordinator.cancel(run, reason)
        except CoordinatorErrorV1 as error:
            failure = (
                RunOperationCode.INTEGRITY_ERROR
                if error.code is CoordinatorCodeV1.STORE_INTEGRITY
                else RunOperationCode.CANCEL_INELIGIBLE
                if error.code is CoordinatorCodeV1.INVALID_INPUT
                else RunOperationCode.STATE_CONFLICT
            )
        except Exception:
            failure = RunOperationCode.STATE_CONFLICT
        if failure is not None:
            raise _closed(failure)
        return project_run_outcome(retained)

    def reconcile(self, run: TrainingRunRef) -> RunOutcome:
        workflow = self._workflow(run)
        descriptor = self._descriptor(workflow)
        self._capability(descriptor, "reconcile")
        failure = None
        retained = None
        try:
            retained = self._coordinator.reconcile(run)
        except CoordinatorErrorV1 as error:
            failure = (
                RunOperationCode.INTEGRITY_ERROR
                if error.code is CoordinatorCodeV1.STORE_INTEGRITY
                else RunOperationCode.STATE_CONFLICT
            )
        except Exception:
            failure = RunOperationCode.STATE_CONFLICT
        if failure is not None:
            raise _closed(failure)
        return project_run_outcome(retained)

    def verify(self, run: TrainingRunRef) -> RunVerification:
        workflow = self._workflow(run)
        descriptor = self._descriptor(workflow)
        self._capability(descriptor, "artifact_streaming")
        request = self._request(workflow, ProviderReadPurposeV1.ARTIFACTS)
        manifest = _invoke(
            lambda: self._reader.artifacts(request),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        receipt = _invoke(
            lambda: self._artifact_verifier.verify(workflow, manifest),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        replacement = _invoke(
            lambda: apply_artifact_verification(
                workflow, manifest, receipt, self._artifact_verifier
            ),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        retained = self._cas(
            workflow,
            replacement,
            ApplyArtifactVerificationTransitionV1(manifest, receipt),
        )
        return RunVerification(
            retained.run,
            retained.phase is WorkflowPhaseV1.VERIFIED,
            self._clock.now_iso(),
        )

    def reverify(self, run: TrainingRunRef) -> RunVerification:
        workflow = self._workflow(run)
        descriptor = self._descriptor(workflow)
        self._capability(descriptor, "artifact_streaming")
        if (
            workflow.phase is not WorkflowPhaseV1.VERIFIED
            or workflow.artifact_manifest is None
            or not workflow.verification_receipts
        ):
            raise _closed(RunOperationCode.ARTIFACTS_UNVERIFIED)
        receipt = _invoke(
            lambda: self._artifact_verifier.replay(
                workflow,
                workflow.artifact_manifest,
                workflow.verification_receipts[-1],
            ),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        replacement = _invoke(
            lambda: apply_reverification(
                workflow,
                workflow.artifact_manifest,
                receipt,
                self._artifact_verifier,
            ),
            RunOperationCode.PROVIDER_READ_INVALID,
        )
        retained = self._cas(
            workflow,
            replacement,
            ApplyReverificationTransitionV1(workflow.artifact_manifest, receipt),
        )
        return RunVerification(retained.run, True, self._clock.now_iso())

    def artifacts(self, public_request: RunArtifactRequest) -> RunArtifactStream:
        if type(public_request) is not RunArtifactRequest:
            raise _closed(RunOperationCode.ARTIFACT_CONTENT_INVALID)
        workflow = self._workflow(public_request.run)
        descriptor = self._descriptor(workflow)
        self._capability(descriptor, "artifact_streaming")
        if workflow.phase is not WorkflowPhaseV1.VERIFIED or workflow.artifact_manifest is None:
            raise _closed(RunOperationCode.ARTIFACTS_UNVERIFIED)
        matches = tuple(
            artifact
            for artifact in workflow.verified_artifacts
            if artifact.role == public_request.role
        )
        if len(matches) != 1:
            raise _closed(RunOperationCode.ARTIFACT_ROLE_MISSING)
        artifact = matches[0]
        if artifact.size_bytes > public_request.maximum_bytes:
            raise _closed(RunOperationCode.ARTIFACT_LIMIT_EXCEEDED)
        if (
            artifact.size_bytes == 0
            and artifact.sha256 != hashlib.sha256(b"").hexdigest()
        ):
            raise _closed(RunOperationCode.ARTIFACT_CONTENT_INVALID)
        request = self._request(workflow, ProviderReadPurposeV1.ARTIFACTS)
        return _ArtifactStreamV1(
            workflow.run,
            artifact,
            public_request.maximum_bytes,
            self._reader,
            request,
            workflow.artifact_manifest,
        )


__all__: list[str] = []
