"""Private, source-free read material for one already authenticated HF job."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.providers import ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.runs_facade import RunOperationError, RunOperationCode
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.training.contracts import ResourceSpec
from tuner.project.execution_source import ExecutionSourceV1
from tuner.training.contracts import execution_material_digest
from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest
from tuner.training.contracts import CanonicalDocument, _json_object, _bound_json
from tuner.training.methods.sft import compile_sft_workload
from .model import HFJobsError, document_bytes

_SCHEMA = "synaptic-hf-read-material/v1"
_LIMITS = {
    "material_base64": 1048576,
    "bundle_projection_base64": 3145728,
    "bootstrap_base64": 65536,
}
_MAX_RECORD = 6291456


def _document(raw):
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
    if type(value) is not dict or document_bytes(value) != raw:
        raise HFJobsError("binding_mismatch")
    return value


def _decode(value, maximum):
    if type(value) is not str or len(value) > ((maximum + 2) // 3) * 4:
        raise HFJobsError("binding_mismatch")
    raw = base64.b64decode(value, validate=True)
    if not raw or len(raw) > maximum or base64.b64encode(raw).decode("ascii") != value:
        raise HFJobsError("binding_mismatch")
    return raw


def _digest(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise HFJobsError("binding_mismatch")
    return value


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _validate_selection(value):
    """Validate consumed original projection, without a new executable profile."""
    fields = {
        "schema_version",
        "provider",
        "scope",
        "runtime_profile",
        "runtime_profile_sha256",
        "compatibility",
        "image",
        "inventory_sha256",
        "bucket_id",
        "source_prefix",
        "run_prefix",
        "flavor",
        "timeout_seconds",
        "resources",
        "host_token",
        "job_token",
        "job_secret_key",
        "model_preparation_token",
        "quote_digest",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] != "synaptic-hf-jobs-selection/v1"
    ):
        raise HFJobsError("binding_mismatch")
    provider = ProviderRef.from_dict(value["provider"])
    ExecutionScopeV1(**value["scope"])
    SecretRef.from_dict(value["host_token"])
    SecretRef.from_dict(value["job_token"])
    resources = ResourceSpec(**value["resources"])
    if (
        provider.provider_id != "hf_jobs"
        or value["job_secret_key"] != "HF_TOKEN"
        or value["model_preparation_token"] is not False
        or type(value["timeout_seconds"]) is not int
        or not 1 <= value["timeout_seconds"] <= 86400
        or resources.timeout_seconds != value["timeout_seconds"]
        or type(value["resources"]["accelerator_count"]) is not int
    ):
        raise HFJobsError("binding_mismatch")
    if type(value["runtime_profile_sha256"]) is not str or not value[
        "runtime_profile_sha256"
    ].startswith("sha256:"):
        raise HFJobsError("binding_mismatch")
    _digest(value["runtime_profile_sha256"][7:])
    for name in ("inventory_sha256", "quote_digest"):
        _digest(value[name])
    if (
        type(value["runtime_profile"]) is not str
        or not 1 <= len(value["runtime_profile"]) <= 128
    ):
        raise HFJobsError("binding_mismatch")
    if (
        type(value["image"]) is not str
        or not 1 <= len(value["image"]) <= 512
        or "@sha256:" not in value["image"]
    ):
        raise HFJobsError("binding_mismatch")
    image, digest = value["image"].rsplit("@sha256:", 1)
    if not image or any(character.isspace() for character in image):
        raise HFJobsError("binding_mismatch")
    _digest(digest)
    token = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}"
    if (
        type(value["bucket_id"]) is not str
        or re.fullmatch(token + "/" + token, value["bucket_id"]) is None
    ):
        raise HFJobsError("binding_mismatch")
    if type(value["flavor"]) is not str or re.fullmatch(token, value["flavor"]) is None:
        raise HFJobsError("binding_mismatch")
    for name in ("source_prefix", "run_prefix"):
        prefix = value[name]
        if (
            type(prefix) is not str
            or not 1 <= len(prefix) <= 256
            or any(
                part in {".", ".."} or re.fullmatch(token, part) is None
                for part in prefix.split("/")
            )
        ):
            raise HFJobsError("binding_mismatch")
    a, b = value["source_prefix"], value["run_prefix"]
    if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
        raise HFJobsError("binding_mismatch")
    compatibility = value["compatibility"]
    if (
        type(compatibility) is not dict
        or set(compatibility) != {"models", "methods"}
        or type(compatibility["methods"]) is not list
        or "sft" not in compatibility["methods"]
        or any(type(method) is not str for method in compatibility["methods"])
        or type(compatibility["models"]) is not dict
        or not compatibility["models"]
    ):
        raise HFJobsError("binding_mismatch")
    for model, revisions in compatibility["models"].items():
        if (
            type(model) is not str
            or not 1 <= len(model) <= 512
            or type(revisions) is not list
            or not revisions
            or any(
                type(revision) is not str
                or not revision
                or revision != revision.strip()
                for revision in revisions
            )
        ):
            raise HFJobsError("binding_mismatch")


def _parse_capture(raw):
    """Parse three bounded blobs; no source, lease, SDK, or issued bundle object."""
    try:
        if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_RECORD:
            raise HFJobsError("binding_mismatch")
        envelope = _document(raw)
        if (
            set(envelope) != {"schema_version", "original_descriptor", *_LIMITS}
            or envelope["schema_version"] != _SCHEMA
        ):
            raise HFJobsError("binding_mismatch")
        if len(document_bytes(envelope["original_descriptor"])) > 16384:
            raise HFJobsError("binding_mismatch")
        descriptor = ProviderDescriptor.from_dict(envelope["original_descriptor"])
        if descriptor.provider_id != "hf_jobs":
            raise HFJobsError("binding_mismatch")
        decoded = {
            key: _decode(envelope[key], maximum) for key, maximum in _LIMITS.items()
        }
        material = _document(decoded["material_base64"])
        _bound_json(material)
        if set(material) != {
            "schema_version",
            "selection",
            "rich_plan_fingerprint",
            "host_sdk_lock_sha256",
            "bundle_projection_sha256",
            "members",
            "input",
            "authorization",
        }:
            raise HFJobsError("binding_mismatch")
        if material["schema_version"] != "synaptic-hf-jobs-full-material/v1":
            raise HFJobsError("binding_mismatch")
        _validate_selection(material["selection"])
        for key in (
            "rich_plan_fingerprint",
            "host_sdk_lock_sha256",
            "bundle_projection_sha256",
        ):
            _digest(material[key])
        projection = _document(decoded["bundle_projection_base64"])
        if set(projection) != {
            "schema_version",
            "plan_fingerprint",
            "workload_fingerprint",
            "workload",
            "closure_manifest",
            "dispatch",
        }:
            raise HFJobsError("binding_mismatch")
        # Large payload strings have explicit byte limits; the remaining tree
        # uses the existing finite structural validator.
        metadata = {
            key: value
            for key, value in projection.items()
            if key not in {"workload", "closure_manifest"}
        }
        _bound_json(metadata)
        if (
            projection["schema_version"] != "synaptic-worker-bundle-materialization/v1"
            or _sha(decoded["bundle_projection_base64"])
            != material["bundle_projection_sha256"]
            or projection["plan_fingerprint"] != material["rich_plan_fingerprint"]
        ):
            raise HFJobsError("binding_mismatch")
        workload_slot = projection["workload"]
        if type(workload_slot) is not dict or set(workload_slot) != {
            "payload_base64",
            "byte_count",
            "sha256",
        }:
            raise HFJobsError("binding_mismatch")
        workload_raw = _decode(workload_slot["payload_base64"], 1048576)
        if (
            type(workload_slot["byte_count"]) is not int
            or workload_slot["byte_count"] != len(workload_raw)
            or workload_slot["sha256"] != _sha(workload_raw)
        ):
            raise HFJobsError("binding_mismatch")
        workload_document = _document(workload_raw)
        _bound_json(workload_document)
        source = ExecutionSourceV1.from_dict(workload_document["execution_source"])
        config = CanonicalDocument(
            document_bytes(workload_document["configuration"]["document"]).decode(
                "utf-8"
            )
        )
        workload = compile_sft_workload(resolved_config=config, execution_source=source)
        if (
            workload.canonical_bytes != workload_raw
            or workload.fingerprint != projection["workload_fingerprint"]
        ):
            raise HFJobsError("binding_mismatch")
        closure_slot = projection["closure_manifest"]
        if type(closure_slot) is not dict or set(closure_slot) != {
            "payload_base64",
            "source",
            "byte_count",
            "sha256",
            "closure_digest",
            "runtime_path",
        }:
            raise HFJobsError("binding_mismatch")
        closure_raw = _decode(closure_slot["payload_base64"], 1048576)
        closure = parse_offline_sft_worker_manifest(
            closure_raw,
            source_ref=closure_slot["source"],
            manifest_path=Path(closure_slot["runtime_path"]),
            expected_digest=closure_slot["closure_digest"],
        )
        if (
            type(closure_slot["byte_count"]) is not int
            or closure_slot["byte_count"] != len(closure_raw)
            or closure_slot["sha256"] != _sha(closure_raw)
            or closure_slot["closure_digest"] != closure.closure.closure_digest
        ):
            raise HFJobsError("binding_mismatch")
        members = material["members"]
        if type(members) is not list or not 1 <= len(members) <= 4096:
            raise HFJobsError("binding_mismatch")
        indexed = {}
        for member in members:
            if type(member) is not dict or set(member) != {
                "path",
                "size_bytes",
                "sha256",
            }:
                raise HFJobsError("binding_mismatch")
            name = member["path"]
            if type(name) is not str or not 1 <= len(name) <= 512:
                raise HFJobsError("binding_mismatch")
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or path.as_posix() != name
                or "\\" in name
                or any(part in {".", ".."} for part in path.parts)
                or name in indexed
                or type(member["size_bytes"]) is not int
                or member["size_bytes"] < 0
            ):
                raise HFJobsError("binding_mismatch")
            _digest(member["sha256"])
            indexed[name] = member
        for name, payload in (
            ("control/workload.json", workload_raw),
            ("control/offline-sft-worker-v1.json", closure_raw),
            (
                "parent/tuner/execution/providers/hf_jobs_provider_v1/bootstrap.py",
                decoded["bootstrap_base64"],
            ),
        ):
            if indexed.get(name) != {
                "path": name,
                "size_bytes": len(payload),
                "sha256": _sha(payload),
            }:
                raise HFJobsError("binding_mismatch")
        runtime_inventory = indexed.get("control/runtime-inventory.json")
        if (
            runtime_inventory is None
            or runtime_inventory["sha256"] != material["selection"]["inventory_sha256"]
        ):
            raise HFJobsError("binding_mismatch")
        selected_model = workload_document["configuration"]["document"]["model"]
        if selected_model["revision"] not in material["selection"]["compatibility"][
            "models"
        ].get(selected_model["ref"], []):
            raise HFJobsError("binding_mismatch")
        decoded["bootstrap_base64"].decode("utf-8")
        identity = PreparedTrainingInputIdentity.from_dict(material["input"])
        if (
            workload_document["configuration"]["document"]["dataset"]
            != identity.to_dict()
        ):
            raise HFJobsError("binding_mismatch")
        return {
            "descriptor": descriptor,
            "material": material,
            "projection": projection,
            "workload": workload,
            "source": source,
            "source_digest": execution_material_digest(source),
            "bootstrap": decoded["bootstrap_base64"],
            "identity": identity,
            "material_digest": _sha(decoded["material_base64"]),
        }
    except Exception:
        raise HFJobsError("binding_mismatch") from None


def _capture(preparation, records):
    """Retain genuine admitted bytes; identical warm reuse never replays effects."""
    from .preparation import HFJobsPreparation

    if type(preparation) is not HFJobsPreparation:
        raise HFJobsError("input_invalid")
    raw = document_bytes(
        {
            "schema_version": _SCHEMA,
            "original_descriptor": preparation.descriptor.to_dict(),
            "material_base64": base64.b64encode(preparation.material_bytes).decode(
                "ascii"
            ),
            "bundle_projection_base64": base64.b64encode(
                preparation.bundle.canonical_projection_bytes
            ).decode("ascii"),
            "bootstrap_base64": base64.b64encode(
                preparation.source_files[
                    "parent/tuner/execution/providers/hf_jobs_provider_v1/bootstrap.py"
                ]
            ).decode("ascii"),
        }
    )
    captured = _parse_capture(raw)
    if (
        captured["material_digest"] != preparation.material_digest
        or captured["descriptor"] != preparation.descriptor
    ):
        raise HFJobsError("binding_mismatch")
    key = (
        "hf-jobs/"
        + preparation.material_digest
        + "/read-material/"
        + preparation.material_digest
    )
    stored = records.put_if_absent(partition="effects", key=key, canonical=raw)
    if type(stored) is not bool:
        raise HFJobsError("binding_mismatch")
    if stored is not True:
        existing = records.read(partition="effects", key=key)
        if existing is None or existing.canonical != raw:
            raise HFJobsError("binding_mismatch")
    return captured


class _ReadOperations:
    """Expose one run's existing read verbs, never effect/reconciliation verbs."""

    def __init__(self, operations, workflows, run):
        self._operations, self._workflows, self._run = operations, workflows, run

    def _check(self, run):
        if run != self._run:
            raise RunOperationError(RunOperationCode.RUN_MISSING)

    def list(self, request):
        if request.project_ref != self._run.project_ref:
            raise RunOperationError(RunOperationCode.RUN_MISSING)
        self._check(self._run)
        result = self._operations.list(request)
        if any(outcome.run != self._run for outcome in result.outcomes):
            raise RunOperationError(RunOperationCode.INTEGRITY_ERROR)
        return result

    def show(self, run):
        self._check(run)
        return self._operations.show(run)

    def outcome(self, run):
        self._check(run)
        return self._operations.outcome(run)

    def verify(self, run):
        from tuner.execution.coordinator_v1.model import WorkflowPhaseV1

        self._check(run)
        if self._workflows.get(run).phase not in {
            WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,
            WorkflowPhaseV1.VERIFICATION_FAILED,
        }:
            raise RunOperationError(RunOperationCode.READ_INELIGIBLE)
        return self._operations.verify(run)

    def artifacts(self, request):
        self._check(request.run)
        return self._operations.artifacts(request)

    def _unavailable(self, *args, **kwargs):
        raise RunOperationError(RunOperationCode.CAPABILITY_UNAVAILABLE)

    logs = cancel = reconcile = reverify = _unavailable


class _RunsView:
    __slots__ = ("run", "runs")

    def __init__(self, run, runs):
        self.run, self.runs = run, runs


class _LifecycleOperations(_ReadOperations):
    """Only the original known run can acquire cancellation/lookup authority."""

    def cancel(self, run, reason):
        self._check(run)
        return self._operations.cancel(run, reason)

    def reconcile(self, run):
        self._check(run)
        workflow = self._workflows.get(run)
        if workflow is None or workflow.cancel is None:
            raise RunOperationError(RunOperationCode.READ_INELIGIBLE)
        return self._operations.reconcile(run)


def _cancel_coordinator(*, ports, authority, transport, workflows, plans,
                        plan, context, descriptor, command, preparation,
                        binding, adapter, allow_current_cancel_authorization):
    """Reuse existing coordinator/Foundation ports, without executable preparation."""
    from synaptic_tuner.api.v1.reference.repositories import (
        DurablePreparationStoreV1, DurableExecutionGrantStoreV1,
        DurableReconciliationGrantStoreV1, DurableEffectRepositoryV1,
    )
    from synaptic_tuner.api.v1.reference.authority import (
        UnavailableRecoveryVerifierV1, UnavailableQuiescenceEvidenceV1,
    )
    from synaptic_tuner.api.v1.reference.training import ReferenceAuthorizationV1
    from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
    from tuner.execution.coordinator_v1.foundation import ComposedEffectFoundationV1
    from tuner.execution.foundation_v2.broker import EffectBrokerV2
    from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
    from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1
    from tuner.execution.foundation_v2.identities import EffectKind
    from .composition import (
        _BoundGrants, _Executor, _ExecutorResolver, _Reconciliation,
        _ReconciliationResolver, _CancelAccess,
    )

    class RetainedPorts:
        def describe(self, provider):
            if provider != context.provider:
                raise HFJobsError("binding_mismatch")
            return descriptor

        def resolve(self, provider, retained_context):
            if provider != context.provider or retained_context != context:
                raise HFJobsError("binding_mismatch")
            return binding

        def prepare(self, *args, **kwargs):
            raise HFJobsError("capability_unavailable")

        def payload(self, retained_preparation, kind):
            if retained_preparation != preparation or kind is not EffectKind.CANCEL:
                raise HFJobsError("binding_mismatch")
            return CanonicalProviderPayloadV1.build(
                "hf_jobs", "cancel-payload/v2", command.payload.input_digest)

    retained_ports = RetainedPorts()
    recovery = UnavailableRecoveryVerifierV1()
    repository = DurableEffectRepositoryV1(
        ports.records, receipt_authority=authority.receipt_authority,
        invalid_evidence_authority=authority.invalid_evidence_authority,
        recovery_verifier=recovery, finality_verifier=recovery,
        grant_authority=authority.grant_authority,
    )
    executor = _Executor(None, transport, authority.clock, descriptor=command.executor)
    broker = EffectBrokerV2(
        repository, _ExecutorResolver(executor), authority.grant_authority,
        authority.receipt_authority, authority.invalid_evidence_authority)
    reconciliation = ReconciliationServiceV1(
        repository, authority.grant_authority,
        _ReconciliationResolver(_Reconciliation(None, transport, descriptor=adapter)),
        authority.receipt_authority, authority.invalid_evidence_authority)
    foundation = ComposedEffectFoundationV1(
        repository, broker, reconciliation,
        grant_authority=authority.grant_authority,
        receipt_authority=authority.receipt_authority,
        invalid_evidence_authority=authority.invalid_evidence_authority,
        assessment_authority=authority.assessment_authority,
        trusted_quiescence_evidence=UnavailableQuiescenceEvidenceV1(),
    )
    authorization = ReferenceAuthorizationV1(
        records=ports.records, grants=_BoundGrants(ports.grants, transport),
        authority=authority.grant_authority, clock=authority.clock,
        allow_current_cancel_authorization=allow_current_cancel_authorization,
    )
    coordinator = TrainingCoordinatorV1(
        retained_ports, plans, workflows, DurablePreparationStoreV1(ports.records),
        DurableExecutionGrantStoreV1(ports.records, authority.grant_authority),
        DurableReconciliationGrantStoreV1(ports.records, authority.grant_authority),
        retained_ports, retained_ports, authorization, foundation,
        authority.foundation_authenticator, authority.clock, None,
    )
    transport.bind_cancel_access(_CancelAccess(
        None, None, authority, transport, plan=plan, context=context,
        descriptor=descriptor, workflows=workflows, foundation=foundation))
    return coordinator, foundation


def _open_retained_hf_runs(
    *,
    run,
    records,
    streams,
    authority,
    secrets,
    sdk,
    api,
    scratch_root,
    reader_authority_ref,
    reader_key_ref,
    reader_key,
    deadline_seconds=30,
    lifecycle_ports=None,
    allow_current_cancel_authorization=False,
):
    """Authenticate the original basis shared by reads and explicit cancellation.

    No input/source preparation or start authority is reconstructed. Only the
    lifecycle assembly accepts existing host grants for the original known run.
    """
    from synaptic_tuner.api.v1.results import TrainingRunRef
    from synaptic_tuner.api.v1.providers import ProviderRef
    from synaptic_tuner.api.v1.secrets import SecretRef
    from synaptic_tuner.api.v1.runs_facade import RunsAPI
    from synaptic_tuner.api.v1.reference.provider_family import require_methods
    from synaptic_tuner.api.v1.reference.repositories import (
        DurableEffectRepositoryV1,
        DurablePlanningStoreV1,
        DurableWorkflowStoreV1,
    )
    from synaptic_tuner.api.v1.reference.authority import (
        UnavailableRecoveryVerifierV1,
        UnavailableQuiescenceEvidenceV1,
    )
    from synaptic_tuner.api.v1.reference.runs import build_run_operations
    from tuner.execution.coordinator_v1.foundation import ComposedEffectFoundationV1
    from tuner.execution.coordinator_v1.model import (
        WorkflowPhaseV1,
        FoundationDispositionV1,
        AuthenticatedFoundationRecordAssessmentV1,
    )
    from tuner.execution.coordinator_v1.state_machine import _derive_foundation
    from tuner.execution.foundation_v2.commands import parse_exact_command
    from tuner.execution.foundation_v2.references import ExecutionScopeV1
    from tuner.training.modal_host_authority import (
        ReaderEvidenceAuthority,
        HMACAuthenticator,
        ObservationAuthenticator,
        LogAuthenticator,
    )
    from .reader import HFJobsReader, HFJobsArtifactVerifier
    from .transport import HFJobsTransport

    try:
        if type(allow_current_cancel_authorization) is not bool:
            raise HFJobsError("input_invalid")
        if lifecycle_ports is not None:
            from synaptic_tuner.api.v1.reference.provider_family import ReferenceHostPortsV1
            if (type(lifecycle_ports) is not ReferenceHostPortsV1
                    or lifecycle_ports.records is not records
                    or lifecycle_ports.streams is not streams
                    or lifecycle_ports.secrets is not secrets):
                raise HFJobsError("input_invalid")
            require_methods(lifecycle_ports.grants, "authorize", "bind")
        if type(run) is not TrainingRunRef:
            raise HFJobsError("input_invalid")
        require_methods(
            records, "read", "list_page", "create", "compare_and_swap", "put_if_absent"
        )
        require_methods(streams, "read_page", "append")
        if type(reader_key) is not bytes or len(reader_key) < 32:
            raise HFJobsError("input_invalid")
        evidence = ReaderEvidenceAuthority(
            reader_authority_ref,
            reader_key_ref,
            HMACAuthenticator(
                {reader_key_ref: reader_key},
                allowed_purposes=frozenset(
                    {"provider-run-observation/v1", "provider-log-page/v1"}
                ),
            ),
        )
        reader = HFJobsReader(
            preparation=None,
            transport=None,
            foundation_authenticator=authority.foundation_authenticator,
            assessment_authenticator=authority.assessment_authority,
            evidence_authority=evidence,
            clock=authority.clock,
        )
        verifier = HFJobsArtifactVerifier(
            reader=reader,
            authority_ref=reader_authority_ref,
            key_ref=reader_key_ref,
            key=reader_key,
            clock=authority.clock,
        )
        workflows = DurableWorkflowStoreV1(
            records,
            streams,
            foundation_authenticator=authority.foundation_authenticator,
            assessment_authenticator=authority.assessment_authority,
            observation_authenticator=ObservationAuthenticator(evidence),
            artifact_verifier=verifier,
        )
        workflow = workflows.get(run)
        allowed = {
            WorkflowPhaseV1.QUEUED,
            WorkflowPhaseV1.RUNNING,
            WorkflowPhaseV1.CANCEL_REQUESTED,
            WorkflowPhaseV1.CANCEL_RECONCILE_REQUIRED,
            WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,
            WorkflowPhaseV1.VERIFIED,
            WorkflowPhaseV1.VERIFICATION_FAILED,
            WorkflowPhaseV1.FAILED,
            WorkflowPhaseV1.CANCELLED,
        }
        if lifecycle_ports is not None:
            allowed.add(WorkflowPhaseV1.CANCEL_INTENT_RECORDED)
        if (
            workflow is None
            or workflow.phase not in allowed
            or workflow.submit is None
            or workflow.provider_run_ref is None
        ):
            raise HFJobsError("binding_mismatch")
        plans = DurablePlanningStoreV1(records)
        plan = plans.get_plan(workflow.plan_fingerprint)
        context = plans.get_context(workflow.provider_context_digest)
        if plan is None or context is None:
            raise HFJobsError("binding_mismatch")
        material_digest = plan.basis.resolved_config_digest
        retained = records.read(
            partition="effects",
            key="hf-jobs/" + material_digest + "/read-material/" + material_digest,
        )
        if retained is None:
            raise HFJobsError("binding_mismatch")
        captured = _parse_capture(retained.canonical)
        selection = captured["material"]["selection"]
        provider = ProviderRef(**selection["provider"])
        scope = ExecutionScopeV1(**selection["scope"])
        SecretRef.from_dict(selection["host_token"])
        SecretRef.from_dict(selection["job_token"])
        descriptor = captured["descriptor"]
        if (
            captured["material_digest"] != material_digest
            or captured["workload"].fingerprint != plan.basis.workload_digest
            or captured["source_digest"] != plan.basis.source_digest
            or context.profile_digest != _sha(document_bytes(selection))
            or context.provider != provider
            or workflow.provider != provider
            or context.descriptor_digest != descriptor.descriptor_digest
            or workflow.provider_descriptor_digest != descriptor.descriptor_digest
        ):
            raise HFJobsError("binding_mismatch")
        recovery = UnavailableRecoveryVerifierV1()
        repository = DurableEffectRepositoryV1(
            records,
            receipt_authority=authority.receipt_authority,
            invalid_evidence_authority=authority.invalid_evidence_authority,
            recovery_verifier=recovery,
            finality_verifier=recovery,
            grant_authority=authority.grant_authority,
        )
        foundation = ComposedEffectFoundationV1(
            repository,
            None,
            None,
            grant_authority=authority.grant_authority,
            receipt_authority=authority.receipt_authority,
            invalid_evidence_authority=authority.invalid_evidence_authority,
            assessment_authority=authority.assessment_authority,
            trusted_quiescence_evidence=UnavailableQuiescenceEvidenceV1(),
        )
        submit = workflow.submit
        current = foundation.get(submit.effect_id)
        assessment = AuthenticatedFoundationRecordAssessmentV1.parse(
            submit.foundation_bindings[-1].canonical_assessment_bytes
        )
        derived = _derive_foundation(
            submit,
            current,
            assessment,
            authority.foundation_authenticator,
            authority.assessment_authority,
            None,
        )
        if (
            derived
            != (
                submit.foundation_bindings[-1],
                submit.foundation_outcomes[-1],
                workflow.provider_run_ref,
            )
            or derived[1].disposition is not FoundationDispositionV1.FOUND
        ):
            raise HFJobsError("binding_mismatch")
        command = parse_exact_command(submit.canonical_command_bytes)
        ref = workflow.provider_run_ref.reference
        if (
            (ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref)
            != (
                provider.provider_id,
                provider.profile_ref,
                scope.account_ref,
                scope.namespace_ref,
            )
            or command.preparation.plan_fingerprint != plan.plan_fingerprint
            or command.payload.input_digest != captured["workload"].fingerprint
        ):
            raise HFJobsError("binding_mismatch")
        transport_constructor = HFJobsTransport.for_retained_read
        if lifecycle_ports is not None:
            if (descriptor.capabilities.cancel is not True
                    or descriptor.capabilities.reconcile is not True):
                raise HFJobsError("capability_unavailable")
            transport_constructor = HFJobsTransport.for_retained_cancellation
        transport = transport_constructor(
            captured=captured,
            submit=command,
            sdk=sdk,
            api=api,
            records=records,
            secrets=secrets,
            scratch_root=scratch_root,
            deadline_seconds=deadline_seconds,
        )
        catalog = transport.retained("command", command.digest)
        if (
            base64.b64decode(catalog["command_base64"], validate=True)
            != command.canonical_bytes
        ):
            raise HFJobsError("binding_mismatch")
        receipt = transport.retained("submit", command.digest)
        if receipt["job_id"] != ref.provider_job_ref:
            raise HFJobsError("binding_mismatch")
        expected = transport.job_request(command)
        expected["volumes"] = [item.to_dict() for item in expected["volumes"]]
        if receipt["request"] != expected:
            raise HFJobsError("binding_mismatch")
        coordinator = None
        if lifecycle_ports is not None:
            from .preparation import _execution_binding
            from tuner.execution.foundation_v2.executors import AdapterDescriptorV1
            from synaptic_tuner.api.v1.reference.repositories import DurablePreparationStoreV1
            from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1

            adapter = AdapterDescriptorV1("hf_jobs", "hf-jobs-unavailable-reconcile", "0.1.0")
            binding = _execution_binding(selection, descriptor, command.executor, adapter)
            preparation = DurablePreparationStoreV1(records).get(workflow.preparation_digest)
            if preparation is None or preparation != command.preparation:
                raise HFJobsError("binding_mismatch")
            TrainingCoordinatorV1._validate_preparation(preparation, plan, workflow, binding)
            coordinator, foundation = _cancel_coordinator(
                ports=lifecycle_ports, authority=authority, transport=transport,
                workflows=workflows, plans=plans, plan=plan, context=context,
                descriptor=descriptor, command=command, preparation=preparation,
                binding=binding, adapter=adapter,
                allow_current_cancel_authorization=allow_current_cancel_authorization)
        reader.transport = transport
        verifier.bind_foundation(foundation)

        class PlanningReads:
            def describe(self, selected):
                if selected != provider:
                    raise HFJobsError("binding_mismatch")
                return descriptor

        operations = build_run_operations(
            planning=PlanningReads(),
            planning_store=plans,
            workflow_store=workflows,
            coordinator=coordinator,
            foundation=foundation,
            foundation_authenticator=authority.foundation_authenticator,
            assessment_authenticator=authority.assessment_authority,
            reader=reader,
            observation_authenticator=ObservationAuthenticator(evidence),
            log_authenticator=LogAuthenticator(evidence),
            artifact_verifier=verifier,
            cursor_authority=authority.cursor_authority,
            clock=authority.clock,
        )
        wrapper = _ReadOperations if lifecycle_ports is None else _LifecycleOperations
        return _RunsView(run, RunsAPI(wrapper(operations, workflows, run)))
    except Exception:
        raise HFJobsError("read_unavailable") from None


def open_retained_hf_runs(
    *, run, records, streams, authority, secrets, sdk, api, scratch_root,
    reader_authority_ref, reader_key_ref, reader_key, deadline_seconds=30,
):
    """Original known-run read surface; cancellation/reconciliation stay unavailable."""
    return _open_retained_hf_runs(
        run=run, records=records, streams=streams, authority=authority, secrets=secrets,
        sdk=sdk, api=api, scratch_root=scratch_root,
        reader_authority_ref=reader_authority_ref, reader_key_ref=reader_key_ref,
        reader_key=reader_key, deadline_seconds=deadline_seconds)


def open_retained_hf_lifecycle(
    *, run, ports, authority, sdk, api, scratch_root,
    reader_authority_ref, reader_key_ref, reader_key, deadline_seconds=30,
    allow_current_cancel_authorization=False,
):
    """Original configured known run only; no source/start authority is reconstructed."""
    from synaptic_tuner.api.v1.reference.provider_family import ReferenceHostPortsV1
    if (type(ports) is not ReferenceHostPortsV1
            or type(allow_current_cancel_authorization) is not bool):
        raise HFJobsError("input_invalid")
    return _open_retained_hf_runs(
        run=run, records=ports.records, streams=ports.streams, authority=authority,
        secrets=ports.secrets, sdk=sdk, api=api, scratch_root=scratch_root,
        reader_authority_ref=reader_authority_ref, reader_key_ref=reader_key_ref,
        reader_key=reader_key, deadline_seconds=deadline_seconds,
        lifecycle_ports=ports,
        allow_current_cancel_authorization=allow_current_cancel_authorization)


def read_local_stage_diagnostic(*, records, streams, run, authority):
    """Authenticated local staging failure only; no known-job or replay authority."""
    from synaptic_tuner.api.v1.reference.repositories import (
        DurableWorkflowStoreV1, DurablePlanningStoreV1, DurableEffectRepositoryV1,
    )
    from synaptic_tuner.api.v1.reference.authority import UnavailableRecoveryVerifierV1
    from tuner.execution.coordinator_v1.model import (
        WorkflowPhaseV1, AuthenticatedFoundationRecordAssessmentV1,
    )
    from tuner.execution.coordinator_v1.state_machine import _derive_foundation
    from tuner.execution.foundation_v2.commands import parse_exact_command

    # This untrusted head selects a reader only. Every admitted stage result
    # below replays authenticated history; ordinary runs retain their full reader.
    head = records.read(partition="workflow", key=DurableWorkflowStoreV1._head_key(run))
    if head is None or len(head.canonical) > _MAX_RECORD:
        raise HFJobsError("binding_mismatch")
    routing = json.loads(head.canonical)
    if routing.get("record", {}).get("phase") != "stage_reconcile_required":
        return None
    workflows = DurableWorkflowStoreV1(records, streams,
        foundation_authenticator=authority.foundation_authenticator,
        assessment_authenticator=authority.assessment_authority,
        observation_authenticator=None, artifact_verifier=None)
    workflow = workflows.get(run)
    if workflow is None:
        raise HFJobsError("binding_mismatch")
    if workflow.phase is not WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED:
        return None
    if workflow.stage is None or workflow.submit is not None or workflow.provider_run_ref is not None:
        raise HFJobsError("binding_mismatch")
    plans = DurablePlanningStoreV1(records)
    plan = plans.get_plan(workflow.plan_fingerprint)
    context = plans.get_context(workflow.provider_context_digest)
    if plan is None or context is None or context.provider != workflow.provider:
        raise HFJobsError("binding_mismatch")
    material_digest = plan.basis.resolved_config_digest
    key = "hf-jobs/" + material_digest + "/"
    record = records.read(partition="effects", key=key + "read-material/" + material_digest)
    if record is None:
        raise HFJobsError("binding_mismatch")
    captured = _parse_capture(record.canonical)
    selection = captured["material"]["selection"]
    if (captured["material_digest"] != material_digest
            or captured["workload"].fingerprint != plan.basis.workload_digest
            or captured["source_digest"] != plan.basis.source_digest
            or context.profile_digest != _sha(document_bytes(selection))
            or context.provider != ProviderRef(**selection["provider"])
            or context.descriptor_digest != captured["descriptor"].descriptor_digest
            or workflow.provider_descriptor_digest != context.descriptor_digest):
        raise HFJobsError("binding_mismatch")
    stage = workflow.stage
    command = parse_exact_command(stage.canonical_command_bytes)
    claim = records.read(partition="effects", key=key + "claim/stage")
    expected_claim = document_bytes({"schema_version": "synaptic-hf-attempt/v1",
        "material_digest": material_digest, "command_digest": command.digest, "kind": "stage"})
    if claim is None or claim.canonical != expected_claim:
        raise HFJobsError("binding_mismatch")
    recovery = UnavailableRecoveryVerifierV1()
    repository = DurableEffectRepositoryV1(records,
        receipt_authority=authority.receipt_authority,
        invalid_evidence_authority=authority.invalid_evidence_authority,
        recovery_verifier=recovery, finality_verifier=recovery,
        grant_authority=authority.grant_authority)
    assessment = AuthenticatedFoundationRecordAssessmentV1.parse(
        stage.foundation_bindings[-1].canonical_assessment_bytes)
    _derive_foundation(stage, repository.get(stage.effect_id), assessment,
        authority.foundation_authenticator, authority.assessment_authority, None)
    record = records.read(partition="effects", key=key + "stage-diagnostic/" + command.digest)
    diagnostic = None
    if record is not None:
        if len(record.canonical) > 1024:
            raise HFJobsError("binding_mismatch")
        diagnostic = json.loads(record.canonical)
        if document_bytes(diagnostic) != record.canonical:
            raise HFJobsError("binding_mismatch")
        if (type(diagnostic) is not dict
                or set(diagnostic) != {"command_digest", "material_digest", "phase", "failure_class"}
                or diagnostic["command_digest"] != command.digest
                or diagnostic["material_digest"] != material_digest
                or diagnostic["phase"] not in {"empty_slots", "input_custody", "upload",
                    "readback_inventory", "readback_bytes", "receipt"}
                or diagnostic["failure_class"] not in {"deadline", "sdk", "binding", "local"}):
            raise HFJobsError("binding_mismatch")
    return {"run": run.to_dict(), "state": "reconcile_required",
        "local_workflow_phase": workflow.phase.value, "local_stage_diagnostic": diagnostic,
        "provider_evidence": False, "replay_authorized": False}
