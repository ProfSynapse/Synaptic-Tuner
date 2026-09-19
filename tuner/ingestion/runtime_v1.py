"""Lean process-local composition for the V1 ingestion facade.

This module intentionally implements only the synchronous local-selection to
Markdown-bundle path.  Host paths and source bytes remain in private catalogs;
the public lifecycle records contain only content-addressed identities and
closed diagnostic codes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Protocol

from synaptic_tuner.api.v1._contract import canonical_bytes
from synaptic_tuner.api.v1.ingestion_facade import (
    INGESTION_PLAN_SCHEMA_VERSION,
    INGESTION_RESULT_SCHEMA_VERSION,
    AuthorizedSourceRef,
    BindingPreview,
    IngestionDiagnosticCode,
    IngestionListRequest,
    IngestionObservationPage,
    IngestionObservationsRequest,
    IngestionOperationCode,
    IngestionOperationError,
    IngestionOutcome,
    IngestionPage,
    IngestionPlan,
    IngestionPreflight,
    IngestionPreview,
    IngestionRequest,
    IngestionResult,
    IngestionRunRef,
    IngestionRunState,
    IngestionStart,
    IngestionVerification,
    NormalizedBundleRef,
    SourceAdmissionKind,
    SourceAdmissionRequest,
    SourceSnapshotRef,
    StructureProposal,
    StructureProposalRequest,
    run_authority_digest,
)
from tuner.ingestion.bundle_v1 import (
    BundleCollisionError,
    BundleDurabilityError,
    BundlePublicationUncertainV1,
    BundlePublicationUncertaintyPhaseV1,
    BundleSemanticIdentityV1,
    BundleValidationError,
    NormalizedItemInputV1,
    VerifiedNormalizedBundleV1,
    retry_bundle_root_durability_v1,
    verify_normalized_bundle_v1,
    write_normalized_bundle_v1,
)
from tuner.ingestion.local_selection_v1 import (
    ImmutableLocalSnapshotV1,
    LocalDiscoveryPolicyV1,
    LocalSelectionErrorV1,
    LocalSelectionRootV1,
    ProcessLocalSelectionRegistryV1,
    glob_matches_v1,
)
from tuner.ingestion.markdown_v1 import (
    MarkdownParseErrorV1,
    map_markdown_fields_v1,
    parse_markdown_v1,
)


_PREFLIGHT_TTL = timedelta(minutes=5)


class _Clock(Protocol):
    def now(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _SnapshotRecord:
    reference: SourceSnapshotRef
    snapshot: ImmutableLocalSnapshotV1


@dataclass(frozen=True, slots=True)
class _PlanRecord:
    plan: IngestionPlan
    items: tuple[NormalizedItemInputV1, ...]
    diagnostics: tuple[IngestionDiagnosticCode, ...]


@dataclass(frozen=True, slots=True)
class _RunRecord:
    run: IngestionRunRef
    outcome: IngestionOutcome
    bundles_root: Path
    semantic_identity: BundleSemanticIdentityV1 | None = None
    uncertainty_phase: BundlePublicationUncertaintyPhaseV1 | None = None


def _closed(code: IngestionOperationCode) -> IngestionOperationError:
    return IngestionOperationError(code)


def _digest(domain: bytes, document: object) -> str:
    return hashlib.sha256(domain + b"\0" + canonical_bytes(document)).hexdigest()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _format_time(value: datetime) -> str:
    rendered = value.isoformat()
    return rendered[:-6] + "Z" if rendered.endswith("+00:00") else rendered


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain_json(item) for item in value]
    return value


class ProcessLocalIngestionOperationsV1:
    """Synchronous, process-local implementation of ``IngestionOperations``.

    Local selection authorization is exposed as a composition helper rather
    than an ingestion verb.  Retaining the exact issued reference here closes
    the gap between the opaque public authority and the registry's deliberately
    one-use consumption API.
    """

    def __init__(self, *, outputs: Mapping[str, Path], clock: _Clock) -> None:
        if not isinstance(outputs, Mapping) or not outputs:
            raise ValueError("outputs must be a nonempty mapping")
        retained_outputs: dict[str, Path] = {}
        for output_ref, path in outputs.items():
            if type(output_ref) is not str or not output_ref:
                raise TypeError("output refs must be nonempty exact strings")
            if not isinstance(path, Path):
                raise TypeError("output paths must be pathlib.Path values")
            retained_outputs[output_ref] = path.absolute()
        if not callable(getattr(clock, "now", None)):
            raise TypeError("clock must provide now()")
        self._outputs = retained_outputs
        self._clock = clock
        self._selections = ProcessLocalSelectionRegistryV1()
        self._authorized: dict[str, AuthorizedSourceRef] = {}
        self._admissions: dict[str, SourceSnapshotRef] = {}
        self._snapshots: dict[str, _SnapshotRecord] = {}
        self._plans: dict[str, _PlanRecord] = {}
        self._preflights: dict[str, IngestionPreflight] = {}
        self._runs: dict[str, _RunRecord] = {}
        self._runs_by_start: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    def authorize_local_selection(
        self,
        project_ref: str,
        roots: tuple[LocalSelectionRootV1, ...],
        policy: LocalDiscoveryPolicyV1,
    ) -> AuthorizedSourceRef:
        issued = self._selections.authorize(project_ref, roots, policy)
        retained = AuthorizedSourceRef.from_dict(issued.to_dict())
        returned = AuthorizedSourceRef.from_dict(issued.to_dict())
        with self._lock:
            self._authorized[retained.source_ref] = retained
        return returned

    def admit(self, request: SourceAdmissionRequest) -> SourceSnapshotRef:
        if request.source.kind is not SourceAdmissionKind.LOCAL_SELECTION:
            raise _closed(IngestionOperationCode.ADMISSION_INELIGIBLE)
        with self._lock:
            previous = self._admissions.get(request.admission_fingerprint)
            if previous is not None:
                return previous
            issued = self._authorized.get(request.source.source_ref)
            if issued != request.source:
                raise _closed(IngestionOperationCode.ADMISSION_INELIGIBLE)
            self._authorized.pop(request.source.source_ref)
            try:
                snapshot = self._selections.consume_snapshot(request.source.source_ref)
            except LocalSelectionErrorV1:
                raise _closed(IngestionOperationCode.ADMISSION_INELIGIBLE) from None
            except BaseException:
                raise _closed(IngestionOperationCode.OPERATION_FAILED) from None
            if (
                snapshot.project_ref != request.project_ref
                or snapshot.source_ref != request.source.source_ref
            ):
                raise _closed(IngestionOperationCode.INTEGRITY_ERROR)
            snapshot_id = "snapshot-" + _digest(
                b"synaptic-process-local-source-snapshot/v1",
                {
                    "project_ref": request.project_ref,
                    "manifest_digest": snapshot.manifest_digest,
                    "admission_fingerprint": request.admission_fingerprint,
                },
            )
            reference = SourceSnapshotRef(
                request.project_ref,
                snapshot_id,
                SourceAdmissionKind.LOCAL_SELECTION,
                snapshot.manifest_digest,
                len(snapshot.entries),
                request.admission_fingerprint,
            )
            collision = self._snapshots.get(snapshot_id)
            if collision is not None and collision.snapshot != snapshot:
                raise _closed(IngestionOperationCode.INTEGRITY_ERROR)
            self._snapshots[snapshot_id] = _SnapshotRecord(reference, snapshot)
            self._admissions[request.admission_fingerprint] = reference
            return reference

    def propose(
        self, request: StructureProposalRequest
    ) -> tuple[StructureProposal, ...]:
        return ()

    @staticmethod
    def _supported(request: IngestionRequest) -> bool:
        structures = request.structures.structures
        bindings = request.structures.bindings
        if len(structures) != 1 or len(bindings) != 1:
            return False
        structure = structures[0]
        binding = bindings[0]
        return (
            request.metadata_policy is None
            and structure.metadata_policy is None
            and binding.metadata_policy is None
            and structure.schema_ref is None
            and not structure.relationships
            and binding.structure_ref == structure.ref
        )

    def plan(self, request: IngestionRequest) -> IngestionPlan:
        with self._lock:
            snapshot_record = self._snapshots.get(request.snapshot.snapshot_id)
            if snapshot_record is None or snapshot_record.reference != request.snapshot:
                raise _closed(IngestionOperationCode.INTEGRITY_ERROR)
            if not self._supported(request):
                raise _closed(IngestionOperationCode.OPERATION_FAILED)
            structure = request.structures.structures[0]
            binding = request.structures.bindings[0]
            matched = 0
            unmatched = 0
            parse_failed = False
            items: list[NormalizedItemInputV1] = []
            for entry in snapshot_record.snapshot.entries:
                if not glob_matches_v1(binding.matcher.pattern, entry.logical_path):
                    unmatched += 1
                    continue
                matched += 1
                try:
                    parsed = parse_markdown_v1(
                        entry.content, structure.markdown.frontmatter_mode
                    )
                    fields = map_markdown_fields_v1(
                        parsed, entry.logical_path, structure
                    )
                    plain_fields = _plain_json(fields)
                    if type(plain_fields) is not dict:
                        raise TypeError
                    items.append(
                        NormalizedItemInputV1(
                            entry.logical_path,
                            entry.sha256,
                            entry.size_bytes,
                            structure.ref.to_dict(),
                            plain_fields,
                        )
                    )
                except (MarkdownParseErrorV1, BundleValidationError, TypeError, ValueError):
                    parse_failed = True
            preview = IngestionPreview(
                len(snapshot_record.snapshot.entries),
                matched,
                unmatched,
                0,
                (BindingPreview(binding.binding_id, matched),),
            )
            plan = IngestionPlan(INGESTION_PLAN_SCHEMA_VERSION, request, preview)
            diagnostics: list[IngestionDiagnosticCode] = []
            if unmatched:
                diagnostics.append(IngestionDiagnosticCode.UNMATCHED_SOURCE)
            if parse_failed:
                diagnostics.append(IngestionDiagnosticCode.PARSE_FAILED)
            if request.output_ref not in self._outputs:
                diagnostics.append(IngestionDiagnosticCode.OUTPUT_CONFLICT)
            record = _PlanRecord(plan, tuple(items), tuple(diagnostics))
            existing = self._plans.get(plan.plan_fingerprint)
            if existing is not None and existing != record:
                raise _closed(IngestionOperationCode.INTEGRITY_ERROR)
            self._plans[plan.plan_fingerprint] = record
            return plan

    def preflight(self, plan: IngestionPlan) -> IngestionPreflight:
        with self._lock:
            record = self._plans.get(plan.plan_fingerprint)
            if record is None or record.plan != plan:
                raise _closed(IngestionOperationCode.INTEGRITY_ERROR)
            try:
                checked = _parse_time(self._clock.now())
            except BaseException:
                raise _closed(IngestionOperationCode.OPERATION_FAILED) from None
            cached = self._preflights.get(plan.plan_fingerprint)
            if cached is not None and (
                _parse_time(cached.checked_at)
                <= checked
                < _parse_time(cached.expires_at)
            ):
                return cached
            diagnostics = list(record.diagnostics)
            if plan.preview.ambiguous_sources:
                diagnostics.append(IngestionDiagnosticCode.AMBIGUOUS_SOURCE)
            ready = plan.preview.ready and not diagnostics and bool(record.items)
            if not ready and not diagnostics:
                diagnostics.append(IngestionDiagnosticCode.PARSE_FAILED)
            preflight = IngestionPreflight(
                plan.plan_fingerprint,
                ready,
                _format_time(checked),
                _format_time(checked + _PREFLIGHT_TTL),
                (),
                tuple(diagnostics),
            )
            self._preflights[plan.plan_fingerprint] = preflight
            return preflight

    def start(
        self, plan: IngestionPlan, preflight: IngestionPreflight
    ) -> IngestionStart:
        with self._lock:
            plan_record = self._plans.get(plan.plan_fingerprint)
            retained_preflight = self._preflights.get(plan.plan_fingerprint)
            if (
                plan_record is None
                or plan_record.plan != plan
                or retained_preflight != preflight
                or not preflight.ready
            ):
                raise _closed(IngestionOperationCode.START_INELIGIBLE)
            start_key = (plan.plan_fingerprint, preflight.preflight_fingerprint)
            previous_run_id = self._runs_by_start.get(start_key)
            if previous_run_id is not None:
                return IngestionStart(self._runs[previous_run_id].run, True)
            run_id = "run-" + _digest(
                b"synaptic-process-local-ingestion-run/v1",
                {
                    "plan_fingerprint": plan.plan_fingerprint,
                    "preflight_fingerprint": preflight.preflight_fingerprint,
                },
            )
            authority = run_authority_digest(
                run_id,
                plan.request.project_ref,
                plan.plan_fingerprint,
                preflight.preflight_fingerprint,
            )
            run = IngestionRunRef(
                run_id,
                plan.request.project_ref,
                plan.plan_fingerprint,
                preflight.preflight_fingerprint,
                authority,
            )
            bundles_root = self._outputs[plan.request.output_ref]
            semantic_identity: BundleSemanticIdentityV1 | None = None
            uncertainty_phase: BundlePublicationUncertaintyPhaseV1 | None = None
            try:
                verified = write_normalized_bundle_v1(
                    bundles_root,
                    plan.request.structures.to_dict(),
                    plan_record.items,
                )
                semantic_identity = verified.semantic_identity
                bundle = NormalizedBundleRef(
                    semantic_identity.bundle_id,
                    semantic_identity.bundle_digest,
                    semantic_identity.item_count,
                    plan.plan_fingerprint,
                    run.authority_digest,
                )
                outcome = IngestionOutcome(
                    run,
                    IngestionRunState.SUCCEEDED,
                    plan.preview.source_count,
                    plan.preview.source_count,
                    semantic_identity.item_count,
                    bundle,
                )
            except BundlePublicationUncertainV1 as uncertain:
                semantic_identity = uncertain.semantic_identity
                uncertainty_phase = uncertain.phase
                outcome = IngestionOutcome(
                    run,
                    IngestionRunState.RECONCILE_REQUIRED,
                    plan.preview.source_count,
                    plan.preview.source_count,
                    0,
                    diagnostic_code=IngestionDiagnosticCode.EFFECT_UNCERTAIN,
                )
            except (BundleValidationError, BundleCollisionError):
                outcome = IngestionOutcome(
                    run,
                    IngestionRunState.FAILED,
                    plan.preview.source_count,
                    0,
                    0,
                    diagnostic_code=IngestionDiagnosticCode.BUNDLE_INVALID,
                )
            except BaseException:
                outcome = IngestionOutcome(
                    run,
                    IngestionRunState.FAILED,
                    plan.preview.source_count,
                    0,
                    0,
                    diagnostic_code=IngestionDiagnosticCode.EXECUTION_FAILED,
                )
            self._runs[run.run_id] = _RunRecord(
                run,
                outcome,
                bundles_root,
                semantic_identity,
                uncertainty_phase,
            )
            self._runs_by_start[start_key] = run.run_id
            return IngestionStart(run, True)

    @staticmethod
    def _verified_bundle(record: _RunRecord) -> VerifiedNormalizedBundleV1:
        bundle = record.outcome.bundle
        identity = record.semantic_identity
        if bundle is None or identity is None:
            raise BundleValidationError("terminal outcome has no bundle")
        verified = verify_normalized_bundle_v1(record.bundles_root / bundle.bundle_id)
        if (
            verified.semantic_identity != identity
            or identity.bundle_id != bundle.bundle_id
            or identity.bundle_digest != bundle.bundle_digest
            or identity.item_count != bundle.document_count
        ):
            raise BundleValidationError("bundle authority mismatch")
        return verified

    def _record(self, run: IngestionRunRef) -> _RunRecord:
        record = self._runs.get(run.run_id)
        if record is None or record.run != run:
            raise _closed(IngestionOperationCode.RUN_MISSING)
        return record

    def show(self, run: IngestionRunRef) -> IngestionOutcome:
        with self._lock:
            return self._record(run).outcome

    def result(self, outcome: IngestionOutcome) -> IngestionResult:
        with self._lock:
            record = self._record(outcome.run)
            if record.outcome != outcome:
                raise _closed(IngestionOperationCode.INTEGRITY_ERROR)
            return IngestionResult(
                INGESTION_RESULT_SCHEMA_VERSION,
                record.outcome,
                record.outcome.outcome_digest,
            )

    def cancel(self, outcome: IngestionOutcome) -> IngestionOutcome:
        raise _closed(IngestionOperationCode.CANCEL_INELIGIBLE)

    def resume(self, outcome: IngestionOutcome) -> IngestionOutcome:
        raise _closed(IngestionOperationCode.RESUME_INELIGIBLE)

    def reconcile(self, outcome: IngestionOutcome) -> IngestionOutcome:
        with self._lock:
            record = self._record(outcome.run)
            identity = record.semantic_identity
            if (
                record.outcome != outcome
                or outcome.state is not IngestionRunState.RECONCILE_REQUIRED
                or identity is None
                or record.uncertainty_phase is None
            ):
                raise _closed(IngestionOperationCode.RECONCILE_INELIGIBLE)
            failed = False
            try:
                retry_bundle_root_durability_v1(record.bundles_root)
                verified = verify_normalized_bundle_v1(
                    record.bundles_root / identity.bundle_id
                )
                if verified.semantic_identity != identity:
                    failed = True
            except (BundleDurabilityError, BundleValidationError, OSError):
                failed = True
            except BaseException:
                failed = True
            if failed:
                reconciled = IngestionOutcome(
                    record.run,
                    IngestionRunState.FAILED,
                    outcome.source_count,
                    outcome.sources_processed,
                    0,
                    diagnostic_code=IngestionDiagnosticCode.BUNDLE_INVALID,
                )
            else:
                bundle = NormalizedBundleRef(
                    identity.bundle_id,
                    identity.bundle_digest,
                    identity.item_count,
                    record.run.plan_fingerprint,
                    record.run.authority_digest,
                )
                reconciled = IngestionOutcome(
                    record.run,
                    IngestionRunState.SUCCEEDED,
                    outcome.source_count,
                    outcome.sources_processed,
                    identity.item_count,
                    bundle,
                )
            self._runs[record.run.run_id] = _RunRecord(
                record.run,
                reconciled,
                record.bundles_root,
                identity,
            )
            return reconciled

    def verify(self, result: IngestionResult) -> IngestionVerification:
        with self._lock:
            record = self._record(result.outcome.run)
            if record.outcome != result.outcome or result.outcome.bundle is None:
                raise _closed(IngestionOperationCode.INTEGRITY_ERROR)
            verified = True
            diagnostics: tuple[IngestionDiagnosticCode, ...] = ()
            try:
                self._verified_bundle(record)
            except BaseException:
                verified = False
                diagnostics = (IngestionDiagnosticCode.BUNDLE_INVALID,)
            try:
                checked_at = _format_time(_parse_time(self._clock.now()))
            except BaseException:
                raise _closed(IngestionOperationCode.OPERATION_FAILED) from None
            return IngestionVerification(
                record.run,
                result.outcome_digest,
                result.outcome.bundle,
                verified,
                checked_at,
                diagnostics,
            )

    def list(self, request: IngestionListRequest) -> IngestionPage:
        raise _closed(IngestionOperationCode.OPERATION_FAILED)

    def observations(
        self, request: IngestionObservationsRequest
    ) -> IngestionObservationPage:
        raise _closed(IngestionOperationCode.OPERATION_FAILED)


__all__ = ["ProcessLocalIngestionOperationsV1"]
