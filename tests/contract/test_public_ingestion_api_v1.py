"""Hostile contract tests for the blob-free Markdown IngestionAPI v1."""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1 import ingestion_facade as api
from synaptic_tuner.api.v1._contract import canonical_bytes
from synaptic_tuner.api.v1.ingestion_facade import (
    AuthorizedSourceRef, BindingPreview, FieldMapping, FieldSelector, FieldSelectorKind,
    FieldValueKind, FrontmatterMode, IngestionAPI, IngestionDiagnosticCode,
    IngestionListRequest, IngestionObservation, IngestionObservationKind,
    IngestionObservationPage, IngestionObservationsRequest, IngestionOperationCode,
    IngestionOperationError, IngestionOperations, IngestionOutcome, IngestionPage,
    IngestionPlan, IngestionPreflight, IngestionPreview, IngestionRequest,
    IngestionResult, IngestionRunRef, IngestionRunState, IngestionStart,
    IngestionVerification, MarkdownProfileV1, MetadataDeclaration, MetadataPolicyRef,
    NormalizedBundleRef, ProposalEvidenceCode, RelationshipDeclaration,
    SchemaRef, SourceAdmissionKind, SourceAdmissionRequest, SourceMatcher,
    SourceSnapshotRef, StructureBinding, StructureDefinition, StructureProposal,
    StructureProposalRequest, StructureRef, StructureSet, TextProjection,
    run_authority_digest,
)
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement


ROOT = Path(__file__).resolve().parents[2]
AT = "2026-09-19T12:00:00Z"
EXPIRES = "2026-09-19T12:05:00Z"


def _admission_request() -> SourceAdmissionRequest:
    source = AuthorizedSourceRef("acme", SourceAdmissionKind.LOCAL_SELECTION, "selection_01", "a" * 64)
    return SourceAdmissionRequest("admit_01", "acme", source)


def _snapshot(count: int = 2) -> SourceSnapshotRef:
    request = _admission_request()
    return SourceSnapshotRef("acme", "snapshot_01", request.source.kind, "b" * 64, count, request.admission_fingerprint)


def _definition(*, name: str = "MarkdownNote", title_kind: FieldValueKind = FieldValueKind.STRING,
                relationships: tuple[RelationshipDeclaration, ...] = ()) -> StructureDefinition:
    fields = (
        FieldMapping("body", FieldSelector(FieldSelectorKind.DOCUMENT_BODY), FieldValueKind.STRING, True),
        FieldMapping("title", FieldSelector(FieldSelectorKind.FRONTMATTER_FIELD, "title"), title_kind, False),
        FieldMapping("logical_path", FieldSelector(FieldSelectorKind.LOGICAL_PATH), FieldValueKind.STRING, True),
    )
    return StructureDefinition.define(
        name=name, version="1", markdown=MarkdownProfileV1(FrontmatterMode.OPTIONAL), fields=fields,
        text_projections=(TextProjection("text", "body"),),
        metadata=(MetadataDeclaration("title", "title"),), relationships=relationships,
        metadata_policy=MetadataPolicyRef("structure_policy", "c" * 64),
    )


def _structures() -> StructureSet:
    definition = _definition()
    binding = StructureBinding("markdown", SourceMatcher("**/*.md"), definition.ref, MetadataPolicyRef("binding_policy", "d" * 64))
    return StructureSet((definition,), (binding,))


def _request(**changes: object) -> IngestionRequest:
    values: dict[str, object] = {
        "request_id": "ingest_request_01", "project_ref": "acme", "snapshot": _snapshot(),
        "structures": _structures(), "output_ref": "bundle_output_01",
        "metadata_policy": MetadataPolicyRef("request_policy", "e" * 64), "tags": ("private",),
    }
    values.update(changes)
    return IngestionRequest(**values)  # type: ignore[arg-type]


def _plan(request: IngestionRequest | None = None, *, matched: int | None = None,
          unmatched: int = 0, ambiguous: int = 0) -> IngestionPlan:
    request = request or _request()
    matched = request.snapshot.source_count - unmatched - ambiguous if matched is None else matched
    preview = IngestionPreview(request.snapshot.source_count, matched, unmatched, ambiguous,
                               (BindingPreview("markdown", matched),))
    return IngestionPlan("synaptic-ingestion-plan/v1", request, preview)


def _preflight(plan: IngestionPlan, *, ready: bool = True,
               diagnostics: tuple[IngestionDiagnosticCode, ...] = ()) -> IngestionPreflight:
    return IngestionPreflight(
        plan.plan_fingerprint, ready, AT, EXPIRES,
        (AuthorizationRequirement("ingestion.start", False),), diagnostics,
    )


def _run(plan: IngestionPlan | None = None, preflight: IngestionPreflight | None = None) -> IngestionRunRef:
    plan = plan or _plan()
    preflight = preflight or _preflight(plan)
    authority = run_authority_digest("run_01", "acme", plan.plan_fingerprint, preflight.preflight_fingerprint)
    return IngestionRunRef("run_01", "acme", plan.plan_fingerprint, preflight.preflight_fingerprint, authority)


def _bundle(run: IngestionRunRef, count: int = 2) -> NormalizedBundleRef:
    return NormalizedBundleRef(
        "bundle_01", "f" * 64, count, run.plan_fingerprint, run.authority_digest,
    )


def _timestamp_with_size(year: int, size: int) -> str:
    prefix = f"{year:04d}-09-19T12:00:00."
    return prefix + ("0" * (size - len(prefix) - 1)) + "Z"


def _outcome(state: IngestionRunState, *, run: IngestionRunRef | None = None,
             processed: int = 0, written: int = 0,
             diagnostic: IngestionDiagnosticCode | None = None) -> IngestionOutcome:
    run = run or _run()
    bundle = _bundle(run, written) if state is IngestionRunState.SUCCEEDED else None
    return IngestionOutcome(run, state, 2, processed, written, bundle, diagnostic)


class _Clock:
    def __init__(self, value: str = "2026-09-19T12:01:00Z") -> None: self.value = value
    def now(self) -> str: return self.value


class _Operations:
    def __init__(self) -> None:
        self.plan_value = _plan()
        self.preflight_value = _preflight(self.plan_value)
        self.run = _run(self.plan_value, self.preflight_value)
        self.outcome = _outcome(IngestionRunState.RUNNING, run=self.run)

    def admit(self, request):
        return SourceSnapshotRef(request.project_ref, "snapshot_01", request.source.kind, "b" * 64, 2, request.admission_fingerprint)
    def propose(self, request):
        return (StructureProposal("proposal_01", request.snapshot, _structures(), (ProposalEvidenceCode.MARKDOWN_EXTENSION,)),)
    def plan(self, request): return self.plan_value
    def preflight(self, plan): return self.preflight_value
    def start(self, plan, preflight): return IngestionStart(self.run, True)
    def show(self, run): return self.outcome
    def result(self, outcome): return IngestionResult("synaptic-ingestion-result/v1", outcome, outcome.outcome_digest)
    def cancel(self, outcome): return _outcome(IngestionRunState.CANCEL_REQUESTED, run=outcome.run, processed=outcome.sources_processed, written=outcome.documents_written)
    def resume(self, outcome): return _outcome(IngestionRunState.RUNNING, run=outcome.run, processed=outcome.sources_processed, written=outcome.documents_written)
    def reconcile(self, outcome): return _outcome(IngestionRunState.RUNNING, run=outcome.run, processed=outcome.sources_processed, written=outcome.documents_written)
    def verify(self, result): return IngestionVerification(result.outcome.run, result.outcome_digest, result.outcome.bundle, True, AT)  # type: ignore[arg-type]
    def list(self, request): return IngestionPage(request, (self.outcome,))
    def observations(self, request):
        return IngestionObservationPage(request, (IngestionObservation(request.run, 1, AT, IngestionObservationKind.SOURCE_PROCESSED, 1),))


def test_exports_and_exact_verb_surface() -> None:
    for name in api.__all__:
        assert getattr(v1, name) is getattr(api, name), name
    assert set(api.__all__) <= set(v1.__all__)
    verbs = {"admit", "propose", "plan", "preflight", "start", "show", "result", "cancel", "resume", "reconcile", "verify", "list", "observations"}
    for owner in (IngestionAPI, IngestionOperations):
        actual = {name for name, member in vars(owner).items() if not name.startswith("_") and inspect.isfunction(member)}
        assert actual == verbs
    for name in verbs:
        assert tuple(inspect.signature(getattr(IngestionAPI, name)).parameters) == tuple(inspect.signature(getattr(IngestionOperations, name)).parameters)
    assert {item.value for item in IngestionOperationCode} == {
        "input_mutated", "operation_failed", "result_invalid", "result_unbound",
        "admission_ineligible", "start_ineligible", "cancel_ineligible",
        "resume_ineligible", "reconcile_ineligible", "result_unavailable",
        "verify_ineligible", "run_missing", "cursor_invalid", "state_conflict",
        "integrity_error",
    }


def test_public_records_are_frozen_slotted_and_exact_field_parsers_reject_unknowns() -> None:
    record_types = (
        AuthorizedSourceRef, SourceAdmissionRequest, SourceSnapshotRef,
        MetadataPolicyRef, SchemaRef, StructureRef, MarkdownProfileV1, SourceMatcher, FieldSelector,
        FieldMapping, TextProjection, MetadataDeclaration, RelationshipDeclaration,
        StructureDefinition, StructureBinding, StructureSet, StructureProposalRequest,
        StructureProposal, IngestionRequest, BindingPreview, IngestionPreview, IngestionPlan,
        IngestionPreflight, IngestionRunRef, IngestionStart, NormalizedBundleRef,
        IngestionOutcome, IngestionResult, IngestionVerification, IngestionListRequest,
        IngestionPage, IngestionObservation, IngestionObservationsRequest,
        IngestionObservationPage,
    )
    for record in record_types:
        assert dataclasses.is_dataclass(record)
        assert "__slots__" in vars(record)
        assert record.__dataclass_params__.frozen
    raw = _request().to_dict()
    raw["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        IngestionRequest.from_dict(raw)


def test_admission_is_blob_free_and_exactly_bound() -> None:
    request = _admission_request()
    admitted = IngestionAPI(_Operations(), clock=_Clock()).admit(request)
    assert admitted.admission_fingerprint == request.admission_fingerprint
    forbidden = {"bytes", "content", "path", "metadata", "value"}
    for contract in (AuthorizedSourceRef, SourceAdmissionRequest, SourceSnapshotRef):
        assert forbidden.isdisjoint({field.name for field in dataclasses.fields(contract)})
    assert "SourceAdmission" not in api.__all__ and "SourceAdmission" not in v1.__all__

    class WrongKind(_Operations):
        def admit(self, presented):
            snapshot = SourceSnapshotRef(
                presented.project_ref, "snapshot_01", SourceAdmissionKind.FINALIZED_UPLOAD,
                "b" * 64, 2, presented.admission_fingerprint,
            )
            return snapshot

    with pytest.raises(IngestionOperationError) as unbound:
        IngestionAPI(WrongKind(), clock=_Clock()).admit(request)
    assert unbound.value.code is IngestionOperationCode.RESULT_UNBOUND


@pytest.mark.parametrize("value", ["/private", "C:/private", "..", "a/b", "a\\b", "x." + "a" * 128])
def test_opaque_refs_reject_paths_and_out_of_grammar_values(value: str) -> None:
    with pytest.raises(ValueError):
        AuthorizedSourceRef("acme", SourceAdmissionKind.LOCAL_SELECTION, value, "a" * 64)


@pytest.mark.parametrize("pattern", ["/x/*.md", "x\\*.md", "x:/*.md", "../*.md", "x//*.md", "x/**foo.md", "x/[a].md", "x/{a}.md", "x/(a).md"])
def test_glob_contract_rejects_unsafe_or_unsupported_syntax(pattern: str) -> None:
    with pytest.raises(ValueError):
        SourceMatcher(pattern)
    with pytest.raises(ValueError):
        SourceMatcher("a/" * 64 + "x.md")
    with pytest.raises(ValueError):
        SourceMatcher("a" * 513)


def test_v1_declarations_are_markdown_file_only_and_bounded() -> None:
    structure = _definition()
    assert structure.parsing_profile.value == "markdown_yaml_frontmatter_v1"
    assert structure.unit.value == "file"
    assert StructureDefinition.from_dict(structure.to_dict()) == structure
    with pytest.raises(ValueError):
        _request(tags=tuple(f"tag{i}" for i in range(33)))
    with pytest.raises(ValueError):
        StructureSet(tuple(_definition(name=f"Note{i}") for i in range(33)), (_structures().bindings[0],))
    with pytest.raises(ValueError):
        FieldSelector(FieldSelectorKind.FRONTMATTER_FIELD, "a" * 65)
    assert FieldSelector(FieldSelectorKind.FRONTMATTER_FIELD, "3rd party").key == "3rd party"
    with pytest.raises(ValueError):
        _request(tags=("a" * 65,))
    with pytest.raises(ValueError):
        IngestionPreflight(
            "a" * 64, True, AT, EXPIRES,
            tuple(AuthorizationRequirement(f"op{i}", False) for i in range(17)),
        )


def test_preflight_has_an_inclusive_canonical_byte_bound() -> None:
    assert api.MAX_INGESTION_PREFLIGHT_BYTES == 65_536
    empty_operation_document = {
        "plan_fingerprint": "a" * 64,
        "ready": True,
        "checked_at": AT,
        "expires_at": EXPIRES,
        "authorization": [{
            "operation": "",
            "paid_effect": False,
            "maximum_cost_minor_units": None,
            "currency": None,
        }],
        "diagnostic_codes": [],
    }
    operation_length = (
        api.MAX_INGESTION_PREFLIGHT_BYTES
        - len(canonical_bytes(empty_operation_document))
    )
    exact = IngestionPreflight(
        "a" * 64, True, AT, EXPIRES,
        (AuthorizationRequirement("x" * operation_length, False),),
    )
    assert len(canonical_bytes(exact.to_dict())) == api.MAX_INGESTION_PREFLIGHT_BYTES
    assert len(exact.preflight_fingerprint) == 64
    with pytest.raises(ValueError, match="exceeds 65536 canonical bytes"):
        IngestionPreflight(
            "a" * 64, True, AT, EXPIRES,
            (AuthorizationRequirement("x" * (operation_length + 1), False),),
        )
    with pytest.raises(ValueError, match="exceeds 65536 canonical bytes"):
        IngestionPreflight(
            "a" * 64, True, AT, EXPIRES,
            (AuthorizationRequirement("x" * 1_000_000, False),),
        )
    with pytest.raises(ValueError, match="document_body"):
        StructureDefinition.define(
            name="NoBody", version="1", markdown=MarkdownProfileV1(FrontmatterMode.OPTIONAL),
            fields=(FieldMapping("path", FieldSelector(FieldSelectorKind.LOGICAL_PATH), FieldValueKind.STRING, True),),
            text_projections=(TextProjection("text", "path"),),
        )


def test_all_ingestion_timestamps_share_an_inclusive_utf8_byte_bound() -> None:
    assert api.MAX_INGESTION_TIMESTAMP_BYTES == 64
    checked_at = _timestamp_with_size(2026, api.MAX_INGESTION_TIMESTAMP_BYTES)
    expires_at = _timestamp_with_size(2027, api.MAX_INGESTION_TIMESTAMP_BYTES)
    assert len(checked_at.encode("utf-8")) == api.MAX_INGESTION_TIMESTAMP_BYTES
    assert len(expires_at.encode("utf-8")) == api.MAX_INGESTION_TIMESTAMP_BYTES

    preflight = IngestionPreflight("a" * 64, True, checked_at, expires_at)
    assert preflight.checked_at == checked_at
    assert preflight.expires_at == expires_at

    run = _run()
    verification = IngestionVerification(
        run, "a" * 64, _bundle(run), True, checked_at,
    )
    assert verification.checked_at == checked_at
    observation = IngestionObservation(
        run, 1, checked_at, IngestionObservationKind.SOURCE_PROCESSED, 1,
    )
    assert observation.occurred_at == checked_at


def test_every_ingestion_timestamp_field_rejects_oversized_values() -> None:
    oversized_checked = _timestamp_with_size(
        2026, api.MAX_INGESTION_TIMESTAMP_BYTES + 1,
    )
    oversized_expires = _timestamp_with_size(
        2027, api.MAX_INGESTION_TIMESTAMP_BYTES + 1,
    )
    run = _run()

    cases = (
        lambda: IngestionPreflight("a" * 64, True, oversized_checked, EXPIRES),
        lambda: IngestionPreflight("a" * 64, True, AT, oversized_expires),
        lambda: IngestionVerification(
            run, "a" * 64, _bundle(run), True, oversized_checked,
        ),
        lambda: IngestionObservation(
            run, 1, oversized_checked, IngestionObservationKind.SOURCE_PROCESSED, 1,
        ),
    )
    for construct in cases:
        with pytest.raises(ValueError, match="exceeds 64 UTF-8 bytes"):
            construct()

    hostile = "2026-09-19T12:00:00." + ("0" * 1_000_000) + "Z"
    with pytest.raises(ValueError, match="exceeds 64 UTF-8 bytes"):
        IngestionObservation(
            run, 1, hostile, IngestionObservationKind.SOURCE_PROCESSED, 1,
        )


def test_parser_semantics_and_limits_are_public_and_closed() -> None:
    assert api.MARKDOWN_ENCODING == "utf-8"
    assert api.MARKDOWN_FRONTMATTER_OPENER == "---\n"
    assert api.METADATA_PRECEDENCE == ("frontmatter", "structure_policy", "binding_policy", "request_policy")
    assert api.YAML_PRESENTATION_SUBSET == "yaml_1_2_presentation_subset"
    assert api.YAML_BOOLEAN_LITERALS == ("false", "true")
    assert api.YAML_INTEGER_PATTERN == r"-?(0|[1-9][0-9]*)"
    assert api.YAML_FLOAT_PATTERN == r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
    assert api.YAML_NUMBER_CLASSIFICATION == (
        "classify_integer_first", "float_requires_dot_or_exponent", "float_must_be_finite",
    )
    assert api.YAML_UNQUOTED_NULL_LITERALS_REJECTED == ("null", "Null", "NULL", "~")
    assert api.YAML_KEY_RULE == "all_nested_keys_nonempty_nfc_string_1_to_64_utf8_bytes"
    assert api.YAML_DUPLICATE_KEY_RULE == "reject_duplicate_keys_after_nfc_normalization"
    assert "typed_timestamp" in api.YAML_REJECTED_FEATURES
    assert api.MAX_FRONTMATTER_BYTES == 65_536 and api.MAX_YAML_DEPTH == 8
    assert api.MARKDOWN_BODY_POLICY == "normalized_verbatim_nonempty"
    assert api.GLOB_MATCH_MODEL == "whole_path_case_sensitive_unicode_codepoints_dotfiles_ordinary"
    assert api.GLOB_DOUBLE_STAR == "complete_segment_matching_zero_or_more_path_segments"
    assert ("**/*.md", "note.md", True) in api.GLOB_EXAMPLES
    assert api.GLOB_DP_ALGORITHM[-1] == "match iff dp[len(pattern_segments)][len(path_segments)]"


def test_relationship_target_field_and_type_are_validated() -> None:
    target = _definition(name="Target", title_kind=FieldValueKind.INT64)
    good = _definition(name="Source", title_kind=FieldValueKind.INT64, relationships=(
        RelationshipDeclaration("parent", target.ref, "title", "title"),
    ))
    StructureSet((good, target), (
        StructureBinding("source", SourceMatcher("source/*.md"), good.ref),
        StructureBinding("target", SourceMatcher("target/*.md"), target.ref),
    ))
    missing = _definition(name="Missing", relationships=(RelationshipDeclaration("bad", target.ref, "title", "absent"),))
    with pytest.raises(ValueError, match="target field"):
        StructureSet((missing, target), (StructureBinding("missing", SourceMatcher("missing/*.md"), missing.ref),))
    mismatch = _definition(name="Mismatch", relationships=(RelationshipDeclaration("bad", target.ref, "title", "title"),))
    with pytest.raises(ValueError, match="same value kind"):
        StructureSet((mismatch, target), (StructureBinding("mismatch", SourceMatcher("mismatch/*.md"), mismatch.ref),))
    placeholder = StructureRef("SelfLinked", "1", "0" * 64)
    self_linked = _definition(name="SelfLinked", relationships=(RelationshipDeclaration("self_link", placeholder, "title", "title"),))
    assert self_linked.relationships[0].target_structure == self_linked.ref
    StructureSet((self_linked,), (StructureBinding("self_linked", SourceMatcher("self/*.md"), self_linked.ref),))


def test_proposals_are_advisory_only() -> None:
    proposal = IngestionAPI(_Operations(), clock=_Clock()).propose(StructureProposalRequest(_snapshot()))[0]
    assert proposal.suggested_structures == _structures()
    assert "proposal" not in {field.name for field in dataclasses.fields(IngestionRequest)}
    before = _plan().plan_fingerprint
    assert _plan().plan_fingerprint == before


def test_preview_requires_exactly_one_binding_per_source() -> None:
    assert _plan().preview.ready
    assert not _plan(unmatched=1).preview.ready
    assert not _plan(ambiguous=1).preview.ready
    with pytest.raises(ValueError, match="sum"):
        IngestionPreview(2, 2, 1, 0, (BindingPreview("markdown", 2),))


def test_authority_chain_rejects_every_mismatch() -> None:
    plan = _plan()
    preflight = _preflight(plan)
    authority = run_authority_digest("run_01", "acme", plan.plan_fingerprint, preflight.preflight_fingerprint)
    with pytest.raises(ValueError, match="authority"):
        IngestionRunRef("run_01", "acme", plan.plan_fingerprint, preflight.preflight_fingerprint, "0" * 64)
    run = IngestionRunRef("run_01", "acme", plan.plan_fingerprint, preflight.preflight_fingerprint, authority)
    assert set(_bundle(run).to_dict()) == {
        "bundle_id", "bundle_digest", "document_count", "plan_fingerprint", "run_authority_digest",
    }
    mismatched = NormalizedBundleRef("bundle_01", "f" * 64, 1, "0" * 64, run.authority_digest)
    with pytest.raises(ValueError, match="bind"):
        IngestionOutcome(run, IngestionRunState.SUCCEEDED, 2, 2, 1, mismatched)
    bundle = _bundle(run, 2)
    with pytest.raises(ValueError, match="bind"):
        IngestionOutcome(run, IngestionRunState.SUCCEEDED, 2, 2, 1, bundle)


@pytest.mark.parametrize(
    ("state", "processed", "written", "diagnostic", "valid"),
    [
        (IngestionRunState.PLANNED, 0, 0, None, True),
        (IngestionRunState.PLANNED, 1, 0, None, False),
        (IngestionRunState.RUNNING, 1, 1, None, True),
        (IngestionRunState.CANCEL_REQUESTED, 1, 0, None, True),
        (IngestionRunState.RECONCILE_REQUIRED, 1, 1, IngestionDiagnosticCode.EFFECT_UNCERTAIN, True),
        (IngestionRunState.RECONCILE_REQUIRED, 1, 1, IngestionDiagnosticCode.EXECUTION_FAILED, False),
        (IngestionRunState.FAILED, 1, 0, IngestionDiagnosticCode.INTERRUPTED, True),
        (IngestionRunState.FAILED, 1, 0, None, False),
        (IngestionRunState.CANCELLED, 1, 0, None, True),
    ],
)
def test_outcome_state_matrix(state, processed, written, diagnostic, valid) -> None:
    if valid:
        _outcome(state, processed=processed, written=written, diagnostic=diagnostic)
    else:
        with pytest.raises(ValueError, match="state matrix"):
            _outcome(state, processed=processed, written=written, diagnostic=diagnostic)
    succeeded = _outcome(IngestionRunState.SUCCEEDED, processed=2, written=2)
    assert succeeded.bundle is not None
    with pytest.raises(ValueError, match="state matrix"):
        IngestionOutcome(_run(), IngestionRunState.SUCCEEDED, 2, 1, 1, _bundle(_run(), 1))
    with pytest.raises(ValueError, match="state matrix"):
        _outcome(IngestionRunState.FAILED, diagnostic=IngestionDiagnosticCode.EFFECT_UNCERTAIN)


def test_start_requires_exact_ready_unexpired_authority() -> None:
    operations = _Operations()
    api_client = IngestionAPI(operations, clock=_Clock())
    assert api_client.start(operations.plan_value, operations.preflight_value).run == operations.run
    with pytest.raises(IngestionOperationError) as stale:
        IngestionAPI(operations, clock=_Clock("2026-09-19T12:06:00Z")).start(operations.plan_value, operations.preflight_value)
    assert stale.value.code is IngestionOperationCode.START_INELIGIBLE
    blocked = _plan(unmatched=1)
    not_ready = _preflight(blocked, ready=False, diagnostics=(IngestionDiagnosticCode.UNMATCHED_SOURCE,))
    with pytest.raises(IngestionOperationError) as rejected:
        api_client.start(blocked, not_ready)
    assert rejected.value.code is IngestionOperationCode.START_INELIGIBLE
    operations.plan_value = blocked
    operations.preflight_value = _preflight(
        blocked, ready=False, diagnostics=(IngestionDiagnosticCode.STRUCTURE_INVALID,),
    )
    with pytest.raises(IngestionOperationError) as unbound:
        api_client.preflight(blocked)
    assert unbound.value.code is IngestionOperationCode.RESULT_UNBOUND


def test_preflight_temporal_boundaries_distinguish_provider_results_from_caller_inputs() -> None:
    operations = _Operations()
    client = IngestionAPI(operations, clock=_Clock("2026-09-19T12:01:00Z"))
    exact_now = IngestionPreflight(
        operations.plan_value.plan_fingerprint, True, "2026-09-19T12:01:00Z", EXPIRES,
        (AuthorizationRequirement("ingestion.start", False),),
    )
    operations.preflight_value = exact_now
    assert client.preflight(operations.plan_value).checked_at == exact_now.checked_at

    future = IngestionPreflight(
        operations.plan_value.plan_fingerprint, True, "2026-09-19T12:02:00Z", EXPIRES,
        (AuthorizationRequirement("ingestion.start", False),),
    )
    operations.preflight_value = future
    with pytest.raises(IngestionOperationError) as provider_future:
        client.preflight(operations.plan_value)
    assert provider_future.value.code is IngestionOperationCode.RESULT_INVALID
    assert provider_future.value.__cause__ is None and provider_future.value.__context__ is None

    with pytest.raises(IngestionOperationError) as caller_future:
        client.start(operations.plan_value, future)
    assert caller_future.value.code is IngestionOperationCode.START_INELIGIBLE
    with pytest.raises(IngestionOperationError) as caller_expired:
        IngestionAPI(operations, clock=_Clock(EXPIRES)).start(operations.plan_value, exact_now)
    assert caller_expired.value.code is IngestionOperationCode.START_INELIGIBLE

    operations.preflight_value = exact_now
    with pytest.raises(IngestionOperationError) as provider_expired:
        IngestionAPI(operations, clock=_Clock(EXPIRES)).preflight(operations.plan_value)
    assert provider_expired.value.code is IngestionOperationCode.RESULT_INVALID


@pytest.mark.parametrize("mode", ["raise", "malformed"])
def test_hostile_clock_is_sanitized_for_preflight_and_start(mode: str) -> None:
    operations = _Operations()

    class HostileClock:
        def now(self):
            if mode == "raise":
                raise RuntimeError("private clock detail C:/secret")
            return "not-a-timestamp private detail"

    client = IngestionAPI(operations, clock=HostileClock())
    for invoke in (
        lambda: client.preflight(operations.plan_value),
        lambda: client.start(operations.plan_value, operations.preflight_value),
    ):
        with pytest.raises(IngestionOperationError) as failed:
            invoke()
        assert failed.value.code is IngestionOperationCode.OPERATION_FAILED
        assert str(failed.value) == "operation_failed"
        assert failed.value.__cause__ is None and failed.value.__context__ is None


@pytest.mark.parametrize("verb", (
    "admit", "propose", "plan", "preflight", "start", "show", "result",
    "cancel", "resume", "reconcile", "verify", "list", "observations",
))
def test_hostile_operation_lookup_is_sanitized_for_every_public_verb(verb: str) -> None:
    class HostileLookup:
        def __getattribute__(self, name: str):
            if name == verb:
                raise RuntimeError("private lookup detail C:/secret")
            return object.__getattribute__(self, name)

    client = IngestionAPI(HostileLookup(), clock=_Clock())  # type: ignore[arg-type]
    plan = _plan()
    preflight = _preflight(plan)
    run = _run(plan, preflight)
    succeeded = _outcome(
        IngestionRunState.SUCCEEDED, run=run, processed=2, written=2,
    )
    result = IngestionResult(
        "synaptic-ingestion-result/v1", succeeded, succeeded.outcome_digest,
    )
    calls = {
        "admit": lambda: client.admit(_admission_request()),
        "propose": lambda: client.propose(StructureProposalRequest(_snapshot())),
        "plan": lambda: client.plan(_request()),
        "preflight": lambda: client.preflight(plan),
        "start": lambda: client.start(plan, preflight),
        "show": lambda: client.show(run),
        "result": lambda: client.result(succeeded),
        "cancel": lambda: client.cancel(_outcome(IngestionRunState.PLANNED, run=run)),
        "resume": lambda: client.resume(_outcome(
            IngestionRunState.FAILED, run=run, processed=1,
            diagnostic=IngestionDiagnosticCode.INTERRUPTED,
        )),
        "reconcile": lambda: client.reconcile(_outcome(
            IngestionRunState.RECONCILE_REQUIRED, run=run, processed=1,
            diagnostic=IngestionDiagnosticCode.EFFECT_UNCERTAIN,
        )),
        "verify": lambda: client.verify(result),
        "list": lambda: client.list(IngestionListRequest("acme")),
        "observations": lambda: client.observations(IngestionObservationsRequest(run)),
    }

    with pytest.raises(IngestionOperationError) as failed:
        calls[verb]()
    assert failed.value.code is IngestionOperationCode.OPERATION_FAILED
    assert str(failed.value) == "operation_failed"
    assert "private" not in repr(failed.value).lower()
    assert failed.value.__cause__ is None and failed.value.__context__ is None


def test_operation_lookup_failure_keeps_input_mutation_precedence() -> None:
    request = _request()

    class MutatingLookup:
        def __getattribute__(self, name: str):
            if name == "plan":
                object.__setattr__(request, "output_ref", "changed")
                raise RuntimeError("private lookup detail")
            return object.__getattribute__(self, name)

    with pytest.raises(IngestionOperationError) as failed:
        IngestionAPI(MutatingLookup(), clock=_Clock()).plan(request)  # type: ignore[arg-type]
    assert failed.value.code is IngestionOperationCode.INPUT_MUTATED
    assert failed.value.__cause__ is None and failed.value.__context__ is None


def test_lifecycle_eligibility_and_transitions() -> None:
    operations = _Operations()
    client = IngestionAPI(operations, clock=_Clock())
    planned = _outcome(IngestionRunState.PLANNED, run=operations.run)
    assert client.cancel(planned).state is IngestionRunState.CANCEL_REQUESTED
    interrupted = _outcome(IngestionRunState.FAILED, run=operations.run, processed=1, diagnostic=IngestionDiagnosticCode.INTERRUPTED)
    assert client.resume(interrupted).state is IngestionRunState.RUNNING
    uncertain = _outcome(IngestionRunState.RECONCILE_REQUIRED, run=operations.run, processed=1, diagnostic=IngestionDiagnosticCode.EFFECT_UNCERTAIN)
    assert client.reconcile(uncertain).state is IngestionRunState.RUNNING
    with pytest.raises(IngestionOperationError) as invalid:
        client.resume(planned)
    assert invalid.value.code is IngestionOperationCode.RESUME_INELIGIBLE


def test_result_and_verification_are_terminal_and_exactly_bound() -> None:
    operations = _Operations()
    client = IngestionAPI(operations, clock=_Clock())
    succeeded = _outcome(IngestionRunState.SUCCEEDED, run=operations.run, processed=2, written=2)
    result = client.result(succeeded)
    assert result.outcome_digest == succeeded.outcome_digest
    assert client.verify(result).verified
    with pytest.raises(IngestionOperationError) as unavailable:
        client.result(operations.outcome)
    assert unavailable.value.code is IngestionOperationCode.RESULT_UNAVAILABLE
    failed = _outcome(IngestionRunState.FAILED, run=operations.run, diagnostic=IngestionDiagnosticCode.EXECUTION_FAILED)
    failed_result = IngestionResult("synaptic-ingestion-result/v1", failed, failed.outcome_digest)
    with pytest.raises(IngestionOperationError) as ineligible:
        client.verify(failed_result)
    assert ineligible.value.code is IngestionOperationCode.VERIFY_INELIGIBLE


def test_callback_failures_are_sanitized_and_mutation_wins() -> None:
    request = _request()

    class Fails(_Operations):
        def plan(self, presented):
            raise RuntimeError("private path C:/secret")

    with pytest.raises(IngestionOperationError) as failed:
        IngestionAPI(Fails(), clock=_Clock()).plan(request)
    assert failed.value.code is IngestionOperationCode.OPERATION_FAILED
    assert failed.value.__cause__ is None and failed.value.__context__ is None

    class MutatesThenFails(_Operations):
        def plan(self, presented):
            object.__setattr__(presented, "output_ref", "changed")
            raise RuntimeError("secret")

    with pytest.raises(IngestionOperationError) as mutated:
        IngestionAPI(MutatesThenFails(), clock=_Clock()).plan(request)
    assert mutated.value.code is IngestionOperationCode.INPUT_MUTATED
    assert request.output_ref == "bundle_output_01"

    class MutatesSecondStartInput(_Operations):
        def start(self, plan, preflight):
            object.__setattr__(preflight, "ready", False)
            raise RuntimeError("secret")

    operations = MutatesSecondStartInput()
    with pytest.raises(IngestionOperationError) as second:
        IngestionAPI(operations, clock=_Clock()).start(operations.plan_value, operations.preflight_value)
    assert second.value.code is IngestionOperationCode.INPUT_MUTATED


def test_typed_callback_codes_are_reissued_without_retaining_exceptions() -> None:
    class Collision(_Operations):
        def admit(self, request):
            raise IngestionOperationError(IngestionOperationCode.INTEGRITY_ERROR)

    with pytest.raises(IngestionOperationError) as collision:
        IngestionAPI(Collision(), clock=_Clock()).admit(_admission_request())
    assert collision.value.code is IngestionOperationCode.INTEGRITY_ERROR
    assert collision.value.__cause__ is None and collision.value.__context__ is None


def test_result_rebuild_and_binding_failures_are_sanitized() -> None:
    class BadToDict(_Operations):
        def plan(self, request):
            value = self.plan_value
            object.__setattr__(value, "request", object())
            return value

    with pytest.raises(IngestionOperationError) as invalid:
        IngestionAPI(BadToDict(), clock=_Clock()).plan(_request())
    assert invalid.value.code is IngestionOperationCode.RESULT_INVALID
    assert invalid.value.__cause__ is None and invalid.value.__context__ is None

    class Unbound(_Operations):
        def plan(self, request): return _plan(_request(request_id="other_request"))

    with pytest.raises(IngestionOperationError) as unbound:
        IngestionAPI(Unbound(), clock=_Clock()).plan(_request())
    assert unbound.value.code is IngestionOperationCode.RESULT_UNBOUND


def test_all_callback_owned_collection_transformations_drop_exception_context() -> None:
    class BadProposal(_Operations):
        def propose(self, request):
            value = StructureProposal("proposal_01", request.snapshot, _structures(), ())
            object.__setattr__(value, "snapshot", object())
            return (value,)

    class BadPage(_Operations):
        def list(self, request):
            value = IngestionPage(request, (self.outcome,))
            object.__setattr__(value, "request", object())
            return value

    class BadObservations(_Operations):
        def observations(self, request):
            value = IngestionObservationPage(
                request,
                (IngestionObservation(request.run, 1, AT, IngestionObservationKind.SOURCE_PROCESSED, 1),),
            )
            object.__setattr__(value, "request", object())
            return value

    cases = (
        lambda: IngestionAPI(BadProposal(), clock=_Clock()).propose(StructureProposalRequest(_snapshot())),
        lambda: IngestionAPI(BadPage(), clock=_Clock()).list(IngestionListRequest("acme")),
        lambda: IngestionAPI(BadObservations(), clock=_Clock()).observations(IngestionObservationsRequest(_run())),
    )
    for invoke in cases:
        with pytest.raises(IngestionOperationError) as invalid:
            invoke()
        assert invalid.value.code is IngestionOperationCode.RESULT_INVALID
        assert invalid.value.__cause__ is None and invalid.value.__context__ is None


def test_read_pages_are_bounded_and_authority_bound() -> None:
    operations = _Operations()
    client = IngestionAPI(operations, clock=_Clock())
    assert client.list(IngestionListRequest("acme")).outcomes == (operations.outcome,)
    page = client.observations(IngestionObservationsRequest(operations.run))
    assert page.records[0].run == operations.run
    with pytest.raises(ValueError):
        IngestionListRequest("acme", limit=101)
    with pytest.raises(ValueError):
        IngestionObservationsRequest(operations.run, limit=201)


def test_settled_host_and_storage_contracts_are_not_expanded() -> None:
    from synaptic_tuner.api.v1.host import HostPorts
    from synaptic_tuner.api.v1.ports import StoragePartition

    assert tuple(HostPorts.__dataclass_fields__) == (
        "training", "runs", "artifacts", "evaluation", "chat", "data", "pipelines", "clock",
    )
    assert "ingestion" not in {member.value for member in StoragePartition}


def test_import_closure_remains_runtime_and_provider_free() -> None:
    script = """
import json, sys
import synaptic_tuner.api.v1.ingestion_facade
import synaptic_tuner.api.v1 as v1
_ = v1.IngestionAPI
blocked = [name for name in sys.modules if name == 'tuner' or name.startswith('tuner.') or name.startswith('modal') or name.startswith('SynthChat') or name.startswith('sqlite3')]
print(json.dumps(blocked))
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert completed.stdout.strip() == "[]"
