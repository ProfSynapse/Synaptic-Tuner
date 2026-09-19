from __future__ import annotations

from pathlib import Path

import pytest

from synaptic_tuner.api.v1.ingestion_facade import (
    AuthorizedSourceRef,
    FieldMapping,
    FieldSelector,
    FieldSelectorKind,
    FieldValueKind,
    FrontmatterMode,
    IngestionAPI,
    IngestionDiagnosticCode,
    IngestionOperationCode,
    IngestionOperationError,
    IngestionRequest,
    IngestionRunState,
    MarkdownProfileV1,
    SourceAdmissionKind,
    SourceAdmissionRequest,
    SourceMatcher,
    StructureBinding,
    StructureDefinition,
    StructureSet,
    TextProjection,
)
from tuner.ingestion.local_selection_v1 import (
    LocalDiscoveryPolicyV1,
    LocalSelectionRootV1,
)
from tuner.ingestion.bundle_v1 import (
    BundleDurabilityError,
    BundlePublicationUncertainV1,
    BundlePublicationUncertaintyPhaseV1,
)
from tuner.ingestion import runtime_v1
from tuner.ingestion.runtime_v1 import ProcessLocalIngestionOperationsV1


class _Clock:
    value = "2026-09-19T12:00:00Z"

    def now(self) -> str:
        return self.value


def _structure(*, frontmatter: FrontmatterMode = FrontmatterMode.OPTIONAL) -> StructureSet:
    definition = StructureDefinition.define(
        name="MarkdownNote",
        version="1",
        markdown=MarkdownProfileV1(frontmatter),
        fields=(
            FieldMapping(
                "body",
                FieldSelector(FieldSelectorKind.DOCUMENT_BODY),
                FieldValueKind.STRING,
                True,
            ),
            FieldMapping(
                "title",
                FieldSelector(FieldSelectorKind.FRONTMATTER_FIELD, "title"),
                FieldValueKind.STRING,
                False,
            ),
        ),
        text_projections=(TextProjection("text", "body"),),
    )
    return StructureSet(
        (definition,),
        (
            StructureBinding(
                "markdown", SourceMatcher("**/*.md"), definition.ref
            ),
        ),
    )


def _composition(
    tmp_path: Path,
    files: dict[str, bytes],
    *,
    frontmatter: FrontmatterMode = FrontmatterMode.OPTIONAL,
    clock: _Clock | None = None,
):
    source = tmp_path / "source"
    source.mkdir()
    for name, content in files.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    clock = clock or _Clock()
    bundles = tmp_path / "bundles"
    operations = ProcessLocalIngestionOperationsV1(
        outputs={"primary": bundles}, clock=clock
    )
    api = IngestionAPI(operations, clock=clock)
    authorized = operations.authorize_local_selection(
        "project",
        (LocalSelectionRootV1("notes", source),),
        LocalDiscoveryPolicyV1(("**/*",), (), False),
    )
    snapshot = api.admit(SourceAdmissionRequest("admit-1", "project", authorized))
    request = IngestionRequest(
        "request-1",
        "project",
        snapshot,
        _structure(frontmatter=frontmatter),
        "primary",
    )
    return api, operations, bundles, request


def _successful(tmp_path: Path):
    api, operations, bundles, request = _composition(
        tmp_path,
        {"note.md": b"---\ntitle: Example\n---\nBody\n"},
    )
    plan = api.plan(request)
    preflight = api.preflight(plan)
    started = api.start(plan, preflight)
    outcome = api.show(started.run)
    assert outcome.state is IngestionRunState.SUCCEEDED
    return api, operations, bundles, plan, preflight, started, outcome


def test_synchronous_success_and_independent_disk_result(tmp_path: Path) -> None:
    api, _, bundles, plan, _, started, outcome = _successful(tmp_path)

    result = api.result(outcome)
    verification = api.verify(result)

    assert outcome.sources_processed == outcome.documents_written == 1
    assert outcome.bundle is not None
    assert outcome.bundle.plan_fingerprint == plan.plan_fingerprint
    assert (bundles / outcome.bundle.bundle_id / "manifest.json").is_file()
    assert api.show(started.run) == outcome
    assert verification.verified
    assert verification.diagnostic_codes == ()


def test_preview_blocker_closes_preflight_and_start(tmp_path: Path) -> None:
    api, _, _, request = _composition(
        tmp_path,
        {"note.md": b"Body\n", "ignored.txt": b"Not Markdown\n"},
    )

    plan = api.plan(request)
    preflight = api.preflight(plan)

    assert plan.preview.matched_sources == 1
    assert plan.preview.unmatched_sources == 1
    assert not preflight.ready
    assert preflight.diagnostic_codes == (IngestionDiagnosticCode.UNMATCHED_SOURCE,)
    with pytest.raises(IngestionOperationError) as failure:
        api.start(plan, preflight)
    assert failure.value.code is IngestionOperationCode.START_INELIGIBLE


def test_malformed_frontmatter_is_a_sanitized_preflight_blocker(
    tmp_path: Path,
) -> None:
    api, _, _, request = _composition(
        tmp_path,
        {"bad.md": b"---\ntitle: [unterminated\n---\nBody\n"},
        frontmatter=FrontmatterMode.REQUIRED,
    )

    plan = api.plan(request)
    preflight = api.preflight(plan)

    assert plan.preview.ready
    assert not preflight.ready
    assert preflight.diagnostic_codes == (IngestionDiagnosticCode.PARSE_FAILED,)
    assert "unterminated" not in repr(preflight.to_dict())


def test_only_explicit_verify_reopens_a_later_tampered_bundle(tmp_path: Path) -> None:
    api, _, bundles, _, _, started, outcome = _successful(tmp_path)
    result = api.result(outcome)
    assert outcome.bundle is not None
    items_path = bundles / outcome.bundle.bundle_id / "items.jsonl"
    items_path.chmod(0o600)
    items_path.write_bytes(b"tampered\n")

    assert api.show(started.run) == outcome
    assert api.result(outcome) == result
    verification = api.verify(result)
    assert not verification.verified
    assert verification.diagnostic_codes == (IngestionDiagnosticCode.BUNDLE_INVALID,)


def test_admission_rejects_authority_digest_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "note.md"
    source.write_bytes(b"Body\n")
    clock = _Clock()
    operations = ProcessLocalIngestionOperationsV1(
        outputs={"primary": tmp_path / "bundles"}, clock=clock
    )
    api = IngestionAPI(operations, clock=clock)
    issued = operations.authorize_local_selection(
        "project",
        (LocalSelectionRootV1("note.md", source),),
        LocalDiscoveryPolicyV1(("*.md",), (), False),
    )
    forged = AuthorizedSourceRef(
        issued.project_ref,
        SourceAdmissionKind.LOCAL_SELECTION,
        issued.source_ref,
        "0" * 64,
    )

    with pytest.raises(IngestionOperationError) as failure:
        api.admit(SourceAdmissionRequest("admit-1", "project", forged))
    assert failure.value.code is IngestionOperationCode.ADMISSION_INELIGIBLE
    admitted = api.admit(SourceAdmissionRequest("admit-2", "project", issued))
    assert admitted.project_ref == "project"


def test_returned_authority_is_detached_from_retained_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "note.md"
    source.write_bytes(b"Body\n")
    clock = _Clock()
    operations = ProcessLocalIngestionOperationsV1(
        outputs={"primary": tmp_path / "bundles"}, clock=clock
    )
    api = IngestionAPI(operations, clock=clock)
    returned = operations.authorize_local_selection(
        "project",
        (LocalSelectionRootV1("note.md", source),),
        LocalDiscoveryPolicyV1(("*.md",), (), False),
    )
    legitimate = AuthorizedSourceRef.from_dict(returned.to_dict())
    object.__setattr__(returned, "authority_digest", "0" * 64)

    with pytest.raises(IngestionOperationError) as failure:
        api.admit(SourceAdmissionRequest("admit-1", "project", returned))
    assert failure.value.code is IngestionOperationCode.ADMISSION_INELIGIBLE
    admitted = api.admit(SourceAdmissionRequest("admit-2", "project", legitimate))
    assert admitted.project_ref == "project"


def test_preflight_reuses_valid_cache_and_refreshes_at_expiry(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    api, _, _, request = _composition(
        tmp_path, {"note.md": b"Body\n"}, clock=clock
    )
    plan = api.plan(request)

    first = api.preflight(plan)
    clock.value = "2026-09-19T12:04:59Z"
    reused = api.preflight(plan)
    clock.value = first.expires_at
    refreshed = api.preflight(plan)

    assert reused == first
    assert refreshed != first
    assert refreshed.checked_at == first.expires_at
    assert refreshed.expires_at == "2026-09-19T12:10:00Z"


def test_repeated_plan_preflight_and_start_are_deterministic(tmp_path: Path) -> None:
    api, _, _, request = _composition(tmp_path, {"note.md": b"Body\n"})

    first_plan = api.plan(request)
    second_plan = api.plan(request)
    first_preflight = api.preflight(first_plan)
    second_preflight = api.preflight(second_plan)
    first_start = api.start(first_plan, first_preflight)
    second_start = api.start(second_plan, second_preflight)

    assert first_plan == second_plan
    assert first_preflight == second_preflight
    assert first_start == second_start
    assert api.show(first_start.run) == api.show(second_start.run)


def test_uncertain_publication_reconciles_exact_bundle_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_write = runtime_v1.write_normalized_bundle_v1

    def uncertain_write(*args, **kwargs):
        verified = actual_write(*args, **kwargs)
        raise BundlePublicationUncertainV1(
            verified.semantic_identity,
            BundlePublicationUncertaintyPhaseV1.PARENT_DURABILITY,
        )

    monkeypatch.setattr(runtime_v1, "write_normalized_bundle_v1", uncertain_write)
    api, _, _, request = _composition(tmp_path, {"note.md": b"Body\n"})
    plan = api.plan(request)
    preflight = api.preflight(plan)
    started = api.start(plan, preflight)
    uncertain = api.show(started.run)

    assert uncertain.state is IngestionRunState.RECONCILE_REQUIRED
    assert uncertain.diagnostic_code is IngestionDiagnosticCode.EFFECT_UNCERTAIN
    assert uncertain.sources_processed == uncertain.source_count == 1
    assert uncertain.documents_written == 0
    assert uncertain.bundle is None

    reconciled = api.reconcile(uncertain)
    assert reconciled.state is IngestionRunState.SUCCEEDED
    assert reconciled.bundle is not None
    assert reconciled.bundle.plan_fingerprint == plan.plan_fingerprint
    assert reconciled.bundle.run_authority_digest == started.run.authority_digest
    assert api.show(started.run) == reconciled
    with pytest.raises(IngestionOperationError) as failure:
        api.reconcile(uncertain)
    assert failure.value.code is IngestionOperationCode.RECONCILE_INELIGIBLE


def test_uncertain_publication_reconcile_fails_closed_for_invalid_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_write = runtime_v1.write_normalized_bundle_v1

    def uncertain_write(*args, **kwargs):
        verified = actual_write(*args, **kwargs)
        items_path = verified.path / "items.jsonl"
        items_path.chmod(0o600)
        items_path.write_bytes(b"tampered\n")
        raise BundlePublicationUncertainV1(
            verified.semantic_identity,
            BundlePublicationUncertaintyPhaseV1.FINAL_VERIFICATION,
        )

    monkeypatch.setattr(runtime_v1, "write_normalized_bundle_v1", uncertain_write)
    api, _, _, request = _composition(tmp_path, {"note.md": b"Body\n"})
    plan = api.plan(request)
    started = api.start(plan, api.preflight(plan))
    uncertain = api.show(started.run)

    reconciled = api.reconcile(uncertain)
    assert reconciled.state is IngestionRunState.FAILED
    assert reconciled.diagnostic_code is IngestionDiagnosticCode.BUNDLE_INVALID
    assert reconciled.sources_processed == reconciled.source_count == 1
    assert reconciled.documents_written == 0
    with pytest.raises(IngestionOperationError) as failure:
        api.reconcile(uncertain)
    assert failure.value.code is IngestionOperationCode.RECONCILE_INELIGIBLE


def test_uncertain_publication_requires_durability_retry_before_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_write = runtime_v1.write_normalized_bundle_v1
    verification_calls = 0

    def uncertain_write(*args, **kwargs):
        verified = actual_write(*args, **kwargs)
        raise BundlePublicationUncertainV1(
            verified.semantic_identity,
            BundlePublicationUncertaintyPhaseV1.PARENT_DURABILITY,
        )

    def failed_durability(root: Path) -> None:
        raise BundleDurabilityError()

    def forbidden_verification(path: Path):
        nonlocal verification_calls
        verification_calls += 1

    monkeypatch.setattr(runtime_v1, "write_normalized_bundle_v1", uncertain_write)
    monkeypatch.setattr(
        runtime_v1, "retry_bundle_root_durability_v1", failed_durability
    )
    monkeypatch.setattr(
        runtime_v1, "verify_normalized_bundle_v1", forbidden_verification
    )
    api, _, _, request = _composition(tmp_path, {"note.md": b"Body\n"})
    plan = api.plan(request)
    started = api.start(plan, api.preflight(plan))

    reconciled = api.reconcile(api.show(started.run))
    assert reconciled.state is IngestionRunState.FAILED
    assert reconciled.diagnostic_code is IngestionDiagnosticCode.BUNDLE_INVALID
    assert verification_calls == 0
