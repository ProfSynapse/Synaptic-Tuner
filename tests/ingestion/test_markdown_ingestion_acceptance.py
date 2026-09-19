from __future__ import annotations

import json
from pathlib import Path

from synaptic_tuner.api.v1.ingestion_facade import (
    FieldMapping,
    FieldSelector,
    FieldSelectorKind,
    FieldValueKind,
    FrontmatterMode,
    IngestionAPI,
    IngestionRequest,
    IngestionRunState,
    MarkdownProfileV1,
    MetadataDeclaration,
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
    ProcessLocalSelectionRegistryV1,
)
from tuner.ingestion.runtime_v1 import ProcessLocalIngestionOperationsV1


_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "ingestion" / "markdown_v1"


class _FixedClock:
    def now(self) -> str:
        return "2026-09-19T12:00:00Z"


def _config() -> dict[str, object]:
    return json.loads((_FIXTURE_ROOT / "ingestion.json").read_text(encoding="utf-8"))


def _structure(config: dict[str, object]) -> StructureSet:
    structure_config = config["structures"]
    binding_config = config["bindings"]
    assert type(structure_config) is list and len(structure_config) == 1
    assert type(binding_config) is list and len(binding_config) == 1
    definition_config = structure_config[0]
    binding = binding_config[0]
    assert type(definition_config) is dict and type(binding) is dict
    markdown = definition_config["markdown"]
    fields = definition_config["fields"]
    projections = definition_config["text_projections"]
    metadata = definition_config["metadata"]
    matcher = binding["matcher"]
    assert type(markdown) is dict
    assert type(fields) is list
    assert type(projections) is list
    assert type(metadata) is list
    assert type(matcher) is dict
    definition = StructureDefinition.define(
        name=definition_config["name"],  # type: ignore[arg-type]
        version=definition_config["version"],  # type: ignore[arg-type]
        markdown=MarkdownProfileV1(FrontmatterMode(markdown["frontmatter_mode"])),
        fields=tuple(
            FieldMapping(
                field["name"],  # type: ignore[arg-type]
                FieldSelector(
                    FieldSelectorKind(field["selector"]["kind"]),  # type: ignore[index,arg-type]
                    field["selector"]["key"],  # type: ignore[index,arg-type]
                ),
                FieldValueKind(field["value_kind"]),  # type: ignore[arg-type]
                field["required"],  # type: ignore[arg-type]
            )
            for field in fields
        ),
        text_projections=tuple(
            TextProjection(item["name"], item["field_ref"])  # type: ignore[arg-type]
            for item in projections
        ),
        metadata=tuple(
            MetadataDeclaration(item["name"], item["field_ref"])  # type: ignore[arg-type]
            for item in metadata
        ),
    )
    return StructureSet(
        (definition,),
        (
            StructureBinding(
                binding["binding_id"],  # type: ignore[arg-type]
                SourceMatcher(matcher["pattern"]),  # type: ignore[arg-type]
                definition.ref,
            ),
        ),
    )


def _selection(config: dict[str, object]) -> tuple[LocalSelectionRootV1, LocalDiscoveryPolicyV1]:
    selection = config["selection"]
    assert type(selection) is dict
    roots = selection["roots"]
    assert type(roots) is list and len(roots) == 1 and type(roots[0]) is dict
    root = roots[0]
    include = selection["include"]
    exclude = selection["exclude"]
    assert type(include) is list and type(exclude) is list
    return (
        LocalSelectionRootV1(root["alias"], _FIXTURE_ROOT / root["path"]),  # type: ignore[arg-type,operator]
        LocalDiscoveryPolicyV1(
            tuple(include),  # type: ignore[arg-type]
            tuple(exclude),  # type: ignore[arg-type]
            selection["include_hidden"],  # type: ignore[arg-type]
        ),
    )


def _run(
    tmp_path: Path, config: dict[str, object]
) -> tuple[object, bytes, bytes, str]:
    root, policy = _selection(config)
    operations = ProcessLocalIngestionOperationsV1(
        outputs={config["output_ref"]: tmp_path / "private-bundles"},  # type: ignore[dict-item]
        clock=_FixedClock(),
    )
    api = IngestionAPI(operations, clock=_FixedClock())
    authorized = operations.authorize_local_selection(
        config["project_ref"], (root,), policy  # type: ignore[arg-type]
    )
    snapshot = api.admit(
        SourceAdmissionRequest(
            "fixture_admission", config["project_ref"], authorized  # type: ignore[arg-type]
        )
    )
    plan = api.plan(
        IngestionRequest(
            config["request_id"],  # type: ignore[arg-type]
            config["project_ref"],  # type: ignore[arg-type]
            snapshot,
            _structure(config),
            config["output_ref"],  # type: ignore[arg-type]
        )
    )
    preflight = api.preflight(plan)
    assert plan.preview.source_count == 3
    assert plan.preview.matched_sources == 3
    assert plan.preview.unmatched_sources == plan.preview.ambiguous_sources == 0
    assert preflight.ready
    assert preflight.diagnostic_codes == ()
    started = api.start(plan, preflight)
    outcome = api.show(started.run)
    result = api.result(outcome)
    verification = api.verify(result)
    assert verification.verified
    assert verification.diagnostic_codes == ()
    assert outcome.bundle is not None
    bundle_path = tmp_path / "private-bundles" / outcome.bundle.bundle_id
    assert {entry.name for entry in bundle_path.iterdir()} == {"manifest.json", "items.jsonl"}
    lifecycle = json.dumps(
        [
            snapshot.to_dict(),
            plan.to_dict(),
            preflight.to_dict(),
            started.to_dict(),
            outcome.to_dict(),
            result.to_dict(),
            verification.to_dict(),
        ],
        sort_keys=True,
    )
    return outcome, (bundle_path / "manifest.json").read_bytes(), (bundle_path / "items.jsonl").read_bytes(), lifecycle


def test_markdown_fixture_ingests_privately_and_deterministically(tmp_path: Path) -> None:
    config = _config()
    assert set(config) == {
        "schema_version",
        "project_ref",
        "request_id",
        "output_ref",
        "selection",
        "structures",
        "bindings",
    }
    assert config["schema_version"] == "synaptic-ingestion-config/v1"
    assert b"\r" not in (_FIXTURE_ROOT / "alpha.md").read_bytes()
    beta_bytes = (_FIXTURE_ROOT / "nested" / "beta.md").read_bytes()
    assert beta_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in beta_bytes
    assert b"\n" not in beta_bytes.replace(b"\r\n", b"")

    root, policy = _selection(config)
    registry = ProcessLocalSelectionRegistryV1()
    authorized = registry.authorize(config["project_ref"], (root,), policy)  # type: ignore[arg-type]
    selection = registry.consume_snapshot(authorized.source_ref)
    assert selection.report.admitted_file_count == 3
    assert selection.report.excluded_file_count == 2
    assert tuple(item.logical_path for item in selection.entries) == (
        "notes/alpha.md",
        "notes/nested/beta.md",
        "notes/plain.md",
    )

    first, first_manifest, first_items, first_lifecycle = _run(tmp_path / "first", config)
    second, second_manifest, second_items, second_lifecycle = _run(tmp_path / "second", config)

    for outcome in (first, second):
        assert outcome.state is IngestionRunState.SUCCEEDED
        assert outcome.sources_processed == 3
        assert outcome.documents_written == 3
        assert outcome.bundle is not None
    assert first.bundle is not None and second.bundle is not None
    assert first.run.run_id != second.run.run_id
    assert first.run.plan_fingerprint != second.run.plan_fingerprint
    assert first.run.authority_digest != second.run.authority_digest
    first_rows = {
        row["logical_path"]: row["fields"]
        for row in (json.loads(line) for line in first_items.splitlines())
    }
    assert first_rows["notes/alpha.md"]["published"] is True
    assert first_rows["notes/alpha.md"]["tags"] == ["acceptance", "markdown"]
    assert first_rows["notes/nested/beta.md"]["rank"] == 2
    assert "\r" not in first_rows["notes/nested/beta.md"]["body"]
    assert set(first_rows["notes/plain.md"]) == {"body", "path"}
    assert first.bundle.bundle_digest == second.bundle.bundle_digest
    assert first_manifest == second_manifest
    assert first_items == second_items

    for serialized in (first_manifest.decode("utf-8"), first_items.decode("utf-8"), first_lifecycle, second_lifecycle):
        assert str(_FIXTURE_ROOT.resolve()) not in serialized
        assert str(tmp_path.resolve()) not in serialized
        assert "F:\\" not in serialized
        assert ":\\" not in serialized
