from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from synaptic_tuner.api.v1.ingestion_facade import (
    FieldMapping,
    FieldSelector,
    FieldSelectorKind,
    FieldValueKind,
    FrontmatterMode,
    MarkdownProfileV1,
    SourceMatcher,
    StructureBinding,
    StructureDefinition,
    StructureSet,
    TextProjection,
)
from tuner.cli.main import main as cli_main
from tuner.cli.parser import create_parser
from tuner.cli.router import route_command
from tuner.dataset_prep import (
    DatasetPrepValidationError,
    verify_prepared_dataset_v1,
    verify_prepared_dataset_v2,
)
from tuner.ingestion import (
    NormalizedItemInputV1,
    load_verified_normalized_bundle_v1,
    write_normalized_bundle_v1,
)
from tuner.project import ProjectContext


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.standalone(engine_root=tmp_path, invocation_cwd=tmp_path)


def _bundle(tmp_path: Path):
    definition = StructureDefinition.define(
        name="MarkdownNote",
        version="1",
        markdown=MarkdownProfileV1(FrontmatterMode.OPTIONAL),
        fields=(
            FieldMapping(
                "body",
                FieldSelector(FieldSelectorKind.DOCUMENT_BODY),
                FieldValueKind.STRING,
                True,
            ),
        ),
        text_projections=(TextProjection("text", "body"),),
    )
    structures = StructureSet(
        (definition,),
        (StructureBinding("markdown", SourceMatcher("**/*.md"), definition.ref),),
    )
    items = tuple(
        NormalizedItemInputV1(
            logical_path=f"chapter-{index}.md",
            source_sha256=(str(index + 1) * 64)[:64],
            source_size_bytes=len(text.encode("utf-8")),
            structure_ref=definition.ref.to_dict(),
            fields={"body": text},
        )
        for index, text in enumerate(("Private alpha", "Private beta", "Private gamma"))
    )
    return structures, write_normalized_bundle_v1(tmp_path / "bundles", structures.to_dict(), items)


def _write_config(tmp_path: Path, structures: StructureSet, bundle) -> Path:
    relative_bundle = os.path.relpath(bundle.path, tmp_path)
    config = {
        "schema_version": "syntunia-dataset-prep/v1",
        "source": {
            "bundle_path": relative_bundle,
            "expected_bundle_digest": bundle.semantic_identity.bundle_digest,
        },
        "format": "raw_text",
        "projection": {
            "structure_ref": structures.structures[0].ref.to_dict(),
            "name": "text",
        },
        "ordering": {"kind": "source_order"},
        "split": {
            "kind": "hash_rank",
            "seed": "private-fixture-split",
            "allocations": [
                {"name": "train", "weight": 2},
                {"name": "validation", "weight": 1},
            ],
        },
    }
    path = tmp_path / "dataset-prep.json"
    path.write_text(json.dumps(config, separators=(",", ":")), encoding="utf-8")
    return path


def _args(config: Path, *, json_mode: bool = True):
    return create_parser().parse_args(
        ["prepare-dataset", "--config", str(config), *(["--json"] if json_mode else [])]
    )


def test_prepare_dataset_help_and_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    structures, bundle = _bundle(tmp_path)
    config = _write_config(tmp_path, structures, bundle)
    handled: list[tuple[str, Path]] = []

    def fake_handle(self) -> int:
        handled.append((self.name, self.context.tracking_root))
        return 7

    monkeypatch.setattr(
        "tuner.handlers.dataset_prepare_handler.DatasetPrepareHandler.handle", fake_handle
    )
    parser = create_parser()

    assert "prepare-dataset" in parser.format_help()
    assert route_command(_args(config), context=_context(tmp_path)) == 7
    assert handled == [("prepare-dataset", (tmp_path / ".tracking").resolve())]


def test_prepare_dataset_cli_builds_private_verified_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structures, bundle = _bundle(tmp_path)
    config = _write_config(tmp_path, structures, bundle)

    exit_code = route_command(_args(config), context=_context(tmp_path))
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    if os.name == "nt":
        assert exit_code == 1
        assert payload["status"] == "publication_uncertain"
        assert payload["phase"] == "parent_durability"
        assert captured.err == ""
        exit_code = route_command(_args(config), context=_context(tmp_path))
        captured = capsys.readouterr()
        payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert payload["success"] is True
    assert payload["status"] == "verified"
    assert payload["row_count"] == 3
    assert payload["split_counts"] == {"train": 2, "validation": 1}
    assert payload["source_bundle_digest"] == bundle.semantic_identity.bundle_digest
    artifact = tmp_path / ".tracking" / "datasets" / payload["dataset_ref"]
    assert {entry.name for entry in artifact.iterdir()} == {"manifest.json", "dataset.jsonl"}
    assert "Private alpha" not in captured.out
    assert str(bundle.path) not in captured.out
    assert str(config) not in captured.out


@pytest.mark.parametrize(
    "content",
    [
        b'{"schema_version":"syntunia-dataset-prep/v1","schema_version":"duplicate"}',
        b"{not-json}",
        b"[1,2,3]",
    ],
)
def test_prepare_dataset_rejects_invalid_config_without_leaking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: bytes,
) -> None:
    config = tmp_path / "private-config.json"
    config.write_bytes(content)

    assert route_command(_args(config), context=_context(tmp_path)) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "success": False,
        "status": "failed",
        "error_code": "invalid_input",
    }
    assert captured.err == ""
    assert str(config) not in captured.out


def test_prepare_dataset_requires_json_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structures, bundle = _bundle(tmp_path)
    config = _write_config(tmp_path, structures, bundle)

    assert route_command(_args(config, json_mode=False), context=_context(tmp_path)) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "invalid_input"


def test_prepare_dataset_sanitizes_backend_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structures, bundle = _bundle(tmp_path)
    config = _write_config(tmp_path, structures, bundle)
    private_detail = str(tmp_path / "private-prose-location")

    def fail(*_args, **_kwargs):
        raise RuntimeError(private_detail)

    monkeypatch.setattr("tuner.handlers.dataset_prepare_handler.prepare_dataset_v1", fail)
    assert route_command(_args(config), context=_context(tmp_path)) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error_code"] == "dataset_prep_failed"
    assert private_detail not in captured.out
    assert captured.err == ""


def test_prepare_dataset_sanitizes_bootstrap_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structures, bundle = _bundle(tmp_path)
    config = _write_config(tmp_path, structures, bundle)
    private_detail = str(tmp_path / "private-bootstrap-location")

    def fail_context(*_args, **_kwargs):
        raise RuntimeError(private_detail)

    main_module = importlib.import_module("tuner.cli.main")
    monkeypatch.setattr(main_module, "build_project_context", fail_context)
    with pytest.raises(SystemExit) as stopped:
        cli_main(["prepare-dataset", "--config", str(config), "--json"])

    captured = capsys.readouterr()
    assert stopped.value.code == 1
    assert json.loads(captured.out)["error_code"] == "bootstrap_failed"
    assert private_detail not in captured.out
    assert str(config) not in captured.out
    assert captured.err == ""


def test_prepare_dataset_rejects_redirected_config_ancestor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structures, bundle = _bundle(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    config = _write_config(real, structures, bundle)
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    redirected = linked / config.name
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNAPTIC_ENGINE_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as stopped:
        cli_main(["prepare-dataset", "--config", str(redirected), "--json"])

    captured = capsys.readouterr()
    assert stopped.value.code == 1
    assert json.loads(captured.out)["error_code"] == "bootstrap_failed"
    assert str(real) not in captured.out
    assert str(linked) not in captured.out
    assert captured.err == ""


def test_prepare_dataset_cli_auto_dispatches_v2_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structures, bundle = _bundle(tmp_path)
    loaded = load_verified_normalized_bundle_v1(bundle.path)
    ids = [item.item_id for item in loaded.items]
    config = {
        "schema_version": "syntunia-dataset-prep/v2",
        "source": {
            "bundle_path": os.path.relpath(bundle.path, tmp_path),
            "expected_bundle_digest": bundle.semantic_identity.bundle_digest,
        },
        "format": "messages",
        "target_projection": {
            "structure_ref": structures.structures[0].ref.to_dict(),
            "name": "text",
        },
        "context_package": {
            "ordering": {"kind": "configured_whole_documents"},
            "lineage": [
                {
                    "item_id": ids[0],
                    "revision_family": "target-a",
                    "sequence": 2,
                    "derived_from": [],
                    "group_id": "group-a",
                },
                {
                    "item_id": ids[1],
                    "revision_family": "target-b",
                    "sequence": 3,
                    "derived_from": [],
                    "group_id": "group-b",
                },
                {
                    "item_id": ids[2],
                    "revision_family": "reference",
                    "sequence": 1,
                    "derived_from": [],
                    "group_id": "reference",
                },
            ],
            "packages": [
                {
                    "target_item_id": ids[0],
                    "context_item_ids": [ids[2]],
                    "prompt_variant": "plain",
                },
                {
                    "target_item_id": ids[1],
                    "context_item_ids": [ids[2]],
                    "prompt_variant": "plain",
                },
            ],
            "prompt_variants": [
                {"name": "plain", "prompt": "Use this context.", "separator": "\n\n"}
            ],
            "target_transforms": {
                "drop_fenced_block_info_strings": [],
                "drop_standalone_line_prefixes": [],
            },
        },
        "split": {
            "kind": "group_hash_rank",
            "seed": "fixture",
            "allocations": [
                {"name": "train", "weight": 1},
                {"name": "validation", "weight": 1},
            ],
        },
    }
    path = tmp_path / "dataset-prep-v2.json"
    path.write_text(json.dumps(config, separators=(",", ":")), encoding="utf-8")

    exit_code = route_command(_args(path), context=_context(tmp_path))
    payload = json.loads(capsys.readouterr().out)
    if os.name == "nt":
        assert payload["status"] == "publication_uncertain"
        exit_code = route_command(_args(path), context=_context(tmp_path))
        payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "verified"
    artifact = tmp_path / ".tracking" / "datasets" / payload["dataset_ref"]
    rows = [json.loads(line) for line in (artifact / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["format"] for row in rows} == {"messages"}
    assert all([message["role"] for message in row["messages"]] == ["user", "assistant"] for row in rows)
    assert verify_prepared_dataset_v2(artifact).semantic_identity.dataset_id == payload["dataset_ref"]
    with pytest.raises(DatasetPrepValidationError, match="requested verifier"):
        verify_prepared_dataset_v1(artifact)
