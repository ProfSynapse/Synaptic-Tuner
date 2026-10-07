from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

import SynthChat.scripts.structured_document_export as export_module
from SynthChat.scripts.structured_document_bakeoff import (
    collect_openrouter_batch,
    judge_collected_batch,
    submit_openrouter_batch,
)
from SynthChat.scripts.structured_document_export import (
    dry_run_export,
    execute_export,
    preflight_export,
    verify_export,
)
from shared.llm.usage import LLMStructuredV1


MODEL = "openai/luna"
JUDGE = "openai/terra"


class _BatchClient:
    def __init__(self, observations):
        self.observations = observations

    def submit_batch(self, items, *, endpoint):
        assert endpoint == "/v1/chat/completions"
        return {"id": "batch-export", "status": "validating", "created_at": 123}

    def observe_batch(self, batch_id):
        assert batch_id == "batch-export"
        return self.observations[batch_id]


class _JudgeClient:
    def __init__(self, verdicts):
        self._verdicts = iter(verdicts)

    def structured_output(self, messages, schema, temperature, max_tokens):
        return LLMStructuredV1({"verdict": next(self._verdicts), "score": 95})


def _now():
    return datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def _make_batch(tmp_path: Path, payloads: list[dict], verdicts: list[str] | None = None):
    documents = []
    for index, payload in enumerate(payloads):
        source = tmp_path / f"source-{index}.md"
        source.write_text(f"---\nprivate: hidden-{index}\n---\nSource {index} body.\n", encoding="utf-8")
        documents.append(
            {
                "id": f"doc-{index}",
                "source_path": source.name,
                "strip_yaml_frontmatter": True,
                "metadata": {"collection": "portable-corpus", "ordinal": index},
            }
        )
    bakeoff = {
        "documents": documents,
        "prompt_template": "Transform {document} with {metadata}",
        "response_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "details": {},
            },
            "required": ["title", "body", "details"],
            "additionalProperties": False,
        },
        "models": [MODEL],
        "temperature": 0,
        "max_tokens": 500,
        "output_dir": "batch-results",
        "batch": {"max_requests": 20},
        "judge": {
            "model": JUDGE,
            "prompt_template": "SOURCE={document}\nCANDIDATE={candidate}",
            "response_schema": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["verdict", "score"],
                "additionalProperties": False,
            },
            "temperature": 0,
            "max_tokens": 100,
        },
    }
    bakeoff_path = tmp_path / "bakeoff.yaml"
    bakeoff_path.write_text(yaml.safe_dump(bakeoff, sort_keys=False), encoding="utf-8")

    observations = {}
    factory = lambda **_: _BatchClient(observations)
    state = submit_openrouter_batch(bakeoff_path, client_factory=factory, now=_now)
    custom_ids = list(state["batches"][0]["custom_ids"])
    observations["batch-export"] = {
        "id": "batch-export",
        "status": "completed",
        "request_counts": {"total": len(payloads), "completed": len(payloads), "failed": 0},
        "results": [
            {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": json.dumps(payload)}}]},
                },
                "error": None,
            }
            for custom_id, payload in zip(custom_ids, payloads)
        ],
    }
    collected = collect_openrouter_batch(bakeoff_path, client_factory=factory, now=_now)
    assert collected["ready"] is True

    judgment_manifest = None
    if verdicts is not None:
        judged = judge_collected_batch(
            bakeoff_path,
            client_factory=lambda **_: _JudgeClient(verdicts),
        )
        assert judged["ready"] is True
        judgment_manifest = tmp_path / "batch-results" / "judgments" / "manifest.json"
    return bakeoff_path, judgment_manifest


def _payload(index: int, *, details=None):
    return {
        "title": f"Portable record {index}",
        "body": f"Evidence-backed body {index}.",
        "details": details if details is not None else {"measure": index, "observations": ["alpha", "beta"]},
    }


def _export_config(
    tmp_path: Path,
    bakeoff_path: Path,
    judgment_manifest: Path | None,
    *,
    count: int,
    mode: str = "required",
    when_absent: str = "review",
    emit: dict[str, bool] | None = None,
    relative_paths: list[str] | None = None,
):
    if relative_paths is None:
        relative_paths = [f"records/item-{index}.md" for index in range(count)]
    config = {
        "kind": "structured_document_export/v1",
        "input": {
            "bakeoff_config": bakeoff_path.name,
            "judgments": {
                "mode": mode,
                "manifest_path": (
                    os.path.relpath(judgment_manifest, tmp_path).replace("\\", "/")
                    if judgment_manifest is not None
                    else None
                ),
                "disposition_pointer": "/judgment/payload/verdict",
                "disposition_map": {"accept": "accept", "review": "review", "reject": "reject"},
                "when_absent": when_absent,
            },
        },
        "destination": {
            "root": "published",
            "manifest": "export-manifest.jsonl",
            "emit": emit or {"accept": True, "review": True, "reject": False},
        },
        "outputs": [
            {"document_id": f"doc-{index}", "model": MODEL, "relative_path": relative_paths[index]}
            for index in range(count)
        ],
        "render": {
            "frontmatter": {
                "artifact_kind": {"$literal": "portable-record"},
                "document_id": {"$select": "/identity/document_id"},
                "model": {"$select": "/identity/model"},
                "collection": {"$select": "/source/metadata/collection"},
                "disposition": {"$select": "/quality/disposition"},
                "hashes": {
                    "source": {"$select": "/hashes/source_sha256"},
                    "generation": {"$select": "/hashes/generation_result_sha256"},
                },
            },
            "body_template": "# {{ title }}\n\n{{ body }}\n\n{{ details }}",
            "variables": {
                "title": {"select": "/generation/payload/title", "format": "text"},
                "body": {"select": "/generation/payload/body", "format": "text"},
                "details": {"select": "/generation/payload/details", "format": "markdown"},
            },
        },
        "limits": {
            "max_documents": max(1, count),
            "max_output_bytes_per_document": 100_000,
            "max_total_output_bytes": 1_000_000,
        },
    }
    if mode == "none":
        config["input"]["judgments"]["manifest_path"] = None
    path = tmp_path / "export.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, config


def _write_config(path: Path, config: dict):
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_execute_is_deterministic_generic_explicit_and_idempotent(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0), _payload(1)], ["accept", "review"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=2)

    first = execute_export(config_path)
    first_bytes = {
        path.relative_to(tmp_path / "published").as_posix(): path.read_bytes()
        for path in (tmp_path / "published").glob("**/*")
        if path.is_file()
    }
    second = execute_export(config_path)
    second_bytes = {
        path.relative_to(tmp_path / "published").as_posix(): path.read_bytes()
        for path in (tmp_path / "published").glob("**/*")
        if path.is_file()
    }

    assert first["created"] == 2
    assert first["manifest_created"] is True
    assert second["created"] == 0
    assert second["existing_identical"] == 2
    assert second["manifest_created"] is False
    assert first_bytes == second_bytes
    rendered = first_bytes["records/item-0.md"].decode("utf-8")
    frontmatter = yaml.safe_load(rendered.split("---", 2)[1])
    assert frontmatter["artifact_kind"] == "portable-record"
    assert frontmatter["document_id"] == "doc-0"
    assert frontmatter["collection"] == "portable-corpus"
    assert frontmatter["disposition"] == "accept"
    assert "# Portable record 0" in rendered
    assert "## measure" in rendered
    assert "chapter" not in rendered.casefold()
    assert "fiction" not in rendered.casefold()
    assert verify_export(config_path)["ok"] is True


def test_disposition_emit_policy_is_applied_and_manifest_lists_only_emitted_rows(tmp_path):
    bakeoff, judgments = _make_batch(
        tmp_path,
        [_payload(0), _payload(1), _payload(2)],
        ["accept", "review", "reject"],
    )
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=3)

    summary = execute_export(config_path)

    assert summary["documents"] == 3
    assert summary["emitted"] == 2
    assert (tmp_path / "published" / "records" / "item-0.md").is_file()
    assert (tmp_path / "published" / "records" / "item-1.md").is_file()
    assert not (tmp_path / "published" / "records" / "item-2.md").exists()
    lines = [json.loads(line) for line in (tmp_path / "published" / "export-manifest.jsonl").read_text().splitlines()]
    assert lines[0]["output_count"] == 2
    assert [line["disposition"] for line in lines[1:]] == ["accept", "review"]


@pytest.mark.parametrize("mode", ["optional", "none"])
def test_absent_judgments_use_declared_policy(tmp_path, mode):
    bakeoff, _ = _make_batch(tmp_path, [_payload(0)], verdicts=None)
    config_path, _ = _export_config(
        tmp_path,
        bakeoff,
        None,
        count=1,
        mode=mode,
        when_absent="reject",
        emit={"accept": True, "review": True, "reject": False},
    )

    summary = execute_export(config_path)

    assert summary["documents"] == 1
    assert summary["emitted"] == 0
    assert summary["created"] == 0
    assert (tmp_path / "published" / "export-manifest.jsonl").is_file()
    assert not (tmp_path / "published" / "records" / "item-0.md").exists()


def test_required_judgment_manifest_must_exist_and_no_destination_is_created(tmp_path):
    bakeoff, _ = _make_batch(tmp_path, [_payload(0)], verdicts=None)
    missing = tmp_path / "missing-judgments.json"
    config_path, _ = _export_config(tmp_path, bakeoff, missing, count=1)

    with pytest.raises(ValueError, match="required judgment manifest is absent"):
        execute_export(config_path)

    assert not (tmp_path / "published").exists()


def test_unmapped_judgment_fails_closed_before_writes(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["manual_hold"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=1)

    with pytest.raises(ValueError, match="disposition is not mapped"):
        execute_export(config_path)

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("drift", ["source", "generation", "judgment", "collection_manifest"])
def test_bound_input_drift_is_rejected_before_destination_writes(tmp_path, drift):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=1)
    if drift == "source":
        (tmp_path / "source-0.md").write_text("changed private source", encoding="utf-8")
    elif drift == "generation":
        result = next((tmp_path / "batch-results").glob("doc-*.json"))
        result.write_bytes(result.read_bytes() + b" ")
    elif drift == "judgment":
        artifact = next((tmp_path / "batch-results" / "judgments").glob("*.judgment.json"))
        artifact.write_bytes(artifact.read_bytes() + b" ")
    else:
        manifest = tmp_path / "batch-results" / "batch-manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(ValueError):
        execute_export(config_path)

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "../escape.md",
        "/absolute.md",
        "C:/drive.md",
        "folder\\windows.md",
        "CON.md",
        "nested/aux.txt",
        "trailing-dot./file.md",
        "trailing-space /file.md",
        "control\x00.md",
    ],
)
def test_nonportable_or_escaping_output_paths_are_rejected(tmp_path, relative_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, _ = _export_config(
        tmp_path, bakeoff, judgments, count=1, relative_paths=[relative_path]
    )

    with pytest.raises(ValueError):
        preflight_export(config_path)

    assert not (tmp_path / "published").exists()


def test_casefolded_output_collision_is_rejected(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0), _payload(1)], ["accept", "accept"])
    config_path, _ = _export_config(
        tmp_path,
        bakeoff,
        judgments,
        count=2,
        relative_paths=["Records/Item.md", "records/item.MD"],
    )

    with pytest.raises(ValueError, match="case-folded path"):
        preflight_export(config_path)


@pytest.mark.parametrize("invalid_character", ["<", ">", ":", '"', "|", "?", "*"])
def test_windows_invalid_component_characters_and_ads_are_rejected_at_admission(
    tmp_path, invalid_character
):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    relative_path = f"records/name{invalid_character}stream.md"
    config_path, _ = _export_config(
        tmp_path,
        bakeoff,
        judgments,
        count=1,
        relative_paths=[relative_path],
    )

    with pytest.raises(ValueError):
        preflight_export(config_path)

    assert not (tmp_path / "published").exists()


def test_case_insensitive_output_ancestor_descendant_conflict_is_rejected_at_admission(
    tmp_path,
):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0), _payload(1)], ["accept", "accept"])
    config_path, _ = _export_config(
        tmp_path,
        bakeoff,
        judgments,
        count=2,
        relative_paths=["Records/Item.md", "records/item.MD/child.md"],
    )

    with pytest.raises(ValueError, match="ancestor|conflict"):
        preflight_export(config_path)

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize(
    "output_path,manifest_path",
    [
        ("records", "records/export-manifest.jsonl"),
        ("Export-Manifest.JSONL/items/record.md", "export-manifest.jsonl"),
    ],
)
def test_output_and_manifest_ancestor_descendant_conflicts_are_rejected_at_admission(
    tmp_path, output_path, manifest_path
):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, config = _export_config(
        tmp_path,
        bakeoff,
        judgments,
        count=1,
        relative_paths=[output_path],
    )
    config["destination"]["manifest"] = manifest_path
    _write_config(config_path, config)

    with pytest.raises(ValueError, match="manifest|ancestor|conflict"):
        preflight_export(config_path)

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate_identity"])
def test_outputs_must_be_an_exact_explicit_identity_mapping(tmp_path, mutation):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0), _payload(1)], ["accept", "accept"])
    config_path, config = _export_config(tmp_path, bakeoff, judgments, count=2)
    if mutation == "missing":
        config["outputs"].pop()
    elif mutation == "extra":
        config["outputs"].append(
            {"document_id": "invented", "model": MODEL, "relative_path": "records/invented.md"}
        )
    else:
        config["outputs"][1]["document_id"] = "doc-0"
    _write_config(config_path, config)

    with pytest.raises(ValueError):
        preflight_export(config_path)


def test_preflight_dry_run_and_failed_verify_do_not_write(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=1)

    assert preflight_export(config_path)["operation"] == "preflight"
    assert dry_run_export(config_path)["operation"] == "dry-run"
    assert not (tmp_path / "published").exists()
    with pytest.raises(ValueError, match="published output is absent"):
        verify_export(config_path)
    assert not (tmp_path / "published").exists()


def test_different_existing_output_blocks_all_new_writes(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0), _payload(1)], ["accept", "accept"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=2)
    collision = tmp_path / "published" / "records" / "item-1.md"
    collision.parent.mkdir(parents=True)
    collision.write_text("USER AUTHORED - DO NOT OVERWRITE", encoding="utf-8")

    with pytest.raises(ValueError, match="different bytes"):
        execute_export(config_path)

    assert collision.read_text(encoding="utf-8") == "USER AUTHORED - DO NOT OVERWRITE"
    assert not (tmp_path / "published" / "records" / "item-0.md").exists()
    assert not (tmp_path / "published" / "export-manifest.jsonl").exists()


def test_different_existing_manifest_blocks_all_document_writes(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=1)
    manifest = tmp_path / "published" / "export-manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("preexisting manifest", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest exists with different bytes"):
        execute_export(config_path)

    assert manifest.read_text(encoding="utf-8") == "preexisting manifest"
    assert not (tmp_path / "published" / "records" / "item-0.md").exists()


def test_interruption_leaves_manifest_absent_and_retry_completes(monkeypatch, tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0), _payload(1)], ["accept", "accept"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=2)
    original = export_module._publish
    calls = 0

    def interrupt_second(path, encoded):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(path, encoded)

    monkeypatch.setattr(export_module, "_publish", interrupt_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute_export(config_path)
    assert (tmp_path / "published" / "records" / "item-0.md").is_file()
    assert not (tmp_path / "published" / "records" / "item-1.md").exists()
    assert not (tmp_path / "published" / "export-manifest.jsonl").exists()
    assert not (tmp_path / "published" / ".structured-document-export.lock").exists()

    monkeypatch.setattr(export_module, "_publish", original)
    completed = execute_export(config_path)
    assert completed["created"] == 1
    assert completed["existing_identical"] == 1
    assert completed["manifest_created"] is True
    assert verify_export(config_path)["ok"] is True


def test_exclusive_claim_prevents_concurrent_publication(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=1)
    root = tmp_path / "published"
    root.mkdir()
    lock = root / ".structured-document-export.lock"
    lock.write_text("other writer", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exclusively claimed"):
        execute_export(config_path)

    assert lock.read_text(encoding="utf-8") == "other writer"
    assert not (root / "records" / "item-0.md").exists()
    assert not (root / "export-manifest.jsonl").exists()


def test_symlink_destination_ancestor_is_rejected_when_supported(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=1)
    target = tmp_path / "outside"
    target.mkdir()
    try:
        (tmp_path / "published").symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating directory symlinks/reparse points is not permitted")

    with pytest.raises(ValueError, match="reparse or symlink ancestor"):
        dry_run_export(config_path)

    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    "limit,expected",
    [
        ("max_output_bytes_per_document", "per-document byte limit"),
        ("max_total_output_bytes", "total byte limit"),
    ],
)
def test_render_byte_limits_fail_before_writes(tmp_path, limit, expected):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0), _payload(1)], ["accept", "accept"])
    config_path, config = _export_config(tmp_path, bakeoff, judgments, count=2)
    config["limits"][limit] = 1
    _write_config(config_path, config)

    with pytest.raises(ValueError, match=expected):
        execute_export(config_path)

    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize(
    "template,variables",
    [
        ("{{ missing }}", {}),
        ("literal only", {"unused": {"select": "/generation/payload/body", "format": "text"}}),
        ("{{ malformed", {}),
    ],
)
def test_body_template_tokens_must_completely_match_declared_variables(
    tmp_path, template, variables
):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, config = _export_config(tmp_path, bakeoff, judgments, count=1)
    config["render"]["body_template"] = template
    config["render"]["variables"] = variables
    _write_config(config_path, config)

    with pytest.raises(ValueError, match="tokens must exactly match variables"):
        preflight_export(config_path)


def test_configured_selection_and_text_type_errors_fail_before_writes(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, config = _export_config(tmp_path, bakeoff, judgments, count=1)
    config["render"]["variables"]["body"] = {
        "select": "/generation/payload/details",
        "format": "text",
    }
    _write_config(config_path, config)

    with pytest.raises(ValueError, match="text format requires a string"):
        execute_export(config_path)

    assert not (tmp_path / "published").exists()


def test_cli_failure_is_closed_json_and_does_not_leak_privacy_canaries(tmp_path, capsys):
    source_canary = "SOURCE-CANARY-91f8e4"
    result_canary = "RESULT-CANARY-721bce"
    config_canary = "CONFIG-CANARY-c2a169"
    bakeoff, judgments = _make_batch(
        tmp_path,
        [{"title": "Private", "body": source_canary, "details": result_canary}],
        ["accept"],
    )
    config_path, config = _export_config(tmp_path, bakeoff, judgments, count=1)
    config["render"]["variables"][config_canary] = {
        "select": "/generation/payload/absent",
        "format": "text",
    }
    config["render"]["body_template"] += f"\n{{{{ {config_canary} }}}}"
    _write_config(config_path, config)

    code = export_module.main(["--execute", "--config", str(config_path)])
    captured = capsys.readouterr()
    error = json.loads(captured.err)

    assert code == 2
    assert captured.out == ""
    assert error == {
        "ok": False,
        "operation": "execute",
        "error": {"type": "ValueError"},
    }
    combined = captured.out + captured.err
    assert source_canary not in combined
    assert result_canary not in combined
    assert config_canary not in combined
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize(
    "flag,expected_code",
    [("--preflight", 0), ("--dry-run", 0), ("--verify", 2)],
)
def test_cli_read_only_phases_are_machine_readable_and_do_not_write(
    tmp_path, capsys, flag, expected_code
):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=1)

    code = export_module.main([flag, "--config", str(config_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out if code == 0 else captured.err)

    assert code == expected_code
    assert payload["ok"] is (expected_code == 0)
    assert not (tmp_path / "published").exists()


def test_verify_detects_post_publish_tampering_without_repairing_it(tmp_path):
    bakeoff, judgments = _make_batch(tmp_path, [_payload(0)], ["accept"])
    config_path, _ = _export_config(tmp_path, bakeoff, judgments, count=1)
    execute_export(config_path)
    output = tmp_path / "published" / "records" / "item-0.md"
    output.write_text("tampered user data", encoding="utf-8")

    with pytest.raises(ValueError):
        verify_export(config_path)

    assert output.read_text(encoding="utf-8") == "tampered user data"
