from __future__ import annotations

import json
from pathlib import Path

import pytest

import Evaluator.cli as evaluator_cli
from Evaluator.reporting import generate_evaluation_model_card_section
from tuner.project import ProjectContext


ENGINE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    """Dry-run the Evaluator CLI without a backend or tracking-registry writes."""
    import shared.experiment_tracking.registry as registry

    class _NoRegistry:
        def register_run(self, *args, **kwargs):
            return None

        def link_runs(self, *args, **kwargs):
            return None

    monkeypatch.setattr(registry, "RunRegistry", _NoRegistry)
    created = {}

    def fake_create_client(*, backend, settings, timeout, retries):
        created["settings"] = settings
        return object()

    monkeypatch.setattr(evaluator_cli, "create_client", fake_create_client)
    context = ProjectContext.standalone(engine_root=ENGINE_ROOT, invocation_cwd=tmp_path)

    def run(*extra: str):
        output = tmp_path / "results.json"
        lineage = tmp_path / "lineage.json"
        code = evaluator_cli.main(
            [
                "--scenario", "tool_prompts.yaml",
                "--limit", "2",
                "--dry-run",
                "--no-dashboard",
                "--output", str(output),
                "--lineage", str(lineage),
                *extra,
            ],
            project_context=context,
        )
        payload = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
        lineage_payload = json.loads(lineage.read_text(encoding="utf-8")) if lineage.exists() else None
        return code, payload, lineage_payload, created

    return run


def test_quantization_detected_from_model_path(run_cli):
    _, payload, lineage, _ = run_cli("--backend", "llamacpp", "--model", "models/nexus-Q5_K_M.gguf")
    artifact = payload["metadata"]["model_artifact"]
    assert artifact["quantization"] == "Q5_K_M"
    assert artifact["quantization_source"] == "detected"
    assert lineage["model_artifact"] == artifact
    assert "| Quantization | Q5_K_M |" in generate_evaluation_model_card_section(lineage)
    assert payload["metadata"]["scenarios"] == ["tool_prompts.yaml"]
    assert payload["metadata"]["preset"] is None


def test_quantization_flag_and_manifest(run_cli, tmp_path):
    manifest = tmp_path / "gguf_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [{"filename": "nexus-Q4_K_M.gguf", "quant_type": "Q4_K_M", "sha256": "ab", "imatrix_used": True}],
                "calibration": {"dataset": "calib.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    _, payload, lineage, _ = run_cli(
        "--backend", "llamacpp",
        "--model", "out/nexus-Q4_K_M.gguf",
        "--quantization", "Q4_K_M",
        "--artifact-manifest", str(manifest),
    )
    artifact = lineage["model_artifact"]
    assert artifact["quantization_source"] == "flag"
    assert artifact["manifest"]["file"]["sha256"] == "ab"
    assert artifact["manifest"]["calibration"] == {"dataset": "calib.jsonl"}
    assert payload["metadata"]["model_artifact"] == artifact


def test_unmatched_manifest_records_warning(run_cli, tmp_path):
    manifest = tmp_path / "gguf_manifest.json"
    manifest.write_text(json.dumps({"files": [{"filename": "a.gguf"}]}), encoding="utf-8")
    code, payload, _, _ = run_cli("--backend", "llamacpp", "--model", "b.gguf", "--artifact-manifest", str(manifest))
    assert payload is not None
    assert "no manifest file entry" in payload["metadata"]["model_artifact"]["warnings"][0]


def test_unsloth_load_settings_recorded(run_cli):
    _, payload, lineage, created = run_cli("--backend", "unsloth", "--model", "final_model")
    assert payload["metadata"]["model_artifact"]["load_settings"]["load_in_4bit"] is True
    assert lineage["model_artifact"]["load_settings"]["load_in_4bit"] is True
    assert created["settings"].load_in_4bit is True

    _, payload, _, created = run_cli("--backend", "unsloth", "--model", "final_model", "--no-load-in-4bit")
    assert payload["metadata"]["model_artifact"]["load_settings"]["load_in_4bit"] is False
    assert created["settings"].load_in_4bit is False


def test_no_load_in_4bit_rejected_for_other_backends(run_cli):
    code, payload, _, _ = run_cli("--backend", "vllm", "--model", "finetuned", "--no-load-in-4bit")
    assert code == 1
    assert payload is None
