from __future__ import annotations

import json
from pathlib import Path

import pytest

from Evaluator.model_artifact import (
    build_model_artifact,
    detect_quantization,
    is_full_precision,
    load_manifest_details,
    match_manifest_entry,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("./models/nexus-Q4_K_M.gguf", "Q4_K_M"),
        ("model.q5_k_m.gguf", "Q5_K_M"),
        ("model-Q8_0.gguf", "Q8_0"),
        ("model-Q6_K.gguf", "Q6_K"),
        ("model-Q3_K_S.gguf", "Q3_K_S"),
        ("model-IQ2_XS.gguf", "IQ2_XS"),
        ("model-IQ2_XXS.gguf", "IQ2_XXS"),
        ("model-IQ4_NL.gguf", "IQ4_NL"),
        ("model-TQ1_0.gguf", "TQ1_0"),
        ("model-Q4_0_4_8.gguf", "Q4_0_4_8"),
        ("model-MXFP4.gguf", "MXFP4"),
        ("model-BF16.gguf", "BF16"),
        ("model-f16.gguf", "F16"),
        ("user/Model-GGUF:Q4_K_M", "Q4_K_M"),
        ("C:\\models\\model-Q5_K_S.gguf", "Q5_K_S"),
        ("model_Q4_K_M_imatrix.gguf", "Q4_K_M"),
    ],
)
def test_detect_quantization_names(name, expected):
    assert detect_quantization(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "qwen2.5-7b-instruct",
        "Trainers/sft/sft_output/20250101/final_model",
        "runs/F16_sweep/final_model",  # directory component is not the artifact
        "gpt-q4k",  # not a llama.cpp type name
        "modelQ4_K_M.gguf",  # glued to other alphanumerics
        "llama-3-8b-Q9_0.gguf",
    ],
)
def test_detect_quantization_no_match(name):
    assert detect_quantization(name) is None


def test_full_precision_labels():
    assert is_full_precision("F16") and is_full_precision("bf16") and is_full_precision("F32")
    assert not is_full_precision("Q8_0")
    assert not is_full_precision(None)


def _manifest(tmp_path: Path, payload) -> Path:
    path = tmp_path / "gguf_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


MANIFEST = {
    "schema_version": "gguf-manifest/v1",
    "source_model": "merged_16bit",
    "files": [
        {"filename": "nexus-Q4_K_M.gguf", "quant_type": "Q4_K_M", "sha256": "aa", "imatrix_used": True},
        {"filename": "nexus-Q8_0.gguf", "quant_type": "Q8_0", "sha256": "bb", "imatrix_used": False},
    ],
    "calibration": {"dataset": "calib.jsonl", "rows": 128},
}


def test_manifest_entry_matches_by_basename():
    entry = match_manifest_entry(MANIFEST, "/abs/path/to/nexus-Q8_0.gguf")
    assert entry == MANIFEST["files"][1]
    assert match_manifest_entry(MANIFEST, "other.gguf") is None


def test_manifest_entry_tolerates_alternative_layouts():
    by_name = {"artifacts": {"q4": {"filename": "dir/nexus-Q4_K_M.gguf", "quant_type": "Q4_K_M"}}}
    assert match_manifest_entry(by_name, "nexus-Q4_K_M.gguf")["quant_type"] == "Q4_K_M"
    # calibration is never treated as a file container
    calibration_only = {"calibration": {"imatrix": {"filename": "nexus-Q4_K_M.gguf"}}}
    assert match_manifest_entry(calibration_only, "nexus-Q4_K_M.gguf") is None


def test_load_manifest_details_copies_entry_and_calibration(tmp_path):
    details = load_manifest_details(_manifest(tmp_path, MANIFEST), "models/nexus-Q4_K_M.gguf")
    assert details["file"]["sha256"] == "aa"
    assert details["file"]["imatrix_used"] is True
    assert details["calibration"] == {"dataset": "calib.jsonl", "rows": 128}
    assert details["header"] == {"schema_version": "gguf-manifest/v1", "source_model": "merged_16bit"}
    assert details["warnings"] == []


def test_load_manifest_details_is_tolerant(tmp_path):
    details = load_manifest_details(_manifest(tmp_path, MANIFEST), "models/unknown.gguf")
    assert "file" not in details
    assert details["calibration"]["rows"] == 128
    assert "no manifest file entry" in details["warnings"][0]

    missing = load_manifest_details(tmp_path / "missing.json", "m.gguf")
    assert "could not read" in missing["warnings"][0]

    not_object = load_manifest_details(_manifest(tmp_path, [1, 2]), "m.gguf")
    assert "not a JSON object" in not_object["warnings"][0]


def test_build_model_artifact_sources(tmp_path):
    manifest = _manifest(tmp_path, MANIFEST)

    flag = build_model_artifact(model="nexus-Q4_K_M.gguf", backend="llamacpp", quantization="Q4_K_M")
    assert (flag["quantization"], flag["quantization_source"]) == ("Q4_K_M", "flag")
    assert "manifest" not in flag and "warnings" not in flag

    detected = build_model_artifact(model="nexus-Q4_K_M.gguf", backend="llamacpp")
    assert (detected["quantization"], detected["quantization_source"]) == ("Q4_K_M", "detected")

    none = build_model_artifact(model="final_model", backend="unsloth", load_settings={"load_in_4bit": True})
    assert (none["quantization"], none["quantization_source"]) == (None, None)
    assert none["load_settings"] == {"load_in_4bit": True}

    from_manifest = build_model_artifact(model="renamed.gguf", backend="llamacpp", manifest_path=manifest)
    assert from_manifest["quantization"] is None
    assert "no manifest file entry" in from_manifest["warnings"][0]

    matched = build_model_artifact(model="x/nexus-Q8_0.gguf", backend="llamacpp", manifest_path=manifest)
    assert (matched["quantization"], matched["quantization_source"]) == ("Q8_0", "manifest")
    assert matched["manifest"]["file"]["sha256"] == "bb"
    assert matched["manifest"]["calibration"]["dataset"] == "calib.jsonl"

    disagree = build_model_artifact(
        model="x/nexus-Q8_0.gguf", backend="llamacpp", quantization="Q4_K_M", manifest_path=manifest
    )
    assert disagree["quantization_source"] == "flag"
    assert "disagrees" in disagree["warnings"][0]
