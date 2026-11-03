"""Extraction interruption, source separation, portable names and probe integrity."""

import json

import pytest

from MechInterp.cli import _load_activation_matrix
from MechInterp.extraction import PositionSpec, extract_rows
from MechInterp.extraction.persistence import safe_key
from tests.mech_interp.test_batched_generation import _build_tiny_model, _build_tiny_tokenizer


def run_capture(root, rows, *, model=None, **kwargs):
    return extract_rows(
        model or _build_tiny_model(), _build_tiny_tokenizer(), rows,
        render_fn=lambda row: row["prompt"], content_end_fn=None,
        spec=PositionSpec(families=["anchor"], layers=[1, 2]),
        out_dir=root, forward_only=True, **kwargs,
    )


ROWS = [{"row_key": "source:one", "prompt": "hello world"},
        {"row_key": "source/two", "prompt": "world hello"}]


def test_interruption_resume_skips_durable_rows_and_matches_clean_capture(tmp_path):
    model = _build_tiny_model()
    original = model.forward
    count = 0

    def interrupted(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    model.forward = interrupted
    root = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="interruption"):
        run_capture(root, ROWS, model=model, resume=True, run_identity={"fixture": "v1"})
    assert len((root / "capture.runlog.jsonl").read_text().splitlines()) == 1
    assert not (root / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        _load_activation_matrix(str(root), "anchor", [{"row_key": ROWS[0]["row_key"], "label": 0}])

    count = 0
    model.forward = interrupted
    manifest = run_capture(root, ROWS, model=model, resume=True, run_identity={"fixture": "v1"})
    assert count == 1  # the committed first row was never evaluated again
    clean = run_capture(tmp_path / "clean", ROWS)
    assert manifest == clean
    for record in manifest["rows"]:
        for info in record["capture_files"].values():
            assert (root / info["file"]).read_bytes() == (tmp_path / "clean" / info["file"]).read_bytes()


def test_resume_refuses_different_patch_inputs_or_source_and_corrupt_files(tmp_path):
    root = tmp_path / "capture"
    run_capture(root, ROWS, resume=True, run_identity={"model": "v1"}, manifest_extra={"weight_patch": None})
    for rows, identity, patch in (
        (ROWS, {"model": "v2"}, None),
        ([{**ROWS[0], "prompt": "changed"}, ROWS[1]], {"model": "v1"}, None),
        (ROWS, {"model": "v1"}, {"patch_digest": "different"}),
    ):
        with pytest.raises(RuntimeError, match="fingerprint mismatch"):
            run_capture(root, rows, resume=True, run_identity=identity, manifest_extra={"weight_patch": patch})
    with pytest.raises(ValueError, match="non-empty"):
        run_capture(root, ROWS)
    artifact = next(root.glob("*.safetensors"))
    artifact.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_capture(root, ROWS, resume=True, run_identity={"model": "v1"}, manifest_extra={"weight_patch": None})


def test_probe_uses_manifest_and_checks_capture_bytes(tmp_path):
    root = tmp_path / "capture"
    manifest = run_capture(root, ROWS)
    labels = [{"row_key": row["row_key"], "label": i} for i, row in enumerate(ROWS)]
    labels.append({"row_key": "stray", "label": 1})
    (root / "stray__anchor.safetensors").write_bytes(b"not a capture")
    matrices, y = _load_activation_matrix(str(root), "anchor", labels)
    assert set(matrices) == {1, 2}
    assert y.tolist() == [0, 1]
    (root / manifest["rows"][0]["capture_files"]["anchor"]["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_activation_matrix(str(root), "anchor", labels)


def test_portable_names_and_key_validation_before_forward(tmp_path):
    keys = ["a/b", "a_b", "a::b", "a__b", "x:y", "../escape", "a\\b", "CON", "NUL", "🌻"]
    manifest = run_capture(tmp_path / "portable", [{"row_key": key, "prompt": "hello"} for key in keys])
    files = [row["capture_files"]["anchor"]["file"] for row in manifest["rows"]]
    assert len(set(files)) == len(keys)
    assert all(not any(char in name for char in '\\/:*?"<>|') for name in files)
    for keys in (["duplicate", "duplicate"], ["A", "a"], ["a/b", safe_key("a/b")]):
        with pytest.raises(ValueError, match="unique"):
            run_capture(tmp_path / "invalid", [{"row_key": key, "prompt": "hello"} for key in keys])
    with pytest.raises(ValueError, match="non-empty"):
        run_capture(tmp_path / "missing", [{"prompt": "hello"}])


def test_torn_journal_tail_is_recovered_and_completed_files_are_checked(tmp_path):
    root = tmp_path / "capture"
    run_capture(root, ROWS, resume=True, run_identity={"fixture": "v1"})
    with (root / "capture.runlog.jsonl").open("ab") as stream:
        stream.write(b'{"row_key":')
    run_capture(root, ROWS, resume=True, run_identity={"fixture": "v1"})
    assert len([json.loads(line) for line in (root / "capture.runlog.jsonl").read_text().splitlines()]) == 2
