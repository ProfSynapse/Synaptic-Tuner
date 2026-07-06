"""CPU tests for the lane-agnostic steer-cell logic: arm resolution, resume,
config sha, and the smoke state file."""

import json

import pytest

from MechInterp.config import SteerCellConfig, SmokeConfig
from MechInterp import cell as cell_mod


def _config():
    return SteerCellConfig(
        surface={"rows_path": "rows.jsonl", "seed": 3},
        readouts=[{"name": "axis", "path": "d.json"}],
        law={"kind": "erase_write", "readout": "axis", "position": "anchor"},
        arms=[
            {"name": "baseline", "strength": 0.0},
            {"name": "primary", "strength": 2.0, "score_field": "s", "threshold": 1.0},
            {
                "name": "control",
                "strength": 2.0,
                "permuted_control_of": "primary",
                "control_seed": 99,
            },
            {"name": "flagged_dose", "strength": 1.0, "flag_field": "flag"},
        ],
        execution={"output_path": "out/rows.jsonl"},
    )


def _rows():
    return [
        {"row_key": "a", "s": 1.5, "flag": True},
        {"row_key": "b", "s": 0.2, "flag": False},
        {"row_key": "c", "s": 1.1, "flag": True},
        {"row_key": "d", "s": 0.9, "flag": False},
    ]


def test_baseline_arm_maps_every_row_to_zero():
    cfg = _config()
    strengths = cell_mod.resolve_all_arms(cfg, _rows())
    assert strengths["baseline"] == {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}


def test_score_threshold_selects_passing_rows():
    cfg = _config()
    strengths = cell_mod.resolve_all_arms(cfg, _rows())
    assert set(strengths["primary"]) == {"a", "c"}  # s >= 1.0
    assert all(v == 2.0 for v in strengths["primary"].values())


def test_flag_field_selection():
    cfg = _config()
    strengths = cell_mod.resolve_all_arms(cfg, _rows())
    assert set(strengths["flagged_dose"]) == {"a", "c"}
    assert all(v == 1.0 for v in strengths["flagged_dose"].values())


def test_permuted_control_is_count_matched_and_seeded():
    cfg = _config()
    rows = _rows()
    s1 = cell_mod.resolve_all_arms(cfg, rows)["control"]
    s2 = cell_mod.resolve_all_arms(cfg, rows)["control"]
    # primary activates 2 rows -> control matches the count
    assert len(s1) == 2
    assert s1 == s2  # seeded determinism


def test_config_sha_is_deterministic_and_content_sensitive():
    cfg = _config()
    sha1 = cell_mod.compute_config_sha(cfg)
    sha2 = cell_mod.compute_config_sha(cfg)
    assert sha1 == sha2
    cfg2 = _config()
    cfg2.law.position = "final"
    assert cell_mod.compute_config_sha(cfg2) != sha1


def test_pending_rows_skips_completed_on_resume(tmp_path):
    out = tmp_path / "rows.jsonl"
    out.write_text(json.dumps({"arm": "primary", "row_key": "a"}) + "\n")
    cfg = _config()
    strengths = cell_mod.resolve_all_arms(cfg, _rows())["primary"]
    pending = cell_mod.pending_rows(_rows(), strengths, "primary", out, resume=True)
    keys = {r["row_key"] for r in pending}
    assert "a" not in keys  # already done
    assert {"b", "c", "d"} <= keys


def test_pending_rows_no_resume_runs_all(tmp_path):
    out = tmp_path / "rows.jsonl"
    out.write_text(json.dumps({"arm": "primary", "row_key": "a"}) + "\n")
    cfg = _config()
    strengths = cell_mod.resolve_all_arms(cfg, _rows())["primary"]
    pending = cell_mod.pending_rows(_rows(), strengths, "primary", out, resume=False)
    assert len(pending) == 4


def test_smoke_state_roundtrip(tmp_path):
    out = tmp_path / "rows.jsonl"
    sha = "abc123"
    assert cell_mod.smoke_passed(out, sha) is False
    cell_mod.record_smoke(out, sha, {"passed": True})
    assert cell_mod.smoke_passed(out, sha) is True
    # a different config sha must NOT count as a passed smoke
    assert cell_mod.smoke_passed(out, "different") is False


def test_evaluate_smoke_readback_erase_write_pass():
    rb = {"commanded": [6.0, 6.0], "measured": [6.0, 6.001], "offtarget_abs_max": 1e-5}
    verdict = cell_mod.evaluate_smoke_readback(rb, SmokeConfig())
    assert verdict["passed"] is True


def test_evaluate_smoke_readback_offtarget_fail():
    rb = {"commanded": [6.0], "measured": [6.0], "offtarget_abs_max": 1.0}
    verdict = cell_mod.evaluate_smoke_readback(rb, SmokeConfig())
    assert verdict["parity_ok"] is False
    assert verdict["passed"] is False


def test_evaluate_smoke_readback_write_error_fail():
    rb = {"commanded": [6.0], "measured": [2.0], "offtarget_abs_max": 0.0}
    verdict = cell_mod.evaluate_smoke_readback(rb, SmokeConfig())
    assert verdict["write_ok"] is False
    assert verdict["passed"] is False


def test_row_key_fallbacks():
    assert cell_mod.row_key_of({"row_key": "x"}) == "x"
    assert cell_mod.row_key_of({"id": "y"}) == "y"
    assert cell_mod.row_key_of({"key": "z"}) == "z"
    with pytest.raises(KeyError):
        cell_mod.row_key_of({"nope": 1})
