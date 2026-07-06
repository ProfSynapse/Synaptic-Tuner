"""CPU tests for config parsing, arm resolution, resume, and smoke gating."""

import json

import pytest

from MechInterp import cell as cell_mod
from MechInterp.config import (
    SteerCellConfig,
    SurfaceConfig,
    LawConfig,
    ArmConfig,
    ExecutionConfig,
    ReadoutRef,
    SmokeConfig,
)


def _mk_config(tmp_path, arms):
    return SteerCellConfig(
        surface=SurfaceConfig(rows_path=str(tmp_path / "rows.jsonl"), seed=1),
        readouts=[ReadoutRef(name="axis", path="d.json")],
        law=LawConfig(kind="erase_write", readout="axis", position="anchor"),
        arms=arms,
        execution=ExecutionConfig(output_path=str(tmp_path / "out.jsonl")),
        smoke=SmokeConfig(n_rows=2),
    )


def _rows():
    return [
        {"row_key": "a", "selection_score": 1.5, "flagged": True},
        {"row_key": "b", "selection_score": 0.2, "flagged": False},
        {"row_key": "c", "selection_score": 1.1, "flagged": True},
        {"row_key": "d", "selection_score": 0.9, "flagged": False},
    ]


def test_law_readout_must_be_declared(tmp_path):
    with pytest.raises(ValueError):
        SteerCellConfig(
            surface=SurfaceConfig(rows_path="r.jsonl"),
            readouts=[ReadoutRef(name="axis", path="d.json")],
            law=LawConfig(kind="additive", readout="missing"),
            arms=[ArmConfig(name="baseline", strength=0.0)],
            execution=ExecutionConfig(output_path="o.jsonl"),
        )


def test_permuted_control_requires_seed():
    with pytest.raises(ValueError):
        ArmConfig(name="ctrl", permuted_control_of="primary")


def test_score_arm_requires_threshold():
    with pytest.raises(ValueError):
        ArmConfig(name="p", score_field="selection_score")


def test_baseline_arm_maps_all_rows_to_zero(tmp_path):
    cfg = _mk_config(tmp_path, [ArmConfig(name="baseline", strength=0.0)])
    strengths = cell_mod.resolve_arm_strengths(
        cfg.arms[0], _rows(), {a.name: a for a in cfg.arms}
    )
    assert set(strengths) == {"a", "b", "c", "d"}
    assert all(v == 0.0 for v in strengths.values())


def test_score_selection_arm_activates_above_threshold(tmp_path):
    arm = ArmConfig(name="primary", strength=2.0, score_field="selection_score", threshold=1.0)
    cfg = _mk_config(tmp_path, [arm])
    strengths = cell_mod.resolve_arm_strengths(arm, _rows(), {arm.name: arm})
    assert set(strengths) == {"a", "c"}
    assert all(v == 2.0 for v in strengths.values())


def test_flag_arm_activates_flagged_rows(tmp_path):
    arm = ArmConfig(name="dose", strength=1.0, flag_field="flagged")
    strengths = cell_mod.resolve_arm_strengths(arm, _rows(), {arm.name: arm})
    assert set(strengths) == {"a", "c"}


def test_permuted_control_is_count_matched_and_seeded(tmp_path):
    primary = ArmConfig(name="primary", strength=2.0, score_field="selection_score", threshold=1.0)
    control = ArmConfig(name="control", strength=2.0, permuted_control_of="primary", control_seed=7)
    arm_by_name = {a.name: a for a in (primary, control)}
    rows = _rows()
    s1 = cell_mod.resolve_arm_strengths(control, rows, arm_by_name)
    s2 = cell_mod.resolve_arm_strengths(control, rows, arm_by_name)
    # count matches primary's active count (2)
    assert len(s1) == 2
    # same seed -> same draw
    assert set(s1) == set(s2)


def test_permuted_control_seed_changes_draw(tmp_path):
    primary = ArmConfig(name="primary", strength=2.0, flag_field="flagged")
    # broaden the pool so different seeds can pick different rows
    rows = [{"row_key": k, "flagged": k in ("a", "b")} for k in "abcdefgh"]
    c7 = ArmConfig(name="c", strength=2.0, permuted_control_of="primary", control_seed=7)
    c9 = ArmConfig(name="c", strength=2.0, permuted_control_of="primary", control_seed=9)
    arm_by_name = {"primary": primary}
    s7 = cell_mod.resolve_arm_strengths(c7, rows, arm_by_name)
    s9 = cell_mod.resolve_arm_strengths(c9, rows, arm_by_name)
    assert len(s7) == len(s9) == 2
    # very likely different draws across 8 rows choosing 2
    assert set(s7) != set(s9)


def test_config_sha_is_stable(tmp_path):
    cfg = _mk_config(tmp_path, [ArmConfig(name="baseline", strength=0.0)])
    assert cell_mod.compute_config_sha(cfg) == cell_mod.compute_config_sha(cfg)


def test_config_sha_changes_with_content(tmp_path):
    a = _mk_config(tmp_path, [ArmConfig(name="baseline", strength=0.0)])
    b = _mk_config(tmp_path, [ArmConfig(name="baseline", strength=1.0)])
    assert cell_mod.compute_config_sha(a) != cell_mod.compute_config_sha(b)


def test_resume_skips_completed_rows(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"arm": "primary", "row_key": "a"}) + "\n")
    strengths = {"a": 2.0, "c": 2.0}
    pending = cell_mod.pending_rows(_rows(), strengths, "primary", out, resume=True)
    keys = {p["row_key"] for p in pending}
    assert "a" not in keys
    assert {"b", "c", "d"} <= keys


def test_resume_disabled_runs_all(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"arm": "primary", "row_key": "a"}) + "\n")
    pending = cell_mod.pending_rows(_rows(), {"a": 2.0}, "primary", out, resume=False)
    assert len(pending) == 4


def test_pending_rows_tag_strength_and_active(tmp_path):
    out = tmp_path / "out.jsonl"
    pending = cell_mod.pending_rows(_rows(), {"a": 2.0}, "primary", out, resume=True)
    by_key = {p["row_key"]: p for p in pending}
    assert by_key["a"]["_strength"] == 2.0 and by_key["a"]["_active"] is True
    assert by_key["b"]["_strength"] == 0.0 and by_key["b"]["_active"] is False


def test_smoke_state_roundtrip(tmp_path):
    out = tmp_path / "out.jsonl"
    assert not cell_mod.smoke_passed(out, "sha1")
    cell_mod.record_smoke(out, "sha1", {"passed": True})
    assert cell_mod.smoke_passed(out, "sha1")
    # different config sha does not count as passed
    assert not cell_mod.smoke_passed(out, "sha2")


def test_evaluate_smoke_readback_erase_write():
    class Cfg:
        write_rel_tol = 0.05
        write_abs_floor = 0.5
        offtarget_tol = 1e-3

    rb = {"commanded": [6.0, 6.0], "measured": [6.0, 6.01], "offtarget_abs_max": 0.0}
    verdict = cell_mod.evaluate_smoke_readback(rb, Cfg())
    assert verdict["passed"]


def test_evaluate_smoke_readback_fails_offtarget():
    class Cfg:
        write_rel_tol = 0.05
        write_abs_floor = 0.5
        offtarget_tol = 1e-3

    rb = {"commanded": [6.0], "measured": [6.0], "offtarget_abs_max": 1.0}
    verdict = cell_mod.evaluate_smoke_readback(rb, Cfg())
    assert not verdict["passed"] and not verdict["parity_ok"]
