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


def test_config_sha_ignores_its_own_expected_value():
    """The drift guard (run_steer: expected_config_sha == compute_config_sha)
    must be satisfiable. Filling in surface.expected_config_sha with the sha
    computed from the unset config must not change the sha -- otherwise the
    field meant to hold "the expected hash of this config" would itself be an
    input to that hash, and no value written into it could ever match."""
    cfg = _config()
    unset_sha = cell_mod.compute_config_sha(cfg)

    cfg.surface.expected_config_sha = unset_sha
    assert cell_mod.compute_config_sha(cfg) == unset_sha

    cfg.surface.expected_config_sha = "not-the-right-sha"
    assert cell_mod.compute_config_sha(cfg) != cfg.surface.expected_config_sha


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


def _gain_config():
    return SteerCellConfig(
        surface={"rows_path": "rows.jsonl", "seed": 3},
        readouts=[{"name": "axis", "path": "d.json"}],
        law={"kind": "erase_write", "readout": "axis", "position": "anchor"},
        arms=[
            {"name": "baseline", "strength": 0.0},
            {"name": "coupled", "strength": 1.0, "gain_field": "prop_z", "gain_clip": 2.0},
            {
                "name": "permuted",
                "strength": 1.0,
                "permuted_control_of": "coupled",
                "control_seed": 20260706,
            },
            {"name": "ablate", "strength": 0.0, "force_active": True},
        ],
        execution={"output_path": "out/rows.jsonl"},
    )


def _gain_rows():
    return [
        {"row_key": "a", "prop_z": 0.5},
        {"row_key": "b", "prop_z": -3.0},  # clips to -2.0 at strength 1.0
        {"row_key": "c", "prop_z": 0.0},   # legit zero gain, still selected
        {"row_key": "d", "prop_z": 1.5},
        {"row_key": "e"},                  # no gain_field value -> not selected
    ]


def test_gain_arm_effective_strength_is_strength_times_field():
    cfg = _gain_config()
    strengths = cell_mod.resolve_all_arms(cfg, _gain_rows())["coupled"]
    assert strengths["a"] == pytest.approx(0.5)
    assert strengths["d"] == pytest.approx(1.5)
    assert strengths["c"] == pytest.approx(0.0)
    assert "e" not in strengths  # row without the gain_field is not selected


def test_gain_arm_clip_is_symmetric_and_sign_preserving():
    cfg = _gain_config()
    strengths = cell_mod.resolve_all_arms(cfg, _gain_rows())["coupled"]
    # prop_z=-3.0 * strength 1.0 = -3.0, clipped to -2.0 (sign preserved, magnitude clamped)
    assert strengths["b"] == pytest.approx(-2.0)


def test_gain_arm_clip_scales_with_negative_strength_alpha():
    cfg = _gain_config()
    cfg.arms[1].strength = -2.0
    strengths = cell_mod.resolve_all_arms(cfg, _gain_rows())["coupled"]
    # -2.0 * 1.5 = -3.0, clipped to -2.0; sign flips relative to the positive-alpha case
    assert strengths["d"] == pytest.approx(-2.0)
    # -2.0 * 0.5 = -1.0, within clip, sign flipped
    assert strengths["a"] == pytest.approx(-1.0)


def test_permuted_gain_is_seed_stable():
    cfg = _gain_config()
    rows = _gain_rows()
    s1 = cell_mod.resolve_all_arms(cfg, rows)["permuted"]
    s2 = cell_mod.resolve_all_arms(cfg, rows)["permuted"]
    assert s1 == s2


def test_permuted_gain_preserves_value_multiset_but_scrambles_pairing():
    cfg = _gain_config()
    rows = _gain_rows()
    coupled = cell_mod.resolve_all_arms(cfg, rows)["coupled"]
    permuted = cell_mod.resolve_all_arms(cfg, rows)["permuted"]
    # same row population (only rows carrying the gain_field are selected)
    assert set(permuted) == set(coupled)
    # identical multiset of values...
    assert sorted(permuted.values()) == pytest.approx(sorted(coupled.values()))
    # ...but the seeded shuffle actually rearranges at least one row's gain
    # (this seed/data combination is not a fixed point of the permutation)
    assert permuted != coupled


def test_permuted_gain_different_seed_gives_different_arrangement():
    cfg = _gain_config()
    cfg.arms[2].control_seed = 1
    rows = _gain_rows()
    permuted_a = cell_mod.resolve_all_arms(cfg, rows)["permuted"]
    cfg.arms[2].control_seed = 2
    permuted_b = cell_mod.resolve_all_arms(cfg, rows)["permuted"]
    assert permuted_a != permuted_b
    assert sorted(permuted_a.values()) == pytest.approx(sorted(permuted_b.values()))


def test_ablate_writes_zero_vs_baseline_is_true_noop():
    cfg = _gain_config()
    rows = _gain_rows()
    strengths_by_arm = cell_mod.resolve_all_arms(cfg, rows)

    baseline_pending = cell_mod.pending_rows(
        rows, strengths_by_arm["baseline"], "baseline", "out/baseline.jsonl", resume=False,
        write_at_zero=False,
    )
    ablate_pending = cell_mod.pending_rows(
        rows, strengths_by_arm["ablate"], "ablate", "out/ablate.jsonl", resume=False,
        write_at_zero=True,
    )
    # both arms resolve every row to strength 0.0...
    assert all(r["_strength"] == 0.0 for r in baseline_pending)
    assert all(r["_strength"] == 0.0 for r in ablate_pending)
    # ...but baseline is a true no-op (never active) while ablate applies the
    # law at every row (active despite the zero value).
    assert all(r["_active"] is False for r in baseline_pending)
    assert all(r["_active"] is True for r in ablate_pending)


def test_gain_arm_write_at_zero_keeps_zero_gain_row_active():
    # a gain arm's own row at a legitimately-computed zero gain (row "c") must
    # still be marked active by the caller's write_at_zero policy, mirroring
    # the couple-with-g-equals-0 nesting in the amendment.
    cfg = _gain_config()
    rows = _gain_rows()
    strengths = cell_mod.resolve_all_arms(cfg, rows)["coupled"]
    pending = cell_mod.pending_rows(
        rows, strengths, "coupled", "out/coupled.jsonl", resume=False, write_at_zero=True,
    )
    by_key = {r["row_key"]: r for r in pending}
    assert by_key["c"]["_strength"] == pytest.approx(0.0)
    assert by_key["c"]["_active"] is True
    assert by_key["e"]["_active"] is False  # never selected at all


def test_row_key_fallbacks():
    assert cell_mod.row_key_of({"row_key": "x"}) == "x"
    assert cell_mod.row_key_of({"id": "y"}) == "y"
    assert cell_mod.row_key_of({"key": "z"}) == "z"
    with pytest.raises(KeyError):
        cell_mod.row_key_of({"nope": 1})
