"""CPU tests for the declarative gates.yaml evaluator."""

import pytest

from MechInterp.stats.evaluator import evaluate_gates, _parse_comparison


def _rows():
    # primary: 3 baseline-positive rows flip to negative (killed); control: 1.
    rows = []
    for k in range(3):
        rows.append({"arm": "primary", "row_key": f"p{k}",
                     "baseline_positive": True, "positive": False, "killed": 1})
    for k in range(2):
        rows.append({"arm": "primary", "row_key": f"pn{k}",
                     "baseline_positive": True, "positive": True, "killed": 0})
    for k in range(1):
        rows.append({"arm": "control", "row_key": f"c{k}",
                     "baseline_positive": True, "positive": False, "killed": 1})
    for k in range(4):
        rows.append({"arm": "control", "row_key": f"cn{k}",
                     "baseline_positive": True, "positive": True, "killed": 0})
    return rows


def test_parse_comparison():
    op, thr = _parse_comparison(">= 5")
    assert op(5, 5) and not op(4, 5)
    assert thr == 5.0
    with pytest.raises(ValueError):
        _parse_comparison("bad")


def test_count_flips_gate_pass():
    cfg = {
        "gates": [
            {"name": "reach", "primitive": "count_flips", "arm": "primary",
             "before": "baseline_positive", "after": "positive",
             "from_state": True, "to_state": False, "pass_if": ">= 3"}
        ]
    }
    report = evaluate_gates(cfg, _rows())
    assert report["gates"]["reach"]["value"] == 3
    assert report["gates"]["reach"]["passed"] is True
    assert report["overall_pass"] is True


def test_count_flips_gate_fail():
    cfg = {
        "gates": [
            {"name": "reach", "primitive": "count_flips", "arm": "primary",
             "before": "baseline_positive", "after": "positive",
             "pass_if": ">= 10"}
        ]
    }
    report = evaluate_gates(cfg, _rows())
    assert report["gates"]["reach"]["passed"] is False
    assert report["overall_pass"] is False


def test_kill_diff_gate_with_ci():
    cfg = {
        "seed": 1,
        "n_boot": 300,
        "gates": [
            {"name": "specificity", "primitive": "kill_diff_vs_control",
             "primary_arm": "primary", "control_arm": "control",
             "primary_indicator": "killed", "control_indicator": "killed",
             "pass_if_diff": ">= 1"}
        ],
    }
    report = evaluate_gates(cfg, _rows())
    # primary killed 3, control killed 1 -> diff 2 >= 1
    assert report["gates"]["specificity"]["value"]["diff"] == pytest.approx(2.0)
    assert report["gates"]["specificity"]["passed"] is True


def test_auroc_floor_gate():
    rows = []
    for i, (lab, sc) in enumerate([(0, 0.1), (0, 0.2), (0, 0.3),
                                   (1, 0.7), (1, 0.8), (1, 0.9)]):
        rows.append({"arm": "primary", "row_key": f"r{i}",
                     "label": lab, "readout_score": sc})
    cfg = {
        "seed": 0,
        "n_boot": 200,
        "gates": [
            {"name": "readout", "primitive": "auroc_floor", "arm": "primary",
             "label": "label", "score": "readout_score",
             "pass_if_auroc": ">= 0.9"}
        ],
    }
    report = evaluate_gates(cfg, rows)
    assert report["gates"]["readout"]["value"]["auroc"] == pytest.approx(1.0)
    assert report["gates"]["readout"]["passed"] is True


def test_overall_pass_requires_all_gates():
    cfg = {
        "gates": [
            {"name": "g1", "primitive": "count_flips", "arm": "primary",
             "before": "baseline_positive", "after": "positive", "pass_if": ">= 3"},
            {"name": "g2", "primitive": "count_flips", "arm": "primary",
             "before": "baseline_positive", "after": "positive", "pass_if": ">= 99"},
        ]
    }
    report = evaluate_gates(cfg, _rows())
    assert report["gates"]["g1"]["passed"] is True
    assert report["gates"]["g2"]["passed"] is False
    assert report["overall_pass"] is False


def test_unknown_primitive_raises():
    cfg = {"gates": [{"name": "x", "primitive": "nope"}]}
    with pytest.raises(ValueError):
        evaluate_gates(cfg, _rows())
