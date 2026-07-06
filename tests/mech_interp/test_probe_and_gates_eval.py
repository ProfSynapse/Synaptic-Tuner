"""CPU tests for probe fitting, direction freezing, and the gate evaluator."""

import numpy as np
import pytest

from MechInterp.probe.fit import (
    fit_pca,
    cv_auroc,
    fit_full_probe,
    score_full_probe,
    sweep_layers,
    freeze_direction,
    load_frozen_direction,
)
from MechInterp.stats.evaluator import evaluate_gates
from MechInterp.extraction.capture import PositionSpec, resolve_capture_positions


def _separable(n=60, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (X @ w > 0).astype(int)
    # push classes apart along w so the probe has real signal
    X += (y[:, None] * 2 - 1) * w[None, :] * 1.5
    return X, y


def test_fit_pca_shapes():
    X, _ = _separable()
    mu, comps = fit_pca(X, n_components=8, seed=0)
    assert mu.shape == (16,)
    assert comps.shape[0] == 8 and comps.shape[1] == 16


def test_cv_auroc_recovers_signal():
    X, y = _separable(seed=1)
    mean_auc, std_auc, oof = cv_auroc(X, y, n_components=8, seed=1)
    assert mean_auc > 0.8
    assert not np.isnan(oof).any()


def test_full_probe_score_matches_direction():
    X, y = _separable(seed=2)
    fp = fit_full_probe(X, y, n_components=8, seed=2)
    s = score_full_probe(fp, X)
    # scores should separate the classes
    assert s[y == 1].mean() > s[y == 0].mean()


def test_sweep_layers_picks_best():
    X, y = _separable(seed=3)
    noise = np.random.default_rng(0).normal(size=X.shape)
    sweep = sweep_layers({0: X, 1: noise}, y, n_components=8, seed=3)
    assert sweep["best_layer"] == 0
    assert sweep["auroc_by_layer"][0] > sweep["auroc_by_layer"][1]


def test_freeze_and_load_direction(tmp_path):
    X, y = _separable(seed=4)
    out = tmp_path / "dir.json"
    rec = freeze_direction(X, y, layer=7, out_path=out, n_components=8, seed=4)
    assert rec["layer"] == 7
    assert rec["normalized"] is True
    loaded = load_frozen_direction(out)
    assert loaded["vector_np"].shape == (16,)
    assert abs(np.linalg.norm(loaded["vector_np"]) - 1.0) < 1e-4
    assert "sigma" in loaded and loaded["sigma"] > 0


def test_evaluate_gates_count_flips_pass():
    rows = [
        {"arm": "primary", "baseline_positive": True, "positive": False},
        {"arm": "primary", "baseline_positive": True, "positive": False},
        {"arm": "primary", "baseline_positive": True, "positive": True},
    ]
    config = {
        "gates": [
            {
                "name": "reach",
                "primitive": "count_flips",
                "arm": "primary",
                "before": "baseline_positive",
                "after": "positive",
                "pass_if": ">= 2",
            }
        ]
    }
    report = evaluate_gates(config, rows)
    assert report["gates"]["reach"]["value"] == 2
    assert report["overall_pass"]


def test_evaluate_gates_kill_diff_ci():
    rows = [{"arm": "primary", "killed": 1} for _ in range(12)]
    rows += [{"arm": "control", "killed": 0} for _ in range(12)]
    config = {
        "seed": 0,
        "n_boot": 300,
        "gates": [
            {
                "name": "spec",
                "primitive": "kill_diff_vs_control",
                "primary_arm": "primary",
                "control_arm": "control",
                "primary_indicator": "killed",
                "control_indicator": "killed",
                "pass_if_diff": ">= 5",
                "pass_if_ci_excludes_zero": True,
            }
        ],
    }
    report = evaluate_gates(config, rows)
    assert report["overall_pass"]
    assert report["gates"]["spec"]["value"]["ci_lo"] > 0


def test_evaluate_gates_overall_fail_when_one_fails():
    rows = [
        {"arm": "primary", "baseline_positive": True, "positive": True},
    ]
    config = {
        "gates": [
            {
                "name": "reach",
                "primitive": "count_flips",
                "arm": "primary",
                "before": "baseline_positive",
                "after": "positive",
                "pass_if": ">= 1",
            }
        ]
    }
    report = evaluate_gates(config, rows)
    assert not report["overall_pass"]


def test_resolve_capture_positions_families():
    spec = PositionSpec(families=["anchor", "first_visible", "answer_end", "every_k"], every_k=2)
    pos = resolve_capture_positions(spec, prompt_len=5, content_end=10, seq_total=11)
    assert pos["anchor"] == [4]
    assert pos["first_visible"] == [5]
    assert pos["answer_end"] == [10]
    assert pos["every_k"] == [5, 7, 9]


def test_resolve_capture_positions_short_completion_drops_out_of_range():
    spec = PositionSpec(families=["first_visible", "answer_end"])
    # completion produced nothing: content_end before prompt_len
    pos = resolve_capture_positions(spec, prompt_len=5, content_end=4, seq_total=5)
    assert pos["first_visible"] == []
    assert pos["answer_end"] == [4]
