"""CPU tests for linear readout fitting and direction freezing (synthetic data)."""

import json

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


def _separable(n=80, d=16, sep=3.0, seed=0):
    """Two Gaussian blobs separated along axis 0; label 1 is the positive blob."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (np.arange(n) % 2).astype(int)
    X[y == 1, 0] += sep
    return X, y


def test_fit_pca_shapes():
    X, _ = _separable()
    mu, comps = fit_pca(X, n_components=8, seed=0)
    assert mu.shape == (X.shape[1],)
    assert comps.shape[0] == 8
    assert comps.shape[1] == X.shape[1]


def test_cv_auroc_separable_is_high_and_scores_every_row():
    X, y = _separable(sep=4.0)
    mean_auc, std_auc, oof = cv_auroc(X, y, n_components=8, n_splits=4, seed=0)
    assert mean_auc > 0.9
    assert not np.isnan(oof).any()  # every row got an out-of-fold score


def test_full_probe_score_separates_classes():
    X, y = _separable(sep=4.0)
    fp = fit_full_probe(X, y, n_components=8, seed=0)
    scores = score_full_probe(fp, X)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_sweep_layers_picks_the_separable_layer():
    X_good, y = _separable(sep=5.0, seed=1)
    rng = np.random.default_rng(2)
    X_noise = rng.standard_normal(X_good.shape)  # no signal
    sweep = sweep_layers({10: X_noise, 20: X_good}, y, n_components=8, n_splits=4, seed=0)
    assert sweep["best_layer"] == 20
    assert sweep["auroc_by_layer"][20] > sweep["auroc_by_layer"][10]


def test_freeze_direction_writes_self_describing_json(tmp_path):
    X, y = _separable(sep=4.0)
    out = tmp_path / "dir.json"
    record = freeze_direction(X, y, layer=20, out_path=out, n_components=8, seed=0)
    assert out.exists()
    assert record["layer"] == 20
    assert record["schema_version"] == "mechinterp-direction/v1"
    assert record["normalized"] is True
    # unit-norm when normalized
    assert np.isclose(np.linalg.norm(record["vector"]), 1.0, atol=1e-6)
    assert "sigma" in record and record["sigma"] > 0
    assert record["calibration"]["separation"] > 0


def test_load_frozen_direction_adds_numpy(tmp_path):
    X, y = _separable(sep=4.0)
    out = tmp_path / "dir.json"
    freeze_direction(X, y, layer=5, out_path=out, n_components=8, seed=0)
    loaded = load_frozen_direction(out)
    assert loaded["vector_np"].dtype == np.float32
    assert loaded["vector_np"].shape[0] == X.shape[1]


def test_bundled_example_direction_loads():
    from pathlib import Path

    p = (Path(__file__).resolve().parents[2] / "MechInterp" / "configs"
         / "templates" / "example_direction.json")
    loaded = load_frozen_direction(p)
    assert loaded["hidden_dim"] == loaded["vector_np"].shape[0]
    assert np.isclose(np.linalg.norm(loaded["vector_np"]), 1.0, atol=1e-4)
