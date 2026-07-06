"""
Linear readout fitting with dimensionality reduction and out-of-fold scoring.

The default recipe reduces each activation matrix with a randomized PCA (fit
label-agnostically) and fits a saga logistic classifier on the reduced features.
Out-of-fold scoring fits PCA and classifier on each training fold only, so the
reported AUROC never sees test-fold information. The full-data direction folds
the classifier weight back through the PCA basis into the original activation
space, giving a single vector that scores raw (mean-centered) activations by a
dot product.

The frozen direction is a self-describing JSON: the layer it was read at, the
vector, the mean offset, the class-projection statistics, and provenance fields.
This is the object the intervention engine and downstream readers consume.

The recipe is configurable: n_components, classifier solver / tol / C, and the
number of CV folds are all parameters, with PCA-k randomized + saga as defaults
because they fit high-dimensional activations quickly without a per-feature
scaler.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def fit_pca(X: np.ndarray, n_components: int = 128, seed: int = 0):
    """Fit a randomized PCA on mean-centered X (label-agnostic).

    Returns (mu, components) where mu is the feature mean and components has
    shape (k, d). k is clipped to min(n_components, n_samples - 1, n_features).
    """
    mu = X.mean(axis=0)
    k = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    pca.fit(X - mu)
    return mu, pca.components_


def _make_classifier(solver: str, tol: float, C: float, max_iter: int):
    return LogisticRegression(solver=solver, tol=tol, C=C, max_iter=max_iter)


def cv_auroc(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 128,
    n_splits: int = 5,
    seed: int = 0,
    solver: str = "saga",
    tol: float = 1e-3,
    C: float = 1.0,
    max_iter: int = 2000,
):
    """Out-of-fold AUROC. PCA and classifier are fit on train folds only.

    Returns (mean_auc, std_auc, oof_scores) where oof_scores are the held-out
    decision-function values for every row.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    aucs = []
    for tr, te in skf.split(X, y):
        mu = X[tr].mean(axis=0)
        k = min(n_components, len(tr) - 1, X.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
        ztr = pca.fit_transform(X[tr] - mu)
        zte = pca.transform(X[te] - mu)
        clf = _make_classifier(solver, tol, C, max_iter)
        clf.fit(ztr, y[tr])
        s = clf.decision_function(zte)
        oof[te] = s
        aucs.append(roc_auc_score(y[te], s))
    assert not np.isnan(oof).any(), "every row must receive an out-of-fold score"
    return float(np.mean(aucs)), float(np.std(aucs)), oof


def fit_full_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 128,
    seed: int = 0,
    solver: str = "saga",
    tol: float = 1e-3,
    C: float = 1.0,
    max_iter: int = 2000,
) -> dict:
    """Fit on all data and fold the classifier weight into activation space.

    Returns a dict with:
      coef      (d,) direction in raw activation space
      intercept scalar such that score(x) = x @ coef + intercept
      mu        feature mean used for centering
    The score is equivalent to the reduced-space logistic decision function.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    mu, comps = fit_pca(X, n_components=n_components, seed=seed)
    Z = (X - mu) @ comps.T
    clf = _make_classifier(solver, tol, C, max_iter)
    clf.fit(Z, y)
    coef_full = comps.T @ clf.coef_.ravel()
    intercept = float(clf.intercept_[0]) - float(mu @ coef_full)
    return {"coef": coef_full, "intercept": intercept, "mu": mu}


def score_full_probe(fp: dict, X: np.ndarray) -> np.ndarray:
    """Score rows with a full-data probe: X @ coef + intercept."""
    X = np.asarray(X, dtype=np.float64)
    return X @ fp["coef"] + fp["intercept"]


def sweep_layers(
    layer_activations: dict[int, np.ndarray],
    y: np.ndarray,
    n_components: int = 128,
    n_splits: int = 5,
    seed: int = 0,
    **clf_kwargs,
) -> dict:
    """Run cv_auroc per layer and return {layer: mean_auc} plus the best layer.

    layer_activations maps a layer index to its (n_rows, d) activation matrix.
    """
    surface = {}
    for layer, X in layer_activations.items():
        mean_auc, _, _ = cv_auroc(
            X, y, n_components=n_components, n_splits=n_splits, seed=seed, **clf_kwargs
        )
        surface[layer] = mean_auc
    best_layer = max(surface, key=surface.get)
    return {"auroc_by_layer": surface, "best_layer": best_layer}


def _projection_stats(scores: np.ndarray, y: np.ndarray) -> dict:
    """Class-conditioned statistics of the readout score."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    return {
        "positive_mean": float(pos.mean()) if len(pos) else float("nan"),
        "positive_std": float(pos.std()) if len(pos) else float("nan"),
        "negative_mean": float(neg.mean()) if len(neg) else float("nan"),
        "negative_std": float(neg.std()) if len(neg) else float("nan"),
        "sigma": float(scores.std()),
        "separation": (
            float(pos.mean() - neg.mean()) if len(pos) and len(neg) else float("nan")
        ),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
    }


def freeze_direction(
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    out_path: str | Path,
    n_components: int = 128,
    seed: int = 0,
    normalize: bool = True,
    provenance: Optional[dict] = None,
    **clf_kwargs,
) -> dict:
    """Fit a full-data direction at one layer and write it to a JSON file.

    The written JSON carries the unit (or raw) direction vector, the mean offset,
    the setpoint scale sigma (the score standard deviation), the class-projection
    statistics, the layer, and any provenance the caller supplies. This is the
    object the intervention engine loads to steer or to erase-and-write.
    """
    fp = fit_full_probe(X, y, n_components=n_components, seed=seed, **clf_kwargs)
    scores = score_full_probe(fp, X)
    stats = _projection_stats(scores, np.asarray(y, dtype=int))

    coef = np.asarray(fp["coef"], dtype=np.float64)
    norm = float(np.linalg.norm(coef))
    if normalize:
        if norm < 1e-12:
            raise ValueError("direction has near-zero norm; cannot normalize")
        vector = (coef / norm).astype(np.float64)
    else:
        vector = coef

    record = {
        "schema_version": "mechinterp-direction/v1",
        "layer": int(layer),
        "hidden_dim": int(coef.shape[0]),
        "normalized": bool(normalize),
        "vector": vector.tolist(),
        "raw_norm": norm,
        "intercept": float(fp["intercept"]),
        "mu": np.asarray(fp["mu"], dtype=np.float64).tolist(),
        "sigma": stats["sigma"],
        "calibration": stats,
        "recipe": {
            "n_components": int(n_components),
            "seed": int(seed),
            **{k: v for k, v in clf_kwargs.items()},
        },
        "provenance": provenance or {},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return record


def load_frozen_direction(path: str | Path) -> dict:
    """Load a frozen direction JSON, returning the record with numpy arrays.

    The returned dict adds a "vector_np" (float32) and "mu_np" (float64) for
    direct use by the intervention engine.
    """
    with open(path) as f:
        record = json.load(f)
    record["vector_np"] = np.asarray(record["vector"], dtype=np.float32)
    if "mu" in record and record["mu"] is not None:
        record["mu_np"] = np.asarray(record["mu"], dtype=np.float64)
    return record
