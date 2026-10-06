"""Calibration metrics (shared/ml/calibration_metrics.py) against hand-computed values."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.ml.calibration_metrics import (  # noqa: E402
    accuracy,
    brier,
    calibration_report,
    expected_calibration_error,
    fit_temperature,
    grouped_reports,
    nll,
    softmax,
)

APPROX = 1e-9


def test_accuracy_nll_brier_on_ragged_rows():
    probs = [[0.8, 0.2], [0.1, 0.3, 0.6], [0.5, 0.25, 0.25]]
    labels = [0, 1, 0]
    # argmax: 0 (right), 2 (wrong), 0 (right)
    assert accuracy(probs, labels) == pytest.approx(2 / 3, abs=APPROX)
    expected_nll = -(math.log(0.8) + math.log(0.3) + math.log(0.5)) / 3
    assert nll(probs, labels) == pytest.approx(expected_nll, abs=APPROX)
    # row0: (0.8-1)^2 + 0.2^2 = 0.08; row1: 0.01 + 0.49 + 0.36 = 0.86; row2: 0.25 + 0.0625*2 = 0.375
    assert brier(probs, labels) == pytest.approx((0.08 + 0.86 + 0.375) / 3, abs=APPROX)


def test_ece_two_bins_hand_computed():
    # confidences 0.9 (right), 0.9 (wrong), 0.6 (right), 0.6 (right); 2 bins split at 0.5
    probs = [[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]]
    labels = [0, 1, 0, 0]
    # all four fall in bin (0.5, 1.0]: acc 0.75, mean conf 0.75 -> ECE 0
    assert expected_calibration_error(probs, labels, n_bins=2) == pytest.approx(0.0, abs=APPROX)
    # 10 bins: bin(0.8,0.9] acc .5 conf .9 weight .5 -> .2; bin(0.5,0.6] acc 1 conf .6 weight .5 -> .2
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.4, abs=APPROX)


def test_softmax_temperature():
    p = softmax([2.0, 0.0], temperature=2.0)
    assert p[0] == pytest.approx(math.exp(1) / (math.exp(1) + 1), abs=APPROX)
    with pytest.raises(ValueError):
        softmax([1.0], temperature=0.0)


def test_fit_temperature_recovers_generating_temperature():
    rng = random.Random(0)
    true_t = 2.5
    logits, labels = [], []
    for _ in range(3000):
        k = rng.choice([2, 3, 5])
        row = [rng.gauss(0, 3) for _ in range(k)]
        p = softmax(row, true_t)
        labels.append(rng.choices(range(k), weights=p)[0])
        logits.append(row)
    fitted = fit_temperature(logits, labels)
    assert fitted == pytest.approx(true_t, rel=0.15)
    # Fitting can only lower held-in NLL relative to T=1.
    assert nll([softmax(r, fitted) for r in logits], labels) <= nll([softmax(r) for r in logits], labels)


def test_fit_temperature_empty_is_identity():
    assert fit_temperature([], []) == 1.0


def test_report_and_groups():
    probs = [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]
    labels = [0, 1, 0]
    rep = calibration_report(probs, labels)
    assert rep.n == 3 and rep.accuracy == pytest.approx(2 / 3)
    groups = grouped_reports(probs, labels, ["a", "a", "b"])
    assert set(groups) == {"a", "b"}
    assert groups["a"]["accuracy"] == 1.0 and groups["b"]["accuracy"] == 0.0
    with pytest.raises(ValueError):
        calibration_report(probs, labels[:2])


# ---- confidence-quality + conformal tools (confidence analysis) --------------

from shared.ml.calibration_metrics import (  # noqa: E402
    aurc,
    auroc,
    confident_wrong_rate,
    conformal_lac_qhat,
    lac_set,
    shortest_ordinal_interval,
    underconfident_right_rate,
)


def test_auroc_hand_computed_with_ties():
    # pos scores {0.9, 0.5}, neg {0.5, 0.1}: pairs (0.9>0.5)=1,(0.9>0.1)=1,(0.5=0.5)=.5,(0.5>0.1)=1 -> 3.5/4
    assert auroc([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1]) == pytest.approx(0.875)
    assert math.isnan(auroc([1, 1], [0.2, 0.3]))


def test_aurc_perfect_vs_inverted_ranking():
    correct = [1, 1, 0, 0]
    good = aurc([0.9, 0.8, 0.2, 0.1], correct)   # risks 0,0,1/3,1/2 -> mean 0.2083
    bad = aurc([0.1, 0.2, 0.8, 0.9], correct)    # risks 1,1,2/3,1/2 -> mean 0.7917
    assert good == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert bad == pytest.approx((1 + 1 + 2 / 3 + 0.5) / 4)


def test_confident_wrong_and_underconfident_right_are_shares_of_all_rows():
    conf = [0.95, 0.9, 0.3, 0.6]
    correct = [0, 1, 1, 0]
    assert confident_wrong_rate(conf, correct, 0.8) == pytest.approx(0.25)
    assert underconfident_right_rate(conf, correct, 0.5) == pytest.approx(0.25)


def test_lac_set_never_empty_and_contains_threshold_members():
    # member iff 1 - p <= qhat, i.e. p >= 1 - qhat
    assert lac_set([0.5, 0.3, 0.2], qhat=0.85) == [0, 1, 2]
    assert lac_set([0.5, 0.3, 0.2], qhat=0.75) == [0, 1]
    assert lac_set([0.4, 0.35, 0.25], qhat=0.1) == [0]   # nothing passes -> argmax kept


def test_conformal_coverage_holds_on_exchangeable_rows():
    rng = random.Random(1)

    def draw(n):
        rows = []
        for _ in range(n):
            k = rng.choice([3, 4, 6])
            p = softmax([rng.gauss(0, 2) for _ in range(k)])
            y = rng.choices(range(k), weights=p)[0]   # labels drawn from the model: calibrated
            rows.append((p, y))
        return rows

    cal, test = draw(2000), draw(4000)
    for alpha in (0.1, 0.2):
        q = conformal_lac_qhat([p[y] for p, y in cal], alpha)
        cov = sum(y in lac_set(p, q) for p, y in test) / len(test)
        assert cov >= 1 - alpha - 0.02, (alpha, cov)


def test_shortest_ordinal_interval():
    assert shortest_ordinal_interval([0.05, 0.6, 0.3, 0.05], 0.8) == (1, 2)
    assert shortest_ordinal_interval([0.05, 0.9, 0.05], 0.8) == (1, 1)
    assert shortest_ordinal_interval([0.25, 0.25, 0.25, 0.25], 0.8) == (0, 3)
