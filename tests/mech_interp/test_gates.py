"""CPU tests for gate primitives against hand-computed fixtures."""

import numpy as np
import pytest

from MechInterp.stats.gates import (
    count_flips,
    kill_diff_vs_control,
    permutation_p,
    auroc_floor,
    hanley_mcneil_se,
    _roc_auc,
)


def test_count_flips_true_to_false():
    before = [True, True, True, False]
    after = [False, True, False, False]
    # rows 0 and 2 moved True->False
    assert count_flips(before, after) == 2


def test_count_flips_custom_states():
    before = [False, False, True]
    after = [True, False, True]
    # count False->True: only row 0
    assert count_flips(before, after, from_state=False, to_state=True) == 1


def test_count_flips_length_mismatch():
    with pytest.raises(ValueError):
        count_flips([True], [True, False])


def test_kill_diff_point_estimate():
    primary = [1, 1, 1, 0, 0]  # 3 positives
    control = [1, 0, 0, 0, 0]  # 1 positive
    res = kill_diff_vs_control(primary, control, seed=0, n_boot=200)
    assert res["diff"] == pytest.approx(2.0)
    assert res["ci_lo"] <= res["diff"] <= res["ci_hi"]


def test_kill_diff_ci_excludes_zero_when_strongly_separated():
    primary = [1] * 20
    control = [0] * 20
    res = kill_diff_vs_control(primary, control, seed=1, n_boot=500)
    assert res["diff"] == pytest.approx(20.0)
    assert res["ci_lo"] > 0


def test_kill_diff_reproducible_by_seed():
    primary = [1, 0, 1, 0, 1, 0]
    control = [0, 1, 0, 1, 0, 1]
    a = kill_diff_vs_control(primary, control, seed=42, n_boot=300)
    b = kill_diff_vs_control(primary, control, seed=42, n_boot=300)
    assert a["ci_lo"] == b["ci_lo"] and a["ci_hi"] == b["ci_hi"]


def test_roc_auc_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert _roc_auc(labels, scores) == pytest.approx(1.0)


def test_roc_auc_reversed_is_zero():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert _roc_auc(labels, scores) == pytest.approx(0.0)


def test_roc_auc_ties_give_half():
    # all identical scores -> AUROC 0.5 (Mann-Whitney with all ties)
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert _roc_auc(labels, scores) == pytest.approx(0.5)


def test_hanley_mcneil_se_known_shape():
    # SE must be positive and shrink as sample size grows
    se_small = hanley_mcneil_se(0.8, 10, 10)
    se_large = hanley_mcneil_se(0.8, 100, 100)
    assert se_small > se_large > 0


def test_auroc_floor_perfect():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    res = auroc_floor(labels, scores, seed=0, n_boot=200)
    assert res["auroc"] == pytest.approx(1.0)
    assert res["ci_lb"] <= 1.0
    assert res["n_pos"] == 3 and res["n_neg"] == 3


def test_permutation_p_strong_effect_is_small():
    # primary took the 5 positives; pool has only 5 positives among 20
    pool = [1] * 5 + [0] * 15
    res = permutation_p(primary_positive=5, pool_ind=pool, n_primary=5, seed=0, n_perm=500)
    # matching 5/5 by chance is rare
    assert res["p_value"] < 0.05


def test_permutation_p_null_effect_is_large():
    pool = [1] * 10 + [0] * 10
    res = permutation_p(primary_positive=5, pool_ind=pool, n_primary=10, seed=0, n_perm=500)
    # expected positives in a draw of 10 is ~5, so p should be well above 0.05
    assert res["p_value"] > 0.2


def test_permutation_p_never_zero():
    pool = [1] * 5 + [0] * 5
    res = permutation_p(primary_positive=5, pool_ind=pool, n_primary=5, seed=0, n_perm=100)
    assert res["p_value"] > 0.0
