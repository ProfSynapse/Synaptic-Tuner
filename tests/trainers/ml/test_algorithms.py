"""Tests for Trainers/ml/algorithms/ — registry and LightGBM wrapper."""
from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator

from Trainers.ml.algorithms import get_algorithm, list_algorithms
from Trainers.ml.algorithms.lightgbm_wrapper import LightGBMWrapper
from Trainers.ml.algorithms.registry import AlgorithmWrapper


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_lightgbm_registered(self):
        assert "lightgbm" in list_algorithms()

    def test_get_algorithm_returns_wrapper(self):
        wrapper = get_algorithm("lightgbm")
        assert isinstance(wrapper, AlgorithmWrapper)
        assert isinstance(wrapper, LightGBMWrapper)

    def test_get_unknown_algorithm_raises(self):
        with pytest.raises(KeyError, match="Unknown algorithm 'xgboost'"):
            get_algorithm("xgboost")

    def test_list_algorithms_sorted(self):
        algs = list_algorithms()
        assert algs == sorted(algs)


# ---------------------------------------------------------------------------
# LightGBM Wrapper
# ---------------------------------------------------------------------------

class TestLightGBMWrapper:
    @pytest.fixture
    def wrapper(self):
        return LightGBMWrapper()

    def test_name(self, wrapper):
        assert wrapper.name == "lightgbm"

    def test_supports_classification(self, wrapper):
        assert wrapper.supports_classification is True

    def test_supports_regression(self, wrapper):
        assert wrapper.supports_regression is True

    def test_supports_gpu(self, wrapper):
        assert wrapper.supports_gpu is True

    def test_create_classifier(self, wrapper):
        est = wrapper.create_estimator("classification", {})
        assert isinstance(est, BaseEstimator)
        from lightgbm import LGBMClassifier
        assert isinstance(est, LGBMClassifier)

    def test_create_regressor(self, wrapper):
        est = wrapper.create_estimator("regression", {})
        assert isinstance(est, BaseEstimator)
        from lightgbm import LGBMRegressor
        assert isinstance(est, LGBMRegressor)

    def test_user_params_override_defaults(self, wrapper):
        est = wrapper.create_estimator("classification", {"n_estimators": 100})
        assert est.get_params()["n_estimators"] == 100

    def test_default_params_applied(self, wrapper):
        est = wrapper.create_estimator("classification", {})
        assert est.get_params()["n_estimators"] == 500
        assert est.get_params()["learning_rate"] == 0.05

    def test_unsupported_task_type_raises(self, wrapper):
        with pytest.raises(ValueError, match="Unsupported task type"):
            wrapper.create_estimator("clustering", {})

    def test_get_default_params_classification(self, wrapper):
        params = wrapper.get_default_params("classification")
        assert params["objective"] == "binary"
        assert "n_estimators" in params

    def test_get_default_params_regression(self, wrapper):
        params = wrapper.get_default_params("regression")
        assert params["objective"] == "regression"

    def test_get_default_params_multiclass(self, wrapper):
        params = wrapper.get_default_params("classification", n_classes=4)
        assert params["objective"] == "multiclass"
        assert params["metric"] == "multi_logloss"
        assert params["num_class"] == 4

    def test_get_default_params_binary_explicit(self, wrapper):
        params = wrapper.get_default_params("classification", n_classes=2)
        assert params["objective"] == "binary"
        assert params["metric"] == "binary_logloss"

    def test_create_multiclass_estimator(self, wrapper):
        est = wrapper.create_estimator("classification", {}, n_classes=3)
        from lightgbm import LGBMClassifier
        assert isinstance(est, LGBMClassifier)
        assert est.get_params()["objective"] == "multiclass"
        assert est.get_params()["num_class"] == 3
