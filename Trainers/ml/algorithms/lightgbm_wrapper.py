# Trainers/ml/algorithms/lightgbm_wrapper.py
# LightGBM algorithm wrapper — Phase 1's sole algorithm.
# Supports both classification and regression tasks.
# Registered via @register_algorithm("lightgbm") decorator.

from typing import Any

from sklearn.base import BaseEstimator

from .registry import AlgorithmWrapper, register_algorithm


@register_algorithm("lightgbm")
class LightGBMWrapper(AlgorithmWrapper):
    """LightGBM wrapper supporting classification and regression.

    Creates LGBMClassifier or LGBMRegressor based on task_type.
    User params are merged over sensible defaults.
    """

    @property
    def name(self) -> str:
        return "lightgbm"

    @property
    def supports_classification(self) -> bool:
        return True

    @property
    def supports_regression(self) -> bool:
        return True

    @property
    def supports_gpu(self) -> bool:
        return True

    def get_default_params(
        self, task_type: str, n_classes: int | None = None,
    ) -> dict[str, Any]:
        """Return sensible defaults for LightGBM.

        Classification uses binary crossentropy for 2 classes, multiclass
        softmax for >2. Regression uses RMSE.
        """
        base = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "random_state": 42,
            "verbosity": -1,
        }
        if task_type == "classification":
            if n_classes is not None and n_classes > 2:
                base["objective"] = "multiclass"
                base["metric"] = "multi_logloss"
                base["num_class"] = n_classes
            else:
                base["objective"] = "binary"
                base["metric"] = "binary_logloss"
        else:
            base["objective"] = "regression"
            base["metric"] = "rmse"
        return base

    def create_estimator(
        self, task_type: str, params: dict[str, Any],
        n_classes: int | None = None,
    ) -> BaseEstimator:
        """Create a LightGBM estimator for the given task type.

        Args:
            task_type: "classification" or "regression".
            params: User overrides merged over defaults.
            n_classes: Number of target classes (classification only).

        Returns:
            LGBMClassifier or LGBMRegressor.

        Raises:
            ValueError: If task_type is not supported.
        """
        from lightgbm import LGBMClassifier, LGBMRegressor

        if task_type not in ("classification", "regression"):
            raise ValueError(f"Unsupported task type: {task_type}")

        merged = {**self.get_default_params(task_type, n_classes), **params}

        if task_type == "classification":
            return LGBMClassifier(**merged)
        return LGBMRegressor(**merged)
