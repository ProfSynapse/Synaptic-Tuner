# Trainers/ml/algorithms/registry.py
# AlgorithmWrapper ABC and @register_algorithm decorator for pluggable algorithms.
# Strategy pattern: each wrapper knows how to create a scikit-learn estimator
# for classification or regression. Used by pipeline_builder.py.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sklearn.base import BaseEstimator


class AlgorithmWrapper(ABC):
    """Base class for all algorithm wrappers.

    Each wrapper knows how to create a scikit-learn compatible estimator
    for a given task type, and can report its capabilities and defaults.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry key (e.g., 'lightgbm')."""
        ...

    @property
    @abstractmethod
    def supports_classification(self) -> bool:
        ...

    @property
    @abstractmethod
    def supports_regression(self) -> bool:
        ...

    @property
    @abstractmethod
    def supports_gpu(self) -> bool:
        ...

    @abstractmethod
    def create_estimator(
        self,
        task_type: str,
        params: dict[str, Any],
        n_classes: int | None = None,
    ) -> BaseEstimator:
        """Create a configured scikit-learn estimator.

        Args:
            task_type: "classification" or "regression".
            params: User-provided hyperparameters from config.
                    Merged over defaults (user params win).
            n_classes: Number of target classes (classification only).
                       Used to select binary vs multiclass objective.

        Returns:
            A scikit-learn compatible estimator (implements fit/predict).

        Raises:
            ValueError: If task_type is unsupported by this algorithm.
        """
        ...

    @abstractmethod
    def get_default_params(
        self, task_type: str, n_classes: int | None = None,
    ) -> dict[str, Any]:
        """Return sensible default hyperparameters for this task type.

        Args:
            task_type: "classification" or "regression".
            n_classes: Number of target classes (classification only).
        """
        ...


# ---------------------------------------------------------------------------
# Registry: name -> wrapper class
# ---------------------------------------------------------------------------

_ALGORITHM_REGISTRY: dict[str, type[AlgorithmWrapper]] = {}


def register_algorithm(name: str):
    """Decorator to register an algorithm wrapper.

    Usage:
        @register_algorithm("lightgbm")
        class LightGBMWrapper(AlgorithmWrapper):
            ...
    """
    def decorator(cls: type[AlgorithmWrapper]) -> type[AlgorithmWrapper]:
        _ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator


def get_algorithm(name: str) -> AlgorithmWrapper:
    """Look up and instantiate an algorithm wrapper by name.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {available}"
        )
    return _ALGORITHM_REGISTRY[name]()


def list_algorithms() -> list[str]:
    """Return sorted list of registered algorithm names."""
    return sorted(_ALGORITHM_REGISTRY.keys())
