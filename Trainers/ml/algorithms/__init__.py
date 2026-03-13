# Trainers/ml/algorithms/__init__.py
# Algorithm registry: ABC + decorator pattern for pluggable ML algorithms.
# Importing this module triggers registration of all bundled wrappers.

from .registry import AlgorithmWrapper, get_algorithm, list_algorithms
from . import lightgbm_wrapper  # Triggers @register_algorithm("lightgbm")

__all__ = ["AlgorithmWrapper", "get_algorithm", "list_algorithms"]
