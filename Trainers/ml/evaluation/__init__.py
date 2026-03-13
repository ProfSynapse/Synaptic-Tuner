# Trainers/ml/evaluation/__init__.py
# Evaluation: metric computation, report generation, and optional plots.

from .metrics import evaluate_model
from .report import build_metrics_report

__all__ = ["evaluate_model", "build_metrics_report"]
