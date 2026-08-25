"""Regression tests for terminal UI widgets."""

from math import inf, nan

from shared.ui.widgets import sparkline


def test_sparkline_marks_non_finite_metrics_without_crashing():
    assert sparkline([nan, 1.0, 2.0, inf]) == "·▁█·"


def test_sparkline_handles_all_non_finite_metrics():
    assert sparkline([nan, -inf]) == "··"
