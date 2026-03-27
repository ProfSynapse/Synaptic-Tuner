"""Tests for PivotRL functional equivalence verifier."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from functional_verifier import (
    _normalize_value,
    _normalize_tool_call,
    _extract_tool_call,
    _compare_args,
    functional_equivalence_reward,
)


# ---------------------------------------------------------------------------
# _normalize_value
# ---------------------------------------------------------------------------

class TestNormalizeValue:

    @pytest.mark.parametrize("inp,expected", [
        ("true", True), ("false", False), ("True", True), ("FALSE", False),
    ])
    def test_bool_coercion(self, inp, expected):
        assert _normalize_value(inp) is expected

    @pytest.mark.parametrize("inp,expected", [
        ("42", 42), ("3.14", 3.14), ("not_a_number", "not_a_number"),
    ])
    def test_numeric_coercion(self, inp, expected):
        result = _normalize_value(inp)
        if isinstance(expected, float):
            assert result == pytest.approx(expected)
        else:
            assert result == expected

    def test_path_normalization(self):
        assert _normalize_value("folder\\subfolder") == "folder/subfolder"

    def test_whitespace_strip(self):
        assert _normalize_value("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# _normalize_tool_call
# ---------------------------------------------------------------------------

class TestNormalizeToolCall:

    def test_sorts_keys(self):
        name, args = _normalize_tool_call("tool", {"b": 1, "a": 2})
        assert list(args.keys()) == ["a", "b"]

    def test_lowercases_name(self):
        name, _ = _normalize_tool_call("CreateFile", {})
        assert name == "createfile"


# ---------------------------------------------------------------------------
# _extract_tool_call
# ---------------------------------------------------------------------------

class TestExtractToolCall:

    def test_qwen_format(self):
        text = '<tool_call>{"name": "test", "arguments": {"key": "val"}}</tool_call>'
        result = _extract_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "test"
        assert args == {"key": "val"}

    def test_mistral_format(self):
        text = '[TOOL_CALLS] [{"name": "test", "arguments": {"key": "val"}}]'
        result = _extract_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "test"
        assert args == {"key": "val"}

    def test_plain_format(self):
        text = 'tool_call: test\narguments: {"key": "val"}'
        result = _extract_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "test"
        assert args == {"key": "val"}

    def test_no_tool_call(self):
        assert _extract_tool_call("Just some plain text, no tools here.") is None


# ---------------------------------------------------------------------------
# _compare_args
# ---------------------------------------------------------------------------

class TestCompareArgs:

    def test_exact_match(self):
        assert _compare_args({"a": 1, "b": "x"}, {"a": 1, "b": "x"}) == pytest.approx(1.0)

    def test_empty_both(self):
        assert _compare_args({}, {}) == pytest.approx(1.0)

    def test_partial_match(self):
        score = _compare_args({"a": 1, "b": 2}, {"a": 1, "b": 99})
        assert 0.0 < score < 1.0

    def test_no_match(self):
        score = _compare_args({"x": 1}, {"y": 2})
        # key_overlap = 0/2 = 0, so score = 0*0.4 = 0
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# functional_equivalence_reward
# ---------------------------------------------------------------------------

class TestFunctionalEquivalenceReward:

    def test_matching(self):
        completion = '<tool_call>{"name": "readFile", "arguments": {"path": "/a.txt"}}</tool_call>'
        scores = functional_equivalence_reward(
            [completion],
            ground_truth_tool=["readFile"],
            ground_truth_args_json=[json.dumps({"path": "/a.txt"})],
        )
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0)

    def test_wrong_tool(self):
        completion = '<tool_call>{"name": "writeFile", "arguments": {"path": "/a"}}</tool_call>'
        scores = functional_equivalence_reward(
            [completion],
            ground_truth_tool=["readFile"],
            ground_truth_args_json=[json.dumps({"path": "/a"})],
        )
        assert scores[0] == pytest.approx(0.0)

    def test_no_tool_call(self):
        scores = functional_equivalence_reward(
            ["I cannot do that."],
            ground_truth_tool=["readFile"],
            ground_truth_args_json=[json.dumps({"path": "/a"})],
        )
        assert scores[0] == pytest.approx(0.0)

    def test_no_ground_truth(self):
        completion = '<tool_call>{"name": "readFile", "arguments": {}}</tool_call>'
        scores = functional_equivalence_reward([completion])
        assert scores[0] == pytest.approx(0.0)

    def test_type_coercion(self):
        completion = '<tool_call>{"name": "setFlag", "arguments": {"verbose": "true", "count": "5"}}</tool_call>'
        scores = functional_equivalence_reward(
            [completion],
            ground_truth_tool=["setFlag"],
            ground_truth_args_json=[json.dumps({"verbose": True, "count": 5})],
        )
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0)
