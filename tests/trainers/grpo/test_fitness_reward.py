"""Tests for the fitness_reward function in GRPO rewards.

Validates that FitnessEvaluator integration works correctly as a
GRPO reward signal for structural correctness of tool calls.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add repo root to path so shared modules are importable
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# Also add GRPO src so rewards module is importable directly
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from rewards import fitness_reward, build_fitness_reward, _coerce_to_text


# --- Valid Qwen-format tool call output for testing ---
VALID_TOOL_CALL = (
    '<tool_call>\n'
    '{"name": "useTools", "arguments": '
    '{"context": {"sessionId": "abc123", "workspaceId": "ws1"}, '
    '"calls": [{"agent": "fileManager", "tool": "readFile", '
    '"params": {"path": "/readme.md"}}]}}\n'
    '</tool_call>'
)

# Output with no tool call at all
NO_TOOL_CALL = "I can help you with that! Let me think about it."

# Malformed tool call (invalid JSON inside tags)
MALFORMED_TOOL_CALL = '<tool_call>\n{this is not valid json}\n</tool_call>'


class TestFitnessReward:
    """Tests for the fitness_reward() function."""

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_valid_tool_call_returns_high_score(self, mock_evaluator_cls):
        """Valid tool call output should score 1.0."""
        mock_result = MagicMock()
        mock_result.score = 1.0
        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = mock_result
        mock_evaluator_cls.return_value = mock_instance

        score = fitness_reward(VALID_TOOL_CALL, config_path="dummy.yaml")

        assert score == 1.0
        mock_evaluator_cls.assert_called_once_with(config_path="dummy.yaml")
        mock_instance.evaluate.assert_called_once_with(VALID_TOOL_CALL)

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_no_tool_call_returns_zero(self, mock_evaluator_cls):
        """Output with no tool calls should score 0.0."""
        mock_result = MagicMock()
        mock_result.score = 0.0
        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = mock_result
        mock_evaluator_cls.return_value = mock_instance

        score = fitness_reward(NO_TOOL_CALL, config_path="dummy.yaml")

        assert score == 0.0

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_partial_validity_returns_intermediate_score(self, mock_evaluator_cls):
        """Partially valid output should return intermediate score."""
        mock_result = MagicMock()
        mock_result.score = 0.6
        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = mock_result
        mock_evaluator_cls.return_value = mock_instance

        score = fitness_reward(MALFORMED_TOOL_CALL, config_path="dummy.yaml")

        assert score == 0.6

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_exception_returns_zero(self, mock_evaluator_cls):
        """Exceptions during evaluation should return 0.0 gracefully."""
        mock_evaluator_cls.side_effect = RuntimeError("Config not found")

        score = fitness_reward(VALID_TOOL_CALL, config_path="nonexistent.yaml")

        assert score == 0.0

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_evaluate_exception_returns_zero(self, mock_evaluator_cls):
        """Exception during evaluate() call should return 0.0."""
        mock_instance = MagicMock()
        mock_instance.evaluate.side_effect = ValueError("Parse error")
        mock_evaluator_cls.return_value = mock_instance

        score = fitness_reward(VALID_TOOL_CALL, config_path="dummy.yaml")

        assert score == 0.0

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_default_config_path(self, mock_evaluator_cls):
        """When no config_path given, should use the default flywheel path."""
        mock_result = MagicMock()
        mock_result.score = 1.0
        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = mock_result
        mock_evaluator_cls.return_value = mock_instance

        fitness_reward(VALID_TOOL_CALL)

        mock_evaluator_cls.assert_called_once_with(
            config_path="configs/flywheel/fitness_rules.yaml"
        )

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_custom_config_path(self, mock_evaluator_cls):
        """Custom config_path should be passed through to FitnessEvaluator."""
        mock_result = MagicMock()
        mock_result.score = 0.8
        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = mock_result
        mock_evaluator_cls.return_value = mock_instance

        score = fitness_reward(VALID_TOOL_CALL, config_path="custom/rules.yaml")

        assert score == 0.8
        mock_evaluator_cls.assert_called_once_with(config_path="custom/rules.yaml")


class TestBuildFitnessReward:
    """Tests for build_fitness_reward() TRL wrapper."""

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_returns_list_of_scores(self, mock_evaluator_cls):
        """Should return a list of scores matching completions length."""
        mock_result = MagicMock()
        mock_result.score = 0.9
        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = mock_result
        mock_evaluator_cls.return_value = mock_instance

        reward_fn = build_fitness_reward(config_path="dummy.yaml")
        scores = reward_fn([VALID_TOOL_CALL, NO_TOOL_CALL, VALID_TOOL_CALL])

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_handles_dict_completions(self, mock_evaluator_cls):
        """Should handle dict-format completions via _coerce_to_text."""
        mock_result = MagicMock()
        mock_result.score = 1.0
        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = mock_result
        mock_evaluator_cls.return_value = mock_instance

        completion_dict = [{"content": VALID_TOOL_CALL}]
        reward_fn = build_fitness_reward(config_path="dummy.yaml")
        scores = reward_fn(completion_dict)

        assert len(scores) == 1
        assert scores[0] == 1.0

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_empty_completions(self, mock_evaluator_cls):
        """Empty completions list should return empty scores list."""
        reward_fn = build_fitness_reward(config_path="dummy.yaml")
        scores = reward_fn([])

        assert scores == []

    @patch("shared.validation.fitness.FitnessEvaluator", autospec=False)
    def test_passes_config_path_through(self, mock_evaluator_cls):
        """Config path should be forwarded to FitnessEvaluator."""
        mock_result = MagicMock()
        mock_result.score = 0.5
        mock_instance = MagicMock()
        mock_instance.evaluate.return_value = mock_result
        mock_evaluator_cls.return_value = mock_instance

        reward_fn = build_fitness_reward(config_path="my/custom/rules.yaml")
        reward_fn([VALID_TOOL_CALL])

        mock_evaluator_cls.assert_called_with(config_path="my/custom/rules.yaml")
