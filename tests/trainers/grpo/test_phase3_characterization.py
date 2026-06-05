"""Phase 3 characterization tests — pin CURRENT numeric behavior.

These tests capture the exact numeric outputs of the GRPO scoring building
blocks BEFORE the Phase 3 dedup refactor, so the refactor can be proven
byte-identical. They must pass against the unrefactored code AND against the
refactored code (where the duplicated logic is consolidated into
``shared.verifiers.builtins.args_match``).

Covered:
- ``functional_equivalence_reward`` on representative cases.
- ``RewardRubric._score_weighted`` for a mapping-based AND a legacy config.
- ``RewardRubric._compare_values`` for all four strategies.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from functional_verifier import functional_equivalence_reward
import rewards as R


def _make_rubric(cfg: dict) -> "R.RewardRubric":
    d = tempfile.mkdtemp()
    p = Path(d) / "r.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return R.RewardRubric(p)


# ---------------------------------------------------------------------------
# functional_equivalence_reward
# ---------------------------------------------------------------------------

class TestFunctionalEquivalenceRewardCharacterization:

    def test_representative_vector(self):
        exact = '<tool_call>{"name": "readFile", "arguments": {"path": "/a.txt"}}</tool_call>'
        partial = '<tool_call>{"name": "readFile", "arguments": {"path": "/a.txt", "extra": "z"}}</tool_call>'
        wrong = '<tool_call>{"name": "writeFile", "arguments": {"path": "/a.txt"}}</tool_call>'
        missing_gt = '<tool_call>{"name": "readFile", "arguments": {"path": "/a.txt"}}</tool_call>'

        scores = functional_equivalence_reward(
            [exact, partial, wrong, missing_gt],
            ground_truth_tool=["readFile", "readFile", "readFile", None],
            ground_truth_args_json=[
                json.dumps({"path": "/a.txt"}),
                json.dumps({"path": "/a.txt"}),
                json.dumps({"path": "/a.txt"}),
                json.dumps({"path": "/a.txt"}),
            ],
        )
        # Pinned current outputs:
        # - exact tool + perfect args -> 1.0
        # - partial key overlap (extra arg) -> 0.8
        # - wrong tool -> 0.0
        # - missing ground_truth -> 0.0
        assert scores == [1.0, 0.8, 0.0, 0.0]


# ---------------------------------------------------------------------------
# RewardRubric._compare_values — all four strategies
# ---------------------------------------------------------------------------

class TestCompareValuesCharacterization:

    @pytest.fixture(scope="class")
    def rubric(self):
        return _make_rubric({"name": "x"})

    def test_equals(self, rubric):
        assert rubric._compare_values("a", "a", "equals") == 1.0
        assert rubric._compare_values("a", "b", "equals") == 0.0

    def test_contains(self, rubric):
        assert rubric._compare_values("hello world", "world", "contains") == 1.0
        assert rubric._compare_values("hi", "world", "contains") == 0.0

    def test_key_overlap(self, rubric):
        # pred keys {a,b}, gt keys {a,c}; overlap {a} = 1; len(gt) = 2 -> 0.5
        assert rubric._compare_values(
            {"a": 1, "b": 2}, {"a": 9, "c": 3}, "key_overlap"
        ) == 0.5

    def test_tool_name_match_exact(self, rubric):
        assert rubric._compare_values("agent_tool", "agent_tool", "tool_name_match") == 1.0

    def test_tool_name_match_partial(self, rubric):
        # same agent prefix -> 0.3
        assert rubric._compare_values("agent_x", "agent_y", "tool_name_match") == 0.3

    def test_tool_name_match_none(self, rubric):
        assert rubric._compare_values("a_x", "b_y", "tool_name_match") == 0.0

    def test_none_input(self, rubric):
        assert rubric._compare_values(None, "a", "equals") == 0.0


# ---------------------------------------------------------------------------
# RewardRubric._score_weighted — mapping-based AND legacy
# ---------------------------------------------------------------------------

class TestScoreWeightedCharacterization:

    def test_mapping_based(self):
        cfg = {
            "name": "mapped",
            "scoring": {"strategy": "weighted"},
            "ground_truth": {"field": "ground_truth_args_json", "parse": "json"},
            "comparison": {
                "mappings": [
                    {"use_tool_name": True, "strategy": "tool_name_match", "weight": 2.0},
                    {"pred_path": "path", "gt_path": "path", "strategy": "equals", "weight": 1.0},
                    {"pred_path": "", "gt_path": "calls.0.params", "strategy": "key_overlap", "weight": 1.0},
                ]
            },
        }
        rb = _make_rubric(cfg)
        data = {"tool_name": "agent_read", "parsed_args": {"path": "/a.txt", "mode": "r"}}
        kwargs = {
            "ground_truth_tool": "agent_read",
            "ground_truth_args_json": json.dumps(
                {"path": "/a.txt", "calls": [{"params": {"mode": 1, "path": 2}}]}
            ),
        }
        # tool_name 1.0*2 + equals(path) 1.0*1 + key_overlap(mode,path vs mode,path) 1.0*1
        # = 4.0 / 4.0 = 1.0
        assert rb._score_weighted(data, kwargs) == 1.0

    def test_mapping_missing_ground_truth(self):
        cfg = {
            "name": "mapped",
            "scoring": {"strategy": "weighted"},
            "ground_truth": {"field": "ground_truth_args_json", "parse": "json"},
            "comparison": {"mappings": [{"pred_path": "p", "gt_path": "p", "weight": 1.0}]},
        }
        rb = _make_rubric(cfg)
        assert rb._score_weighted({"parsed_args": {}}, {}) == 0.0

    def test_legacy(self):
        cfg = {
            "name": "legacy",
            "scoring": {
                "strategy": "weighted",
                "weights": {"context_match": 0.4, "tool_match": 0.3, "params_match": 0.3},
            },
            "ground_truth": {"field": "ground_truth_args_json", "parse": "json"},
            "comparison": {"context_fields": ["userId"], "call_fields": []},
        }
        rb = _make_rubric(cfg)
        data = {"tool_name": "agent_x", "parsed_args": {"userId": "u1", "query": "hi"}}
        kwargs = {
            "ground_truth_tool": "agent_x",
            "ground_truth_args_json": json.dumps(
                {"context": {"userId": "u1"}, "calls": [{"params": {"query": 1, "limit": 2}}]}
            ),
        }
        # context 0.4*(1/1) + tool 0.3*1.0 + params 0.3*(1/2) = 0.4 + 0.3 + 0.15 = 0.85
        assert rb._score_weighted(data, kwargs) == pytest.approx(0.85)

    def test_legacy_missing_ground_truth(self):
        cfg = {
            "name": "legacy",
            "scoring": {"strategy": "weighted"},
            "ground_truth": {"field": "ground_truth_args_json", "parse": "json"},
            "comparison": {"context_fields": ["userId"]},
        }
        rb = _make_rubric(cfg)
        assert rb._score_weighted({"parsed_args": {}}, {}) == 0.0
