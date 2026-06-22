"""Runner verifier-dispatch test (Test 3) + aggregator integration seam (Test 4).

Test 3 proves Evaluator/runner.py::_evaluate_single_case builds the args_match
verifier from case.metadata['verifiers'], scores the RAW completion, populates
record.verifier, and that record.status does NOT crash when the case carries no
correctness/scoring block (it returns 'fail', which is fine — the verifier metric
is intentionally NOT wired into status).

Test 4 proves the integration seam: a record carrying a populated verifier,
serialized via Evaluator/reporting.py::record_to_dict + build_run_payload and
written to a run JSON, can be aggregated by Tools/aggregate_toolcall_report.py
into correct per-tool + overall stats. THE KEY QUESTION is whether the aggregator
can resolve the REFERENCE TOOL per record; record_to_dict does NOT emit
case.metadata.ground_truth, so this verifies the surviving path (verifier
detail.gt_tool) actually works.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Evaluator.prompt_sets import PromptCase
from Evaluator.protocols import BackendResponse
from Evaluator.runner import EvaluationRecord, _evaluate_single_case
from Evaluator import reporting
from Tools import aggregate_toolcall_report as A


_SPEC = {
    "type": "args_match",
    "params": {
        "scheme": "overlap",
        "gt_tool_field": "tool_name",
        "gt_args_field": "arguments",
        "pass_threshold": 0.5,
    },
}

_TOOL_CALL_TEXT = '<tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>'


class _StubClient:
    """Backend stub returning a fixed BackendResponse for chat()."""

    def __init__(self, message: str):
        self._message = message

    def chat(self, messages):
        return BackendResponse(message=self._message, raw={"stub": True}, latency_s=0.01)


def _toolcall_case(case_id: str, gt_tool: str, gt_args: dict, question: str = "q") -> PromptCase:
    return PromptCase(
        case_id=case_id,
        question=question,
        tags=["toolcall_holdout", f"tool:{gt_tool}"],
        metadata={
            "verifiers": [_SPEC],
            "ground_truth": {"tool_name": gt_tool, "arguments": gt_args, "all_calls": []},
            "system": "You are a tool-using assistant.",
        },
    )


# ---------------------------------------------------------------------------
# Test 3 — runner verifier dispatch
# ---------------------------------------------------------------------------

def test_runner_populates_verifier_from_metadata_spec():
    case = _toolcall_case("tc_000000", "search", {"q": "x"})
    client = _StubClient(_TOOL_CALL_TEXT)

    record = _evaluate_single_case(case, client, dry_run=False)

    assert record.verifier is not None
    assert record.verifier["name"] == "args_match"
    assert record.verifier["score"] == 1.0
    assert record.verifier["passed"] is True
    # detail carries the reference tool — the aggregator seam.
    assert record.verifier["detail"]["gt_tool"] == "search"


def test_runner_status_does_not_crash_without_correctness_block():
    case = _toolcall_case("tc_000001", "search", {"q": "x"})
    client = _StubClient(_TOOL_CALL_TEXT)

    record = _evaluate_single_case(case, client, dry_run=False)

    # Accessing status must NOT raise even though there is no correctness/scoring/
    # judge/environment/retrieval block. (It returns 'fail' by design — the verifier
    # metric is a standalone continuous signal, not wired into pass/fail.)
    status = record.status
    assert status in {"pass", "warn", "fail"}
    assert status == "fail"  # documents the intentional decoupling


def test_runner_verifier_name_mismatch_scores_zero():
    case = _toolcall_case("tc_000002", "different_tool", {"q": "x"})
    client = _StubClient(_TOOL_CALL_TEXT)

    record = _evaluate_single_case(case, client, dry_run=False)
    assert record.verifier["score"] == 0.0
    assert record.verifier["detail"]["gt_tool"] == "different_tool"


# ---------------------------------------------------------------------------
# Test 4 — integration seam: record_to_dict -> run JSON -> aggregator
# ---------------------------------------------------------------------------

def _record_for(gt_tool: str, gt_args: dict, completion: str) -> EvaluationRecord:
    """Drive a real record through the runner so verifier is populated authentically."""
    case = _toolcall_case(f"tc_{gt_tool}", gt_tool, gt_args)
    client = _StubClient(completion)
    return _evaluate_single_case(case, client, dry_run=False)


def test_record_to_dict_does_not_emit_ground_truth_metadata():
    """Documents the serialization gap the aggregator must work around.

    record_to_dict emits NO case.metadata / ground_truth block; the reference tool
    is therefore resolvable ONLY via verifier.detail.gt_tool. This test pins that
    gap so a future change that adds ground_truth is a deliberate decision.
    """
    record = _record_for("search", {"q": "x"}, _TOOL_CALL_TEXT)
    d = reporting.record_to_dict(record)
    assert "verifier" in d and d["verifier"] is not None
    assert "ground_truth" not in d
    assert "case" not in d
    assert "metadata" not in d
    # The ONLY surviving reference-tool path:
    assert d["verifier"]["detail"]["gt_tool"] == "search"


def test_aggregator_resolves_reference_tool_via_verifier_detail(tmp_path):
    # Two 'search' records (one exact=1.0, one name-miss=0.0) + one 'lookup' record.
    records = [
        _record_for("search", {"q": "x"}, _TOOL_CALL_TEXT),                       # score 1.0
        _record_for("search", {"q": "x"}, '<tool_call>{"name":"nope","arguments":{}}</tool_call>'),  # 0.0
        _record_for("lookup", {"id": 1}, '<tool_call>{"name":"lookup","arguments":{"id":1}}</tool_call>'),  # 1.0
    ]
    payload = reporting.build_run_payload(records, metadata={"model": "stub"})

    run_json = tmp_path / "scratch" / "run.json"
    run_json.parent.mkdir(parents=True, exist_ok=True)
    run_json.write_text(json.dumps(payload), encoding="utf-8")

    report = A.build_report(run_json)

    # Reference tool resolved from verifier.detail.gt_tool — NOT __unknown__.
    assert set(report["per_tool"].keys()) == {"search", "lookup"}

    search = report["per_tool"]["search"]
    assert search["n"] == 2
    assert search["name_match_rate"] == 0.5      # 1 of 2 scored > 0
    assert search["mean_arg_overlap"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2

    lookup = report["per_tool"]["lookup"]
    assert lookup["n"] == 1
    assert lookup["name_match_rate"] == 1.0
    assert lookup["mean_arg_overlap"] == pytest.approx(1.0)

    overall = report["overall"]
    assert overall["n"] == 3
    assert overall["mean_arg_overlap"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)
    assert overall["name_match_rate"] == pytest.approx(2 / 3)


def test_aggregator_multi_call_scored_is_always_false(tmp_path):
    # Even when ground truth carries multiple calls, the report must declare
    # multi_call_scored == False (only the first call is ever scored).
    rec = _record_for("lookup", {"id": 1}, '<tool_call>{"name":"lookup","arguments":{"id":1}}</tool_call>')
    # Inject a multi-call ground_truth into the case metadata to attempt to flip it.
    rec.case.metadata["ground_truth"]["all_calls"] = [
        {"tool_name": "lookup", "arguments": {"id": 1}},
        {"tool_name": "fetch", "arguments": {"url": "u"}},
    ]
    payload = reporting.build_run_payload([rec], metadata={"model": "stub"})
    run_json = tmp_path / "scratch" / "run_mc.json"
    run_json.parent.mkdir(parents=True, exist_ok=True)
    run_json.write_text(json.dumps(payload), encoding="utf-8")

    report = A.build_report(run_json)
    assert report["meta"]["multi_call_scored"] is False


def test_aggregator_privacy_guard_no_freetext_keys(tmp_path):
    # The per_tool keys must look like tool identifiers; a real run produces clean keys.
    rec = _record_for("search", {"q": "x"}, _TOOL_CALL_TEXT)
    payload = reporting.build_run_payload([rec], metadata={"model": "stub"})
    run_json = tmp_path / "scratch" / "run_pg.json"
    run_json.parent.mkdir(parents=True, exist_ok=True)
    run_json.write_text(json.dumps(payload), encoding="utf-8")

    # Should not raise.
    report = A.build_report(run_json)
    A.assert_no_freetext_keys(report["per_tool"])
