from __future__ import annotations

import time

from Evaluator.prompt_sets import PromptCase
from Evaluator.protocols import BackendResponse
from Evaluator.runner import evaluate_cases


class _DelayedClient:
    def __init__(self, delays: dict[str, float]):
        self.delays = delays

    def chat(self, messages):
        question = messages[-1]["content"]
        time.sleep(self.delays.get(question, 0.0))
        return BackendResponse(message="Plain text response", raw={"message": "Plain text response"}, latency_s=0.01)


def test_parallel_evaluate_cases_preserves_input_order():
    cases = [
        PromptCase(case_id="case-1", question="slow"),
        PromptCase(case_id="case-2", question="fast"),
        PromptCase(case_id="case-3", question="medium"),
    ]
    client = _DelayedClient({"slow": 0.04, "fast": 0.0, "medium": 0.02})

    observed_order: list[str] = []
    records = evaluate_cases(
        cases,
        client=client,
        parallel=True,
        max_workers=3,
        on_record=lambda record: observed_order.append(record.case.case_id),
    )

    assert [record.case.case_id for record in records] == ["case-1", "case-2", "case-3"]
    assert observed_order == ["case-1", "case-2", "case-3"]


def test_serial_and_parallel_record_counts_match():
    cases = [
        PromptCase(case_id="case-1", question="one"),
        PromptCase(case_id="case-2", question="two"),
        PromptCase(case_id="case-3", question="three"),
    ]
    client = _DelayedClient({})

    serial = evaluate_cases(cases, client=client, parallel=False)
    parallel = evaluate_cases(cases, client=client, parallel=True, max_workers=2)

    assert len(serial) == len(parallel) == 3
    assert [record.case.case_id for record in parallel] == [record.case.case_id for record in serial]
