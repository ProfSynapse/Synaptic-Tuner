from types import SimpleNamespace

from shared.cloud_eval_progress import EvaluationDashboardReplayer, extract_record_progress


def test_extract_record_progress_uses_issue_formatter():
    record = SimpleNamespace(
        status="warn",
        error=None,
        latency_s=1.25,
        case=SimpleNamespace(case_id="case-1"),
        validator=SimpleNamespace(issues=[SimpleNamespace(message="raw message")]),
        environment=None,
        behavior=SimpleNamespace(passed=False, issues=[SimpleNamespace(message="behavior issue")]),
    )

    payload = extract_record_progress(
        record,
        issue_formatter=lambda message: f"clean:{message}",
    )

    assert payload["event"] == "result"
    assert payload["name"] == "case-1"
    assert payload["status"] == "warn"
    assert payload["reason"] == "clean:raw message"
    assert payload["behavior_tested"] is True
    assert payload["behavior_passed"] is False


def test_dashboard_replayer_applies_meta_and_result_events():
    dashboard = SimpleNamespace(
        title="Initial",
        metrics=SimpleNamespace(total_tests=0),
        updates=[],
    )
    dashboard.update = lambda **kwargs: dashboard.updates.append(kwargs)

    replayer = EvaluationDashboardReplayer(dashboard)
    replayer.apply_event({"event": "meta", "title": "Cloud Evaluation", "total_tests": 12})
    replayer.apply_event(
        {
            "event": "result",
            "status": "pass",
            "name": "case-2",
            "latency": 0.5,
            "reason": None,
            "behavior_tested": False,
            "behavior_passed": False,
        }
    )

    assert dashboard.title == "Cloud Evaluation"
    assert dashboard.metrics.total_tests == 12
    assert dashboard.updates == [
        {
            "status": "pass",
            "name": "case-2",
            "latency": 0.5,
            "reason": None,
            "behavior_tested": False,
            "behavior_passed": False,
        }
    ]
