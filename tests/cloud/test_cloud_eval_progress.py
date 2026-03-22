import json
from types import SimpleNamespace

from shared.cloud_stage_logging import CloudStageLogger
from shared.cloud_eval_progress import CloudEvaluationProgressWriter, EvaluationDashboardReplayer, extract_record_progress


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


def test_progress_writer_writes_failure_event(tmp_path):
    path = tmp_path / "logs" / "eval_progress.jsonl"
    sync_calls = []
    stage_logger = CloudStageLogger(path.parent, stage="evaluation", provider="hf_jobs")
    writer = CloudEvaluationProgressWriter(path, sync_callback=lambda p: sync_calls.append(p), stage_logger=stage_logger)

    writer.write_failure("runtime blew up badly")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    summary = json.loads((path.parent / "stage_summary.json").read_text(encoding="utf-8"))
    assert payload["event"] == "failure"
    assert "runtime blew up badly" in payload["reason"]
    assert summary["event"] == "artifacts_synced"
    assert summary["details"]["error_message"] == "runtime blew up badly"
    assert sync_calls == [path.parent]


def test_progress_writer_bridges_progress_into_stage_events(tmp_path):
    path = tmp_path / "logs" / "eval_progress.jsonl"
    stage_logger = CloudStageLogger(path.parent, stage="evaluation", provider="hf_jobs", run_prefix="runs/demo")
    writer = CloudEvaluationProgressWriter(path, stage_logger=stage_logger)

    record = SimpleNamespace(
        status="pass",
        latency_s=0.75,
        error=None,
        case=SimpleNamespace(case_id="case-9"),
        validator=None,
        environment=None,
        behavior=None,
    )

    writer.write_metadata(total_tests=3, backend="unsloth", model="/tmp/final_model")
    writer.write_record(record)
    writer.write_complete()

    events = [json.loads(line) for line in (path.parent / "stage_events.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((path.parent / "stage_summary.json").read_text(encoding="utf-8"))

    assert [event["event"] for event in events] == ["work_started", "progress", "completed"]
    assert summary["event"] == "completed"
    assert summary["details"]["cases_done"] == 1
    assert summary["details"]["cases_total"] == 3
    assert summary["details"]["passed"] == 1
    assert summary["details"]["current_case"] == "case-9"
