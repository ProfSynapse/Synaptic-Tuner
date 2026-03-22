import json

from shared.cloud_stage_logging import CloudStageLogger


def test_stage_logger_appends_events_and_rolls_summary(tmp_path):
    logger = CloudStageLogger(
        tmp_path / "logs",
        stage="evaluation",
        provider="hf_jobs",
        run_prefix="runs/hf_jobs/sft/demo",
        job_ref="job-123",
        bucket_id="bucket-1",
    )

    logger.emit("bootstrap_started", message="boot")
    logger.emit("progress", details={"cases_done": 1, "cases_total": 4})
    logger.emit("completed", status="completed", message="done", details={"results_written": True})

    events_path = tmp_path / "logs" / "stage_events.jsonl"
    summary_path = tmp_path / "logs" / "stage_summary.json"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [event["event"] for event in events] == ["bootstrap_started", "progress", "completed"]
    assert summary["event"] == "completed"
    assert summary["status"] == "completed"
    assert summary["health"] == "healthy"
    assert summary["details"]["cases_done"] == 1
    assert summary["details"]["cases_total"] == 4
    assert summary["details"]["results_written"] is True
    assert summary["event_count"] == 3


def test_stage_logger_normalizes_failures(tmp_path):
    logger = CloudStageLogger(tmp_path / "logs", stage="loss", provider="hf_jobs")

    logger.emit_failure(RuntimeError("boom"), traceback_text="traceback text")

    summary = json.loads((tmp_path / "logs" / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["event"] == "failed"
    assert summary["status"] == "failed"
    assert summary["health"] == "failed"
    assert summary["details"]["error_type"] == "RuntimeError"
    assert summary["details"]["error_message"] == "boom"
    assert summary["details"]["traceback"] == "traceback text"
