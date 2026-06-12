import json
import os

from shared.cloud_stage_logging import (
    ENV_STAGE_BUCKET_ID,
    ENV_STAGE_JOB_REF,
    ENV_STAGE_LOG_DIR,
    ENV_STAGE_NAME,
    ENV_STAGE_PROVIDER,
    ENV_STAGE_RUN_PREFIX,
    CloudStageLogger,
    apply_stage_logging_env,
    stage_logger_from_env,
)


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


def test_stage_logger_from_env_uses_env_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    env_keys = [
        ENV_STAGE_NAME,
        ENV_STAGE_PROVIDER,
        ENV_STAGE_RUN_PREFIX,
        ENV_STAGE_JOB_REF,
        ENV_STAGE_BUCKET_ID,
        ENV_STAGE_LOG_DIR,
    ]
    previous_env = {key: os.environ.get(key) for key in env_keys}
    try:
        apply_stage_logging_env(
            stage="evaluation",
            provider="hf_jobs",
            run_prefix="runs/demo",
            job_ref="job-123",
            bucket_id="bucket-1",
            log_dir=log_dir,
        )

        logger = stage_logger_from_env()
        logger.emit("model_load_started", details={"backend": "unsloth"})
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    summary = json.loads((log_dir / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "evaluation"
    assert summary["provider"] == "hf_jobs"
    assert summary["job_ref"] == "job-123"
    assert summary["run_prefix"] == "runs/demo"
    assert summary["bucket_id"] == "bucket-1"
    assert summary["event"] == "model_load_started"
