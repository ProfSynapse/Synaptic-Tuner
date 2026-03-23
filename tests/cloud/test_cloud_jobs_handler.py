import json
from argparse import Namespace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

from tuner.handlers.cloud_jobs_handler import CloudJobsHandler


def _make_job(command):
    return SimpleNamespace(
        id="job-123",
        status=SimpleNamespace(stage="running", message="Job is running"),
        owner=SimpleNamespace(name="professorsynapse"),
        created_at=datetime(2026, 3, 22, 15, 30, tzinfo=timezone.utc),
        docker_image="unsloth/unsloth:latest",
        flavor="a10g-small",
        url="https://hf.co/jobs/job-123",
        labels={"task": "evaluation", "provider": "hf_jobs"},
        command=command,
    )


def test_resolve_stage_artifact_root_from_eval_command(repo_root):
    handler = CloudJobsHandler(args=Namespace())
    handler._repo_root = repo_root

    job = {
        "command": [
            "bash",
            "-c",
            "cd /workspace/repo && python -m Evaluator.cloud_hf_job "
            "--bucket-id test-user/toolset-training-artifacts "
            "--run-prefix runs/hf_jobs/sft/20260322_111111-abc12345 "
            "--eval-prefix runs/hf_jobs/sft/20260322_111111-abc12345/evaluations/vllm/20260322_120000",
        ],
        "labels": {},
    }

    artifact_root = handler._resolve_stage_artifact_root(job)

    assert artifact_root == {
        "bucket_id": "test-user/toolset-training-artifacts",
        "prefix": "runs/hf_jobs/sft/20260322_111111-abc12345/evaluations/vllm/20260322_120000",
    }


def test_resolve_stage_artifact_root_decodes_encoded_labels(repo_root):
    """Labels encoded by sanitize_hf_job_labels are decoded when reading back."""
    handler = CloudJobsHandler(args=Namespace())
    handler._repo_root = repo_root

    job = {
        "command": ["bash", "-c", "echo placeholder"],
        "labels": {
            "bucket_id": "test-user..toolset-training-artifacts",
            "artifact_prefix": "runs..hf_jobs..sft..20260322_111111-abc12345",
        },
    }

    artifact_root = handler._resolve_stage_artifact_root(job)

    assert artifact_root == {
        "bucket_id": "test-user/toolset-training-artifacts",
        "prefix": "runs/hf_jobs/sft/20260322_111111-abc12345",
    }


def test_handle_show_prefers_structured_stage_summary(repo_root, clean_env):
    clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
    args = Namespace(
        json=False,
        subcommand="show",
        job="professorsynapse/job-123",
        namespace=None,
        tail=25,
        limit=20,
        follow=False,
        auto_confirm=False,
    )
    handler = CloudJobsHandler(args=args)
    handler._repo_root = repo_root

    summary = {
        "stage": "evaluation",
        "health": "healthy",
        "status": "running",
        "message": "vLLM server is ready",
        "last_event": "runtime_ready",
        "progress": {"cases_done": 17, "cases_total": 77},
    }
    command = [
        "bash",
        "-c",
        "cd /workspace/repo && python -m Evaluator.cloud_hf_job_vllm "
        "--bucket-id test-user/toolset-training-artifacts "
        "--run-prefix runs/hf_jobs/sft/20260322_111111-abc12345 "
        "--eval-prefix runs/hf_jobs/sft/20260322_111111-abc12345/evaluations/vllm/20260322_120000",
    ]
    mock_hub = MagicMock()
    mock_hub.inspect_job.return_value = _make_job(command)
    fs = MagicMock()
    fs.open = mock_open(read_data=json.dumps(summary))
    mock_hub.HfFileSystem.return_value = fs

    with patch.object(handler, "_validate_environment", return_value=(mock_hub, "hf_test_token_12345")):
        with patch("tuner.handlers.cloud_jobs_handler.print_header"):
            with patch("tuner.handlers.cloud_jobs_handler.print_info"):
                with patch("tuner.handlers.cloud_jobs_handler.print_config") as mock_print_config:
                    exit_code = handler.handle()

    assert exit_code == 0
    fs.open.assert_called_once_with(
        "hf://buckets/test-user/toolset-training-artifacts/"
        "runs/hf_jobs/sft/20260322_111111-abc12345/evaluations/vllm/20260322_120000/"
        "logs/stage_summary.json",
        "r",
    )
    mock_hub.fetch_job_logs.assert_not_called()
    assert mock_print_config.call_count == 2
    structured_display = mock_print_config.call_args_list[1].args[0]
    assert structured_display["Health"] == "healthy"
    assert structured_display["Status"] == "running"
    assert structured_display["Artifacts"] == (
        "hf://buckets/test-user/toolset-training-artifacts/"
        "runs/hf_jobs/sft/20260322_111111-abc12345/evaluations/vllm/20260322_120000"
    )


def test_handle_show_falls_back_to_raw_logs_when_summary_missing(repo_root, clean_env):
    clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
    args = Namespace(
        json=False,
        subcommand="show",
        job="job-123",
        namespace="professorsynapse",
        tail=10,
        limit=20,
        follow=False,
        auto_confirm=False,
    )
    handler = CloudJobsHandler(args=args)
    handler._repo_root = repo_root

    command = [
        "bash",
        "-c",
        "python train_sft.py "
        "--artifact-bucket test-user/toolset-training-artifacts "
        "--artifact-prefix runs/hf_jobs/sft/20260322_111111-abc12345",
    ]
    mock_hub = MagicMock()
    mock_hub.inspect_job.return_value = _make_job(command)
    mock_hub.HfFileSystem.return_value.open.side_effect = FileNotFoundError("missing stage summary")
    mock_hub.fetch_job_logs.return_value = "bootstrap line\nprogress line"

    with patch.object(handler, "_validate_environment", return_value=(mock_hub, "hf_test_token_12345")):
        with patch("tuner.handlers.cloud_jobs_handler.print_header"):
            with patch("tuner.handlers.cloud_jobs_handler.print_config"):
                with patch("tuner.handlers.cloud_jobs_handler.print_info") as mock_print_info:
                    with patch("builtins.print"):
                        exit_code = handler.handle()

    assert exit_code == 0
    mock_hub.fetch_job_logs.assert_called_once_with(
        job_id="job-123",
        namespace="professorsynapse",
        follow=False,
        token="hf_test_token_12345",
    )
    info_messages = [call.args[0] for call in mock_print_info.call_args_list]
    assert "Structured stage summary not available; falling back to recent raw logs." in info_messages
    assert "Recent raw logs:" in info_messages
