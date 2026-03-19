from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tuner.cloud import (
    CloudJobSpec,
    HFJobExecutor,
    RepoCheckoutSpec,
    build_bash_command,
    build_hf_job_secrets,
    build_repo_checkout_steps,
    format_timeout_hours,
    resolve_hf_bucket_id,
)


def test_build_repo_checkout_steps_pins_exact_commit():
    steps = build_repo_checkout_steps(
        RepoCheckoutSpec(
            url="https://github.com/test/repo.git",
            branch="main",
            commit="abc12345def67890",
        )
    )

    assert steps == [
        "git clone --branch main --depth 1 https://github.com/test/repo.git /workspace/repo",
        "cd /workspace/repo && git fetch --depth 1 origin abc12345def67890 && git checkout abc12345def67890",
    ]


def test_format_timeout_hours_normalizes_integers_and_floats():
    assert format_timeout_hours(4.0) == "4h"
    assert format_timeout_hours(2.5) == "2.5h"
    assert format_timeout_hours(None) is None


def test_build_hf_job_secrets_returns_both_key_names(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")

    assert build_hf_job_secrets() == {
        "HF_TOKEN": "hf_test_token_12345",
        "HF_API_KEY": "hf_test_token_12345",
    }


def test_hf_job_executor_submits_shared_job_spec():
    mock_hub = MagicMock()
    mock_hub.run_job.return_value = SimpleNamespace(id="job-123", url="https://hf.co/jobs/job-123")

    submission = HFJobExecutor(mock_hub).submit(
        CloudJobSpec(
            provider="hf_jobs",
            image="unsloth/test:latest",
            command=build_bash_command(["echo hi"]),
            flavor="a10g-small",
            timeout_hours=4.0,
            secrets={"HF_TOKEN": "hf_test_token_12345"},
            env={"FOO": "bar"},
            labels={"task": "gym"},
        )
    )

    assert submission.job_id == "job-123"
    assert submission.job_url == "https://hf.co/jobs/job-123"
    kwargs = mock_hub.run_job.call_args.kwargs
    assert kwargs["image"] == "unsloth/test:latest"
    assert kwargs["command"] == ["bash", "-c", "echo hi"]
    assert kwargs["flavor"] == "a10g-small"
    assert kwargs["timeout"] == "4h"
    assert kwargs["secrets"] == {"HF_TOKEN": "hf_test_token_12345"}
    assert kwargs["env"] == {"FOO": "bar"}
    assert kwargs["labels"] == {"task": "gym"}


def test_resolve_hf_bucket_id_returns_namespaced_bucket():
    mock_hub = MagicMock()
    mock_hub.create_bucket.return_value = SimpleNamespace(bucket_id="test-user/toolset-training-artifacts")

    bucket_id = resolve_hf_bucket_id(
        mock_hub,
        "toolset-training-artifacts",
        token="hf_test_token_12345",
    )

    assert bucket_id == "test-user/toolset-training-artifacts"
    mock_hub.create_bucket.assert_called_once_with(
        "toolset-training-artifacts",
        exist_ok=True,
        private=True,
        token="hf_test_token_12345",
    )
