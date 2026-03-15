from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from tuner.handlers.cloud_inspect_handler import CloudInspectHandler


def test_extract_failure_reason_prefers_validator_issue(repo_root):
    handler = CloudInspectHandler(args=Namespace())
    handler._repo_root = repo_root

    reason = handler._extract_failure_reason(
        {
            "error": None,
            "validator": {"issues": [{"message": "wrong tool selected"}]},
            "behavior": {"issues": [{"message": "behavior miss"}]},
        }
    )

    assert reason == "wrong tool selected"


def test_list_eval_runs_sorts_newest_first(repo_root, clean_env):
    clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
    handler = CloudInspectHandler(args=Namespace())
    handler._repo_root = repo_root

    mock_hub = ModuleType("huggingface_hub")
    api = MagicMock()
    api.list_bucket_tree.return_value = [
        SimpleNamespace(type="directory", path="runs/hf_jobs/sft/20260315_010000-abc/evaluations/vllm/20260315_020000"),
        SimpleNamespace(type="directory", path="runs/hf_jobs/sft/20260315_010000-abc/evaluations/vllm/20260315_015000"),
    ]
    mock_hub.HfApi = MagicMock(return_value=api)

    with patch.object(handler._cloud_eval, "_validate_environment", return_value=mock_hub):
        eval_runs = handler._list_eval_runs(
            mock_hub,
            "test-user/toolset-training-artifacts",
            "runs/hf_jobs/sft/20260315_010000-abc",
        )

    assert [run["slug"] for run in eval_runs] == ["20260315_020000", "20260315_015000"]


def test_handle_loads_and_reports_eval_results(repo_root, clean_env):
    clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
    args = Namespace(
        json=False,
        run="latest",
        eval_run="latest",
        method="sft",
        bucket=None,
    )
    handler = CloudInspectHandler(args=args)
    handler._repo_root = repo_root

    mock_hub = MagicMock()
    payload = {
        "summary": {"passed": 8, "warned": 1, "failed": 2, "total": 11, "request_errors": 0},
        "records": [
            {
                "case_id": "bad-case",
                "tags": ["storageManager"],
                "passed": False,
                "response_text": "model response",
                "validator": {"issues": [{"message": "wrong tool selected"}]},
            }
        ],
    }

    with patch.object(handler._cloud_eval, "_validate_environment", return_value=mock_hub):
        with patch.object(handler._cloud_eval, "_resolve_bucket_id", return_value="test-user/toolset-training-artifacts"):
            with patch.object(
                handler._cloud_eval,
                "_list_remote_runs",
                return_value=[{"method": "sft", "slug": "20260315_010000-abc", "prefix": "runs/hf_jobs/sft/20260315_010000-abc"}],
            ):
                with patch.object(handler, "_list_eval_runs", return_value=[{"slug": "20260315_020000", "prefix": "runs/hf_jobs/sft/20260315_010000-abc/evaluations/vllm/20260315_020000"}]):
                    with patch.object(handler, "_download_eval_payload", return_value=payload):
                        exit_code = handler.handle()

    assert exit_code == 0
