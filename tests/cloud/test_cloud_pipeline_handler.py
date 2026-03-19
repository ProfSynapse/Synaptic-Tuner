from argparse import Namespace
from unittest.mock import MagicMock, patch

from tuner.handlers.cloud_pipeline_handler import CloudPipelineHandler


def test_cloud_pipeline_runs_training_then_eval(repo_root, clean_env):
    clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
    args = Namespace(
        json=False,
        method="sft",
        preset="full",
        scenario=None,
        tags=None,
        upload_to_hf=None,
        update_model_card=False,
        gpu=None,
        timeout_hours=2.0,
    )
    handler = CloudPipelineHandler(args=args)
    handler._repo_root = repo_root

    backend = MagicMock()
    backend.validate_environment.return_value = (True, "")
    backend.get_available_methods.return_value = ["sft", "kto"]
    backend.load_config.return_value = MagicMock(
        method="sft",
        model_name="test-org/test-model-sft",
        dataset_file="../../Datasets/test.jsonl",
        provider="hf_jobs",
        gpu_type="a10g-small",
        timeout_hours=4.0,
        epochs=2,
        batch_size=4,
        learning_rate=2e-4,
        artifact_identifier="test-user/toolset-training-artifacts",
    )
    backend.execute.return_value = 0
    backend.last_artifact_prefix = "runs/hf_jobs/sft/20260315_010000-abc12345"

    with patch("tuner.handlers.cloud_pipeline_handler.TrainingBackendRegistry.get", return_value=backend):
        with patch("tuner.handlers.cloud_pipeline_handler.confirm", return_value=True):
            with patch("tuner.handlers.cloud_pipeline_handler.CloudEvalHandler") as mock_eval_handler_cls:
                mock_eval_handler = MagicMock()
                mock_eval_handler.handle.return_value = 0
                mock_eval_handler_cls.return_value = mock_eval_handler
                exit_code = handler.handle()

    assert exit_code == 0
    assert backend.show_post_training_actions is False
    backend.execute.assert_called_once()
    eval_args = mock_eval_handler_cls.call_args.kwargs["args"]
    assert eval_args.run == "runs/hf_jobs/sft/20260315_010000-abc12345"
    assert eval_args.method == "sft"
    assert eval_args.bucket == "test-user/toolset-training-artifacts"
    assert eval_args.auto_confirm is True


def test_cloud_pipeline_skips_confirmation_when_auto_confirm_enabled(repo_root, clean_env):
    clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
    args = Namespace(
        json=False,
        method="sft",
        preset="full",
        scenario=None,
        tags=None,
        upload_to_hf=None,
        update_model_card=False,
        gpu=None,
        timeout_hours=2.0,
        auto_confirm=True,
    )
    handler = CloudPipelineHandler(args=args)
    handler._repo_root = repo_root

    backend = MagicMock()
    backend.validate_environment.return_value = (True, "")
    backend.get_available_methods.return_value = ["sft", "kto"]
    backend.load_config.return_value = MagicMock(
        method="sft",
        model_name="test-org/test-model-sft",
        dataset_file="../../Datasets/test.jsonl",
        provider="hf_jobs",
        gpu_type="a10g-small",
        timeout_hours=4.0,
        epochs=2,
        batch_size=4,
        learning_rate=2e-4,
        artifact_identifier="test-user/toolset-training-artifacts",
    )
    backend.execute.return_value = 0
    backend.last_artifact_prefix = "runs/hf_jobs/sft/20260315_010000-abc12345"

    with patch("tuner.handlers.cloud_pipeline_handler.TrainingBackendRegistry.get", return_value=backend):
        with patch("tuner.handlers.cloud_pipeline_handler.confirm") as mock_confirm:
            with patch("tuner.handlers.cloud_pipeline_handler.CloudEvalHandler") as mock_eval_handler_cls:
                mock_eval_handler = MagicMock()
                mock_eval_handler.handle.return_value = 0
                mock_eval_handler_cls.return_value = mock_eval_handler
                exit_code = handler.handle()

    assert exit_code == 0
    mock_confirm.assert_not_called()
