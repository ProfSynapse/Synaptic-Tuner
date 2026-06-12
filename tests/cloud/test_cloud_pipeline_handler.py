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
        eval_timeout_hours=5.0,
        eval_runtime="vllm",
        eval_image_profile="fast_vllm",
        eval_cloud_image="custom/eval:latest",
        eval_pip_packages=["vllm==0.12.0"],
        with_loss=True,
        loss_dataset_name="test/dataset",
        loss_dataset_file="train.jsonl",
        loss_max_seq_length=1024,
        loss_no_completion_only=True,
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
    assert eval_args.timeout_hours == 5.0
    assert eval_args.eval_timeout_hours == 5.0
    assert eval_args.eval_runtime == "vllm"
    assert eval_args.eval_image_profile == "fast_vllm"
    assert eval_args.eval_cloud_image == "custom/eval:latest"
    assert eval_args.eval_pip_packages == ["vllm==0.12.0"]
    assert eval_args.with_loss is True
    assert eval_args.loss_dataset_name == "test/dataset"
    assert eval_args.loss_dataset_file == "train.jsonl"
    assert eval_args.loss_max_seq_length == 1024
    assert eval_args.loss_no_completion_only is True
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
