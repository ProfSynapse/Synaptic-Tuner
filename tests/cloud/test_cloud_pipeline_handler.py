import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

from tuner.handlers.cloud_pipeline_handler import CloudPipelineHandler


def test_cloud_pipeline_json_is_metadata_only(repo_root, capsys):
    handler = CloudPipelineHandler(args=Namespace(json=True))
    handler._repo_root = repo_root

    with patch(
        "tuner.handlers.cloud_pipeline_handler.TrainingBackendRegistry.list",
        return_value=["rtx", "hf_jobs"],
    ) as registry_list, patch(
        "tuner.handlers.cloud_pipeline_handler.TrainingBackendRegistry.get",
        side_effect=AssertionError("inspection constructed a provider backend"),
    ) as backend_get, patch(
        "tuner.handlers.cloud_pipeline_handler.require_current_hf_source_submission_authorization",
        side_effect=AssertionError("inspection crossed submission authorization"),
    ) as authorize, patch(
        "tuner.handlers.cloud_pipeline_handler.CloudTrainHandler",
        side_effect=AssertionError("inspection entered training configuration"),
    ) as train_handler, patch(
        "tuner.handlers.cloud_pipeline_handler.CloudEvalHandler",
        side_effect=AssertionError("inspection entered evaluation configuration"),
    ) as eval_handler:
        exit_code = handler.handle()

    assert exit_code == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["success"] is True
    assert set(envelope) == {"success", "data", "timestamp"}
    assert envelope["data"] == {
        "command": "cloud-pipeline",
        "status": "inspection_only",
        "inspection_only": True,
        "submission_enabled": False,
        "credentials_checked": False,
        "provider_registry": {
            "id": "hf_jobs",
            "name": "HuggingFace Jobs",
            "registered": True,
        },
        "pipeline_stages": ["training", "evaluation"],
    }
    registry_list.assert_called_once_with()
    backend_get.assert_not_called()
    authorize.assert_not_called()
    train_handler.assert_not_called()
    eval_handler.assert_not_called()


def test_cloud_pipeline_json_does_not_claim_unregistered_provider_is_ready(capsys):
    handler = CloudPipelineHandler(args=Namespace(json=True))

    with patch(
        "tuner.handlers.cloud_pipeline_handler.TrainingBackendRegistry.list",
        return_value=["rtx"],
    ), patch(
        "tuner.handlers.cloud_pipeline_handler.TrainingBackendRegistry.get",
        side_effect=AssertionError("inspection constructed a provider backend"),
    ):
        assert handler.handle() == 0

    data = json.loads(capsys.readouterr().out)["data"]
    assert data["provider_registry"]["registered"] is False
    assert "ready" not in data
    assert "available" not in data
    assert "env_ready" not in data["provider_registry"]


def test_cloud_pipeline_runs_training_then_eval(repo_root, clean_env, capsys):
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

    training_config = MagicMock(
        method="sft",
        artifact_identifier="test-user/toolset-training-artifacts",
    )
    source_preparation = object()
    eval_args = handler._resolve_eval_args(
        training_config=training_config,
        artifact_prefix="runs/hf_jobs/sft/20260315_010000-abc12345",
        source_preparation=source_preparation,
    )
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
    assert eval_args._source_preparation is source_preparation

    with patch("tuner.handlers.cloud_pipeline_handler.TrainingBackendRegistry.get") as backend_get, \
         patch("tuner.handlers.cloud_pipeline_handler.confirm") as mock_confirm, \
         patch("tuner.handlers.cloud_pipeline_handler.CloudEvalHandler") as eval_handler_cls:
        exit_code = handler.handle()

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "exact-run" in output
    assert "no provider-facing operation" in output
    backend_get.assert_not_called()
    mock_confirm.assert_not_called()
    eval_handler_cls.assert_not_called()


def test_cloud_pipeline_skips_confirmation_when_auto_confirm_enabled(
    repo_root, clean_env, capsys
):
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

    with patch("tuner.handlers.cloud_pipeline_handler.TrainingBackendRegistry.get") as backend_get, \
         patch("tuner.handlers.cloud_pipeline_handler.confirm") as mock_confirm, \
         patch("tuner.handlers.cloud_pipeline_handler.CloudEvalHandler") as eval_handler_cls:
        exit_code = handler.handle()

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "exact-run" in output
    assert "no provider-facing operation" in output
    mock_confirm.assert_not_called()
    backend_get.assert_not_called()
    eval_handler_cls.assert_not_called()
