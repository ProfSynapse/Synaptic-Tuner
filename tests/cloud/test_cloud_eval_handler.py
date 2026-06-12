from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from tuner.handlers.cloud_eval_handler import CloudEvalHandler


def test_build_eval_command_uses_cloud_job_helper(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    command = handler._build_eval_command(
        helper_module="Evaluator.cloud_hf_job",
        bucket_id="test-user/toolset-training-artifacts",
        run_prefix="runs/hf_jobs/sft/20260314_191223-abc12345",
        eval_prefix="runs/hf_jobs/sft/20260314_191223-abc12345/evaluations/vllm/20260314_200000",
        preset="full",
        scenarios=None,
        tags=None,
        pip_packages=None,
        env_backend="local",
        env_template=None,
        env_tool_schema=None,
        env_exec_config=None,
        upload_to_hf=None,
        update_model_card=False,
        with_loss=False,
        loss_dataset_name=None,
        loss_dataset_file=None,
        loss_max_seq_length=None,
        loss_completion_only=True,
    )

    assert "Evaluator.cloud_hf_job" in command
    assert "--bucket-id" in command
    assert "test-user/toolset-training-artifacts" in command
    assert "--run-prefix" in command
    assert "runs/hf_jobs/sft/20260314_191223-abc12345" in command
    assert "--preset" in command
    assert "full" in command
    assert "--env-backend" in command
    assert "local" in command
    assert "huggingface_hub>=1.5.0" in command
    assert "hf_transfer" in command
    assert "hf_xet" in command
    assert "pip install --upgrade --no-deps --target /tmp/hf-eval-runtime-site -r Evaluator/requirements.txt" in command
    assert (
        "pip install --upgrade --no-deps --target /tmp/hf-eval-bucket-sync-site "
        "'huggingface_hub>=1.5.0' hf_transfer hf_xet"
    ) in command
    runtime_install = command.split("pip install --upgrade --no-deps --target /tmp/hf-eval-runtime-site", 1)[1].split(" && ", 1)[0]
    bucket_sync_install = command.split("pip install --upgrade --no-deps --target /tmp/hf-eval-bucket-sync-site", 1)[1].split(" && ", 1)[0]
    assert "huggingface_hub" not in runtime_install
    assert "hf_transfer" not in runtime_install
    assert "hf_xet" not in runtime_install
    for overlay_install in (runtime_install, bucket_sync_install):
        assert " peft" not in overlay_install
        assert " torch" not in overlay_install
        assert " transformers" not in overlay_install
        assert " numpy" not in overlay_install
    assert "$(command -v python3 || command -v python)" in command
    assert "python3 -m Evaluator.cloud_hf_job" in command
    assert "export PYTHONPATH=/tmp/hf-eval-runtime-site${PYTHONPATH:+:$PYTHONPATH}" in command
    assert "export HF_BUCKET_SYNC_PYTHONPATH=/tmp/hf-eval-bucket-sync-site" in command
    assert "export PYTHONPATH=/tmp/hf-eval-bucket-sync-site" not in command
    assert "vllm==0.11.0" not in command


def test_build_eval_command_can_include_same_job_loss(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    command = handler._build_eval_command(
        helper_module="Evaluator.cloud_hf_job",
        bucket_id="test-user/toolset-training-artifacts",
        run_prefix="runs/hf_jobs/sft/20260314_191223-abc12345",
        eval_prefix="runs/hf_jobs/sft/20260314_191223-abc12345/evaluations/vllm/20260314_200000",
        preset="full",
        scenarios=None,
        tags=None,
        pip_packages=None,
        env_backend="none",
        env_template=None,
        env_tool_schema=None,
        env_exec_config=None,
        upload_to_hf=None,
        update_model_card=False,
        with_loss=True,
        loss_dataset_name="professorsynapse/claudesidian-synthetic-dataset",
        loss_dataset_file="train.jsonl",
        loss_max_seq_length=2048,
        loss_completion_only=True,
    )

    assert "--with-loss" in command
    assert "--loss-dataset-name" in command
    assert "professorsynapse/claudesidian-synthetic-dataset" in command
    assert "--loss-dataset-file" in command
    assert "train.jsonl" in command
    assert "export PYTHONPATH=/tmp/hf-eval-runtime-site${PYTHONPATH:+:$PYTHONPATH}" in command


def test_build_eval_command_installs_stage_pip_packages(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    command = handler._build_eval_command(
        helper_module="Evaluator.cloud_hf_job_vllm",
        bucket_id="test-user/toolset-training-artifacts",
        run_prefix="runs/hf_jobs/sft/20260314_191223-abc12345",
        eval_prefix="runs/hf_jobs/sft/20260314_191223-abc12345/evaluations/vllm/20260314_200000",
        preset="full",
        scenarios=None,
        tags=None,
        pip_packages=["vllm==0.12.0", "transformers==5.3.0"],
        env_backend="none",
        env_template=None,
        env_tool_schema=None,
        env_exec_config=None,
        upload_to_hf=None,
        update_model_card=False,
        with_loss=False,
        loss_dataset_name=None,
        loss_dataset_file=None,
        loss_max_seq_length=None,
        loss_completion_only=True,
    )

    assert "pip install --upgrade vllm==0.12.0 transformers==5.3.0" in command


def test_list_remote_runs_sorts_newest_first(repo_root, clean_env):
    clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    mock_hub = ModuleType("huggingface_hub")
    api = MagicMock()
    api.list_bucket_tree.return_value = [
        SimpleNamespace(type="directory", path="runs/hf_jobs/sft/20260314_191223-abc12345"),
        SimpleNamespace(type="directory", path="runs/hf_jobs/sft/20260314_181223-aaaabbbb"),
        SimpleNamespace(type="file", path="runs/hf_jobs/sft/README.md"),
    ]
    mock_hub.HfApi = MagicMock(return_value=api)

    runs = handler._list_remote_runs(mock_hub, "test-user/toolset-training-artifacts", "sft")

    assert [run["slug"] for run in runs] == [
        "20260314_191223-abc12345",
        "20260314_181223-aaaabbbb",
    ]


def test_select_run_latest_returns_newest(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root
    runs = [
        {"method": "sft", "slug": "20260314_191223-abc12345", "prefix": "runs/hf_jobs/sft/20260314_191223-abc12345"},
        {"method": "sft", "slug": "20260314_181223-aaaabbbb", "prefix": "runs/hf_jobs/sft/20260314_181223-aaaabbbb"},
    ]

    selected = handler._select_run(runs, "latest")

    assert selected["slug"] == "20260314_191223-abc12345"


def test_resolve_display_scenarios_uses_preset_when_no_explicit_scenarios(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    with patch("tuner.handlers.cloud_eval_handler.ConfigLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load_eval_run.return_value = SimpleNamespace(
            scenarios=["tool_prompts.yaml", "behavior_prompts.yaml"]
        )
        mock_loader_cls.return_value = mock_loader

        scenarios = handler.resolve_display_scenarios(preset="full", scenarios=None)

    assert scenarios == ["tool_prompts.yaml", "behavior_prompts.yaml"]


def test_resolve_eval_image_uses_eval_profile_defaults(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    image, profile = handler._resolve_eval_image()

    assert profile == "stable_unsloth"
    assert image.startswith("unsloth/unsloth:")


def test_resolve_eval_image_maps_stable_alias_for_unsloth(repo_root):
    handler = CloudEvalHandler(
        args=Namespace(eval_runtime="unsloth", eval_cloud_image=None, eval_image_profile="stable")
    )
    handler._repo_root = repo_root

    image, profile = handler._resolve_eval_image()

    assert profile == "stable_unsloth"
    assert image.startswith("unsloth/unsloth:")


def test_resolve_eval_image_uses_fast_vllm_profile_for_vllm_runtime(repo_root):
    handler = CloudEvalHandler(args=Namespace(eval_runtime="vllm", eval_cloud_image=None, eval_image_profile=None))
    handler._repo_root = repo_root

    image, profile = handler._resolve_eval_image()

    assert profile == "fast_vllm"
    assert image.startswith("vllm/vllm-openai:")


def test_resolve_eval_helper_module_supports_vllm(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    assert handler._resolve_eval_helper_module("vllm") == "Evaluator.cloud_hf_job_vllm"


def test_resolve_eval_timeout_prefers_explicit_eval_timeout(repo_root):
    handler = CloudEvalHandler(args=Namespace(eval_timeout_hours=8.0, timeout_hours=2.0))
    handler._repo_root = repo_root

    assert handler._resolve_eval_timeout_hours() == 8.0


def test_resolve_eval_timeout_uses_eval_config_before_provider(repo_root):
    handler = CloudEvalHandler(args=Namespace(eval_timeout_hours=None, timeout_hours=None))
    handler._repo_root = repo_root

    with patch.object(handler, "_hf_eval_settings", return_value={"timeout": "90m"}):
        with patch.object(handler, "_hf_jobs_settings", return_value={"timeout": "6h"}):
            assert handler._resolve_eval_timeout_hours() == 1.5


def test_resolve_eval_timeout_uses_provider_timeout(repo_root):
    handler = CloudEvalHandler(args=Namespace(eval_timeout_hours=None, timeout_hours=None))
    handler._repo_root = repo_root

    with patch.object(handler, "_hf_eval_settings", return_value={}):
        with patch.object(handler, "_hf_jobs_settings", return_value={"timeout": "6h"}):
            assert handler._resolve_eval_timeout_hours() == 6.0


def test_new_eval_timestamp_includes_nonce(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    timestamp = handler._new_eval_timestamp()
    prefix, nonce = timestamp.rsplit("_", 1)

    assert prefix.count("_") == 1
    assert len(nonce) == 4
    int(nonce, 16)


def test_handle_submits_hf_eval_job(repo_root, clean_env):
    clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
    args = Namespace(
        json=False,
        run="latest",
        method="sft",
        bucket=None,
        preset="full",
        scenario=None,
        tags=None,
        upload_to_hf=None,
        update_model_card=False,
        gpu=None,
        timeout_hours=2.0,
        eval_timeout_hours=None,
        eval_runtime=None,
        eval_image_profile=None,
        eval_cloud_image=None,
        with_loss=False,
        loss_dataset_name=None,
        loss_dataset_file=None,
        loss_max_seq_length=None,
        loss_no_completion_only=False,
        env_backend=None,
        env_template=None,
        env_tool_schema=None,
        env_exec_config=None,
    )
    handler = CloudEvalHandler(args=args)
    handler._repo_root = repo_root

    mock_hub = MagicMock()
    mock_job = MagicMock()
    mock_job.id = "eval-job-123"
    mock_job.url = "https://hf.co/jobs/eval-job-123"
    mock_hub.run_job.return_value = mock_job

    with patch.object(handler, "_validate_environment", return_value=mock_hub):
        with patch.object(handler, "_resolve_bucket_id", return_value="test-user/toolset-training-artifacts"):
            with patch.object(
                handler,
                "_list_remote_runs",
                return_value=[
                    {
                        "method": "sft",
                        "slug": "20260314_191223-abc12345",
                        "prefix": "runs/hf_jobs/sft/20260314_191223-abc12345",
                    }
                ],
            ):
                with patch("tuner.handlers.cloud_eval_handler.confirm", return_value=True):
                    with patch.object(handler, "_poll_job", return_value=0):
                        exit_code = handler.handle()

    assert exit_code == 0
    kwargs = mock_hub.run_job.call_args.kwargs
    assert kwargs["secrets"] == {
        "HF_TOKEN": "hf_test_token_12345",
        "HF_API_KEY": "hf_test_token_12345",
    }
    assert kwargs["flavor"] == "a10g-small"
    assert kwargs["image"].startswith("unsloth/unsloth:")
    assert "Evaluator.cloud_hf_job" in kwargs["command"][2]


def test_local_training_run_dir_uses_primary_output_dir(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    run_dir = handler._local_training_run_dir(
        "sft",
        "runs/hf_jobs/sft/20260314_191223-abc12345",
    )

    assert run_dir == repo_root / "Trainers" / "sft" / "sft_output" / "20260314_191223-abc12345"
