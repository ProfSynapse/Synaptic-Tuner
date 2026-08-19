from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from tuner.backends.training.cloud.base_cloud import RepoSource
from tuner.handlers.cloud_eval_handler import CloudEvalHandler
from tuner.handlers.stages.hf_eval_stage import HFEvalStageRunner
from tuner.handlers.stages._util import HFSourcePreparation
from tuner.cloud.hf_volume_transport import HFVerifiedVolume, HFVerifiedVolumeSpec
from tuner.cloud.runtime_layout import build_runtime_layout
from tuner.project import ProjectContext
from tuner.project.source_bundle import GitSource, RepositoryLocation


_FIXTURE_HEAD = "0123456789abcdef0123456789abcdef01234567"


def _canonical_repo_source() -> RepoSource:
    source = GitSource(
        location=RepositoryLocation.parse("https://github.com/test/repo.git"),
        branch="main",
        commit=_FIXTURE_HEAD,
        dirty=False,
        pushed=True,
    )


def _source_preparation(repo_root) -> HFSourcePreparation:
    lock = SimpleNamespace(
        mode="standalone",
        run_id="eval-test",
        project_source=SimpleNamespace(commit=_FIXTURE_HEAD),
        engine_source=SimpleNamespace(commit=_FIXTURE_HEAD, submodule_path=None),
    )
    volume_spec = HFVerifiedVolumeSpec(
        source="test-user/bootstrap",
        capsule_path="capsule",
        capsule_manifest_sha256="a" * 64,
        source_lock_path="source-lock.json",
        source_lock_sha256="b" * 64,
        checkout_policy_path="checkout-policy.json",
        checkout_policy_sha256="c" * 64,
        local_root=repo_root.resolve(),
    )
    preparation = HFSourcePreparation(
        lock, "b" * 64, "tracking://test/source-lock.json", volume_spec,
        build_runtime_layout(ProjectContext.standalone(engine_root=repo_root)),
        descriptor_uri="tracking://test/source-transport/descriptor.json",
        descriptor_sha256="d" * 64,
        provisioning_evidence_uri="tracking://test/source-transport/evidence.json",
        provisioning_evidence_sha256="e" * 64,
        source_transport_state="CONSUMABLE",
    )
    object.__setattr__(
        preparation,
        "prove_volume",
        lambda hub: HFVerifiedVolume(volume_spec, "verified-volume", "d" * 64, "e" * 64),
    )
    return preparation
    return RepoSource(
        url=source.location.canonical_url,
        branch=source.branch,
        commit=source.commit,
        canonical_source=source,
    )


def test_build_eval_command_uses_cloud_job_helper(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    command = handler._build_eval_command(
            source_preparation=_source_preparation(repo_root),
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
    assert "pip install --upgrade --no-deps --target /workspace/cache/hf-eval-runtime-site -r Evaluator/requirements.txt" in command
    assert (
        "pip install --upgrade --no-deps --target /workspace/cache/hf-eval-bucket-sync-site "
        "'huggingface_hub>=1.5.0' hf_transfer hf_xet"
    ) in command
    runtime_install = command.split("pip install --upgrade --no-deps --target /workspace/cache/hf-eval-runtime-site", 1)[1].split(" && ", 1)[0]
    bucket_sync_install = command.split("pip install --upgrade --no-deps --target /workspace/cache/hf-eval-bucket-sync-site", 1)[1].split(" && ", 1)[0]
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
    assert "export PYTHONPATH=/workspace/cache/hf-eval-runtime-site${PYTHONPATH:+:$PYTHONPATH}" in command
    assert "export HF_BUCKET_SYNC_PYTHONPATH=/workspace/cache/hf-eval-bucket-sync-site" in command
    assert "export PYTHONPATH=/workspace/cache/hf-eval-bucket-sync-site" not in command
    assert "vllm==0.11.0" not in command


def test_build_eval_command_can_include_same_job_loss(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    command = handler._build_eval_command(
            source_preparation=_source_preparation(repo_root),
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
    assert "export PYTHONPATH=/workspace/cache/hf-eval-runtime-site${PYTHONPATH:+:$PYTHONPATH}" in command


def test_build_eval_command_installs_stage_pip_packages(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    command = handler._build_eval_command(
            source_preparation=_source_preparation(repo_root),
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


def test_read_only_eval_inspection_does_not_claim_volume_launch_capability(clean_env):
    clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
    handler = CloudEvalHandler(args=Namespace())
    with patch("tuner.handlers.cloud_eval_handler.load_huggingface_hub", return_value=object()) as loader:
        handler._validate_environment(for_launch=False)
    assert loader.call_args.kwargs["require_apis"] == ("create_bucket", "HfApi")


def test_evaluation_plan_args_are_pure_and_preserve_same_job_settings(repo_root):
    runner = HFEvalStageRunner(repo_root=repo_root, tracking_service=MagicMock())
    preparation = object()
    spec = SimpleNamespace(
        method="sft",
        evaluation=SimpleNamespace(
            preset="full",
            scenarios=("one.yaml",),
            tags="smoke",
            runtime="vllm",
            image_profile="fast_vllm",
            cloud_image=None,
            pip_packages=("stage-package==1.0",),
            gpu="a10g-small",
            timeout_hours=2.0,
        ),
        post_training=SimpleNamespace(mode="same_job"),
        loss=SimpleNamespace(
            enabled=True,
            max_seq_length=2048,
            completion_only=False,
        ),
        training=SimpleNamespace(max_seq_length=4096),
        dataset=SimpleNamespace(source="owner/dataset", file="train.jsonl"),
    )
    args = runner.build_evaluation_plan_args(
        spec=spec,
        artifact_prefix="runs/hf_jobs/sft/run-1",
        bucket_id="owner/bucket",
        source_preparation=preparation,
    )
    assert args.eval_runtime == "vllm"
    assert args.eval_pip_packages == ["stage-package==1.0"]
    assert args.with_loss is True
    assert args.loss_dataset_name == "owner/dataset"
    assert args.loss_dataset_file == "train.jsonl"
    assert args.loss_max_seq_length == 2048
    assert args.loss_no_completion_only is True
    assert args._source_preparation is preparation


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


def test_handle_fails_at_exact_run_approval_before_provider_calls(repo_root, clean_env):
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

    with patch.object(handler, "_prepare_source", return_value=_source_preparation(repo_root)):
        with patch.object(handler, "_validate_environment", return_value=mock_hub) as validate_environment:
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

    assert exit_code == 1
    validate_environment.assert_not_called()
    mock_hub.run_job.assert_not_called()


def test_local_training_run_dir_uses_primary_output_dir(repo_root):
    handler = CloudEvalHandler(args=Namespace())
    handler._repo_root = repo_root

    run_dir = handler._local_training_run_dir(
        "sft",
        "runs/hf_jobs/sft/20260314_191223-abc12345",
    )

    assert run_dir == repo_root / "Trainers" / "sft" / "sft_output" / "20260314_191223-abc12345"
