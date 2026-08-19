"""
Tests for tuner/backends/training/cloud/hf_jobs_backend.py

Covers:
- HFJobsBackend.name, get_available_methods
- validate_environment: no token, bad format, hub not installed, no run_job
- load_config: valid, unknown method, missing config
- execute: type check, job submission, polling flow, error masking
- _build_training_command: command structure, repo URL resolution
- _parse_timeout: hours, minutes, bare numbers, invalid
"""

import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tuner.backends.training.cloud.hf_jobs_backend import (
    DEFAULT_FLAVOR,
    DEFAULT_IMAGE,
    DEFAULT_TIMEOUT,
    HFJobsBackend,
    _parse_timeout,
)
from tuner.handlers.cloud_eval_handler import CloudEvalHandler
from tuner.handlers.cloud_pipeline_handler import CloudPipelineHandler
from tuner.handlers.cloud_run_handler import CloudRunHandler
from tuner.backends.training.cloud.base_cloud import RepoSource
from tuner.cloud import HF_BUCKET_SYNC_OVERLAY_PACKAGES
from tuner.core.config import CloudTrainingConfig, TrainingConfig
from tuner.core.exceptions import CloudProviderError, ConfigurationError
from tuner.project.source_bundle import GitSource, RepositoryLocation
from tuner.cloud.hf_volume_transport import HFVerifiedVolume, HFVerifiedVolumeSpec
from tuner.cloud.runtime_layout import build_runtime_layout
from tuner.project import ProjectContext


_FIXTURE_HEAD = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def verified_source_preparation(monkeypatch, tmp_path):
    """Give legacy backend tests an explicit, non-provider verified-source seam."""

    original_init = HFJobsBackend.__init__
    provider_volume = object()

    def initialize(self, repo_root, context=None):
        original_init(self, repo_root, context=context)
        volume_spec = HFVerifiedVolumeSpec(
            source="test-user/bootstrap",
            capsule_path="capsule",
            capsule_manifest_sha256="a" * 64,
            source_lock_path="source-lock.json",
            source_lock_sha256="b" * 64,
            checkout_policy_path="checkout-policy.json",
            checkout_policy_sha256="c" * 64,
            local_root=tmp_path.resolve(),
        )
        source_lock = SimpleNamespace(
            mode="standalone",
            run_id="test-run",
            project_source=SimpleNamespace(commit=_FIXTURE_HEAD),
            engine_source=SimpleNamespace(
                commit=_FIXTURE_HEAD,
                branch="main",
                submodule_path=None,
                location=SimpleNamespace(canonical_url="https://github.com/test/repo.git"),
            ),
        )
        self.source_preparation = SimpleNamespace(
            source_lock=source_lock,
            source_lock_sha256="b" * 64,
            source_lock_uri="tracking://test/source-lock.json",
            volume_spec=volume_spec,
            runtime_layout=build_runtime_layout(ProjectContext.standalone(engine_root=Path(repo_root))),
            remote_project_root="/workspace/project",
            remote_engine_root="/workspace/engine",
            physical_project_root="/workspace/source/project",
            physical_engine_root="/workspace/source/project",
            descriptor_uri="tracking://test/source-transport/descriptor.json",
            descriptor_sha256="d" * 64,
            provisioning_evidence_uri="tracking://test/source-transport/evidence.json",
            provisioning_evidence_sha256="e" * 64,
            source_transport_state="CONSUMABLE",
            require_consumable=lambda: None,
            prove_volume=lambda hub: HFVerifiedVolume(
                volume_spec, provider_volume, "d" * 64, "e" * 64
            ),
        )

    monkeypatch.setattr(HFJobsBackend, "__init__", initialize)


@pytest.fixture
def canonical_repo_source():
    canonical_source = GitSource(
        location=RepositoryLocation.parse("https://github.com/test/repo.git"),
        branch="main",
        commit=_FIXTURE_HEAD,
        dirty=False,
        pushed=True,
    )
    return RepoSource(
        url=canonical_source.location.canonical_url,
        branch=canonical_source.branch,
        commit=canonical_source.commit,
        canonical_source=canonical_source,
    )


def _cloud_config(**overrides):
    config = CloudTrainingConfig(
        method="sft",
        platform="hf_jobs",
        config_path=Path("/fake"),
        trainer_dir=Path("/fake"),
        model_name="test",
        dataset_file="test",
        epochs=1,
        batch_size=4,
        learning_rate=2e-4,
        provider="hf_jobs",
        gpu_type="a10g-small",
        timeout_hours=4.0,
        cloud_image=DEFAULT_IMAGE,
        hf_flavor="a10g-small",
        artifact_backend="hf_bucket",
        artifact_identifier="toolset-training-artifacts",
        artifact_mount_path="/workspace/outputs",
        repo_url="https://github.com/test/repo.git",
        repo_branch="main",
        repo_commit="abc12345def67890",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestHFJobsBackendProperties:
    def test_name(self, repo_root):
        backend = HFJobsBackend(repo_root)
        assert backend.name == "hf_jobs"

    def test_available_methods(self, repo_root):
        backend = HFJobsBackend(repo_root)
        # SSOT-DERIVED (CONTRACTS §5.1.1, backend-coder's #27 dedup): the method
        # list now derives from TRAINING_METHODS via `list(TRAINING_METHODS)`, so
        # we assert equality WITH the SSOT (preserving order) rather than a frozen
        # literal. This auto-tracks new methods (embedding, ace_step, …) without a
        # test edit, while still failing loudly if the derivation breaks.
        from shared.utilities.paths import TRAINING_METHODS

        assert backend.get_available_methods() == list(TRAINING_METHODS)
        # Spot-check the methods this cloud backend must advertise.
        assert {"embedding", "ace_step"} <= set(backend.get_available_methods())


class TestHFJobsValidateEnvironment:
    def test_fails_when_no_hf_token(self, repo_root, clean_env):
        backend = HFJobsBackend(repo_root)
        is_valid, error = backend.validate_environment()
        assert not is_valid
        assert "HF_TOKEN" in error

    def test_fails_when_token_bad_format(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "not-a-valid-token")
        backend = HFJobsBackend(repo_root)
        is_valid, error = backend.validate_environment()
        assert not is_valid
        assert "unexpected format" in error.lower()

    def test_fails_when_hub_not_installed(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                is_valid, error = backend.validate_environment()
        assert not is_valid
        assert "not installed" in error.lower()

    def test_fails_when_no_run_job(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        mock_hub = MagicMock(spec=[])  # No run_job attribute
        mock_hub.__version__ = "0.20.0"
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            is_valid, error = backend.validate_environment()
        assert not is_valid
        assert "does not support" in error.lower()

    def test_succeeds_with_valid_setup(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        mock_hub = MagicMock()
        mock_hub.run_job = MagicMock()
        mock_hub.create_bucket = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            is_valid, error = backend.validate_environment()
        assert is_valid
        assert error == ""

    def test_accepts_hf_api_key_alias(self, repo_root, clean_env):
        clean_env.setenv("HF_API_KEY", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        mock_hub = MagicMock()
        mock_hub.run_job = MagicMock()
        mock_hub.create_bucket = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            is_valid, error = backend.validate_environment()
        assert is_valid
        assert error == ""


class TestHFJobsLoadConfig:
    def test_loads_sft_config(self, repo_root, canonical_repo_source):
        backend = HFJobsBackend(repo_root)
        with patch(
            "tuner.backends.training.cloud.hf_jobs_backend.resolve_repo_source",
            return_value=canonical_repo_source,
        ):
            config = backend.load_config("sft")
        assert isinstance(config, CloudTrainingConfig)
        assert config.method == "sft"
        assert config.platform == "hf_jobs"
        assert config.provider == "hf_jobs"
        assert config.gpu_type == "a10g-small"
        assert config.hf_flavor == "a10g-small"
        assert config.timeout_hours == 4.0
        assert config.cloud_image == "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39"
        assert config.model_name == "test-org/test-model-sft"
        assert config.artifact_backend == "hf_bucket"
        assert config.artifact_identifier == "toolset-training-artifacts"
        assert config.repo_url == "https://github.com/test/repo.git"
        assert config.repo_branch == "main"
        assert config.repo_commit == _FIXTURE_HEAD

    def test_loads_kto_config(self, repo_root, canonical_repo_source):
        backend = HFJobsBackend(repo_root)
        with patch(
            "tuner.backends.training.cloud.hf_jobs_backend.resolve_repo_source",
            return_value=canonical_repo_source,
        ):
            config = backend.load_config("kto")
        assert config.method == "kto"
        assert config.artifact_mount_path == "/workspace/outputs"

    def test_loads_grpo_config(self, repo_root, canonical_repo_source):
        backend = HFJobsBackend(repo_root)
        with patch(
            "tuner.backends.training.cloud.hf_jobs_backend.resolve_repo_source",
            return_value=canonical_repo_source,
        ):
            config = backend.load_config("grpo")
        assert config.method == "grpo"
        assert config.config_path.name == "env_config.yaml"
        assert config.dataset_name == "professorsynapse/nexus-synthetic-data"
        assert config.dataset_file == "professorsynapse/nexus-synthetic-data/environment_rollouts/canonical/vault_shared_seed_dynamic_roles_aggregate_20260316.jsonl"

    def test_raises_on_unknown_method(self, repo_root):
        backend = HFJobsBackend(repo_root)
        with pytest.raises(ConfigurationError, match="Unknown method"):
            backend.load_config("orpo")

    def test_raises_on_missing_config(self, tmp_path):
        backend = HFJobsBackend(tmp_path)
        with pytest.raises(ConfigurationError, match="Training config not found"):
            backend.load_config("sft")


class TestHFJobsExecute:
    def _make_config(self):
        return _cloud_config()

    def test_raises_when_hub_not_installed(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = self._make_config()
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(CloudProviderError, match="exact-run approval"):
                    backend.execute(config, python_path="")

    def test_raises_when_wrong_config_type(self, repo_root):
        backend = HFJobsBackend(repo_root)
        # Pass a plain TrainingConfig instead of CloudTrainingConfig
        config = TrainingConfig(
            method="sft", platform="hf_jobs", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4,
        )
        mock_hub = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            with pytest.raises(CloudProviderError, match="requires CloudTrainingConfig"):
                backend.execute(config, python_path="")

    def test_forged_volume_binding_precedes_provider_error_path(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        config = self._make_config()
        mock_hub = MagicMock()
        mock_hub.run_job.side_effect = Exception("Error with hf_abc123 token")
        bucket_info = MagicMock()
        bucket_info.bucket_id = "test-user/toolset-training-artifacts"
        mock_hub.create_bucket.return_value = bucket_info
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            with pytest.raises(CloudProviderError) as exc_info:
                backend.execute(config, python_path="")
        assert "hf_abc123" not in str(exc_info.value)
        assert "exact-run approval" in str(exc_info.value).lower()
        mock_hub.run_job.assert_not_called()

    def test_volume_proof_failure_precedes_bucket_mutation_and_submission(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)

        def reject(_hub):
            raise CloudProviderError("pre-provisioned member digest mismatch")

        backend.source_preparation.prove_volume = reject
        mock_hub = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            with pytest.raises(CloudProviderError, match="exact-run approval"):
                backend.execute(self._make_config(), python_path="")
        mock_hub.create_bucket.assert_not_called()
        mock_hub.run_job.assert_not_called()

    def test_legacy_success_fixture_cannot_bypass_secure_submission_gate(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        config = self._make_config()

        mock_hub = MagicMock()
        bucket_info = MagicMock()
        bucket_info.bucket_id = "test-user/toolset-training-artifacts"
        mock_hub.create_bucket.return_value = bucket_info
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.url = "https://hf.co/jobs/123"
        mock_hub.run_job.return_value = mock_job

        # inspect_job returns completed on first check
        mock_job_info = MagicMock()
        mock_job_info.status.stage = "completed"
        mock_hub.inspect_job.return_value = mock_job_info
        mock_hub.fetch_job_logs.return_value = "Training done."

        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            with patch("tuner.backends.training.cloud.base_cloud.time.sleep"):
                with pytest.raises(CloudProviderError, match="exact-run approval"):
                    backend.execute(config, python_path="")
        mock_hub.create_bucket.assert_not_called()
        mock_hub.run_job.assert_not_called()

    def test_legacy_failure_fixture_cannot_bypass_secure_submission_gate(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        config = self._make_config()

        mock_hub = MagicMock()
        bucket_info = MagicMock()
        bucket_info.bucket_id = "test-user/toolset-training-artifacts"
        mock_hub.create_bucket.return_value = bucket_info
        mock_job = MagicMock()
        mock_job.id = "job-456"
        mock_job.url = None
        mock_hub.run_job.return_value = mock_job

        mock_job_info = MagicMock()
        mock_job_info.status.stage = "ERROR"
        mock_hub.inspect_job.return_value = mock_job_info
        mock_hub.fetch_job_logs.return_value = ""

        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            with patch("tuner.backends.training.cloud.base_cloud.time.sleep"):
                with pytest.raises(CloudProviderError, match="exact-run approval"):
                    backend.execute(config, python_path="")
        mock_hub.run_job.assert_not_called()

    def test_polling_path_is_unreachable_before_secure_submission_gate(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        config = self._make_config()

        mock_hub = MagicMock()
        bucket_info = MagicMock()
        bucket_info.bucket_id = "test-user/toolset-training-artifacts"
        mock_hub.create_bucket.return_value = bucket_info
        mock_job = MagicMock()
        mock_job.id = "job-789"
        mock_job.url = None
        mock_hub.run_job.return_value = mock_job

        mock_job_info = MagicMock()
        mock_job_info.status.stage = "ERROR"
        mock_hub.inspect_job.return_value = mock_job_info
        mock_hub.fetch_job_logs.return_value = ""

        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            with patch("tuner.backends.training.cloud.base_cloud.time.sleep"):
                with patch.object(backend, "_should_use_remote_dashboard", return_value=False):
                    with patch.object(backend, "_recover_completed_run_from_bucket", return_value=True):
                        with patch.object(backend, "_finalize_completed_job", return_value=0) as mock_finalize:
                            with pytest.raises(CloudProviderError, match="exact-run approval"):
                                backend.execute(config, python_path="")

        mock_finalize.assert_not_called()
        mock_hub.run_job.assert_not_called()

    def test_raises_when_bucket_api_unavailable(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        config = self._make_config()
        mock_hub = MagicMock(spec=["run_job", "__version__"])
        mock_hub.__version__ = "1.2.3"
        with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
            with pytest.raises(CloudProviderError, match="exact-run approval"):
                backend.execute(config, python_path="")


def test_execute_authorization_barrier_precedes_sdk_and_config_helpers(repo_root) -> None:
    backend = HFJobsBackend(repo_root)
    events: list[str] = []

    def barrier(*, route: str):
        events.append(f"barrier:{route}")
        raise CloudProviderError("authorization stopped")

    with patch(
        "tuner.backends.training.cloud.hf_jobs_backend."
        "require_current_hf_source_submission_authorization",
        side_effect=barrier,
    ), patch(
        "tuner.backends.training.cloud.hf_jobs_backend.load_huggingface_hub"
    ) as load_hub, patch.object(
        backend, "prepare_source"
    ) as prepare_source, patch.object(
        backend, "_build_training_command"
    ) as build_command, patch.object(
        backend, "_ensure_hf_bucket"
    ) as ensure_bucket:
        with pytest.raises(CloudProviderError, match="authorization stopped"):
            backend.execute(_cloud_config(), python_path="")

    assert events == ["barrier:backend.execute"]
    load_hub.assert_not_called()
    prepare_source.assert_not_called()
    build_command.assert_not_called()
    ensure_bucket.assert_not_called()


def test_cloud_run_barrier_precedes_defaults_source_provider_and_command_helpers(repo_root) -> None:
    handler = CloudRunHandler(args=SimpleNamespace())
    handler._repo_root = repo_root
    events: list[str] = []

    def barrier(*, route: str):
        events.append(f"barrier:{route}")
        raise CloudProviderError("authorization stopped")

    with patch(
        "tuner.handlers.cloud_run_handler.require_current_hf_source_submission_authorization",
        side_effect=barrier,
    ), patch.object(handler, "_hf_defaults") as defaults, patch(
        "tuner.handlers.cloud_run_handler.preflight_hf_source_lock"
    ) as preflight, patch(
        "tuner.handlers.cloud_run_handler.load_huggingface_hub"
    ) as load_hub, patch(
        "tuner.handlers.cloud_run_handler.resolve_hf_bucket_id"
    ) as resolve_bucket, patch(
        "tuner.handlers.cloud_run_handler.build_bash_command"
    ) as build_command:
        with pytest.raises(CloudProviderError, match="authorization stopped"):
            handler._compile_hf_job(repo_root / "job.yaml", {"provider": "hf_jobs"})

    assert events == ["barrier:cloud-run.compile"]
    defaults.assert_not_called()
    preflight.assert_not_called()
    load_hub.assert_not_called()
    resolve_bucket.assert_not_called()
    build_command.assert_not_called()


def test_cloud_eval_barrier_precedes_settings_source_provider_and_command_helpers(repo_root) -> None:
    handler = CloudEvalHandler(args=SimpleNamespace(json=False))
    handler._repo_root = repo_root
    events: list[str] = []

    def barrier(*, route: str):
        events.append(f"barrier:{route}")
        raise CloudProviderError("authorization stopped")

    with patch(
        "tuner.handlers.cloud_eval_handler.require_current_hf_source_submission_authorization",
        side_effect=barrier,
    ), patch.object(handler, "_hf_jobs_settings") as settings, patch.object(
        handler, "_prepare_source"
    ) as prepare_source, patch.object(
        handler, "_validate_environment"
    ) as validate_environment, patch.object(
        handler, "_build_eval_command"
    ) as build_command:
        assert handler.handle() == 1

    assert events == ["barrier:cloud-eval.handle"]
    settings.assert_not_called()
    prepare_source.assert_not_called()
    validate_environment.assert_not_called()
    build_command.assert_not_called()


def test_cloud_pipeline_barrier_precedes_backend_and_plan_compilation(repo_root) -> None:
    handler = CloudPipelineHandler(args=SimpleNamespace(json=False))
    handler._repo_root = repo_root
    events: list[str] = []

    def barrier(*, route: str):
        events.append(f"barrier:{route}")
        raise CloudProviderError("authorization stopped")

    with patch(
        "tuner.handlers.cloud_pipeline_handler.require_current_hf_source_submission_authorization",
        side_effect=barrier,
    ), patch(
        "tuner.handlers.cloud_pipeline_handler.TrainingBackendRegistry.get"
    ) as backend_get, patch(
        "tuner.handlers.cloud_pipeline_handler.CloudEvalHandler"
    ) as eval_handler:
        assert handler.handle() == 1

    assert events == ["barrier:cloud-pipeline.handle"]
    backend_get.assert_not_called()
    eval_handler.assert_not_called()


class TestBuildTrainingCommand:
    def test_new_run_timestamp_includes_nonce(self, repo_root):
        backend = HFJobsBackend(repo_root)
        timestamp = backend._new_run_timestamp()
        prefix, nonce = timestamp.rsplit("_", 1)

        assert prefix.count("_") == 1
        assert len(nonce) == 4
        int(nonce, 16)

    def test_command_structure(self, repo_root, clean_env):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config()
        cmd = backend._build_training_command(config, timestamp="20260314_181946")
        assert "$(command -v python3 || command -v python) -m pip install --disable-pip-version-check" in cmd
        assert "$(command -v python3 || command -v python) -m pip install --upgrade pyyaml" not in cmd
        assert "mkdir -p /workspace/cache/hf-bucket-sync-site" in cmd
        assert "$(command -v python3 || command -v python) -m pip install --upgrade --no-deps --target /workspace/cache/hf-bucket-sync-site 'huggingface_hub>=1.5.0' hf_transfer hf_xet" in cmd
        assert " huggingface_hub>=1.5.0 " not in cmd
        assert "export HF_BUCKET_SYNC_PYTHON=$(command -v python3 || command -v python)" in cmd
        assert "export HF_BUCKET_SYNC_PYTHONPATH=/workspace/cache/hf-bucket-sync-site" in cmd
        assert "export CLOUD_PROVIDER=hf_jobs" in cmd
        assert "export CLOUD_GPU_TYPE=a10g-small" in cmd
        assert "git clone" not in cmd
        assert "synaptic-bootstrap-capsule.json" in cmd
        assert cmd.index("synaptic-bootstrap-capsule.json") < cmd.index("-m tuner.cloud.hf_volume_transport _verify-identities")
        identity_index = cmd.index("-m tuner.cloud.hf_volume_transport _verify-identities")
        projection_index = cmd.index("-m tuner.cloud.hf_volume_transport _project-layout")
        workload_index = cmd.index("pip install --disable-pip-version-check")
        assert identity_index < projection_index < workload_index
        projection_end = cmd.index(" && ", projection_index)
        assert "/workspace/source" not in cmd[projection_end:]
        assert "export SYNAPTIC_PROJECT_ROOT=/workspace/project" in cmd
        assert "export SYNAPTIC_ENGINE_ROOT=/workspace/engine" in cmd
        assert "cd /workspace/engine/Trainers/sft" in cmd
        assert "python train_sft.py" in cmd
        assert "--output-root /workspace/artifacts" in cmd
        assert "--artifact-backend hf_bucket" in cmd
        assert "--artifact-bucket toolset-training-artifacts" in cmd
        assert "--artifact-prefix runs/hf_jobs/sft/20260314_181946-abc12345" in cmd

    def test_bucket_sync_overlay_packages_include_xet(self):
        assert HF_BUCKET_SYNC_OVERLAY_PACKAGES == (
            "huggingface_hub>=1.5.0",
            "hf_transfer",
            "hf_xet",
        )

    def test_command_installs_stage_pip_packages(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(
            pip_packages=[
                "unsloth==2026.4.2",
                "unsloth-zoo==2026.4.2",
                "transformers==5.3.0",
            ]
        )

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "pip install --upgrade unsloth==2026.4.2 unsloth-zoo==2026.4.2 transformers==5.3.0" in cmd

    def test_command_includes_lora_variant_overrides(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(
            lora_r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            use_dora=True,
            use_rslora=True,
            init_lora_weights="loftq",
            lora_target_modules="all-linear",
        )

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--lora-r 128" in cmd
        assert "--lora-alpha 256" in cmd
        assert "--lora-dropout 0.05" in cmd
        assert "--use-dora" in cmd
        assert "--use-rslora" in cmd
        assert "--init-lora-weights loftq" in cmd
        assert "--lora-target-modules all-linear" in cmd

    def test_command_includes_evolutionary_overrides(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(
            evolutionary_enabled=True,
            evolutionary_candidates=4,
            evolutionary_eval_batch_size=2,
            evolutionary_validation_config="configs/fitness/tool_calling.yaml",
            evolutionary_strategy="antithetic_noise",
            evolutionary_noise_scale=0.03,
            evolutionary_max_grad_norm=1.0,
            evolutionary_scale_factors=[0.5, 1.0, 1.5],
            evolutionary_selection_method="best",
            evolutionary_min_improvement=0.01,
            evolutionary_min_relative_improvement=0.0001,
            evolutionary_noise_floor_epsilon=0.000001,
            evolutionary_eval_frequency=5,
            evolutionary_warmup_steps=200,
            evolutionary_cache_baseline=True,
            evolutionary_log_candidates=False,
            evolutionary_log_selected=True,
        )

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--evolutionary-enabled" in cmd
        assert "--evolutionary-candidates 4" in cmd
        assert "--evolutionary-eval-batch-size 2" in cmd
        assert "--evolutionary-validation-config configs/fitness/tool_calling.yaml" in cmd
        assert "--evolutionary-strategy antithetic_noise" in cmd
        assert "--evolutionary-noise-scale 0.03" in cmd
        assert "--evolutionary-max-grad-norm 1.0" in cmd
        assert "--evolutionary-scale-factors 0.5,1.0,1.5" in cmd
        assert "--evolutionary-selection-method best" in cmd
        assert "--evolutionary-min-improvement 0.01" in cmd
        assert "--evolutionary-min-relative-improvement 0.0001" in cmd
        assert "--evolutionary-noise-floor-epsilon 1e-06" in cmd
        assert "--evolutionary-eval-frequency 5" in cmd
        assert "--evolutionary-warmup-steps 200" in cmd
        assert "--evolutionary-cache-baseline" in cmd
        assert "--evolutionary-no-log-candidates" in cmd
        assert "--evolutionary-log-selected" in cmd

    def test_command_includes_seed_when_set(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(seed=1234)

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--seed 1234" in cmd

    def test_command_includes_seed_zero(self, repo_root):
        # seed=0 is a legitimate seed; emission uses `is not None`, not a truthy
        # guard, so it must reach the trainer command rather than being dropped.
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(seed=0)

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--seed 0" in cmd

    def test_command_omits_seed_when_absent(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config()  # seed defaults to None

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--seed" not in cmd

    def test_command_includes_beta_for_dpo(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(method="dpo", beta=0.5)

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--beta 0.5" in cmd

    def test_command_includes_beta_for_kto(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(method="kto", beta=0.1)

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--beta 0.1" in cmd

    def test_command_omits_beta_for_sft(self, repo_root):
        # beta is a DPO/KTO-only parameter; SFT has no --beta flag, so even a
        # set beta value must not be emitted for an SFT run.
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(method="sft", beta=0.5)

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--beta" not in cmd

    def test_command_omits_beta_when_absent(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(method="dpo")  # beta defaults to None

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--beta" not in cmd

    def test_command_includes_chat_template_kwargs_for_sft(self, repo_root):
        # The cloud lane must forward chat_template_kwargs as the same JSON-string
        # --chat-template-kwargs flag the local lane uses, so a protocol pin like
        # enable_thinking=False reaches the trainer's preprocessing on the cloud
        # path. One wire format across both lanes.
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(
            method="sft", chat_template_kwargs={"enable_thinking": False}
        )

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        # The assembled command string shell-quotes each arg; the JSON payload
        # (which contains spaces and braces) is therefore single-quoted so it
        # reaches the trainer as one argv element.
        assert "--chat-template-kwargs '{\"enable_thinking\": false}'" in cmd

    def test_command_omits_chat_template_kwargs_for_dpo(self, repo_root):
        # chat_template_kwargs is an SFT-only flag: DPO/KTO template internally via
        # TRL and expose no --chat-template-kwargs argument, so even a set value
        # must not be emitted for a DPO run.
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(
            method="dpo", chat_template_kwargs={"enable_thinking": False}
        )

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--chat-template-kwargs" not in cmd

    def test_command_omits_chat_template_kwargs_for_kto(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(
            method="kto", chat_template_kwargs={"enable_thinking": False}
        )

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--chat-template-kwargs" not in cmd

    def test_command_omits_chat_template_kwargs_when_absent(self, repo_root):
        # Default None ⇒ no flag ⇒ byte-identical command for every existing cloud
        # config, preserving the tuner's generic default rendering.
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(method="sft")  # chat_template_kwargs defaults to None

        cmd = backend._build_training_command(config, timestamp="20260314_181946")

        assert "--chat-template-kwargs" not in cmd

    def test_cloud_training_config_exposes_seed_and_beta(self):
        # Regression guard: CloudTrainingConfig must carry seed/beta so recipe
        # values are not silently dropped before the command builder runs.
        config = _cloud_config()
        assert hasattr(config, "seed")
        assert hasattr(config, "beta")
        assert config.seed is None
        assert config.beta is None

    def test_raises_when_verified_source_is_missing(self, repo_root, clean_env):
        backend = HFJobsBackend(repo_root)
        backend.source_preparation = None
        with pytest.raises(CloudProviderError, match="secure source preparation"):
            backend._build_training_command(_cloud_config())

    def test_grpo_command_uses_env_runtime_bootstrap_and_bucket_sync(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(
            method="grpo",
            config_path=repo_root / "Trainers" / "grpo" / "configs" / "env_config.yaml",
            trainer_dir=repo_root / "Trainers" / "grpo",
            model_name="professorsynapse/Nexus-Quark-L2.5.28",
            dataset_name="professorsynapse/nexus-synthetic-data",
            dataset_file="professorsynapse/nexus-synthetic-data/environment_rollouts/canonical/vault_shared_seed_dynamic_roles_aggregate_20260316.jsonl",
        )

        cmd = backend._build_training_command(config, timestamp="20260322_170000")

        assert "$(command -v python3 || command -v python) -m virtualenv --no-download /workspace/cache/venvs/grpo-openenv" in cmd
        assert "cd /workspace/engine/Trainers/grpo && python train_env_grpo.py --config /workspace/engine/Trainers/grpo/configs/env_config.yaml" in cmd
        assert "--model-name professorsynapse/Nexus-Quark-L2.5.28" in cmd
        assert "--dataset-name professorsynapse/nexus-synthetic-data" in cmd
        assert "--dataset-file environment_rollouts/canonical/vault_shared_seed_dynamic_roles_aggregate_20260316.jsonl" in cmd
        assert "--output-dir /workspace/artifacts/grpo/20260322_170000" in cmd
        assert "python -m shared.hf_bucket_sync_helper /workspace/artifacts/grpo/20260322_170000" in cmd

    def test_build_artifact_prefix(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = _cloud_config()
        assert (
            backend._build_artifact_prefix(config, "20260314_181946")
            == "runs/hf_jobs/sft/20260314_181946-abc12345"
        )


class TestHFJobsArtifacts:
    def test_download_completed_run_uses_primary_output_dir(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(artifact_identifier="test-user/toolset-training-artifacts")

        with patch.object(backend, "_sync_bucket_path") as mock_sync:
            local_dir = backend._download_completed_run(
                config=config,
                artifact_prefix="runs/hf_jobs/sft/20260314_191223-abc12345",
            )

        expected_dir = repo_root / "Trainers" / "sft" / "sft_output" / "20260314_191223-abc12345"
        assert local_dir == expected_dir
        mock_sync.assert_called_once_with(
            "hf://buckets/test-user/toolset-training-artifacts/runs/hf_jobs/sft/20260314_191223-abc12345",
            expected_dir,
            token="hf_test_token_12345",
        )

    def test_recover_completed_run_from_bucket_returns_true_when_artifacts_complete(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(artifact_identifier="test-user/toolset-training-artifacts")

        def fake_sync(remote_uri, local_dir, token=None):
            local_dir.mkdir(parents=True, exist_ok=True)
            final_model_dir = local_dir / "final_model"
            final_model_dir.mkdir(parents=True, exist_ok=True)
            (final_model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (local_dir / "training_lineage.json").write_text("{}", encoding="utf-8")

        with patch.object(backend, "_sync_bucket_path", side_effect=fake_sync):
            assert backend._recover_completed_run_from_bucket(
                config=config,
                artifact_prefix="runs/hf_jobs/sft/20260314_191223-abc12345",
            ) is True

    def test_remote_dashboard_timeout_recovers_when_artifacts_are_complete(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_token_12345")
        backend = HFJobsBackend(repo_root)
        config = _cloud_config(
            timeout_hours=0.0,
            artifact_identifier="test-user/toolset-training-artifacts",
        )

        mock_hub_module = ModuleType("huggingface_hub")
        mock_hub_module.sync_bucket = lambda *args, **kwargs: None

        class DummyDashboard:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, **kwargs):
                return None

            def _process_log_line(self, line):
                return None

        mock_shared_ui = ModuleType("shared.ui")
        mock_shared_ui.LiveDashboard = DummyDashboard

        with patch.dict("sys.modules", {"huggingface_hub": mock_hub_module, "shared.ui": mock_shared_ui}):
            with patch.object(backend, "_recover_completed_run_from_bucket", return_value=True):
                with patch.object(backend, "_finalize_completed_job", return_value=0) as mock_finalize:
                    exit_code = backend._watch_job_with_remote_dashboard(
                        config=config,
                        huggingface_hub=MagicMock(),
                        job_id="job-123",
                        artifact_prefix="runs/hf_jobs/sft/20260314_191223-abc12345",
                    )

        assert exit_code == 0
        mock_finalize.assert_called_once_with(
            config=config,
            artifact_prefix="runs/hf_jobs/sft/20260314_191223-abc12345",
        )


# ---------------------------------------------------------------------------
# _parse_timeout (module-level function)
# ---------------------------------------------------------------------------


class TestParseTimeout:
    def test_hours_integer(self):
        assert _parse_timeout("4h") == 4.0

    def test_hours_decimal(self):
        assert _parse_timeout("2.5h") == 2.5

    def test_minutes(self):
        assert _parse_timeout("90m") == 1.5

    def test_minutes_decimal(self):
        assert _parse_timeout("30m") == 0.5

    def test_bare_number(self):
        assert _parse_timeout("6") == 6.0

    def test_bare_decimal(self):
        assert _parse_timeout("3.5") == 3.5

    def test_invalid_hours_returns_default(self):
        assert _parse_timeout("abch") == 4.0

    def test_invalid_minutes_returns_default(self):
        assert _parse_timeout("xyzm") == 4.0

    def test_invalid_bare_returns_default(self):
        assert _parse_timeout("invalid") == 4.0

    def test_whitespace_handling(self):
        assert _parse_timeout("  4h  ") == 4.0

    def test_case_insensitive(self):
        assert _parse_timeout("4H") == 4.0
        assert _parse_timeout("90M") == 1.5
