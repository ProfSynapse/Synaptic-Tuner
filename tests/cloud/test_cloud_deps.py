"""
Cross-backend dependency consistency tests for cloud training.

Verifies that all cloud backends install the same project-specific
dependencies, use correct Docker images, and do NOT install packages
that are pre-installed in the unsloth Docker image.

Covers:
- Cross-backend project dep consistency (HF Jobs, RunPod, RunPod sync)
- Version pin assertions: HF Jobs / RunPod must NOT install unsloth/trl/transformers
- Default image tag validation per backend
- cloud_config.yaml dependencies section loading
- RunPod sync script alignment with RunPod backend
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tuner.backends.training.cloud.hf_jobs_backend import (
    DEFAULT_IMAGE as HF_DEFAULT_IMAGE,
    HFJobsBackend,
)
from tuner.backends.training.cloud.runpod_backend import RunPodBackend
from tuner.core.config import CloudTrainingConfig


# ---------------------------------------------------------------------------
# Expected values — single source of truth for assertions
# ---------------------------------------------------------------------------

EXPECTED_PROJECT_DEPS = {"pyyaml", "wandb", "hf_transfer", "python-dotenv", "rich"}
EXPECTED_UNSLOTH_IMAGE = "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39"

# Packages pre-installed in the unsloth Docker image that backends
# must NOT pip-install (doing so causes version conflicts)
IMAGE_PREINSTALLED = {"unsloth", "trl", "transformers", "torch", "datasets", "peft",
                      "accelerate", "bitsandbytes"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf_cloud_config(**overrides):
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
        cloud_image=HF_DEFAULT_IMAGE,
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


def _runpod_cloud_config(**overrides):
    config = CloudTrainingConfig(
        method="sft",
        platform="runpod",
        config_path=Path("/fake"),
        trainer_dir=Path("/fake"),
        model_name="test",
        dataset_file="test",
        epochs=1,
        batch_size=4,
        learning_rate=2e-4,
        provider="runpod",
        gpu_type="NVIDIA A100 SXM",
        timeout_hours=6,
        artifact_backend="runpod_network_volume",
        artifact_identifier="runpod-vol-123",
        artifact_mount_path="/runpod-volume",
        repo_url="https://github.com/test/repo.git",
        repo_branch="main",
        repo_commit="abc12345def67890",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _extract_pip_packages(command: str) -> set:
    """Extract package names from 'pip install pkg1 pkg2 ...' in a command."""
    packages = set()
    for part in command.split("&&"):
        part = part.strip()
        if part.startswith("pip install"):
            # Everything after 'pip install' are package names/specifiers
            tokens = part.split()[2:]  # skip 'pip' and 'install'
            for tok in tokens:
                # Strip version pins (e.g. "torch==2.7.0" -> "torch")
                name = tok.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0]
                packages.add(name.lower())
    return packages


# ---------------------------------------------------------------------------
# Cross-backend dependency consistency
# ---------------------------------------------------------------------------


class TestCrossBackendDepsConsistency:
    """Verify all backends install the same project-specific dependencies."""

    def test_hf_jobs_installs_project_deps(self, repo_root, clean_env):
        backend = HFJobsBackend(repo_root)
        config = _hf_cloud_config()
        cmd = backend._build_training_command(config)
        packages = _extract_pip_packages(cmd)
        assert EXPECTED_PROJECT_DEPS.issubset(packages), (
            f"HF Jobs missing project deps: {EXPECTED_PROJECT_DEPS - packages}"
        )

    def test_runpod_installs_project_deps(self, repo_root, clean_env):
        backend = RunPodBackend(repo_root)
        config = _runpod_cloud_config()
        cmd = backend._build_startup_command(config, {})
        packages = _extract_pip_packages(cmd)
        assert EXPECTED_PROJECT_DEPS.issubset(packages), (
            f"RunPod missing project deps: {EXPECTED_PROJECT_DEPS - packages}"
        )

    def test_runpod_sync_installs_project_deps(self, repo_root):
        """RunPod sync startup command must install all project deps."""
        from Trainers.cloud.runpod_sync import build_training_startup_command

        # Patch load_project_deps in the sync module's namespace to read
        # from the test fixture's cloud_config.yaml (the real function
        # resolves relative to __file__, which won't be the tmp_path).
        from tuner.backends.training.cloud.base_cloud import load_project_deps
        config_path = repo_root / "Trainers" / "cloud" / "cloud_config.yaml"

        with patch(
            "Trainers.cloud.runpod_sync.load_project_deps",
            side_effect=lambda p=None: load_project_deps(config_path),
        ):
            cmd = build_training_startup_command(
                method="sft",
                repo_url="https://github.com/test/repo.git",
            )
        packages = _extract_pip_packages(cmd)
        assert EXPECTED_PROJECT_DEPS.issubset(packages), (
            f"RunPod sync missing project deps: {EXPECTED_PROJECT_DEPS - packages}"
        )

    def test_hf_jobs_and_runpod_install_same_deps(self, repo_root, clean_env):
        """HF Jobs and RunPod must install identical project-specific packages."""
        hf_backend = HFJobsBackend(repo_root)
        hf_cmd = hf_backend._build_training_command(_hf_cloud_config())
        hf_packages = _extract_pip_packages(hf_cmd)

        rp_backend = RunPodBackend(repo_root)
        rp_cmd = rp_backend._build_startup_command(_runpod_cloud_config(), {})
        rp_packages = _extract_pip_packages(rp_cmd)

        assert hf_packages == rp_packages, (
            f"HF Jobs and RunPod install different packages.\n"
            f"  HF Jobs only: {hf_packages - rp_packages}\n"
            f"  RunPod only:  {rp_packages - hf_packages}"
        )

    def test_runpod_backend_and_sync_deps_match(self, repo_root, clean_env):
        """RunPod backend and sync script must install identical packages.

        Both read from cloud_config.yaml via load_project_deps(), so their
        pip install lines should contain the same set of packages.
        """
        from Trainers.cloud.runpod_sync import build_training_startup_command
        from tuner.backends.training.cloud.base_cloud import load_project_deps

        config_path = repo_root / "Trainers" / "cloud" / "cloud_config.yaml"

        # Get packages from RunPod backend
        backend = RunPodBackend(repo_root)
        rp_config = _runpod_cloud_config()
        backend_cmd = backend._build_startup_command(rp_config, {})
        backend_packages = _extract_pip_packages(backend_cmd)

        # Get packages from RunPod sync
        with patch(
            "Trainers.cloud.runpod_sync.load_project_deps",
            side_effect=lambda p=None: load_project_deps(config_path),
        ):
            sync_cmd = build_training_startup_command(
                method="sft",
                repo_url="https://github.com/test/repo.git",
            )
        sync_packages = _extract_pip_packages(sync_cmd)

        assert backend_packages == sync_packages, (
            f"RunPod backend and sync install different packages.\n"
            f"  Backend only: {backend_packages - sync_packages}\n"
            f"  Sync only:    {sync_packages - backend_packages}"
        )


# ---------------------------------------------------------------------------
# Version pin assertions — must NOT install pre-installed packages
# ---------------------------------------------------------------------------


class TestNoPreinstalledDepsInCommand:
    """HF Jobs and RunPod commands must NOT pip-install packages that are
    already in the unsloth Docker image (unsloth, trl, transformers, etc.)."""

    def test_hf_jobs_does_not_install_preinstalled(self, repo_root, clean_env):
        backend = HFJobsBackend(repo_root)
        config = _hf_cloud_config()
        cmd = backend._build_training_command(config)
        packages = _extract_pip_packages(cmd)
        overlap = packages & IMAGE_PREINSTALLED
        assert not overlap, (
            f"HF Jobs command installs pre-installed packages: {overlap}"
        )

    def test_runpod_does_not_install_preinstalled(self, repo_root, clean_env):
        backend = RunPodBackend(repo_root)
        config = _runpod_cloud_config()
        cmd = backend._build_startup_command(config, {})
        packages = _extract_pip_packages(cmd)
        overlap = packages & IMAGE_PREINSTALLED
        assert not overlap, (
            f"RunPod command installs pre-installed packages: {overlap}"
        )

    def test_hf_jobs_command_no_unsloth_keyword(self, repo_root, clean_env):
        """Double-check: the literal word 'unsloth' should not appear in
        pip install lines (it may appear in the image name, which is fine)."""
        backend = HFJobsBackend(repo_root)
        config = _hf_cloud_config()
        cmd = backend._build_training_command(config)
        for part in cmd.split("&&"):
            part = part.strip()
            if part.startswith("pip install"):
                assert "unsloth" not in part.lower(), (
                    f"HF Jobs pip install contains 'unsloth': {part}"
                )

    def test_runpod_command_no_unsloth_keyword(self, repo_root, clean_env):
        backend = RunPodBackend(repo_root)
        config = _runpod_cloud_config()
        cmd = backend._build_startup_command(config, {})
        for part in cmd.split("&&"):
            part = part.strip()
            if part.startswith("pip install"):
                assert "unsloth" not in part.lower(), (
                    f"RunPod pip install contains 'unsloth': {part}"
                )


# ---------------------------------------------------------------------------
# Default image tag validation
# ---------------------------------------------------------------------------


class TestDefaultImageTags:
    """Each backend must use the correct default Docker image."""

    def test_hf_jobs_default_image(self):
        assert HF_DEFAULT_IMAGE == EXPECTED_UNSLOTH_IMAGE

    def test_hf_jobs_config_loads_unsloth_image(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = backend.load_config("sft")
        assert config.cloud_image == EXPECTED_UNSLOTH_IMAGE

    def test_runpod_hardcoded_fallback_is_unsloth(self):
        """RunPod backend's hardcoded fallback image (used when config
        value is missing) must be the unsloth image."""
        # In execute(), the fallback when runpod_config has no default_image:
        #   runpod_config.get("default_image",
        #       "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39")
        # We verify this by inspecting the source directly.
        import inspect
        source = inspect.getsource(RunPodBackend.execute)
        assert EXPECTED_UNSLOTH_IMAGE in source, (
            "RunPod backend's hardcoded fallback image should be the unsloth image"
        )


# ---------------------------------------------------------------------------
# cloud_config.yaml dependencies section
# ---------------------------------------------------------------------------


class TestCloudConfigDependencies:
    """Verify the dependencies section in cloud_config.yaml is well-formed
    and consistent with what backends actually install."""

    @pytest.fixture(autouse=True)
    def _load_real_config(self):
        """Load the real cloud_config.yaml once for all tests in this class."""
        config_path = (
            Path(__file__).resolve().parents[2]
            / "Trainers" / "cloud" / "cloud_config.yaml"
        )
        if not config_path.exists():
            pytest.skip("Real cloud_config.yaml not found (running outside repo)")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def test_dependencies_section_exists(self):
        """The actual cloud_config.yaml must have a dependencies section."""
        deps = self.config.get("dependencies")
        assert deps is not None, "cloud_config.yaml missing 'dependencies' section"

    def test_dependencies_docker_image_matches_expected(self):
        """dependencies.docker_image must match the expected unsloth image."""
        deps = self.config["dependencies"]
        assert deps["docker_image"] == EXPECTED_UNSLOTH_IMAGE

    def test_dependencies_pip_deps_match_backends(self):
        """dependencies.project_pip_deps must match what backends install."""
        config_deps = set(self.config["dependencies"]["project_pip_deps"])
        assert config_deps == EXPECTED_PROJECT_DEPS, (
            f"cloud_config.yaml deps don't match expected.\n"
            f"  Config: {config_deps}\n"
            f"  Expected: {EXPECTED_PROJECT_DEPS}"
        )

    def test_dependencies_extra_setup_commands_is_list(self):
        """dependencies.extra_setup_commands must be a list."""
        extra = self.config["dependencies"]["extra_setup_commands"]
        assert isinstance(extra, list)

    def test_hf_jobs_image_matches_dependencies_image(self):
        """cloud.hf_jobs.image must match dependencies.docker_image."""
        deps_image = self.config["dependencies"]["docker_image"]
        hf_image = self.config["cloud"]["hf_jobs"]["image"]
        assert hf_image == deps_image, (
            f"HF Jobs image ({hf_image}) doesn't match "
            f"dependencies.docker_image ({deps_image})"
        )


# ---------------------------------------------------------------------------
# Modal backend dependency consistency
# ---------------------------------------------------------------------------


class TestModalDepsConsistency:
    """Verify the Modal backend (train_modal.py) installs the same project
    deps as HF Jobs and RunPod.

    Modal uses modal.Image.pip_install() instead of shell commands, so we
    inspect the source of the training_image definition directly.
    """

    @pytest.fixture(autouse=True)
    def _load_modal_source(self):
        """Read the training_image source from train_modal.py."""
        modal_path = (
            Path(__file__).resolve().parents[2]
            / "Trainers" / "cloud" / "train_modal.py"
        )
        if not modal_path.exists():
            pytest.skip("train_modal.py not found (running outside repo)")
        with open(modal_path) as f:
            self.modal_source = f.read()

    def test_modal_includes_project_deps(self):
        """Modal training image must pip_install all project deps."""
        for dep in EXPECTED_PROJECT_DEPS:
            assert f'"{dep}"' in self.modal_source, (
                f"Modal training image missing project dep: {dep}"
            )

    def test_modal_does_not_install_unpinned_preinstalled(self):
        """Modal should pin pre-installed packages (torch, unsloth, etc.)
        with exact versions, not install them unpinned."""
        import re
        # Find all .pip_install(...) argument strings
        pip_args = re.findall(r'\.pip_install\((.*?)\)', self.modal_source, re.DOTALL)
        full_pip_block = " ".join(pip_args)
        # Project deps are allowed unpinned; pre-installed must have ==
        for dep in IMAGE_PREINSTALLED:
            # If the dep appears in pip_install, it must have a version pin
            if f'"{dep}' in full_pip_block:
                assert f'"{dep}==' in full_pip_block or f'"{dep}[' in full_pip_block, (
                    f"Modal installs pre-installed package '{dep}' without version pin"
                )


# ---------------------------------------------------------------------------
# RunPod extra_setup_commands passthrough
# ---------------------------------------------------------------------------


class TestRunPodExtraSetupCommands:
    """Verify RunPod backend honours extra_setup_commands from config."""

    def test_extra_commands_included_in_startup(self, repo_root, clean_env):
        backend = RunPodBackend(repo_root)
        config = _runpod_cloud_config()
        extra = ["pip install custom-package", "echo setup-done"]
        runpod_config = {"extra_setup_commands": extra}
        cmd = backend._build_startup_command(config, runpod_config)
        assert "pip install custom-package" in cmd
        assert "echo setup-done" in cmd

    def test_empty_extra_commands_no_effect(self, repo_root, clean_env):
        backend = RunPodBackend(repo_root)
        config = _runpod_cloud_config()
        cmd_without = backend._build_startup_command(config, {})
        cmd_with = backend._build_startup_command(
            config, {"extra_setup_commands": []}
        )
        assert cmd_without == cmd_with
