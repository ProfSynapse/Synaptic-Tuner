"""
Shared helpers for local Docker-backed runtimes.

Location: tuner/utils/docker_runtime.py
Purpose: Resolve local runtime images and build Docker commands
Used by: docker_handler, train_handler, eval_handler
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence

from tuner.backends.training.cloud.base_cloud import resolve_cloud_image
from tuner.core.exceptions import CloudProviderError

CONTAINER_REPO_ROOT = Path("/workspace/repo")
BUCKET_HELPER_IMAGE = "toolset-training-bucket-helper:latest"
BUCKET_HELPER_ENV_MARKER = "TUNER_BUCKET_HELPER_ACTIVE"


def get_cloud_config_path(repo_root: Path) -> Path:
    """Return the canonical cloud config path."""
    return repo_root / "Trainers" / "cloud" / "cloud_config.yaml"


def get_bucket_helper_dir(repo_root: Path) -> Path:
    """Return the checked-in Docker helper directory for Buckets support."""
    return repo_root / "docker" / "bucket-helper"


def get_bucket_helper_dockerfile(repo_root: Path) -> Path:
    """Return the Buckets helper Dockerfile path."""
    return get_bucket_helper_dir(repo_root) / "Dockerfile"


def ensure_docker_cli() -> tuple[bool, str]:
    """Check whether Docker is available on the host."""
    if shutil.which("docker") is None:
        return False, "Docker CLI not found. Install Docker Desktop first."
    return True, ""


def bucket_helper_image_present(repo_root: Path, *, image: str = BUCKET_HELPER_IMAGE) -> bool:
    """Return True when the local Buckets helper image already exists."""
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", image],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool((result.stdout or "").strip())


def build_bucket_helper_image_command(
    repo_root: Path,
    *,
    image: str = BUCKET_HELPER_IMAGE,
) -> list[str]:
    """Build the checked-in Buckets helper image."""
    helper_dir = get_bucket_helper_dir(repo_root)
    dockerfile = get_bucket_helper_dockerfile(repo_root)
    return [
        "docker",
        "build",
        "-t",
        image,
        "-f",
        str(dockerfile),
        str(helper_dir),
    ]


def build_bucket_helper_run_command(
    repo_root: Path,
    *,
    helper_args: Sequence[str],
    image: str = BUCKET_HELPER_IMAGE,
    remove: bool = True,
) -> list[str]:
    """Run the Buckets helper image against the mounted repo checkout."""
    cmd = ["docker", "run"]
    if remove:
        cmd.append("--rm")
    cmd.extend(["-v", f"{repo_root}:/workspace/repo"])
    cmd.extend(["-e", f"{BUCKET_HELPER_ENV_MARKER}=1"])
    cmd.extend(["-e", f"PYTHONPATH={CONTAINER_REPO_ROOT}"])
    cmd.extend(["--entrypoint", "python"])
    cmd.append(image)
    cmd.append(str((CONTAINER_REPO_ROOT / "tuner.py").as_posix()))
    cmd.extend(helper_args)
    return cmd


def resolve_training_image(
    repo_root: Path,
    *,
    explicit_image: Optional[str] = None,
    requested_profile: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Resolve the Docker image for local training."""
    return resolve_cloud_image(
        get_cloud_config_path(repo_root),
        explicit_image=explicit_image,
        requested_profile=requested_profile,
        default_profile="stable",
        fallback_image=None,
        profile_section="docker_image_profiles",
    )


def resolve_eval_image(
    repo_root: Path,
    *,
    runtime: str,
    explicit_image: Optional[str] = None,
    requested_profile: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Resolve the Docker image for local evaluation."""
    default_profile = "fast_vllm" if runtime == "vllm" else "stable_unsloth"
    return resolve_cloud_image(
        get_cloud_config_path(repo_root),
        explicit_image=explicit_image,
        requested_profile=requested_profile,
        default_profile=default_profile,
        fallback_image=None,
        profile_section="eval_image_profiles",
    )


def container_repo_path(host_path: Path, repo_root: Path) -> str:
    """Map a host repo-relative path into the mounted container path."""
    resolved_host = host_path.resolve()
    resolved_root = repo_root.resolve()
    relative = resolved_host.relative_to(resolved_root)
    return str((CONTAINER_REPO_ROOT / relative).as_posix())


def build_docker_run_command(
    *,
    image: str,
    repo_root: Path,
    command: Sequence[str],
    workdir: Optional[str] = None,
    entrypoint: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    publish_ports: Optional[Sequence[tuple[int, int]]] = None,
    gpus: bool = True,
    name: Optional[str] = None,
    detach: bool = False,
    remove: bool = True,
) -> list[str]:
    """Build a `docker run` command with the repo mounted into the container."""
    cmd = ["docker", "run"]
    if detach:
        cmd.append("-d")
    if remove:
        cmd.append("--rm")
    if name:
        cmd.extend(["--name", name])
    if gpus:
        cmd.extend(["--gpus", "all"])

    cmd.extend(["-v", f"{repo_root}:/workspace/repo"])

    if workdir:
        cmd.extend(["-w", workdir])
    if publish_ports:
        for host_port, container_port in publish_ports:
            cmd.extend(["-p", f"{host_port}:{container_port}"])
    if env:
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])
    if entrypoint:
        cmd.extend(["--entrypoint", entrypoint])

    cmd.append(image)
    cmd.extend(command)
    return cmd


def resolve_runtime_image(
    repo_root: Path,
    *,
    command_name: str,
    runtime: str,
    explicit_image: Optional[str] = None,
    requested_profile: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Resolve the correct Docker image for a local command/runtime pair."""
    if command_name == "train":
        return resolve_training_image(
            repo_root,
            explicit_image=explicit_image,
            requested_profile=requested_profile,
        )
    if command_name == "eval":
        return resolve_eval_image(
            repo_root,
            runtime=runtime,
            explicit_image=explicit_image,
            requested_profile=requested_profile,
        )
    raise CloudProviderError(f"Unsupported local Docker runtime command: {command_name}")
