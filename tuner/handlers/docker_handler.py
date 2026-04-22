"""
Local Docker runtime helper for Synaptic Tuner.

Location: tuner/handlers/docker_handler.py
Purpose: Validate and manage local Docker-backed model runtimes
Used by: Router when 'docker' command is invoked
"""

from __future__ import annotations

import json
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Optional

from shared.utilities.env import get_hf_token
from tuner.backends.training.cloud.base_cloud import resolve_cloud_image
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler
from tuner.utils.docker_runtime import (
    BUCKET_HELPER_IMAGE,
    bucket_helper_image_present,
    build_bucket_helper_image_command,
)


class DockerHandler(BaseHandler):
    """Handler for ``tuner docker`` subcommands."""

    _SUBCOMMANDS = {
        "build": "_handle_build",
        "bootstrap": "_handle_bootstrap",
        "status": "_handle_status",
        "pull": "_handle_pull",
        "smoke": "_handle_smoke",
    }

    def __init__(self, args: Optional[Namespace] = None):
        super().__init__(args=args)

    @property
    def name(self) -> str:
        return "docker"

    def can_handle_direct_mode(self) -> bool:
        return True

    @property
    def cloud_config_path(self) -> Path:
        return self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"

    def handle(self) -> int:
        action = getattr(self.args, "subcommand", None) if self.args else None
        if not action:
            action = "status"

        method_name = self._SUBCOMMANDS.get(action)
        if not method_name:
            self.output_error(f"Unknown docker subcommand: {action}", code="UNKNOWN_SUBCOMMAND")
            return 1
        return getattr(self, method_name)()

    def _ensure_docker_available(self) -> bool:
        if shutil.which("docker") is None:
            self.output_error(
                "Docker CLI not found. Install Docker Desktop first.",
                code="DOCKER_NOT_FOUND",
            )
            return False
        return True

    def _run(self, cmd: list[str], *, stream: bool = False, log_path: Optional[Path] = None) -> tuple[int, str]:
        if not stream:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            output = (result.stdout or "") + (result.stderr or "")
            return result.returncode, output.strip()

        log_handle = None
        output_lines: list[str] = []
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                output_lines.append(line)
                if not self.json_mode:
                    print(line)
                if log_handle is not None:
                    log_handle.write(raw_line)
            process.wait()
            return process.returncode, "\n".join(output_lines).strip()
        finally:
            if log_handle is not None:
                log_handle.close()

    def _docker_info(self) -> tuple[int, str]:
        return self._run(["docker", "info", "--format", "{{.ServerVersion}}"])

    def _resolve_target_images(self, *, target_override: Optional[str] = None) -> list[tuple[str, str, Optional[str]]]:
        target = target_override or getattr(self.args, "docker_target", None) or "unsloth"
        if target == "all":
            targets = ["unsloth", "vllm", "bucket"]
        else:
            targets = [target]

        explicit_image = getattr(self.args, "docker_image", None)
        requested_profile = getattr(self.args, "docker_profile", None)
        resolved: list[tuple[str, str, Optional[str]]] = []

        for runtime in targets:
            try:
                if explicit_image and len(targets) > 1:
                    image, profile = explicit_image, None
                elif runtime == "bucket":
                    image, profile = BUCKET_HELPER_IMAGE, "local_build"
                elif runtime == "unsloth":
                    image, profile = resolve_cloud_image(
                        self.cloud_config_path,
                        explicit_image=explicit_image,
                        requested_profile=requested_profile,
                        default_profile="latest_unsloth",
                        fallback_image=None,
                        profile_section="eval_image_profiles",
                    )
                else:
                    image, profile = resolve_cloud_image(
                        self.cloud_config_path,
                        explicit_image=explicit_image,
                        requested_profile=requested_profile,
                        default_profile="fast_vllm",
                        fallback_image=None,
                        profile_section="eval_image_profiles",
                    )
            except CloudProviderError as exc:
                raise RuntimeError(str(exc)) from exc
            resolved.append((runtime, image, profile))
        return resolved

    def _inspect_target_image(self, runtime: str, image: str, profile: Optional[str]) -> dict:
        if runtime == "bucket":
            present = bucket_helper_image_present(self.repo_root, image=image)
            image_output = image if present else ""
        else:
            image_code, image_output = self._run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}|{{.ID}}|{{.Size}}", image]
            )
            present = image_code == 0 and bool(image_output)

        return {
            "runtime": runtime,
            "image": image,
            "profile": profile,
            "present": present,
            "local_images": image_output.splitlines() if image_output else [],
        }

    def _build_bucket_helper_image(self) -> tuple[int, str]:
        log_path = self.repo_root / "logs" / "bucket-docker-build.log"
        if not self.json_mode:
            print(f"Building bucket helper image: {BUCKET_HELPER_IMAGE}")
        return self._run(
            build_bucket_helper_image_command(self.repo_root),
            stream=True,
            log_path=log_path,
        )

    def _pull_target(self, runtime: str, image: str, profile: Optional[str]) -> tuple[int, dict]:
        if runtime == "bucket":
            code, output = self._build_bucket_helper_image()
            return code, {
                "runtime": runtime,
                "image": image,
                "profile": profile,
                "success": code == 0,
                "log_path": str(self.repo_root / "logs" / "bucket-docker-build.log"),
                "tail": output.splitlines()[-10:] if output else [],
            }

        log_path = self.repo_root / "logs" / f"{runtime}-docker-pull.log"
        self.output_info(f"Pulling {runtime} image: {image}")
        code, output = self._run(["docker", "pull", image], stream=True, log_path=log_path)
        return code, {
            "runtime": runtime,
            "image": image,
            "profile": profile,
            "success": code == 0,
            "log_path": str(log_path),
            "tail": output.splitlines()[-10:] if output else [],
        }

    def _handle_build(self) -> int:
        if not self._ensure_docker_available():
            return 1

        target = getattr(self.args, "docker_target", None) or "bucket"
        if target not in {"bucket", "all"}:
            self.output_error("Docker build currently supports only the local bucket helper image.", code="DOCKER_BUILD_UNSUPPORTED")
            return 1

        code, output = self._build_bucket_helper_image()
        if code != 0:
            self.output_error("docker build failed for bucket helper image", code="DOCKER_BUILD_FAILED")
            return 1

        payload = {
            "runtime": "bucket",
            "image": BUCKET_HELPER_IMAGE,
            "present": bucket_helper_image_present(self.repo_root),
            "tail": output.splitlines()[-10:] if output else [],
        }
        self.output(payload, f"Bucket helper image ready: {BUCKET_HELPER_IMAGE}")
        return 0

    def _handle_status(self) -> int:
        if not self._ensure_docker_available():
            return 1

        docker_version_code, docker_version = self._run(["docker", "--version"])
        info_code, server_version = self._docker_info()
        if docker_version_code != 0:
            self.output_error(docker_version or "Failed to run docker --version", code="DOCKER_VERSION_ERROR")
            return 1

        entries = []
        try:
            for runtime, image, profile in self._resolve_target_images():
                entries.append(self._inspect_target_image(runtime, image, profile))
        except RuntimeError as exc:
            self.output_error(str(exc), code="DOCKER_IMAGE_RESOLUTION_ERROR")
            return 1

        payload = {
            "docker_cli": docker_version,
            "docker_engine": server_version if info_code == 0 else "unavailable",
            "targets": entries,
        }
        human = (
            f"Docker CLI: {docker_version}\n"
            f"Docker Engine: {server_version if info_code == 0 else 'unavailable'}\n"
            + "\n".join(
                f"{item['runtime']}: {item['image']} ({'present' if item['present'] else 'missing'})"
                for item in entries
            )
        )
        self.output(payload, human)
        return 0 if info_code == 0 else 1

    def _handle_pull(self) -> int:
        if not self._ensure_docker_available():
            return 1

        try:
            targets = self._resolve_target_images()
        except RuntimeError as exc:
            self.output_error(str(exc), code="DOCKER_IMAGE_RESOLUTION_ERROR")
            return 1

        results = []
        for runtime, image, profile in targets:
            code, result = self._pull_target(runtime, image, profile)
            results.append(result)
            if code != 0:
                error_code = "DOCKER_BUILD_FAILED" if runtime == "bucket" else "DOCKER_PULL_FAILED"
                message = (
                    "docker build failed for bucket helper image"
                    if runtime == "bucket"
                    else f"docker pull failed for {image}"
                )
                self.output_error(message, code=error_code)
                if self.json_mode:
                    self.output({"results": results}, success=False)
                return 1

        self.output({"results": results}, "Docker image pull complete.")
        return 0

    def _unsloth_smoke_command(self, image: str) -> list[str]:
        repo_mount = str(self.repo_root)
        smoke_code = (
            "import os, sys, torch; "
            "print('cuda', torch.cuda.is_available()); "
            "print('torch', torch.__version__); "
            "from unsloth import FastLanguageModel; "
            "print('unsloth-ok'); "
            "print('repo-mounted', os.path.exists('/workspace/repo')); "
            "sys.path.insert(0, '/workspace/repo'); "
            "import tuner; "
            "print('tuner-ok')"
        )
        return [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{repo_mount}:/workspace/repo",
            "--entrypoint", "python",
            image,
            "-c", smoke_code,
        ]

    def _vllm_smoke_command(self, image: str) -> list[str]:
        smoke_code = (
            "import torch, vllm; "
            "print('cuda', torch.cuda.is_available()); "
            "print('torch', torch.__version__); "
            "print('vllm', vllm.__version__)"
        )
        return [
            "docker", "run", "--rm", "--gpus", "all",
            "--entrypoint", "python3",
            image,
            "-c", smoke_code,
        ]

    def _bucket_smoke_command(self, image: str) -> list[str]:
        smoke_code = (
            "import huggingface_hub, dotenv, yaml; "
            "print('hf_hub', huggingface_hub.__version__); "
            "print('has_create_bucket', hasattr(huggingface_hub, 'create_bucket')); "
            "print('has_hffs', hasattr(huggingface_hub, 'HfFileSystem'))"
        )
        return [
            "docker", "run", "--rm",
            "--entrypoint", "python",
            image,
            "-c", smoke_code,
        ]

    def _smoke_target(self, runtime: str, image: str, profile: Optional[str]) -> tuple[int, dict]:
        if runtime == "bucket" and not bucket_helper_image_present(self.repo_root, image=image):
            code, _ = self._build_bucket_helper_image()
            if code != 0:
                return 1, {
                    "runtime": runtime,
                    "image": image,
                    "profile": profile,
                    "success": False,
                    "output": ["Bucket helper image is missing and could not be built."],
                }

        if runtime == "unsloth":
            cmd = self._unsloth_smoke_command(image)
        elif runtime == "vllm":
            cmd = self._vllm_smoke_command(image)
        else:
            cmd = self._bucket_smoke_command(image)

        code, output = self._run(cmd)
        return code, {
            "runtime": runtime,
            "image": image,
            "profile": profile,
            "success": code == 0,
            "output": output.splitlines(),
        }

    def _bootstrap_guidance(self, *, cli_ok: bool, engine_ok: bool) -> list[str]:
        guidance: list[str] = []
        if not cli_ok:
            guidance.extend(
                [
                    "Install Docker Desktop for Windows and leave WSL 2 integration enabled.",
                    "Start Docker Desktop and wait for the engine status to show Running.",
                    "Re-run `python tuner.py docker bootstrap --docker-target all`.",
                ]
            )
            return guidance

        if not engine_ok:
            guidance.extend(
                [
                    "Start Docker Desktop and wait for the engine to finish initializing.",
                    "If GPU containers are required, confirm the NVIDIA driver is installed on the host.",
                    "Re-run `python tuner.py docker status` to confirm the engine is reachable.",
                ]
            )
            return guidance

        guidance.extend(
            [
                "Use `python tuner.py train --runtime docker` for local Docker-backed training.",
                "Use `python tuner.py eval --runtime docker` for local Docker-backed evaluation.",
                "Use `python tuner.py bucket pull ...` to bring cloud adapters local; pulled runs under `toolset-training-artifacts/runs/...` are now discoverable in local eval flows.",
            ]
        )
        return guidance

    def _handle_bootstrap(self) -> int:
        cli_ok = shutil.which("docker") is not None
        docker_version = None
        server_version = None
        info_code = 1

        if cli_ok:
            version_code, docker_version_output = self._run(["docker", "--version"])
            if version_code == 0:
                docker_version = docker_version_output
            info_code, server_version_output = self._docker_info()
            if info_code == 0:
                server_version = server_version_output

        engine_ok = cli_ok and info_code == 0
        guidance = self._bootstrap_guidance(cli_ok=cli_ok, engine_ok=engine_ok)

        if not cli_ok or not engine_ok:
            payload = {
                "docker_cli_found": cli_ok,
                "docker_cli": docker_version,
                "docker_engine": server_version,
                "ready": False,
                "guidance": guidance,
            }
            human_lines = [
                f"Docker CLI: {docker_version or 'missing'}",
                f"Docker Engine: {server_version or 'unavailable'}",
                "",
                "Next steps:",
                *[f"  - {line}" for line in guidance],
            ]
            self.output(payload, "\n".join(human_lines), success=False)
            return 1

        try:
            targets = self._resolve_target_images(
                target_override=getattr(self.args, "docker_target", None) or "all"
            )
        except RuntimeError as exc:
            self.output_error(str(exc), code="DOCKER_IMAGE_RESOLUTION_ERROR")
            return 1

        status_entries = [self._inspect_target_image(runtime, image, profile) for runtime, image, profile in targets]
        pull_results = []
        smoke_results = []

        for runtime, image, profile in targets:
            inspected = next((entry for entry in status_entries if entry["runtime"] == runtime), None)
            present = bool(inspected and inspected["present"])
            if present:
                pull_results.append(
                    {
                        "runtime": runtime,
                        "image": image,
                        "profile": profile,
                        "success": True,
                        "skipped": True,
                        "reason": "already_present",
                    }
                )
            else:
                code, result = self._pull_target(runtime, image, profile)
                pull_results.append(result)
                if code != 0:
                    self.output_error(
                        "Docker bootstrap failed while preparing local images.",
                        code="DOCKER_BOOTSTRAP_PULL_FAILED",
                        details={"runtime": runtime, "image": image},
                    )
                    return 1

            code, smoke_result = self._smoke_target(runtime, image, profile)
            smoke_results.append(smoke_result)
            if code != 0:
                self.output_error(
                    f"{runtime} smoke test failed",
                    code="DOCKER_BOOTSTRAP_SMOKE_FAILED",
                    details={"image": image, "output": smoke_result.get("output", [])},
                )
                return 1

        hf_token_available = bool(get_hf_token())
        payload = {
            "docker_cli_found": True,
            "docker_cli": docker_version,
            "docker_engine": server_version,
            "ready": True,
            "targets": [self._inspect_target_image(runtime, image, profile) for runtime, image, profile in targets],
            "pull_results": pull_results,
            "smoke_results": smoke_results,
            "hf_token_available": hf_token_available,
            "guidance": guidance,
        }
        human_lines = [
            f"Docker CLI: {docker_version}",
            f"Docker Engine: {server_version}",
            "Local Docker runtime is ready:",
        ]
        for result in smoke_results:
            human_lines.append(f"  - {result['runtime']}: {result['image']}")
        human_lines.append(f"HF_TOKEN available via env/.env: {'yes' if hf_token_available else 'no'}")
        human_lines.append("Next steps:")
        human_lines.extend(f"  - {line}" for line in guidance)
        self.output(payload, "\n".join(human_lines))
        return 0

    def _handle_smoke(self) -> int:
        if not self._ensure_docker_available():
            return 1

        try:
            targets = self._resolve_target_images()
        except RuntimeError as exc:
            self.output_error(str(exc), code="DOCKER_IMAGE_RESOLUTION_ERROR")
            return 1

        results = []
        for runtime, image, profile in targets:
            code, result = self._smoke_target(runtime, image, profile)
            results.append(result)
            if code != 0:
                self.output_error(
                    f"{runtime} smoke test failed",
                    code="DOCKER_SMOKE_FAILED",
                    details={"image": image, "output": result.get("output", [])},
                )
                return 1

        human_lines = ["Docker smoke tests passed:"]
        for result in results:
            human_lines.append(f"  {result['runtime']}: {result['image']}")
        self.output({"results": results}, "\n".join(human_lines))
        return 0
