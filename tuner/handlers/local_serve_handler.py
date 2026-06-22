"""Serve a local merged model in a vLLM container on an OpenAI-compatible port.

Location: tuner/handlers/local_serve_handler.py
Purpose: Launch / stop / inspect a ``vllm/vllm-openai`` Docker container that
         serves a host-resident merged (16-bit) HF model directory on an
         OpenAI-compatible endpoint, so the existing Evaluator ``--backend vllm``
         path works end-to-end against a locally-trained model.
Used by: tuner/cli/router.py (command: ``local-serve``)

This is the *serving* slice of docs/plans/docker-first-local-runtime-plan.md.
It mirrors the Docker conventions established by ``local_run_handler`` (named
container, ``--gpus all``, ``--init``, fail-loud subprocess checks) but is the
long-lived container variant: it runs detached and waits for ``/v1/models`` to
report ready, then leaves the container running for evaluation.

No vLLM is installed on the host — serving lives entirely in the container.

Image pin
---------
Default image is the standard published ``vllm/vllm-openai:latest``. The image
is fully overridable via ``--serve-image`` for architectures that need a
purpose-built tag. For example, an architecture unsupported by older
``:latest`` images would need a custom tag such as
``vllm/vllm-openai:<your-custom-tag>`` — pass it with
``--serve-image vllm/vllm-openai:<your-custom-tag>``.
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from tuner.handlers.base import BaseHandler
from tuner.ui import confirm
from tuner.utils.docker import (
    DockerEnv,
    DockerError,
    container_logs,
    container_state,
    ensure_daemon,
    image_exists,
    pull_image,
    run_detached,
    stop_container,
    wait_for_http_ready,
)


# Stable container name so serve/stop/status all address the same instance and
# a re-run is idempotent rather than double-launching.
CONTAINER_NAME = "tuner-vllm-serve"

# Standard published vLLM image; fully overridable via --serve-image. A custom
# tag (e.g. vllm/vllm-openai:<your-custom-tag>) can be passed when :latest lacks
# support for the model's architecture. See module docstring.
DEFAULT_IMAGE = "vllm/vllm-openai:latest"

DEFAULT_PORT = 8011
DEFAULT_SERVED_MODEL_NAME = "finetuned"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_READY_TIMEOUT = 900  # vLLM weight load + CUDA graph capture can be slow.

# In-container mount target for the model directory (read-only).
CONTAINER_MODEL_PATH = "/model"


class LocalServeError(RuntimeError):
    """Raised for local-serve configuration errors (distinct from DockerError)."""


class LocalServeHandler(BaseHandler):
    """Launch / stop / inspect a local vLLM serving container."""

    def __init__(self, args: Namespace | None = None):
        super().__init__(args=args)
        self._docker = DockerEnv()

    @property
    def name(self) -> str:
        return "local-serve"

    def can_handle_direct_mode(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Arg resolution
    # ------------------------------------------------------------------
    def _resolve_model_dir(self, raw: str | None) -> Path:
        if not raw:
            raise LocalServeError(
                "local-serve requires --model <dir> (host path to the merged model)."
            )
        path = Path(raw)
        if not path.is_absolute():
            path = (self.repo_root / raw).resolve()
        if not path.exists():
            raise LocalServeError(f"Model directory not found: {path}")
        if not path.is_dir():
            raise LocalServeError(f"--model must be a directory, got a file: {path}")
        if not (path / "config.json").exists():
            raise LocalServeError(
                f"Model directory has no config.json (not an HF model dir?): {path}"
            )
        return path

    def _port(self) -> int:
        return int(getattr(self.args, "serve_port", None) or DEFAULT_PORT)

    def _served_name(self) -> str:
        return str(
            getattr(self.args, "served_model_name", None) or DEFAULT_SERVED_MODEL_NAME
        )

    def _ready_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}/v1/models"

    # ------------------------------------------------------------------
    # docker run argv
    # ------------------------------------------------------------------
    def _build_run_args(self, model_dir: Path, port: int) -> list[str]:
        """Build the ``docker run -d`` argv for the vLLM serving container.

        Mounts the model dir read-only at /model, publishes <port>:8000, and
        passes the vLLM engine flags. ``--ipc=host`` is required by vLLM for
        shared-memory tensor transfer; ``--init`` gives a real PID 1 for clean
        signal handling on stop.
        """
        image = str(getattr(self.args, "serve_image", None) or DEFAULT_IMAGE)
        gpu_util = float(
            getattr(self.args, "gpu_memory_utilization", None)
            or DEFAULT_GPU_MEMORY_UTILIZATION
        )
        max_len = int(
            getattr(self.args, "max_model_len", None) or DEFAULT_MAX_MODEL_LEN
        )
        served_name = self._served_name()

        return [
            "docker",
            "run",
            "-d",
            "--init",
            "--name",
            CONTAINER_NAME,
            "--gpus",
            "all",
            "--ipc=host",
            "-p",
            f"{port}:8000",
            "-v",
            f"{model_dir}:{CONTAINER_MODEL_PATH}:ro",
            image,
            "--model",
            CONTAINER_MODEL_PATH,
            "--served-model-name",
            served_name,
            "--max-model-len",
            str(max_len),
            "--gpu-memory-utilization",
            str(gpu_util),
        ]

    # ------------------------------------------------------------------
    # Management actions
    # ------------------------------------------------------------------
    def _do_stop(self) -> int:
        try:
            stopped = stop_container(self._docker, CONTAINER_NAME)
        except DockerError as exc:
            return self._fail(str(exc), "LOCAL_SERVE_STOP_ERROR")
        # Remove the stopped container so a later serve starts cleanly (the
        # named container would otherwise block a fresh `docker run --name`).
        from tuner.utils.docker import remove_container

        try:
            remove_container(self._docker, CONTAINER_NAME)
        except DockerError as exc:
            return self._fail(str(exc), "LOCAL_SERVE_STOP_ERROR")
        msg = (
            f"Stopped and removed container {CONTAINER_NAME}."
            if stopped
            else f"Container {CONTAINER_NAME} was not running (cleaned up any leftover)."
        )
        if self.json_mode:
            self.output({"container": CONTAINER_NAME, "stopped": stopped}, msg)
        else:
            print(msg)
        return 0

    def _do_status(self) -> int:
        try:
            state = container_state(self._docker, CONTAINER_NAME)
        except DockerError as exc:
            return self._fail(str(exc), "LOCAL_SERVE_STATUS_ERROR")
        data = {"container": CONTAINER_NAME, "state": state}
        if state == "running":
            # Surface the published port from docker inspect for convenience.
            from tuner.utils.docker import _run

            port_probe = _run(
                self._docker,
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{range $p, $conf := .NetworkSettings.Ports}}{{$p}}->"
                    "{{(index $conf 0).HostPort}} {{end}}",
                    CONTAINER_NAME,
                ],
                capture=True,
            )
            ports = (port_probe.stdout or "").strip()
            if ports:
                data["ports"] = ports
        if self.json_mode:
            self.output(data, f"{CONTAINER_NAME}: {state}")
        else:
            line = f"{CONTAINER_NAME}: {state}"
            if data.get("ports"):
                line += f" ({data['ports']})"
            print(line)
        return 0

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------
    def _do_serve(self) -> int:
        try:
            model_dir = self._resolve_model_dir(getattr(self.args, "model", None))
        except LocalServeError as exc:
            return self._fail(str(exc), "LOCAL_SERVE_CONFIG_ERROR")

        port = self._port()
        served_name = self._served_name()
        image = str(getattr(self.args, "serve_image", None) or DEFAULT_IMAGE)

        try:
            server_version = ensure_daemon(self._docker)
        except DockerError as exc:
            return self._fail(str(exc), "DOCKER_DAEMON_UNREACHABLE")

        # Idempotency: a running container is reported, not relaunched.
        state = container_state(self._docker, CONTAINER_NAME)
        if state == "running":
            url = self._ready_url(port)
            msg = (
                f"Container {CONTAINER_NAME} is already running. "
                f"Endpoint (if this port matches): {url}. "
                "Use 'local-serve --stop' to replace it."
            )
            if self.json_mode:
                self.output(
                    {"container": CONTAINER_NAME, "state": "running", "endpoint": url},
                    msg,
                )
            else:
                print(msg)
            return 0
        if state == "exited":
            # A leftover stopped container with this name blocks `docker run
            # --name`; remove it so we always start from a clean slate.
            from tuner.utils.docker import remove_container

            try:
                remove_container(self._docker, CONTAINER_NAME)
            except DockerError as exc:
                return self._fail(str(exc), "LOCAL_SERVE_CLEANUP_ERROR")

        run_args = self._build_run_args(model_dir, port)

        if not self.json_mode:
            print("LOCAL SERVE")
            print("Serve a local merged model in a vLLM container")
            print()
            print(f"  Docker server: {server_version}")
            print(f"  Container: {CONTAINER_NAME}")
            print(f"  Image: {image}")
            print(f"  Model dir: {model_dir} -> {CONTAINER_MODEL_PATH} (ro)")
            print(f"  Served model name: {served_name}")
            print(f"  Endpoint: {self._ready_url(port)}")
            print(f"  Port mapping: {port} -> 8000")
            print(f"  GPU memory utilization: "
                  f"{getattr(self.args, 'gpu_memory_utilization', None) or DEFAULT_GPU_MEMORY_UTILIZATION}")
            print(f"  Max model len: "
                  f"{getattr(self.args, 'max_model_len', None) or DEFAULT_MAX_MODEL_LEN}")
            print()
            if not getattr(self.args, "auto_confirm", False) and not confirm(
                "Start vLLM serving container with this configuration?"
            ):
                print("Local serve cancelled.")
                return 0

        # Pull the image if missing, surfacing layer progress so a large
        # download is distinguishable from a hung daemon.
        try:
            if not image_exists(self._docker, image):
                if not self.json_mode:
                    print(
                        f"Image {image} is not cached; pulling now "
                        "(this is a large download, not a hang)."
                    )
                pull_image(self._docker, image)
        except DockerError as exc:
            return self._fail(str(exc), "LOCAL_SERVE_PULL_ERROR")

        # Launch detached.
        try:
            container_id = run_detached(self._docker, run_args)
        except DockerError as exc:
            return self._fail(str(exc), "LOCAL_SERVE_RUN_ERROR")

        if not self.json_mode:
            print(f"Container started ({container_id[:12]}). Waiting for readiness...")

        url = self._ready_url(port)
        try:
            ready = wait_for_http_ready(
                self._docker,
                url,
                container_name=CONTAINER_NAME,
                timeout=DEFAULT_READY_TIMEOUT,
                progress=(None if self.json_mode else print),
            )
        except DockerError as exc:
            # Container died during load — DockerError carries the log tail.
            return self._fail(str(exc), "LOCAL_SERVE_NOT_READY")

        if not ready:
            logs = container_logs(self._docker, CONTAINER_NAME, tail=40)
            return self._fail(
                f"vLLM server never became ready within {DEFAULT_READY_TIMEOUT}s. "
                f"Container {CONTAINER_NAME} is still running; inspect it with "
                f"'docker logs {CONTAINER_NAME}'. Last log lines:\n{logs}",
                "LOCAL_SERVE_NOT_READY",
            )

        success_data = {
            "container": CONTAINER_NAME,
            "container_id": container_id,
            "endpoint": url,
            "served_model_name": served_name,
            "port": port,
            "image": image,
            "eval_command": (
                f"python -m Evaluator.cli --backend vllm --model {served_name} "
                f"--host 127.0.0.1 --port {port} --prompt-set <cases.jsonl> "
                "--temperature 0 --output <out.json>"
            ),
            "stop_command": "python tuner.py local-serve --stop",
            "status_command": "python tuner.py local-serve --status",
        }
        if self.json_mode:
            self.output(success_data, "vLLM server is ready.")
        else:
            print()
            print("vLLM server is ready.")
            print(f"  Endpoint: {url}")
            print(f"  Served model name: {served_name}")
            print()
            print("Evaluate against it with:")
            print(
                f"  python -m Evaluator.cli --backend vllm --model {served_name} "
                f"--host 127.0.0.1 --port {port} \\"
            )
            print(
                "    --prompt-set <cases.jsonl> --temperature 0 --output <out.json>"
            )
            print()
            print("Manage the container:")
            print("  python tuner.py local-serve --status")
            print("  python tuner.py local-serve --stop")
        return 0

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _fail(self, message: str, code: str) -> int:
        if self.json_mode:
            self.output_error(message, code=code)
        else:
            print(f"Error: {message}")
        return 1

    def handle(self) -> int:
        stop = bool(getattr(self.args, "stop", False))
        status = bool(getattr(self.args, "container_status", False))
        if stop and status:
            return self._fail(
                "--stop and --status are mutually exclusive.", "LOCAL_SERVE_ARG_ERROR"
            )
        if stop:
            return self._do_stop()
        if status:
            return self._do_status()
        return self._do_serve()
