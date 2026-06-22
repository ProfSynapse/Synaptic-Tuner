"""Reusable local Docker helpers for the tuner CLI.

Location: tuner/utils/docker.py
Purpose: Daemon/context resolution, image checks, pull-with-progress, container
         run/stop/remove, status queries, and an HTTP readiness poll. Shared by
         local container handlers (local-serve, and future local container work).
Used by: tuner/handlers/local_serve_handler.py

Why force the default daemon socket
------------------------------------
The shell's *active* Docker context on this host may point at an unrelated /
client-owned engine (for example a stopped colima profile). Container work in
this repo always wants the default Docker Desktop / engine daemon. Rather than
mutate the user's selected context, every command this module runs forces
``DOCKER_HOST=unix:///var/run/docker.sock`` in its environment so it always
hits the default daemon regardless of which context is active. Override with
the ``DOCKER_HOST`` env var only if you know what you are doing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional

import requests


# Default daemon socket. Forced into every command's environment so we never
# accidentally drive a non-default (e.g. colima) context that happens to be
# active in the invoking shell. Callers may override by exporting DOCKER_HOST
# before launching the CLI.
DEFAULT_DOCKER_HOST = "unix:///var/run/docker.sock"

ContainerState = Literal["running", "exited", "absent"]


class DockerError(RuntimeError):
    """Raised for Docker daemon, image, or container errors."""


@dataclass(frozen=True)
class DockerEnv:
    """Resolved Docker invocation environment.

    docker_host: the value forced into ``DOCKER_HOST`` for every command.
    """

    docker_host: str = DEFAULT_DOCKER_HOST

    def env(self) -> dict[str, str]:
        """Return an environment dict with the default daemon socket forced.

        On Windows the unix-socket default is meaningless, so we leave the
        process environment untouched there and let Docker Desktop's named
        pipe resolve through the default context.
        """
        merged = dict(os.environ)
        if sys.platform != "win32":
            merged["DOCKER_HOST"] = self.docker_host
        return merged


def _run(
    docker_env: DockerEnv,
    args: list[str],
    *,
    capture: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run a docker command with the forced daemon environment.

    capture=True captures stdout/stderr as text. check=True raises DockerError
    on a non-zero return code (with captured stderr when available).
    """
    kwargs: dict = {"text": True, "env": docker_env.env()}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    result = subprocess.run(args, **kwargs)
    if check and result.returncode != 0:
        detail = (result.stderr or "").strip() if capture else ""
        suffix = f": {detail}" if detail else ""
        raise DockerError(
            f"Command failed ({result.returncode}): {' '.join(args)}{suffix}"
        )
    return result


def ensure_daemon(docker_env: DockerEnv) -> str:
    """Verify the default Docker daemon is reachable; return its server version.

    Fails loud with an actionable message distinguishing "daemon unreachable"
    from a working engine. Calling this first lets serve commands report a
    clear error instead of a cryptic failure deep inside a pull or run.
    """
    result = _run(
        docker_env,
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture=True,
    )
    if result.returncode != 0 or not (result.stdout or "").strip():
        detail = (result.stderr or result.stdout or "").strip()
        raise DockerError(
            "Docker daemon is not reachable at "
            f"{docker_env.docker_host}. Is Docker Desktop / the engine running? "
            f"({detail})"
        )
    return result.stdout.strip()


def image_exists(docker_env: DockerEnv, image: str) -> bool:
    """Return True iff the image is already present locally (no network)."""
    result = _run(
        docker_env,
        ["docker", "image", "inspect", image],
        capture=True,
    )
    return result.returncode == 0


def pull_image(docker_env: DockerEnv, image: str) -> None:
    """Pull an image, streaming layer progress to stdout.

    Streams ``docker pull`` output line-by-line so the user can distinguish a
    large-but-progressing download from a hung daemon. Raises DockerError with
    the tail of the output on failure (for example: tag/digest not found, or
    auth required).
    """
    print(f"Pulling image {image} (this can be large on first use)...")
    proc = subprocess.Popen(
        ["docker", "pull", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=docker_env.env(),
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  {line}")
            tail.append(line)
            if len(tail) > 20:
                tail.pop(0)
    proc.wait()
    if proc.returncode != 0:
        raise DockerError(
            f"Failed to pull image {image} (exit {proc.returncode}). "
            "Last output:\n  " + "\n  ".join(tail[-10:])
        )


def container_state(docker_env: DockerEnv, name: str) -> ContainerState:
    """Query a container's state by name: running | exited | absent.

    Non-standard states (created/paused/restarting/dead) are coerced to
    "exited" so callers take a recreate-or-restart path rather than assuming
    the container is serving.
    """
    result = _run(
        docker_env,
        ["docker", "inspect", "--format", "{{.State.Status}}", name],
        capture=True,
    )
    if result.returncode != 0:
        return "absent"
    state = (result.stdout or "").strip().lower()
    return "running" if state == "running" else "exited"


def run_detached(docker_env: DockerEnv, run_args: list[str]) -> str:
    """Run ``docker run -d ...`` and return the new container id.

    ``run_args`` is the full argv beginning with ``docker`` (the ``-d`` flag and
    ``--name`` are the caller's responsibility — typically built by the
    handler). Raises DockerError on failure with captured stderr.
    """
    result = _run(docker_env, run_args, capture=True, check=True)
    return (result.stdout or "").strip()


def stop_container(docker_env: DockerEnv, name: str, timeout: int = 30) -> bool:
    """Stop a running container by name. Returns True if a stop was issued.

    Returns False (no error) when the container is absent or already stopped,
    so ``--stop`` is idempotent.
    """
    state = container_state(docker_env, name)
    if state == "absent":
        return False
    if state == "exited":
        return False
    _run(
        docker_env,
        ["docker", "stop", "-t", str(timeout), name],
        capture=True,
        check=True,
    )
    return True


def remove_container(docker_env: DockerEnv, name: str, force: bool = True) -> bool:
    """Remove a container by name. Returns True if a removal was issued.

    Idempotent: returns False when the container is absent.
    """
    if container_state(docker_env, name) == "absent":
        return False
    args = ["docker", "rm"]
    if force:
        args.append("-f")
    args.append(name)
    _run(docker_env, args, capture=True, check=True)
    return True


def container_logs(docker_env: DockerEnv, name: str, tail: int = 50) -> str:
    """Return the last ``tail`` lines of a container's logs (stdout+stderr).

    Best-effort: returns an empty string if the container is gone or logs are
    unavailable, so it is safe to call inside an error-reporting path.
    """
    result = _run(
        docker_env,
        ["docker", "logs", "--tail", str(tail), name],
        capture=True,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "") + (result.stderr or "")


def wait_for_http_ready(
    docker_env: DockerEnv,
    url: str,
    *,
    container_name: str,
    timeout: int = 600,
    poll_interval: float = 3.0,
    progress: Optional[Callable[[str], None]] = None,
) -> bool:
    """Poll an HTTP readiness URL until 200, the container dies, or timeout.

    Returns True when ``url`` answers 200. Returns False on timeout. Raises
    DockerError if the container exits before becoming ready (with the log
    tail), since that is a hard failure rather than a slow start.

    ``progress`` is an optional callback for status lines (defaults to print).
    """
    emit = progress or print
    start = time.time()
    last_note = 0.0
    while time.time() - start < timeout:
        # Hard-fail fast if the container died (crash, OOM, bad args).
        if container_state(docker_env, container_name) != "running":
            logs = container_logs(docker_env, container_name, tail=40)
            raise DockerError(
                f"Container {container_name} exited before becoming ready. "
                "Last log lines:\n" + logs
            )
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        elapsed = time.time() - start
        if elapsed - last_note >= 15:
            emit(f"  ...still loading ({int(elapsed)}s elapsed, timeout {timeout}s)")
            last_note = elapsed
        time.sleep(poll_interval)
    return False
