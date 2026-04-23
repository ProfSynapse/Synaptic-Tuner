"""Config-driven local Docker runner for GPU training jobs.

Bind-mount runs use root inside the container with a chown-on-exit trap so
artifacts written to the host tree land with host-user ownership. Copy-mode
extracts the artifact archive and then rewrites ownership on the host on
Linux/macOS. The ``job.user`` YAML knob overrides this behavior:

  auto  (default) — bind: root + chown-back; copy: image-user + host chown-back
  root           — run as 0:0 inside container, do not chown back
  image          — rely on the image's default user; no chown-back
  "<uid>:<gid>"  — run as literal uid:gid, no chown-back
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal

import yaml

from tuner.handlers.base import BaseHandler
from tuner.ui import BOX, confirm, print_menu


DEFAULT_STOP_TIMEOUT = 60

_USER_FIELD_PATTERN = re.compile(r"^\d+:\d+$")


class LocalRunError(RuntimeError):
    """Raised for local Docker configuration or runtime errors."""


@dataclass(frozen=True)
class UserSpec:
    """Resolved docker-user / host-chown configuration for a run.

    docker_user_flag: value for ``docker run -u`` (None means do not pass -u).
    chown_host_uid / chown_host_gid: host-side chown target (None means skip).
    skip_chown: True when no chown-back should happen (user opted out).
    """

    docker_user_flag: str | None
    chown_host_uid: int | None
    chown_host_gid: int | None
    skip_chown: bool


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _append_flag(args: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    flag = _flag_name(key)
    if isinstance(value, bool):
        if value:
            args.append(flag)
        return
    if isinstance(value, list):
        value = ",".join(str(item) for item in value)
    args.extend([flag, str(value)])


def _validate_user_field(raw: Any) -> str:
    """Normalize the ``job.user`` YAML value.

    Accepts: None / missing -> "auto"; "auto" | "root" | "image";
    "<uid>:<gid>" where both sides are non-negative integers.
    """
    if raw is None:
        return "auto"
    value = str(raw).strip().lower()
    if value in {"auto", "root", "image"}:
        return value
    if _USER_FIELD_PATTERN.match(value):
        return value
    raise LocalRunError(
        f"Invalid job.user: {raw!r}. Expected one of: auto, root, image, or '<uid>:<gid>'."
    )


def _current_host_ids() -> tuple[int, int]:
    """Return (uid, gid) of the host process. Windows reports (0, 0)."""
    if sys.platform == "win32":
        return (0, 0)
    return (os.getuid(), os.getgid())


def _resolve_user_spec(
    job_user: str,
    transfer_mode: str,
    host_uid: int,
    host_gid: int,
    platform: str,
) -> UserSpec:
    """Resolve a validated job.user value into a concrete UserSpec.

    job_user must already be validated (``_validate_user_field``).
    transfer_mode is "bind" or "copy". platform is ``sys.platform``-style.
    """
    if job_user == "root":
        return UserSpec("0:0", None, None, skip_chown=True)
    if job_user == "image":
        return UserSpec(None, None, None, skip_chown=True)
    if _USER_FIELD_PATTERN.match(job_user):
        uid_s, gid_s = job_user.split(":")
        return UserSpec(job_user, int(uid_s), int(gid_s), skip_chown=True)

    # auto
    if platform == "win32":
        # Windows has no POSIX ownership model on the host side.
        if transfer_mode == "bind":
            return UserSpec("0:0", None, None, skip_chown=True)
        return UserSpec(None, None, None, skip_chown=True)

    if transfer_mode == "bind":
        return UserSpec("0:0", host_uid, host_gid, skip_chown=False)
    # copy mode: let the image's default user run inside the container;
    # chown-back happens on the host after tar extraction.
    return UserSpec(None, host_uid, host_gid, skip_chown=False)


def _collect_chown_paths(plan: dict[str, Any]) -> list[str]:
    """Narrow, ordered, deduped list of in-container paths to chown on exit.

    Prefer the artifact path (primary write target) and the workdir, plus
    well-known relative output locations under /workspace/repo.
    """
    raw: list[str] = []

    container_artifact = plan.get("container_artifact_path")
    if container_artifact:
        raw.append(str(container_artifact))

    workdir = plan.get("workdir")
    if workdir:
        raw.append(str(workdir))

    # Common artifact roots the trainer may write under.
    raw.append("/workspace/repo/toolset-training-artifacts")

    return list(dict.fromkeys(raw))


def _build_bash_wrapper(plan: dict[str, Any], user_spec: UserSpec) -> str:
    """Construct the string passed to ``bash -lc`` inside the container.

    When chown-back is active, wrap with ``trap ... EXIT`` and ``exec`` so the
    python command becomes PID 1's child and the trap fires on any exit path.
    pip prelude (if any) runs before ``exec``.
    """
    pip = plan.get("pip") or []
    pip_prelude = ""
    if pip:
        pip_prelude = "pip install --upgrade " + " ".join(shlex.quote(item) for item in pip) + " && "

    command_text = " ".join(shlex.quote(part) for part in plan["command"])

    if user_spec.skip_chown or user_spec.chown_host_uid is None:
        # No chown-back: simple prelude + command.
        return pip_prelude + command_text

    uid = user_spec.chown_host_uid
    gid = user_spec.chown_host_gid
    chown_targets = _collect_chown_paths(plan)
    # Quote each target; ``chown -R`` on a non-existent path fails but the
    # ``|| true`` at the end of the trap swallows it.
    targets_quoted = " ".join(shlex.quote(t) for t in chown_targets)
    trap = f'trap "chown -R {uid}:{gid} {targets_quoted} 2>/dev/null || true" EXIT'

    return f"{trap}; {pip_prelude}exec {command_text}"


def _chown_host_tree(path: Path, uid: int, gid: int) -> None:
    """Recursively chown a host path to (uid, gid). Swallows PermissionError.

    No-op on Windows (os.chown doesn't exist there).
    """
    if sys.platform == "win32":
        return
    if not path.exists():
        return
    try:
        os.chown(path, uid, gid)
    except PermissionError as exc:
        print(f"Warning: chown {path} failed ({exc}); leaving ownership unchanged.")
        return
    except OSError as exc:
        print(f"Warning: chown {path} failed ({exc}); leaving ownership unchanged.")
        return
    if path.is_dir():
        for root, dirs, files in os.walk(path):
            for entry in dirs + files:
                target = os.path.join(root, entry)
                try:
                    os.chown(target, uid, gid)
                except PermissionError:
                    # Keep going — some files (e.g. symlinks to outside the
                    # tree, or root-owned pip caches) may not be chownable.
                    continue
                except OSError:
                    continue


def _validate_tty_field(raw: Any) -> str:
    """Normalize the ``job.tty`` YAML value.

    Accepts None / missing -> "auto"; "auto" | "always" | "never".
    """
    if raw is None:
        return "auto"
    value = str(raw).strip().lower()
    if value in {"auto", "always", "never"}:
        return value
    raise LocalRunError(
        f"Invalid job.tty: {raw!r}. Expected one of: auto, always, never."
    )


def _resolve_tty_flags(tty_mode: str, stdout_isatty: bool) -> list[str]:
    """Resolve ``job.tty`` into the docker `-i`/`-t` flag list.

    ``always`` -> always attach; ``never`` -> never attach;
    ``auto`` -> attach iff the invoking stdout is a tty.
    """
    if tty_mode == "always":
        return ["-i", "-t"]
    if tty_mode == "never":
        return []
    if tty_mode == "auto":
        return ["-i", "-t"] if stdout_isatty else []
    # Defensive: _validate_tty_field should have rejected anything else.
    raise LocalRunError(f"Unexpected tty_mode: {tty_mode!r}")


_BOOL_TRUE_STRINGS = {"true", "yes", "1", "on"}
_BOOL_FALSE_STRINGS = {"false", "no", "0", "off"}


def _validate_bool_field(raw: Any, field_name: str, default: bool) -> bool:
    """Normalize a YAML-ish truthy/falsy value into a bool.

    None -> default; bool -> bool; int 0/1 -> bool; string (case-insensitive,
    trimmed) matched against the true/false string sets. Anything else raises.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        if raw in (0, 1):
            return bool(raw)
        raise LocalRunError(
            f"Invalid job.{field_name}: {raw!r}. Expected a boolean."
        )
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in _BOOL_TRUE_STRINGS:
            return True
        if value in _BOOL_FALSE_STRINGS:
            return False
    raise LocalRunError(
        f"Invalid job.{field_name}: {raw!r}. Expected a boolean."
    )


def _pip_marker_hash(pip_items: Iterable[str]) -> str:
    """Return a stable, order-independent short hash of the pip dep set.

    Empty input -> empty string (caller can treat as "no marker needed").
    Otherwise: sha256 of a sorted, newline-joined representation, truncated
    to 12 hex chars.
    """
    items = sorted(str(item) for item in pip_items if item)
    if not items:
        return ""
    digest = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()
    return digest[:12]


_SLUG_PATTERN = re.compile(r"[^a-z0-9-]+")
_SLUG_COLLAPSE = re.compile(r"-+")


def _derive_container_name(job_name: str) -> str:
    """Derive a stable Docker container name for persistent mode.

    Lowercases, replaces any non-[a-z0-9-] run with a single hyphen, collapses
    hyphen runs, strips leading/trailing hyphens, then prefixes with
    ``local-run-`` iff the slug doesn't already start with it.
    """
    slug = _SLUG_PATTERN.sub("-", str(job_name).lower())
    slug = _SLUG_COLLAPSE.sub("-", slug).strip("-")
    if not slug:
        slug = "job"
    if slug.startswith("local-run-"):
        return slug
    return f"local-run-{slug}"


def _cache_mount_args(plan: dict[str, Any], home_dir: Path) -> list[str]:
    """Build `-v` args for HF + pip host-cache bind mounts.

    Defaults are true-on in ``_compile``; user opts out via
    ``job.mount_hf_cache: false`` / ``job.mount_pip_cache: false``.
    ``home_dir`` is injected so tests don't read the env.
    """
    args: list[str] = []
    if plan.get("mount_hf_cache"):
        args.extend(["-v", f"{home_dir}/.cache/huggingface:/root/.cache/huggingface"])
    if plan.get("mount_pip_cache"):
        args.extend(["-v", f"{home_dir}/.cache/pip:/root/.cache/pip"])
    return args


def _ensure_host_cache_dirs(plan: dict[str, Any], home_dir: Path) -> None:
    """Pre-create ``~/.cache/huggingface`` / ``~/.cache/pip`` on the host.

    Docker auto-creates missing bind-mount sources as root-owned empty dirs,
    which defeats the point of pre-warming. Pre-creating with the invoking
    user's ownership keeps them writable by subsequent host-side tools.
    No-op on Windows (cache dirs don't live under $HOME there and the
    current mount paths wouldn't apply anyway).
    """
    if sys.platform == "win32":
        return
    for field, subpath in (
        ("mount_hf_cache", "huggingface"),
        ("mount_pip_cache", "pip"),
    ):
        if not plan.get(field):
            continue
        target = home_dir / ".cache" / subpath
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"Warning: could not pre-create {target} ({exc}); docker will create it.")


def _build_persistent_docker_run_args(
    plan: dict[str, Any],
    repo_root: Path,
    home_dir: Path,
) -> list[str]:
    """Build ``docker run -d`` argv for creating a persistent container.

    Caller is expected to have resolved ``plan`` via ``_compile`` with
    ``persist=true``. ``home_dir`` is injected (not read from env) so tests
    stay hermetic.
    """
    name = plan["persistent_container_name"]
    stop_timeout = int(plan.get("stop_timeout", DEFAULT_STOP_TIMEOUT))
    user_spec: UserSpec = plan["user_spec"]
    docker_user = user_spec.docker_user_flag or "0:0"

    args: list[str] = [
        "docker",
        "run",
        "-d",
        "--init",
        "--name",
        name,
        "--stop-timeout",
        str(stop_timeout),
        "--gpus",
        "all",
        "-u",
        docker_user,
        "-v",
        f"{repo_root}:/workspace/repo",
    ]
    args.extend(_cache_mount_args(plan, home_dir))
    args.extend(
        [
            "--entrypoint",
            "bash",
            plan["image"],
            "-c",
            "sleep infinity",
        ]
    )
    return args


class LocalRunHandler(BaseHandler):
    """Run config-driven local Docker jobs, starting with SFT training."""

    def __init__(self, args: Namespace | None = None):
        super().__init__(args=args)
        self._container_name: str | None = None

    @property
    def name(self) -> str:
        return "local-run"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _jobs_dir(self) -> Path:
        return self.repo_root / "Trainers" / "local" / "jobs"

    def _list_job_configs(self) -> list[Path]:
        jobs_dir = self._jobs_dir()
        if not jobs_dir.exists():
            return []
        return sorted(path for path in jobs_dir.glob("*.yaml") if path.is_file())

    def _resolve_job_config_path(self, requested: str | None) -> Path:
        if requested:
            candidate = Path(requested)
            if not candidate.is_absolute():
                candidate = self.repo_root / requested
                if not candidate.exists():
                    candidate = self._jobs_dir() / requested
            if candidate.exists():
                return candidate.resolve()
            raise LocalRunError(f"Local job config not found: {requested}")

        configs = self._list_job_configs()
        if self.json_mode:
            raise LocalRunError("JSON mode requires --job-config for local-run.")
        if not configs:
            raise LocalRunError(f"No local job configs found under {self._jobs_dir()}")
        options = [(str(path), f"{BOX['bullet']} {path.stem}") for path in configs]
        choice = print_menu(options, "Select local Docker job config:")
        if not choice:
            raise LocalRunError("Local run cancelled.")
        return Path(choice)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise LocalRunError(f"Local job config must be a YAML object: {path}")
        return data

    def _rel_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def _render_value(self, value: Any, variables: dict[str, str]) -> Any:
        if isinstance(value, str):
            return value.format_map(variables)
        if isinstance(value, list):
            return [self._render_value(item, variables) for item in value]
        if isinstance(value, dict):
            return {str(k): self._render_value(v, variables) for k, v in value.items()}
        return value

    def _build_sft_command(self, cfg: dict[str, Any], variables: dict[str, str]) -> tuple[list[str], str, Path]:
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
        training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
        lora_cfg = cfg.get("lora", {}) if isinstance(cfg.get("lora"), dict) else {}
        run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
        artifacts_cfg = cfg.get("artifacts", {}) if isinstance(cfg.get("artifacts"), dict) else {}

        trainer_path = Path(str(run_cfg.get("trainer", "Trainers/sft/train_sft.py")))
        trainer_dir = trainer_path.parent
        trainer_file = trainer_path.name
        workdir = "/workspace/repo/" + trainer_dir.as_posix()

        command = ["python", trainer_file]
        _append_flag(command, "model_name", model_cfg.get("name") or model_cfg.get("model_name"))
        _append_flag(command, "model_size", model_cfg.get("size"))
        if "load_in_4bit" in model_cfg:
            command.append("--load-in-4bit" if bool(model_cfg["load_in_4bit"]) else "--no-load-in-4bit")
        _append_flag(command, "max_seq_length", model_cfg.get("max_seq_length") or training_cfg.get("max_seq_length"))

        _append_flag(command, "dataset_name", dataset_cfg.get("name") or dataset_cfg.get("dataset_name"))
        _append_flag(command, "dataset_file", dataset_cfg.get("file") or dataset_cfg.get("dataset_file"))
        local_file = dataset_cfg.get("local_file")
        if local_file:
            container_dataset_path = PurePosixPath("/workspace/repo") / Path(str(local_file)).as_posix()
            container_workdir = PurePosixPath(workdir)
            local_file = os.path.relpath(str(container_dataset_path), str(container_workdir)).replace("\\", "/")
        _append_flag(command, "local_file", local_file)
        if bool(dataset_cfg.get("split_dataset", False)):
            command.append("--split-dataset")

        for key in (
            "batch_size",
            "gradient_accumulation",
            "learning_rate",
            "num_epochs",
            "max_steps",
            "save_steps",
            "save_total_limit",
        ):
            _append_flag(command, key, training_cfg.get(key))

        _append_flag(command, "lora_r", lora_cfg.get("r"))
        _append_flag(command, "lora_alpha", lora_cfg.get("alpha") or lora_cfg.get("lora_alpha"))
        _append_flag(command, "lora_dropout", lora_cfg.get("dropout") or lora_cfg.get("lora_dropout"))
        _append_flag(command, "lora_target_modules", lora_cfg.get("target_modules"))
        if bool(lora_cfg.get("use_dora", False)):
            command.append("--use-dora")
        if bool(lora_cfg.get("use_rslora", False)):
            command.append("--use-rslora")
        _append_flag(command, "init_lora_weights", lora_cfg.get("init_lora_weights"))

        output_root = artifacts_cfg.get("output_root", "toolset-training-artifacts/runs/local_docker/sft/{name}")
        output_root = str(self._render_value(output_root, variables))
        run_timestamp = str(
            self._render_value(
                artifacts_cfg.get("run_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")),
                variables,
            )
        )
        command.extend(["--output-root", "../../" + output_root if not output_root.startswith("/") else output_root])
        command.extend(["--run-timestamp", run_timestamp])

        for key in ("tier", "resume_from_checkpoint"):
            _append_flag(command, key, training_cfg.get(key))
        if bool(run_cfg.get("dry_run", False)):
            command.append("--dry-run")
        if not bool(run_cfg.get("dashboard", False)):
            command.append("--no-dashboard")
        if bool(run_cfg.get("quiet", True)):
            command.append("--quiet")
        command.extend(_as_list(run_cfg.get("extra_args")))

        host_artifact_path = self._rel_path(Path(output_root) / run_timestamp)
        return command, workdir, host_artifact_path

    def _compile(self, config_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        provider = str(cfg.get("provider", "local_docker")).strip().lower()
        if provider != "local_docker":
            raise LocalRunError(f"Unsupported local-run provider: {provider}")

        name = str(cfg.get("name") or config_path.stem)
        variables = {
            "name": name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "repo_root": str(self.repo_root),
        }
        variables.update({str(k): str(v) for k, v in (cfg.get("template_vars") or {}).items()})

        job_cfg = cfg.get("job", {}) if isinstance(cfg.get("job"), dict) else {}
        run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
        setup_cfg = cfg.get("setup", {}) if isinstance(cfg.get("setup"), dict) else {}
        artifacts_cfg = cfg.get("artifacts", {}) if isinstance(cfg.get("artifacts"), dict) else {}

        image = str(job_cfg.get("image", "unsloth/unsloth:latest"))
        method = str(run_cfg.get("method", "sft")).lower()
        if run_cfg.get("command"):
            command = _as_list(self._render_value(run_cfg["command"], variables))
            workdir = str(run_cfg.get("workdir", "/workspace/repo"))
            host_artifact_path = self._rel_path(
                artifacts_cfg.get("host_path", f"toolset-training-artifacts/runs/local_docker/custom/{name}")
            )
        elif method == "sft":
            command, workdir, host_artifact_path = self._build_sft_command(cfg, variables)
        else:
            raise LocalRunError("local-run currently supports run.method: sft or an explicit run.command list.")

        transfer_mode = str(job_cfg.get("transfer", "auto")).lower()
        if transfer_mode == "auto":
            transfer_mode = "copy" if os.name == "nt" else "bind"

        job_user = _validate_user_field(job_cfg.get("user"))
        host_uid, host_gid = _current_host_ids()
        user_spec = _resolve_user_spec(job_user, transfer_mode, host_uid, host_gid, sys.platform)

        copy_paths = [Path(path) for path in _as_list(setup_cfg.get("copy"))]
        if transfer_mode == "copy" and not copy_paths:
            copy_paths = [Path("Trainers/sft"), Path("shared"), Path("tuner")]
            dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
            if dataset_cfg.get("local_file"):
                copy_paths.append(Path(str(dataset_cfg["local_file"])))

        stop_timeout = int(job_cfg.get("stop_timeout", DEFAULT_STOP_TIMEOUT))
        tty_mode = _validate_tty_field(job_cfg.get("tty"))

        persist = _validate_bool_field(job_cfg.get("persist"), "persist", default=False)
        if persist and transfer_mode != "bind":
            raise LocalRunError(
                "job.persist=true is only supported with transfer=bind "
                f"(got transfer={transfer_mode!r})."
            )
        mount_hf_cache = _validate_bool_field(
            job_cfg.get("mount_hf_cache"), "mount_hf_cache", default=True
        )
        mount_pip_cache = _validate_bool_field(
            job_cfg.get("mount_pip_cache"), "mount_pip_cache", default=True
        )

        explicit_container_name = job_cfg.get("container_name")
        if explicit_container_name:
            # User-supplied name wins; we still slug/normalize it.
            persistent_container_name = _derive_container_name(str(explicit_container_name))
            ephemeral_container_name = persistent_container_name
        else:
            persistent_container_name = _derive_container_name(name)
            ephemeral_container_name = (
                f"local-run-{name}-{variables['timestamp']}".replace("_", "-")
            )

        pip_items = _as_list(setup_cfg.get("pip"))
        pip_marker_hash = _pip_marker_hash(pip_items)

        return {
            "name": name,
            "config_path": str(config_path),
            "image": image,
            "pull_policy": str(job_cfg.get("pull_policy", "missing")).lower(),
            "transfer": transfer_mode,
            "keep_container": bool(job_cfg.get("keep_container", False)),
            "container_name": ephemeral_container_name,
            "persistent_container_name": persistent_container_name,
            "pip": pip_items,
            "pip_marker_hash": pip_marker_hash,
            "copy_paths": copy_paths,
            "command": command,
            "workdir": workdir,
            "host_artifact_path": host_artifact_path,
            "container_artifact_path": str(
                artifacts_cfg.get(
                    "container_path",
                    "/workspace/repo/" + str(host_artifact_path.relative_to(self.repo_root)).replace("\\", "/"),
                )
            ),
            "job_user": job_user,
            "user_spec": user_spec,
            "stop_timeout": stop_timeout,
            "tty_mode": tty_mode,
            "persist": persist,
            "mount_hf_cache": mount_hf_cache,
            "mount_pip_cache": mount_pip_cache,
        }

    def _run(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=self.repo_root, text=True, **kwargs)

    def _check(self, args: list[str]) -> None:
        result = self._run(args)
        if result.returncode != 0:
            raise LocalRunError(f"Command failed ({result.returncode}): {' '.join(args)}")

    def _pull_image(self, image: str, policy: str) -> None:
        if policy == "never":
            return
        if policy == "missing":
            inspect = self._run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if inspect.returncode == 0:
                return
        if policy not in {"missing", "always"}:
            raise LocalRunError("job.pull_policy must be one of: missing, always, never")
        self._check(["docker", "pull", image])

    def _copy_into_container(self, container: str, paths: Iterable[Path]) -> None:
        for relative in paths:
            src = self._rel_path(relative)
            if not src.exists():
                raise LocalRunError(f"Configured copy path does not exist: {relative}")
            dest = "/workspace/repo/" + Path(relative).as_posix()
            parent = str(Path(dest).parent).replace("\\", "/")
            self._check(["docker", "exec", "-u", "root", container, "mkdir", "-p", parent])
            self._check(["docker", "cp", str(src), f"{container}:{dest}"])
        self._check(["docker", "exec", "-u", "root", container, "chown", "-R", "unsloth:unsloth", "/workspace/repo"])

    def _copy_artifacts_from_container(
        self,
        container: str,
        container_path: str,
        host_path: Path,
        user_spec: UserSpec,
    ) -> None:
        host_parent = host_path.parent
        host_parent.mkdir(parents=True, exist_ok=True)
        if host_path.exists() and any(host_path.iterdir() if host_path.is_dir() else [host_path]):
            raise LocalRunError(f"Artifact destination already exists and is not empty: {host_path}")
        archive_name = f"/tmp/{host_path.name}.tar"
        container_parent = str(Path(container_path).parent).replace("\\", "/")
        container_base = Path(container_path).name
        self._check(["docker", "exec", container, "tar", "-chf", archive_name, "-C", container_parent, container_base])
        host_archive = self.repo_root / "tmp" / f"{host_path.name}.tar"
        host_archive.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._check(["docker", "cp", f"{container}:{archive_name}", str(host_archive)])
            self._check(["tar", "-xf", str(host_archive), "-C", str(host_parent)])
            if (
                user_spec.chown_host_uid is not None
                and user_spec.chown_host_gid is not None
                and sys.platform in {"linux", "darwin"}
            ):
                _chown_host_tree(host_path, user_spec.chown_host_uid, user_spec.chown_host_gid)
        finally:
            if host_archive.exists():
                host_archive.unlink()

    def _execute_copy_mode(self, plan: dict[str, Any]) -> None:
        container = plan["container_name"]
        user_spec: UserSpec = plan["user_spec"]
        tty_flags = _resolve_tty_flags(plan["tty_mode"], sys.stdout.isatty())
        self._container_name = container
        self._check(
            [
                "docker",
                "create",
                "--gpus",
                "all",
                "--stop-timeout",
                str(plan["stop_timeout"]),
                "--entrypoint",
                "sleep",
                "--name",
                container,
                plan["image"],
                "infinity",
            ]
        )
        self._check(["docker", "start", container])
        try:
            self._check(["docker", "exec", "-u", "root", container, "mkdir", "-p", "/workspace/repo"])
            self._copy_into_container(container, plan["copy_paths"])
            if plan["pip"]:
                self._check(["docker", "exec", "-u", "root", container, "pip", "install", "--upgrade", *plan["pip"]])
            command_text = " ".join(shlex.quote(part) for part in plan["command"])
            exec_args = ["docker", "exec", *tty_flags, "-w", plan["workdir"]]
            if user_spec.docker_user_flag is not None:
                exec_args.extend(["-u", user_spec.docker_user_flag])
            exec_args.extend([container, "bash", "-lc", command_text])
            self._check(exec_args)
            self._copy_artifacts_from_container(
                container,
                plan["container_artifact_path"],
                plan["host_artifact_path"],
                user_spec,
            )
        finally:
            if not plan["keep_container"]:
                self._run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._container_name = None

    def _execute_bind_mode(self, plan: dict[str, Any]) -> None:
        user_spec: UserSpec = plan["user_spec"]
        tty_flags = _resolve_tty_flags(plan["tty_mode"], sys.stdout.isatty())
        command_text = _build_bash_wrapper(plan, user_spec)
        home_dir = Path(os.path.expanduser("~"))
        docker_cmd: list[str] = [
            "docker",
            "run",
            "--rm",
            *tty_flags,
            "--gpus",
            "all",
            "--stop-timeout",
            str(plan["stop_timeout"]),
        ]
        if user_spec.docker_user_flag is not None:
            docker_cmd.extend(["-u", user_spec.docker_user_flag])
        docker_cmd.extend(
            [
                "--entrypoint",
                "bash",
                "-v",
                f"{self.repo_root}:/workspace/repo",
            ]
        )
        docker_cmd.extend(_cache_mount_args(plan, home_dir))
        docker_cmd.extend(
            [
                "-w",
                plan["workdir"],
                plan["image"],
                "-lc",
                command_text,
            ]
        )
        self._check(docker_cmd)

    def _container_exists(self, name: str) -> Literal["running", "exited", "absent"]:
        """Query docker for the container's state by name.

        Returns "running" | "exited" | "absent". Unknown/non-standard states
        are coerced to "exited" so callers take the start-then-exec path
        rather than trying to re-create.
        """
        result = self._run(
            ["docker", "inspect", "--format", "{{.State.Status}}", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return "absent"
        state = (result.stdout or "").strip().lower()
        if state == "running":
            return "running"
        return "exited"

    def _ensure_persistent_container(self, plan: dict[str, Any]) -> str:
        """Ensure the persistent container is running; return transition taken.

        Returns one of "reused" (was already running), "started" (was exited),
        "created" (did not exist). Callers use this for summary printing.
        """
        name = plan["persistent_container_name"]
        state = self._container_exists(name)
        if state == "running":
            return "reused"
        if state == "exited":
            self._check(["docker", "start", name])
            return "started"
        # absent
        home_dir = Path(os.path.expanduser("~"))
        self._check(_build_persistent_docker_run_args(plan, self.repo_root, home_dir))
        return "created"

    def _execute_persistent_bind_mode(self, plan: dict[str, Any]) -> None:
        """Run training inside a reusable, long-lived container via ``docker exec``.

        Flow: ensure container -> marker-file-gated pip install -> docker exec
        training with the existing bash wrapper (trap EXIT + chown + exec).
        Container is NOT removed on exit.
        """
        name = plan["persistent_container_name"]
        user_spec: UserSpec = plan["user_spec"]
        tty_flags = _resolve_tty_flags(plan["tty_mode"], sys.stdout.isatty())
        self._ensure_persistent_container(plan)

        # Marker-file-gated pip install. Skip when the exact dep set has
        # already been installed during this container's lifetime.
        if plan["pip"] and plan["pip_marker_hash"]:
            marker = f"/tmp/.pip-installed-{plan['pip_marker_hash']}"
            pip_install_cmd = (
                "pip install --upgrade "
                + " ".join(shlex.quote(item) for item in plan["pip"])
                + f" && touch {shlex.quote(marker)}"
            )
            guarded = (
                f"if [ -f {shlex.quote(marker)} ]; then "
                f"echo 'pip deps unchanged; skipping install'; "
                f"else {pip_install_cmd}; fi"
            )
            self._check(
                ["docker", "exec", "-u", "0:0", name, "bash", "-lc", guarded]
            )

        # Training command. Pip prelude is omitted inside the wrapper because
        # we already ran pip above (and marker-file gating is cheaper than
        # pip's own up-to-date check).
        wrapper_plan = dict(plan)
        wrapper_plan["pip"] = []
        command_text = _build_bash_wrapper(wrapper_plan, user_spec)
        exec_args = ["docker", "exec", *tty_flags, "-w", plan["workdir"]]
        if user_spec.docker_user_flag is not None:
            exec_args.extend(["-u", user_spec.docker_user_flag])
        else:
            exec_args.extend(["-u", "0:0"])
        exec_args.extend([name, "bash", "-lc", command_text])
        self._check(exec_args)

    def _stop_persistent(self, name: str) -> int:
        state = self._container_exists(name)
        if state == "absent":
            print(f"Container {name} does not exist.")
            return 0
        if state == "exited":
            print(f"Container {name} already stopped.")
            return 0
        self._check(["docker", "stop", name])
        print(f"Container {name} stopped.")
        return 0

    def _remove_persistent(self, name: str) -> int:
        state = self._container_exists(name)
        if state == "absent":
            print(f"Container {name} does not exist.")
            return 0
        self._check(["docker", "rm", "-f", name])
        print(f"Container {name} removed.")
        return 0

    def _status_persistent(self, name: str) -> int:
        state = self._container_exists(name)
        print(f"{name}: {state}")
        return 0

    def handle(self) -> int:
        try:
            config_path = self._resolve_job_config_path(getattr(self.args, "job_config", None))
            cfg = self._load_yaml(config_path)
            plan = self._compile(config_path, cfg)
        except Exception as exc:
            if self.json_mode:
                self.output_error(str(exc), code="LOCAL_RUN_CONFIG_ERROR")
            else:
                print(f"Error: {exc}")
            return 1

        # Management actions (persistent-container lifecycle). These short-
        # circuit the normal run path.
        manage_stop = bool(getattr(self.args, "stop", False))
        manage_rm = bool(getattr(self.args, "rm_persistent", False))
        manage_status = bool(getattr(self.args, "container_status", False))
        manage_flags = [manage_stop, manage_rm, manage_status]
        if sum(manage_flags) > 1:
            msg = "--stop, --rm-persistent, --container-status are mutually exclusive."
            if self.json_mode:
                self.output_error(msg, code="LOCAL_RUN_ARG_ERROR")
            else:
                print(f"Error: {msg}")
            return 1
        if any(manage_flags):
            name = plan["persistent_container_name"]
            try:
                if manage_stop:
                    return self._stop_persistent(name)
                if manage_rm:
                    return self._remove_persistent(name)
                return self._status_persistent(name)
            except Exception as exc:
                if self.json_mode:
                    self.output_error(str(exc), code="LOCAL_RUN_MANAGE_ERROR")
                else:
                    print(f"Error: {exc}")
                return 1

        if self.json_mode and not any(
            [
                getattr(self.args, "stop", False),
                getattr(self.args, "rm_persistent", False),
                getattr(self.args, "container_status", False),
            ]
        ):
            serializable = dict(plan)
            serializable["copy_paths"] = [str(path) for path in plan["copy_paths"]]
            serializable["host_artifact_path"] = str(plan["host_artifact_path"])
            user_spec: UserSpec = plan["user_spec"]
            serializable["user_spec"] = {
                "docker_user_flag": user_spec.docker_user_flag,
                "chown_host_uid": user_spec.chown_host_uid,
                "chown_host_gid": user_spec.chown_host_gid,
                "skip_chown": user_spec.skip_chown,
            }
            self.output(serializable)
            return 0

        user_spec = plan["user_spec"]
        chown_back_desc = (
            f"{user_spec.chown_host_uid}:{user_spec.chown_host_gid}"
            if user_spec.chown_host_uid is not None
            else "disabled"
        )
        docker_user_desc = user_spec.docker_user_flag or "image default"
        tty_flags_preview = _resolve_tty_flags(plan["tty_mode"], sys.stdout.isatty())
        tty_attached_desc = "attached" if tty_flags_preview else "detached"

        print("LOCAL RUN")
        print("Run a config-driven local Docker job")
        print()
        print("Local Docker Run Configuration")
        print(f"  Config: {plan['config_path']}")
        print(f"  Name: {plan['name']}")
        print(f"  Image: {plan['image']}")
        print(f"  Pull policy: {plan['pull_policy']}")
        print(f"  Transfer: {plan['transfer']}")
        print(f"  Workdir: {plan['workdir']}")
        print(f"  Artifacts: {plan['host_artifact_path']}")
        print(f"  User mode: {plan['job_user']} (container user: {docker_user_desc})")
        print(f"  Chown back as: {chown_back_desc}")
        print(f"  Stop timeout: {plan['stop_timeout']}s")
        print(f"  TTY: {plan['tty_mode']} ({tty_attached_desc})")
        if plan["persist"]:
            persistent_name = plan["persistent_container_name"]
            reuse_state = self._container_exists(persistent_name)
            reuse_desc = {
                "running": "reusing running container",
                "exited": "reusing stopped container (will start)",
                "absent": "will be created",
            }[reuse_state]
            print(f"  Container: {persistent_name} ({reuse_desc})")
        if plan["transfer"] == "bind":
            print(f"  HF cache mount: {'yes' if plan['mount_hf_cache'] else 'no'}")
            print(f"  pip cache mount: {'yes' if plan['mount_pip_cache'] else 'no'}")
        print(f"  Command: {' '.join(shlex.quote(part) for part in plan['command'])}")

        # WSL drvfs notice — bind-mounting /mnt/<letter>/... may show stale
        # ownership due to drvfs caching; the chown-on-exit trap still runs.
        if (
            plan["transfer"] == "bind"
            and sys.platform == "linux"
            and str(self.repo_root).startswith("/mnt/")
        ):
            print(
                "  Note: repo is under WSL drvfs (/mnt/...). File ownership on Windows "
                "filesystems is a drvfs overlay; chown-back may appear unchanged in "
                "Windows Explorer but will be correct from WSL."
            )
        print()
        if not getattr(self.args, "auto_confirm", False) and not confirm("Start local Docker run with this configuration?"):
            print("Local run cancelled.")
            return 0

        # Ensure the host artifact parent exists before executing — prevents
        # docker from creating it as root on bind mounts.
        plan["host_artifact_path"].parent.mkdir(parents=True, exist_ok=True)
        # Pre-create ~/.cache/huggingface and ~/.cache/pip so docker doesn't
        # bind an empty root-owned dir. Cache mounts apply to bind modes only.
        if plan["transfer"] == "bind":
            _ensure_host_cache_dirs(plan, Path(os.path.expanduser("~")))

        try:
            self._pull_image(plan["image"], plan["pull_policy"])
            if plan["transfer"] == "copy":
                self._execute_copy_mode(plan)
            elif plan["transfer"] == "bind":
                if plan["persist"]:
                    self._execute_persistent_bind_mode(plan)
                else:
                    self._execute_bind_mode(plan)
            else:
                raise LocalRunError("job.transfer must be one of: auto, copy, bind")
        except Exception as exc:
            print(f"Error: {exc}")
            if self._container_name:
                print(f"Temporary container retained for inspection: {self._container_name}")
            return 1

        print(f"Local run completed. Artifacts: {plan['host_artifact_path']}")
        return 0
