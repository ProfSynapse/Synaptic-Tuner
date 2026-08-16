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

from tuner.discovery.recipes import load_recipe
from tuner.handlers.base import BaseHandler
from tuner.project import PathRef, ProjectContext
from tuner.ui import BOX, confirm, print_menu


DEFAULT_STOP_TIMEOUT = 60
RUNTIME_LAYOUT_SCHEMA = "synaptic-runtime-layout/v1"

CONTAINER_ROOTS = {
    "engine": "/workspace/engine",
    "project": "/workspace/project",
    "artifacts": "/workspace/artifacts",
    "state": "/workspace/state",
    "tracking": "/workspace/tracking",
    "cache": "/workspace/cache",
    "tmp": "/workspace/tmp",
}

# Generic gitignored landing dir for a music-training audio corpus when a recipe
# leaves dataset.data_dir empty (build contract §5.1). Repo-relative, NOT
# user-specific — a researcher normally points dataset.data_dir at their own
# out-of-repo corpus instead. Kept in sync with the .gitignore entry.
DEFAULT_ACE_STEP_CORPUS_DIR = "Datasets/ace_step_corpus"

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


@dataclass(frozen=True)
class CopyEntry:
    """Canonical copy-mode input resolved at compile time."""

    source: Path
    destination: str
    source_root: Literal["engine", "project"]

    def to_dict(self) -> dict[str, str]:
        return {
            "source": str(self.source),
            "destination": self.destination,
            "source_root": self.source_root,
        }


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


def _append_bool_flag(args: list[str], key: str, value: Any) -> None:
    """Emit a tri-state boolean as ``--flag`` / ``--no-flag`` (None ⇒ omit).

    Mirrors train_sft.py's paired store_true/store_false dest pattern (e.g.
    --load-in-4bit / --no-load-in-4bit) so a recipe can express False explicitly
    without absence collapsing to False. Distinct from :func:`_append_flag`, whose
    bool path can only emit the positive ``--flag`` and silently drops False.
    """
    if value is None:
        return
    args.append(_flag_name(key) if bool(value) else _flag_name("no_" + key))


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


def _resolve_transfer_mode(raw: Any, *, platform_name: str | None = None) -> str:
    mode = str(raw or "auto").lower()
    if mode == "auto":
        return "copy" if (platform_name or os.name) == "nt" else "bind"
    return mode


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

    # Common artifact roots the trainer may write under. Host-project mode has
    # a split read-only source / writable-runtime layout; legacy mode retains
    # the historical single-repository location.
    if plan.get("runtime_layout") == RUNTIME_LAYOUT_SCHEMA:
        raw.extend(CONTAINER_ROOTS[name] for name in ("artifacts", "state", "tracking", "cache", "tmp"))
    else:
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
    cache_root = plan.get("host_cache_root")
    host_base = Path(cache_root) if cache_root else home_dir / ".cache"
    # ``Path.__str__`` uses backslashes on Windows even for injected POSIX test
    # paths. Docker accepts forward slashes consistently on all supported hosts.
    host_base_text = host_base.as_posix()
    if plan.get("mount_hf_cache"):
        args.extend(["-v", f"{host_base_text}/huggingface:/root/.cache/huggingface"])
    if plan.get("mount_pip_cache"):
        args.extend(["-v", f"{host_base_text}/pip:/root/.cache/pip"])
    return args


def _runtime_mount_args(plan: dict[str, Any], repo_root: Path) -> list[str]:
    """Return portable source and writable-root bind arguments.

    Host mode supplies an explicit mount table. Standalone mode intentionally
    keeps the legacy writable repo mount for backwards compatibility.
    """

    mounts = plan.get("runtime_mounts")
    if not mounts:
        return ["-v", f"{repo_root.as_posix()}:/workspace/repo"]
    args: list[str] = []
    for mount in mounts:
        host = Path(mount["host"]).as_posix()
        container = str(mount["container"])
        mode = str(mount.get("mode", "rw"))
        suffix = ":ro" if mode == "ro" else ""
        args.extend(["-v", f"{host}:{container}{suffix}"])
    return args


def _path_within(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` itself or a descendant of it.

    Thin wrapper over ``Path.is_relative_to`` (py3.9+), which is COMPONENT-aware —
    so it does NOT false-positive on sibling prefixes (``/data-evil`` is correctly
    NOT within ``/data``, unlike a brittle string ``startswith`` check). Both inputs
    are expected to be already-``resolve()``-d absolute paths, so the comparison
    runs on the symlink-followed REAL paths: a data_dir/cache_dir that symlinks out
    of the tree (e.g. -> /etc) collapses to its real target and this check catches
    the escape FOR FREE — no separate symlink logic needed (M-b).
    """
    return path.is_relative_to(root)


def _data_dir_mount_args(plan: dict[str, Any]) -> list[str]:
    """Build `-v` args for the generic ACE-STEP audio corpus + tensor-cache mounts.

    Mirrors ``_cache_mount_args``: a method (today only ``ace_step``) that needs a
    large out-of-repo audio corpus sets ``dataset.data_dir`` / ``dataset.cache_dir``
    in its config; ``_compile`` resolves those to absolute host paths and stores
    them on the plan. The corpus is mounted read-only at ``/workspace/data`` and
    the writable ``.pt`` cache at ``/workspace/cache`` (build contract §5.2). The
    wrapper inside the container reads the rewritten container paths, NOT the host
    paths. Absent keys -> no mounts (every existing recipe is unaffected).

    SECURITY NOTE (M-b): ``/workspace/cache`` is mounted READ-WRITE, and under the
    default bind-mode user model (``job.user: auto`` -> run as root + chown-back),
    files the container writes there land ROOT-OWNED on the host. The host path is
    operator-supplied (``dataset.cache_dir``) and resolved with ``.resolve()``,
    which follows symlinks — so do NOT point ``cache_dir`` at a sensitive host
    directory. ``_resolve_data_dir_paths`` emits a containment WARNING when the
    resolved cache_dir escapes both the repo tree and the corpus root (warn-only;
    operator-trust model, never a hard block).
    """
    args: list[str] = []
    host_data_dir = plan.get("data_dir")
    if host_data_dir:
        args.extend(["-v", f"{host_data_dir}:/workspace/data:ro"])
    host_cache_dir = plan.get("cache_dir")
    if host_cache_dir:
        args.extend(["-v", f"{host_cache_dir}:/workspace/cache"])
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
    base = Path(plan["host_cache_root"]) if plan.get("host_cache_root") else home_dir / ".cache"
    for field, subpath in (
        ("mount_hf_cache", "huggingface"),
        ("mount_pip_cache", "pip"),
    ):
        if not plan.get(field):
            continue
        target = base / subpath
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
    ]
    args.extend(_runtime_mount_args(plan, repo_root))
    args.extend(_cache_mount_args(plan, home_dir))
    args.extend(_data_dir_mount_args(plan))
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

    def __init__(
        self,
        args: Namespace | None = None,
        context: ProjectContext | None = None,
    ):
        super().__init__(args=args, context=context)
        self._container_name: str | None = None

    @property
    def name(self) -> str:
        return "local-run"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _job_dirs(self) -> list[Path]:
        if self.context.mode == "standalone":
            return [self.engine_root / "Trainers" / "recipes"]
        candidates = [
            self.context.config_root,
            self.project_root / "Trainers" / "recipes",
            self.engine_root / "Trainers" / "recipes",
        ]
        return list(dict.fromkeys(path.resolve(strict=False) for path in candidates))

    def _jobs_dir(self) -> Path:
        return self._job_dirs()[0]

    def _list_job_configs(self) -> list[Path]:
        results: list[Path] = []
        seen_names: set[str] = set()
        for jobs_dir in self._job_dirs():
            if not jobs_dir.exists():
                continue
            for path in sorted(jobs_dir.glob("*.yaml")):
                if path.is_file() and path.name not in seen_names:
                    results.append(path)
                    seen_names.add(path.name)
        return results

    def _resolve_job_config_path(self, requested: str | None) -> Path:
        if requested:
            candidate = Path(requested)
            if not candidate.is_absolute():
                roots = [self.context.invocation_cwd, self.project_root, *self._job_dirs()]
                for root in roots:
                    candidate = root / requested
                    if candidate.exists():
                        return candidate.resolve()
            elif candidate.exists():
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
        try:
            data = load_recipe(path, "local")
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise LocalRunError(
                f"Local job config must be a YAML object: {path} ({exc})"
            ) from exc
        if not isinstance(data, dict):
            raise LocalRunError(f"Local job config must be a YAML object: {path}")
        return data

    def _rel_path(
        self,
        path_value: str | Path,
        *,
        declaring_file: Path | None = None,
        access: Literal["read", "write"] = "read",
        output_default: bool = False,
    ) -> Path:
        if self.context.path_mode == "project_v1":
            raw = str(path_value)
            if output_default and "://" not in raw and not Path(raw).is_absolute():
                raw = "artifact://" + raw.replace("\\", "/")
            return PathRef.parse(raw).resolve(
                self.context,
                declaring_file=declaring_file,
                from_cli=declaring_file is None,
                access=access,
            )
        path = Path(path_value)
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def _container_path(self, host_path: Path) -> str:
        """Map a resolved host path into the logical runtime layout."""

        resolved = host_path.resolve(strict=False)
        if self.context.mode == "standalone":
            try:
                relative = resolved.relative_to(self.engine_root)
            except ValueError as exc:
                raise LocalRunError(f"Standalone runtime path is outside the engine: {resolved}") from exc
            return str(PurePosixPath("/workspace/repo") / relative.as_posix())
        roots = (
            (self.engine_root, CONTAINER_ROOTS["engine"]),
            (self.artifact_root, CONTAINER_ROOTS["artifacts"]),
            (self.state_root, CONTAINER_ROOTS["state"]),
            (self.tracking_root, CONTAINER_ROOTS["tracking"]),
            (self.cache_root, CONTAINER_ROOTS["cache"]),
            (self.context.tmp_root, CONTAINER_ROOTS["tmp"]),
            (self.project_root, CONTAINER_ROOTS["project"]),
        )
        for root, container_root in roots:
            try:
                relative = resolved.relative_to(root.resolve(strict=False))
            except ValueError:
                continue
            return str(PurePosixPath(container_root) / relative.as_posix())
        raise LocalRunError(f"Path is outside the declared runtime roots: {resolved}")

    def _external_input_destination(self, source: Path) -> str:
        digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
        root = CONTAINER_ROOTS["project"] if self.context.mode == "host" else "/workspace/repo"
        return str(PurePosixPath(root) / ".inputs" / digest / source.name)

    def _copy_entry_for_source(
        self,
        source: Path,
        *,
        destination: str | None = None,
        source_root: Literal["engine", "project"] | None = None,
    ) -> CopyEntry:
        resolved = source.resolve(strict=False)
        if destination is None:
            try:
                destination = self._container_path(resolved)
            except LocalRunError:
                destination = self._external_input_destination(resolved)
        if source_root is None:
            source_root = (
                "engine"
                if resolved.is_relative_to(self.engine_root.resolve(strict=False))
                else "project"
            )
        return CopyEntry(resolved, destination, source_root)

    def _resolve_dataset_copy_entry(
        self, raw_value: str | Path, *, config_path: Path
    ) -> CopyEntry:
        raw = Path(str(raw_value))
        if self.context.mode == "host":
            source = self._rel_path(
                str(raw_value), declaring_file=config_path, access="read"
            )
        elif raw.is_absolute():
            source = raw.resolve(strict=False)
        else:
            source = (self.engine_root / raw).resolve(strict=False)
        return self._copy_entry_for_source(source)

    def _runtime_identity(self, *, image: str, pip_hash: str, config_path: Path) -> str:
        manifest_hash = "standalone"
        if self.context.manifest_path and self.context.manifest_path.is_file():
            manifest_hash = hashlib.sha256(self.context.manifest_path.read_bytes()).hexdigest()
        engine_identity = str(self.engine_root)
        try:
            result = subprocess.run(
                ["git", "-C", str(self.engine_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                engine_identity = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        payload = {
            "schema": RUNTIME_LAYOUT_SCHEMA if self.context.mode == "host" else "legacy",
            "engine": engine_identity,
            "manifest": manifest_hash,
            "image": image,
            "dependencies": pip_hash,
            "config": str(config_path.resolve(strict=False)),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _runtime_mounts(self) -> list[dict[str, str]]:
        if self.context.mode == "standalone":
            return []
        mounts = [
            {"host": str(self.engine_root), "container": CONTAINER_ROOTS["engine"], "mode": "ro"},
            {"host": str(self.project_root), "container": CONTAINER_ROOTS["project"], "mode": "ro"},
        ]
        for name, root in (
            ("artifacts", self.artifact_root),
            ("state", self.state_root),
            ("tracking", self.tracking_root),
            ("cache", self.cache_root),
            ("tmp", self.context.tmp_root),
        ):
            mounts.append({"host": str(root), "container": CONTAINER_ROOTS[name], "mode": "rw"})
        return mounts

    def _resolve_data_dir_paths(
        self, cfg: dict[str, Any], *, config_path: Path | None = None
    ) -> tuple[str | None, str | None]:
        """Resolve dataset.data_dir / dataset.cache_dir to absolute host paths.

        Generic + config-driven (build contract §5.1 — SACROSANCT, never a
        hardcoded personal path). Returns ``(data_dir_host, cache_dir_host)`` as
        strings, or ``(None, None)`` when the recipe declares neither (so non-audio
        methods are entirely unaffected). Resolution rules:

        - ``dataset.data_dir`` empty AND ``dataset.cache_dir`` empty AND method is
          not ``ace_step`` -> ``(None, None)`` (no mounts; the common case).
        - ``data_dir`` empty but the method needs a corpus -> the gitignored
          default landing dir ``Datasets/ace_step_corpus`` (NOT user-specific).
        - ``cache_dir`` empty -> a ``.cache`` subdir under the resolved data_dir,
          so the ``.pt`` tensor cache survives container restarts alongside the
          corpus.

        Both are resolved with ``_rel_path`` so a researcher may give an absolute
        out-of-repo path (the normal case for a large corpus) or a repo-relative
        one. The directories are pre-created on the host so Docker does not create
        them root-owned.

        SECURITY (M-b): ``cache_dir`` is bind-mounted READ-WRITE and, in the default
        bind-mode user model, container writes land root-owned on the host; the
        resolved paths come from ``.resolve()`` which follows symlinks. This method
        therefore emits a containment WARNING (via ``_warn_mount_containment``) when
        a resolved path escapes its expected roots. It is warn-only — the project's
        local-run trust model is operator-trust, so we never hard-fail here.
        """
        dataset_cfg = cfg.get("dataset", {})
        if not isinstance(dataset_cfg, dict):
            return None, None
        method = str(
            (cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}).get(
                "method", ""
            )
        ).lower()

        raw_data_dir = str(dataset_cfg.get("data_dir") or "").strip()
        raw_cache_dir = str(dataset_cfg.get("cache_dir") or "").strip()

        # Nothing to mount unless this method opts into the corpus seam. ace_step
        # always gets the default landing dir even when data_dir is left blank.
        if not raw_data_dir and not raw_cache_dir and method != "ace_step":
            return None, None

        if self.context.mode == "host":
            data_dir_host = self._rel_path(
                raw_data_dir or f"project://{DEFAULT_ACE_STEP_CORPUS_DIR}",
                declaring_file=config_path,
                access="read",
            )
            cache_ref = raw_cache_dir
            if cache_ref and "://" not in cache_ref and not Path(cache_ref).is_absolute():
                cache_ref = "cache://" + cache_ref.replace("\\", "/")
            cache_dir_host = self._rel_path(
                cache_ref or "cache://.",
                declaring_file=config_path,
                access="write",
            )
        else:
            data_dir_host = self._rel_path(raw_data_dir or DEFAULT_ACE_STEP_CORPUS_DIR)
            cache_dir_host = (
                self._rel_path(raw_cache_dir)
                if raw_cache_dir
                else (data_dir_host / ".cache")
            )

        # Containment heads-up BEFORE we create dirs / mount (M-b). Warn-only.
        self._warn_mount_containment(data_dir_host, cache_dir_host)

        targets = (data_dir_host, cache_dir_host) if self.context.mode == "standalone" else (cache_dir_host,)
        for target in targets:
            try:
                target.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                print(
                    f"Warning: could not pre-create {target} ({exc}); "
                    "docker will create it."
                )

        return str(data_dir_host), str(cache_dir_host)

    def _warn_mount_containment(
        self, data_dir_host: Path, cache_dir_host: Path
    ) -> None:
        """Emit a containment WARNING when a resolved corpus/cache mount escapes
        its expected roots (security finding M-b). Warn-only — operator-trust
        model, never a hard block.

        The bind-mount host paths are operator-supplied (``dataset.data_dir`` /
        ``dataset.cache_dir``) and resolved with ``.resolve()``, which FOLLOWS
        SYMLINKS — so a careless or crafted value can land a mount outside the repo
        tree. The two mounts carry different risk:

        - ``cache_dir`` is mounted READ-WRITE at ``/workspace/cache`` and, under the
          default bind-mode user model (root + chown-back), container writes land
          root-owned on the host. This is the real foot-gun, so the warning is
          prominent and fires when cache_dir escapes BOTH the repo tree AND the
          corpus root (``data_dir``, which the operator chose, self-authorizes a
          cache nested under it).
        - ``data_dir`` is mounted READ-ONLY, so an out-of-repo corpus is the
          documented normal case and low-risk; we surface only a quieter
          informational note (not the prominent ``security`` warning) when it
          leaves the repo tree, to flag a possible symlink escape.

        Expected (safe) roots = repo_root and, for cache_dir, the resolved corpus
        root. ``self.repo_root`` is already absolute; we re-resolve defensively so
        the comparison is symlink-consistent with the resolved mount paths.
        """
        repo_root = self.engine_root.resolve()
        default_corpus = (
            (self.project_root if self.context.mode == "host" else repo_root)
            / DEFAULT_ACE_STEP_CORPUS_DIR
        ).resolve()

        # cache_dir: RW + potentially root-owned -> prominent security warning.
        # The escape test runs on the RESOLVED paths (post-_rel_path/.resolve()),
        # so a symlinked cache_dir collapses to its real target and is caught here.
        cache_is_approved = self.context.mode == "host" and _path_within(
            cache_dir_host, self.cache_root.resolve()
        )
        if not cache_is_approved and not _path_within(cache_dir_host, repo_root) and not _path_within(cache_dir_host, data_dir_host):
            print(
                f"Warning (security): resolved dataset.cache_dir {cache_dir_host} "
                "resolves outside both the repo tree and the corpus root. It is "
                "bind-mounted READ-WRITE at /workspace/cache and, under the default "
                "bind/auto user model, files written there are created as "
                "container-root then chown'd to your host uid. Do NOT point "
                "cache_dir at a sensitive directory (e.g. ~/.ssh, /etc, $HOME) "
                "(path resolution follows symlinks)."
            )

        # data_dir: :ro and out-of-repo is the normal large-corpus case -> quieter
        # informational note only, to flag a possible symlink escape.
        data_is_project = self.context.mode == "host" and _path_within(
            data_dir_host, self.project_root.resolve()
        )
        if not data_is_project and not _path_within(data_dir_host, repo_root) and data_dir_host != default_corpus:
            print(
                f"Note: resolved dataset.data_dir {data_dir_host} is outside the "
                "repo tree. It is bind-mounted READ-ONLY at /workspace/data; this is "
                "expected for an out-of-repo corpus — confirm it is the path you "
                "intend (note: path resolution follows symlinks)."
            )

    def _render_value(self, value: Any, variables: dict[str, str]) -> Any:
        if isinstance(value, str):
            return value.format_map(variables)
        if isinstance(value, list):
            return [self._render_value(item, variables) for item in value]
        if isinstance(value, dict):
            return {str(k): self._render_value(v, variables) for k, v in value.items()}
        return value

    def _build_trainer_command(
        self,
        cfg: dict[str, Any],
        variables: dict[str, str],
        method: str,
        *,
        config_path: Path | None = None,
    ) -> tuple[list[str], str, Path]:
        # Builds the trainer invocation for any registered method. The trainer
        # script is selected by run.trainer; the per-method flag dialect differs,
        # so SFT-only flags (LoRA scalars, dashboard/quiet toggles, 4bit, save
        # cadence) are gated to sft, beta is gated to dpo/kto, and the shared
        # hyperparameter flags forward for all methods. See _flag_dialect below.
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
        training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
        lora_cfg = cfg.get("lora", {}) if isinstance(cfg.get("lora"), dict) else {}
        run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
        artifacts_cfg = cfg.get("artifacts", {}) if isinstance(cfg.get("artifacts"), dict) else {}

        default_trainer = f"Trainers/{method}/train_{method}.py"
        trainer_path = Path(str(run_cfg.get("trainer", default_trainer)))
        trainer_dir = trainer_path.parent
        trainer_file = trainer_path.name
        source_root = CONTAINER_ROOTS["engine"] if self.context.mode == "host" else "/workspace/repo"
        workdir = str(PurePosixPath(source_root) / trainer_dir.as_posix())

        # Flags the dpo/kto trainers do not expose as CLI args (they read these
        # from their YAML config instead). Emitting them would make argparse
        # reject the command, so they are forwarded only for sft.
        sft_only = method == "sft"

        command = ["python", trainer_file]
        _append_flag(command, "model_name", model_cfg.get("name") or model_cfg.get("model_name"))
        _append_flag(command, "model_size", model_cfg.get("size"))
        if sft_only and "load_in_4bit" in model_cfg:
            command.append("--load-in-4bit" if bool(model_cfg["load_in_4bit"]) else "--no-load-in-4bit")
        _append_flag(command, "max_seq_length", model_cfg.get("max_seq_length") or training_cfg.get("max_seq_length"))

        _append_flag(command, "dataset_name", dataset_cfg.get("name") or dataset_cfg.get("dataset_name"))
        _append_flag(command, "dataset_file", dataset_cfg.get("file") or dataset_cfg.get("dataset_file"))
        local_file = dataset_cfg.get("local_file")
        if local_file:
            if config_path is None:
                raise LocalRunError("Dataset path resolution requires a declaring config")
            dataset_entry = self._resolve_dataset_copy_entry(
                str(local_file), config_path=config_path
            )
            if self.context.mode == "host":
                local_file = dataset_entry.destination
            else:
                container_dataset_path = PurePosixPath(dataset_entry.destination)
                container_workdir = PurePosixPath(workdir)
                local_file = os.path.relpath(str(container_dataset_path), str(container_workdir)).replace("\\", "/")
        _append_flag(command, "local_file", local_file)
        if bool(dataset_cfg.get("split_dataset", False)):
            command.append("--split-dataset")

        for key in (
            "batch_size",
            "gradient_accumulation",
            "learning_rate",
            "seed",
            "num_epochs",
            "max_steps",
        ):
            _append_flag(command, key, training_cfg.get(key))
        if sft_only:
            for key in ("save_steps", "save_total_limit"):
                _append_flag(command, key, training_cfg.get(key))
            # chat_template_kwargs is a nested mapping, not a scalar, so it cannot
            # ride the _append_flag scalar/list path. Serialize to a JSON object
            # string and forward via --chat-template-kwargs (sft-only: the dpo/kto
            # trainers template internally via TRL and expose no such flag). Omitted
            # entirely when unset so existing recipes are byte-identical.
            chat_template_kwargs = training_cfg.get("chat_template_kwargs")
            if chat_template_kwargs is not None:
                command.extend(
                    ["--chat-template-kwargs", json.dumps(chat_template_kwargs)]
                )

            # aux_head block forwarding (sft-only; the dpo/kto trainers expose no
            # --aux-head-* flags). Forwarded field-by-field ONLY when the recipe
            # carries an aux_head block, so recipes without one emit ZERO new flags
            # and stay byte-identical. enabled/freeze_base are tri-state booleans
            # (--flag/--no-flag) so a recipe can set them False explicitly; the
            # scalar/str/numeric knobs ride the _append_flag None⇒omit path (a
            # falsy-but-set value like lm_loss_weight: 0.0 is still forwarded).
            aux_head_cfg = cfg.get("aux_head")
            if isinstance(aux_head_cfg, dict):
                _append_bool_flag(command, "aux_head_enabled", aux_head_cfg.get("enabled"))
                _append_bool_flag(command, "aux_head_freeze_base", aux_head_cfg.get("freeze_base"))
                for field_name in (
                    "layer",
                    "token_position",
                    "target_field",
                    "loss",
                    "head_type",
                    "out_activation",
                    "input_norm",
                    "lm_loss_weight",
                    "head_lr",
                ):
                    _append_flag(command, "aux_head_" + field_name, aux_head_cfg.get(field_name))
            # prompt_render is a training-config preprocessing knob (it replaces the
            # masking region, so it lives on training, not aux_head), forwarded via
            # the aux-head-grouped flag independently of the aux_head block. Unset
            # ⇒ omitted ⇒ byte-identical for every existing recipe.
            _append_flag(command, "aux_head_prompt_render", training_cfg.get("prompt_render"))

        # beta forwards only for dpo/kto (sft has no --beta argparse). is not None
        # so an explicit beta: 0.0 is honored, not silently swapped for the trainer
        # default — mirroring the --seed semantics (provenance: no silent override).
        if method in ("dpo", "kto"):
            beta = training_cfg.get("beta")
            if beta is not None:
                _append_flag(command, "beta", beta)

        # LoRA scalars forward for all methods: the recipe's lora budget is the
        # SSOT and must flow end-to-end (gating these would silently run dpo/kto at
        # the trainer-default budget, breaking the identical-LoRA-budget control).
        # All three trainers accept these CLI flags.
        _append_flag(command, "lora_r", lora_cfg.get("r"))
        _append_flag(command, "lora_alpha", lora_cfg.get("alpha") or lora_cfg.get("lora_alpha"))
        _append_flag(command, "lora_dropout", lora_cfg.get("dropout") or lora_cfg.get("lora_dropout"))
        _append_flag(command, "lora_target_modules", lora_cfg.get("target_modules"))
        _append_flag(command, "init_lora_weights", lora_cfg.get("init_lora_weights"))
        if bool(lora_cfg.get("use_dora", False)):
            command.append("--use-dora")
        if bool(lora_cfg.get("use_rslora", False)):
            command.append("--use-rslora")

        default_output_root = (
            "runs/local_docker/" + method + "/{name}"
            if self.context.mode == "host"
            else "toolset-training-artifacts/runs/local_docker/" + method + "/{name}"
        )
        output_root = artifacts_cfg.get("output_root", default_output_root)
        output_root = str(self._render_value(output_root, variables))
        run_timestamp = str(
            self._render_value(
                artifacts_cfg.get("run_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")),
                variables,
            )
        )
        host_output_root = self._rel_path(
            output_root,
            declaring_file=config_path,
            access="write" if self.context.mode == "host" else "read",
            output_default=self.context.mode == "host",
        )
        container_output_root = self._container_path(host_output_root)
        command.extend(
            [
                "--output-root",
                container_output_root
                if self.context.mode == "host"
                else ("../../" + output_root if not output_root.startswith("/") else output_root),
            ]
        )
        command.extend(["--run-timestamp", run_timestamp])

        for key in ("tier", "resume_from_checkpoint"):
            _append_flag(command, key, training_cfg.get(key))
        if bool(run_cfg.get("dry_run", False)):
            command.append("--dry-run")
        # --no-dashboard / --quiet are sft-only CLI toggles; dpo/kto do not expose them.
        if sft_only:
            if not bool(run_cfg.get("dashboard", False)):
                command.append("--no-dashboard")
            if bool(run_cfg.get("quiet", True)):
                command.append("--quiet")
        command.extend(_as_list(run_cfg.get("extra_args")))

        host_artifact_path = (host_output_root / run_timestamp).resolve(strict=False)
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
            "engine_root": str(self.engine_root),
            "project_root": str(self.project_root),
            "artifact_root": str(self.artifact_root),
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
            workdir = str(
                run_cfg.get(
                    "workdir",
                    CONTAINER_ROOTS["engine"] if self.context.mode == "host" else "/workspace/repo",
                )
            )
            host_artifact_path = self._rel_path(
                artifacts_cfg.get(
                    "host_path",
                    f"runs/local_docker/custom/{name}"
                    if self.context.mode == "host"
                    else f"toolset-training-artifacts/runs/local_docker/custom/{name}",
                ),
                declaring_file=config_path,
                access="write" if self.context.mode == "host" else "read",
                output_default=self.context.mode == "host",
            )
        elif method in ("sft", "dpo", "kto"):
            command, workdir, host_artifact_path = self._build_trainer_command(
                cfg, variables, method, config_path=config_path
            )
        else:
            raise LocalRunError(
                "local-run supports run.method: sft, dpo, kto, or an explicit run.command list."
            )

        transfer_mode = _resolve_transfer_mode(job_cfg.get("transfer", "auto"))

        job_user = _validate_user_field(job_cfg.get("user"))
        host_uid, host_gid = _current_host_ids()
        user_spec = _resolve_user_spec(job_user, transfer_mode, host_uid, host_gid, sys.platform)

        copy_paths = [Path(path) for path in _as_list(setup_cfg.get("copy"))]
        if transfer_mode == "copy" and not copy_paths:
            # Copy the trainer directory the dispatched method actually runs from
            # (run.trainer selects it per method) rather than always Trainers/sft.
            trainer_dir = Path(str(run_cfg.get("trainer", f"Trainers/{method}/train_{method}.py"))).parent
            copy_paths = [trainer_dir, Path("shared"), Path("tuner")]
            dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
            if dataset_cfg.get("local_file"):
                copy_paths.append(Path(str(dataset_cfg["local_file"])))

        copy_entries: list[CopyEntry] = []
        if transfer_mode == "copy":
            explicit_copy = _as_list(setup_cfg.get("copy"))
            if explicit_copy:
                for raw_copy in explicit_copy:
                    if self.context.mode == "host":
                        source = self._rel_path(
                            raw_copy, declaring_file=config_path, access="read"
                        )
                    else:
                        raw_path = Path(raw_copy)
                        source = (
                            raw_path.resolve(strict=False)
                            if raw_path.is_absolute()
                            else (self.engine_root / raw_path).resolve(strict=False)
                        )
                    copy_entries.append(self._copy_entry_for_source(source))
            else:
                trainer_dir = Path(
                    str(run_cfg.get("trainer", f"Trainers/{method}/train_{method}.py"))
                ).parent
                for engine_relative in (trainer_dir, Path("shared"), Path("tuner")):
                    source = (self.engine_root / engine_relative).resolve(strict=False)
                    destination = str(
                        PurePosixPath(
                            CONTAINER_ROOTS["engine"]
                            if self.context.mode == "host"
                            else "/workspace/repo"
                        )
                        / engine_relative.as_posix()
                    )
                    copy_entries.append(
                        self._copy_entry_for_source(
                            source, destination=destination, source_root="engine"
                        )
                    )
            dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
            if dataset_cfg.get("local_file"):
                copy_entries.append(
                    self._resolve_dataset_copy_entry(
                        str(dataset_cfg["local_file"]), config_path=config_path
                    )
                )
            copy_entries = list(
                {
                    (entry.source, entry.destination): entry
                    for entry in copy_entries
                }.values()
            )
        elif self.context.mode == "host":
            dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
            if dataset_cfg.get("local_file"):
                entry = self._resolve_dataset_copy_entry(
                    str(dataset_cfg["local_file"]), config_path=config_path
                )
                if not (
                    entry.source.is_relative_to(self.engine_root)
                    or entry.source.is_relative_to(self.project_root)
                ):
                    raise LocalRunError(
                        "Host bind mode requires dataset.local_file below the engine or project root"
                    )

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

        # Generic out-of-repo audio-corpus + tensor-cache mounts (build contract
        # §5). Config-driven and method-agnostic: any recipe that sets
        # dataset.data_dir / dataset.cache_dir gets the host paths resolved here
        # and mounted by _data_dir_mount_args. Today only ace_step uses it; absent
        # keys -> None -> no mounts, so every existing recipe is unaffected.
        data_dir_host, cache_dir_host = self._resolve_data_dir_paths(
            cfg, config_path=config_path
        )
        if data_dir_host and transfer_mode != "bind":
            raise LocalRunError(
                "dataset.data_dir requires job.transfer: bind (an out-of-repo "
                "audio corpus cannot be copied into the build context); "
                f"got transfer={transfer_mode!r}."
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
        runtime_identity = self._runtime_identity(
            image=image, pip_hash=pip_marker_hash, config_path=config_path
        )
        if self.context.mode == "host":
            persistent_container_name = _derive_container_name(
                f"{persistent_container_name}-{runtime_identity[:12]}"
            )

        container_artifact_path = self._container_path(host_artifact_path)
        explicit_container_artifact = artifacts_cfg.get("container_path")
        if explicit_container_artifact:
            explicit_text = str(explicit_container_artifact)
            if self.context.mode == "host" and not any(
                explicit_text == root or explicit_text.startswith(root + "/")
                for root in (
                    CONTAINER_ROOTS["artifacts"],
                    CONTAINER_ROOTS["state"],
                    CONTAINER_ROOTS["tracking"],
                    CONTAINER_ROOTS["cache"],
                    CONTAINER_ROOTS["tmp"],
                )
            ):
                raise LocalRunError(
                    "artifacts.container_path must be below a writable /workspace root"
                )
            container_artifact_path = explicit_text

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
            "copy_entries": copy_entries,
            "command": command,
            "workdir": workdir,
            "host_artifact_path": host_artifact_path,
            "container_artifact_path": container_artifact_path,
            "job_user": job_user,
            "user_spec": user_spec,
            "stop_timeout": stop_timeout,
            "tty_mode": tty_mode,
            "persist": persist,
            "data_dir": data_dir_host,
            "cache_dir": cache_dir_host,
            "mount_hf_cache": mount_hf_cache,
            "mount_pip_cache": mount_pip_cache,
            "runtime_layout": RUNTIME_LAYOUT_SCHEMA if self.context.mode == "host" else "legacy",
            "runtime_identity": runtime_identity,
            "runtime_mounts": self._runtime_mounts(),
            "host_cache_root": str(self.cache_root) if self.context.mode == "host" else None,
        }

    def _run(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=self.engine_root, text=True, **kwargs)

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

    def _copy_into_container(
        self,
        container: str,
        paths: Iterable[Path],
        *,
        copy_entries: Iterable[CopyEntry] | None = None,
    ) -> None:
        entries = list(copy_entries or [])
        if self.context.mode == "host" and not entries:
            raise LocalRunError("Host copy mode requires canonical copy entries")
        if not entries:
            for relative in paths:
                raw = Path(relative)
                src = (
                    raw.resolve(strict=False)
                    if raw.is_absolute()
                    else (self.engine_root / raw).resolve(strict=False)
                )
                destination = (
                    self._external_input_destination(src)
                    if raw.is_absolute() and not src.is_relative_to(self.engine_root)
                    else "/workspace/repo/" + raw.as_posix()
                )
                entries.append(
                    self._copy_entry_for_source(
                        src, destination=destination, source_root="engine"
                    )
                )
        for entry in entries:
            src = entry.source
            if not src.exists():
                raise LocalRunError(f"Configured copy path does not exist: {src}")
            dest = entry.destination
            parent = str(Path(dest).parent).replace("\\", "/")
            self._check(["docker", "exec", "-u", "root", container, "mkdir", "-p", parent])
            self._check(["docker", "cp", str(src), f"{container}:{dest}"])
        if self.context.mode == "host":
            self._check(
                [
                    "docker", "exec", "-u", "root", container, "mkdir", "-p",
                    *[CONTAINER_ROOTS[name] for name in ("artifacts", "state", "tracking", "cache", "tmp")],
                ]
            )
            self._check(
                [
                    "docker", "exec", "-u", "root", container, "chmod", "-R", "a-w",
                    CONTAINER_ROOTS["engine"], CONTAINER_ROOTS["project"],
                ]
            )
            self._check(
                [
                    "docker", "exec", "-u", "root", container, "chown", "-R", "unsloth:unsloth",
                    *[CONTAINER_ROOTS[name] for name in ("artifacts", "state", "tracking", "cache", "tmp")],
                ]
            )
        else:
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
        host_archive = self.context.tmp_root / f"{host_path.name}.tar"
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
            source_dirs = (
                [CONTAINER_ROOTS["engine"], CONTAINER_ROOTS["project"]]
                if self.context.mode == "host"
                else ["/workspace/repo"]
            )
            self._check(["docker", "exec", "-u", "root", container, "mkdir", "-p", *source_dirs])
            self._copy_into_container(
                container,
                plan["copy_paths"],
                copy_entries=plan.get("copy_entries"),
            )
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
        docker_cmd.extend(["--entrypoint", "bash"])
        docker_cmd.extend(_runtime_mount_args(plan, self.repo_root))
        docker_cmd.extend(_cache_mount_args(plan, home_dir))
        docker_cmd.extend(_data_dir_mount_args(plan))
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
            serializable["copy_entries"] = [
                entry.to_dict() for entry in plan.get("copy_entries", [])
            ]
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

        # Ensure declared writable roots exist before Docker sees them. Source
        # roots are intentionally never created or mutated here.
        if self.context.mode == "host":
            for root in self.context.writable_roots:
                root.mkdir(parents=True, exist_ok=True)
        else:
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
