"""Project-root discovery and immutable runtime root context."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from .errors import ProjectRootAmbiguousError

ProjectMode = Literal["standalone", "host"]
PathMode = Literal["legacy", "project_v1"]


@dataclass(frozen=True)
class ProjectContext:
    engine_root: Path
    project_root: Path
    config_root: Path
    artifact_root: Path
    state_root: Path
    tracking_root: Path
    cache_root: Path
    tmp_root: Path
    invocation_cwd: Path
    manifest_path: Path | None
    mode: ProjectMode
    path_mode: PathMode

    def __post_init__(self) -> None:
        for field_name in (
            "engine_root",
            "project_root",
            "config_root",
            "artifact_root",
            "state_root",
            "tracking_root",
            "cache_root",
            "tmp_root",
            "invocation_cwd",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{field_name} must be an absolute Path")
        if self.mode == "host" and self.path_mode != "project_v1":
            raise ValueError("Host mode requires project_v1 path semantics")

    @property
    def writable_roots(self) -> tuple[Path, ...]:
        return (
            self.artifact_root,
            self.state_root,
            self.tracking_root,
            self.cache_root,
            self.tmp_root,
        )

    @classmethod
    def host(
        cls,
        *,
        engine_root: Path,
        project_root: Path,
        invocation_cwd: Path | None = None,
        manifest_path: Path | None = None,
        config_root: Path | None = None,
        artifact_root: Path | None = None,
        state_root: Path | None = None,
        tracking_root: Path | None = None,
        cache_root: Path | None = None,
        tmp_root: Path | None = None,
    ) -> "ProjectContext":
        project = project_root.resolve()
        mutable = project / ".synaptic"
        return cls(
            engine_root=engine_root.resolve(),
            project_root=project,
            config_root=(config_root or project / "experiments").resolve(),
            artifact_root=(artifact_root or mutable / "artifacts").resolve(),
            state_root=(state_root or mutable / "state").resolve(),
            tracking_root=(tracking_root or mutable / "tracking").resolve(),
            cache_root=(cache_root or mutable / "cache").resolve(),
            tmp_root=(tmp_root or mutable / "tmp").resolve(),
            invocation_cwd=(invocation_cwd or Path.cwd()).resolve(),
            manifest_path=(manifest_path or project / "synaptic.yaml").resolve(),
            mode="host",
            path_mode="project_v1",
        )

    @classmethod
    def standalone(
        cls, *, engine_root: Path, invocation_cwd: Path | None = None
    ) -> "ProjectContext":
        engine = engine_root.resolve()
        return cls(
            engine_root=engine,
            project_root=engine,
            config_root=engine,
            artifact_root=engine,
            state_root=engine,
            tracking_root=(engine / ".tracking").resolve(),
            cache_root=(engine / ".cache").resolve(),
            tmp_root=(engine / "tmp").resolve(),
            invocation_cwd=(invocation_cwd or Path.cwd()).resolve(),
            manifest_path=None,
            mode="standalone",
            path_mode="legacy",
        )


def find_nearest_manifest(start: Path | None) -> Path | None:
    if start is None:
        return None
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate_root in (current, *current.parents):
        candidate = candidate_root / "synaptic.yaml"
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_git_superproject(engine_root: Path) -> Path | None:
    """Return Git's superproject root, or None outside a submodule checkout."""

    try:
        result = subprocess.run(
            ["git", "-C", str(engine_root), "rev-parse", "--show-superproject-working-tree"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return Path(value).resolve() if value else None


def discover_project_context(
    *,
    engine_root: Path,
    explicit_project_root: Path | None = None,
    primary_config: Path | None = None,
    invocation_cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    superproject_root: Path | None = None,
) -> ProjectContext:
    """Discover roots using the contract precedence without loading the manifest."""

    env = environment if environment is not None else os.environ
    cwd = (invocation_cwd or Path.cwd()).resolve()
    if explicit_project_root is not None:
        return ProjectContext.host(
            engine_root=engine_root, project_root=explicit_project_root, invocation_cwd=cwd
        )
    env_root = env.get("SYNAPTIC_PROJECT_ROOT", "").strip()
    if env_root:
        return ProjectContext.host(
            engine_root=engine_root, project_root=Path(env_root), invocation_cwd=cwd
        )

    config_manifest = find_nearest_manifest(primary_config)
    cwd_manifest = find_nearest_manifest(cwd)
    if config_manifest and cwd_manifest and config_manifest != cwd_manifest:
        raise ProjectRootAmbiguousError(
            "Primary config and invocation cwd select different projects",
            details={
                "config_manifest": str(config_manifest),
                "cwd_manifest": str(cwd_manifest),
            },
        )
    selected = config_manifest or cwd_manifest
    if selected:
        return ProjectContext.host(
            engine_root=engine_root,
            project_root=selected.parent,
            invocation_cwd=cwd,
            manifest_path=selected,
        )

    superproject = superproject_root or find_git_superproject(engine_root)
    if superproject:
        return ProjectContext.host(
            engine_root=engine_root,
            project_root=superproject,
            invocation_cwd=cwd,
            manifest_path=superproject / "synaptic.yaml",
        )
    return ProjectContext.standalone(engine_root=engine_root, invocation_cwd=cwd)
