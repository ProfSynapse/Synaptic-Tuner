"""Provider-neutral logical filesystem layout for cloud runtimes."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from tuner.project.context import ProjectContext
from tuner.project.errors import SourceLockError


RUNTIME_LAYOUT_SCHEMA = "synaptic-runtime-layout/v1"
_WRITABLE_NAMES = ("artifacts", "state", "tracking", "cache", "tmp")


@dataclass(frozen=True)
class RuntimeMount:
    name: str
    source: Path
    target: PurePosixPath
    read_only: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": str(self.source),
            "target": self.target.as_posix(),
            "read_only": self.read_only,
        }


@dataclass(frozen=True)
class CloudRuntimeLayout:
    """Logical split between immutable source and mutable runtime roots."""

    engine: RuntimeMount
    project: RuntimeMount
    writable: tuple[RuntimeMount, ...]
    project_carveout_root: Path | None = None
    engine_carveout_root: Path | None = None
    schema_version: str = RUNTIME_LAYOUT_SCHEMA

    def __post_init__(self) -> None:
        if self.engine.name != "engine" or self.project.name != "project":
            raise SourceLockError("Runtime layout requires engine and project source mounts")
        if not self.engine.read_only or not self.project.read_only:
            raise SourceLockError("Engine and project roots must be read-only")
        names = tuple(mount.name for mount in self.writable)
        if names != _WRITABLE_NAMES or any(mount.read_only for mount in self.writable):
            raise SourceLockError("Runtime layout requires the five canonical writable roots")
        self._validate_source_separation()
        targets = [self.engine.target, self.project.target, *(mount.target for mount in self.writable)]
        if any(not target.is_absolute() for target in targets) or len(set(targets)) != len(targets):
            raise SourceLockError("Runtime mount targets must be distinct absolute paths")
        for index, target in enumerate(targets):
            for other in targets[index + 1 :]:
                if target in other.parents or other in target.parents:
                    raise SourceLockError("Runtime mount targets cannot overlap")

    def _validate_source_separation(self) -> None:
        """Validate resolved filesystem identity, including symlinks/junctions."""

        engine = self.engine.source.resolve()
        project = self.project.source.resolve()
        carveout = self.project_carveout_root.resolve() if self.project_carveout_root else None
        engine_carveout = self.engine_carveout_root.resolve() if self.engine_carveout_root else None
        if carveout is not None and (
            carveout != (project / ".synaptic").resolve() or project not in carveout.parents
        ):
            raise SourceLockError("Project writable carve-out must remain inside project/.synaptic")
        if engine_carveout is not None and (
            engine_carveout != (engine / ".synaptic").resolve()
            or engine not in engine_carveout.parents
        ):
            raise SourceLockError("Engine writable carve-out must remain inside engine/.synaptic")

        resolved_writable = [(mount.name, mount.source.resolve()) for mount in self.writable]
        for name, writable in resolved_writable:
            for source_name, source in (("engine", engine), ("project", project)):
                if writable == source or writable in source.parents:
                    raise SourceLockError(
                        f"Writable root {name!r} aliases or contains the {source_name} source root"
                    )
            if engine in writable.parents:
                allowed_engine = (
                    engine_carveout is not None
                    and engine_carveout in writable.parents
                    and writable == (engine_carveout / name).resolve()
                )
                if not allowed_engine:
                    raise SourceLockError(f"Writable root {name!r} is inside the engine source root")
            if project in writable.parents:
                allowed = (
                    carveout is not None
                    and carveout in writable.parents
                    and writable == (carveout / name).resolve()
                ) or (
                    project == engine
                    and engine_carveout is not None
                    and writable == (engine_carveout / name).resolve()
                )
                if not allowed:
                    raise SourceLockError(
                        f"Writable root {name!r} is not an approved project/.synaptic carve-out"
                    )

        for index, (name, writable) in enumerate(resolved_writable):
            for other_name, other in resolved_writable[index + 1 :]:
                if writable == other or writable in other.parents or other in writable.parents:
                    raise SourceLockError(
                        f"Writable roots {name!r} and {other_name!r} overlap"
                    )

    @property
    def writable_by_name(self) -> Mapping[str, RuntimeMount]:
        return {mount.name: mount for mount in self.writable}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_roots": [self.engine.to_dict(), self.project.to_dict()],
            "writable_roots": [mount.to_dict() for mount in self.writable],
        }


def build_runtime_layout(
    context: ProjectContext,
    *,
    workspace_root: PurePosixPath = PurePosixPath("/workspace"),
) -> CloudRuntimeLayout:
    """Map a resolved project context onto the canonical cloud paths."""

    workspace = PurePosixPath(workspace_root)
    if not workspace.is_absolute():
        raise SourceLockError("Cloud workspace root must be absolute")
    sources = (
        RuntimeMount("engine", context.engine_root.resolve(), workspace / "engine", True),
        RuntimeMount("project", context.project_root.resolve(), workspace / "project", True),
    )
    if context.mode == "standalone":
        lexical_carveout = context.engine_root / ".synaptic"
        writable_sources = {}
        for name in _WRITABLE_NAMES:
            lexical_writable = lexical_carveout / name
            _assert_no_redirected_components(lexical_writable, context.engine_root)
            writable_sources[name] = lexical_writable.resolve()
    else:
        lexical_carveout = context.project_root / ".synaptic"
        defaults = {name: lexical_carveout / name for name in _WRITABLE_NAMES}
        declared = dict(zip(_WRITABLE_NAMES, context.writable_roots))
        writable_sources = {}
        for name in _WRITABLE_NAMES:
            if declared[name].resolve() == defaults[name].resolve():
                _assert_no_redirected_components(defaults[name], context.project_root)
            writable_sources[name] = declared[name].resolve()
    writable = tuple(
        RuntimeMount(name, writable_sources[name], workspace / name, False)
        for name in _WRITABLE_NAMES
    )
    return CloudRuntimeLayout(
        engine=sources[0],
        project=sources[1],
        writable=writable,
        project_carveout_root=(context.project_root / ".synaptic") if context.mode == "host" else None,
        engine_carveout_root=(context.engine_root / ".synaptic") if context.mode == "standalone" else None,
    )


def _is_redirect(path: Path) -> bool:
    if path.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(path)):
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _assert_no_redirected_components(path: Path, boundary: Path) -> None:
    """Reject lexical symlink/junction/reparse escapes without creating paths."""

    boundary = boundary.resolve()
    lexical = path.absolute()
    try:
        relative = lexical.relative_to(boundary)
    except ValueError as exc:
        raise SourceLockError("Writable carve-out is outside its source boundary") from exc
    current = boundary
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if _is_redirect(current):
                raise SourceLockError("Writable carve-out cannot traverse a symlink or junction")
    resolved = lexical.resolve()
    if resolved == boundary or boundary not in resolved.parents:
        raise SourceLockError("Writable carve-out resolves outside its source boundary")
