"""Scheme-aware path references for project_v1 configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from .context import ProjectContext
from .errors import ExternalPathError, PathEscapeError, PathReferenceError, WriteAccessError

PathAccess = Literal["read", "write"]
_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):\/\/")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SUPPORTED_SCHEMES = {
    "project",
    "engine",
    "config",
    "artifact",
    "state",
    "tracking",
    "cache",
    "tmp",
    "file",
}
_EXTERNAL_PATH_POLICIES = {"deny", "warn_local_deny_cloud", "allow"}


def _contained(path: Path, root: Path) -> bool:
    try:
        return path.resolve(strict=False).is_relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class PathRef:
    raw: str
    scheme: str | None
    value: str

    @classmethod
    def parse(cls, value: str | os.PathLike[str]) -> "PathRef":
        raw = os.fspath(value)
        if not isinstance(raw, str) or not raw:
            raise PathReferenceError("Path reference must be a non-empty string")
        match = None if _WINDOWS_DRIVE_RE.match(raw) else _SCHEME_RE.match(raw)
        if not match:
            return cls(raw=raw, scheme=None, value=raw)
        scheme = match.group(1).lower()
        if scheme not in _SUPPORTED_SCHEMES:
            raise PathReferenceError(f"Unsupported path reference scheme: {scheme}")
        return cls(raw=raw, scheme=scheme, value=raw[match.end() :])

    def resolve(
        self,
        context: ProjectContext,
        *,
        declaring_file: Path | None = None,
        from_cli: bool = False,
        access: PathAccess = "read",
        cloud: bool = False,
        external_paths: str = "warn_local_deny_cloud",
    ) -> Path:
        if external_paths not in _EXTERNAL_PATH_POLICIES:
            raise ExternalPathError(f"Unsupported external path policy: {external_paths}")
        if context.path_mode == "legacy" and self.scheme is None:
            return Path(self.value)

        if self.scheme == "file":
            parsed = urlsplit(self.raw)
            if parsed.netloc not in {"", "localhost"}:
                raise ExternalPathError("file:// network authorities are not supported")
            value = unquote(parsed.path)
            if os.name == "nt" and re.match(r"^/[A-Za-z]:/", value):
                value = value[1:]
            result = Path(value)
            if not result.is_absolute():
                raise ExternalPathError("file:// references must be absolute")
            if external_paths == "deny":
                raise ExternalPathError("External file paths are denied by project policy")
            if cloud:
                raise ExternalPathError("External file paths require a declared cloud transport")
            if access == "write":
                raise WriteAccessError("External file references are input-only")
            return result.resolve(strict=False)

        roots = {
            "project": context.project_root,
            "engine": context.engine_root,
            "artifact": context.artifact_root,
            "state": context.state_root,
            "tracking": context.tracking_root,
            "cache": context.cache_root,
            "tmp": context.tmp_root,
        }
        if self.scheme == "config":
            if declaring_file is None:
                raise PathReferenceError("config:// requires a declaring document")
            root = declaring_file.resolve().parent
        elif self.scheme is not None:
            root = roots[self.scheme]
        elif from_cli:
            root = context.invocation_cwd
        else:
            if declaring_file is None:
                raw_candidate = Path(self.value)
                if not raw_candidate.is_absolute() and not _WINDOWS_DRIVE_RE.match(self.value):
                    raise PathReferenceError("Relative config paths require a declaring document")
                root = context.invocation_cwd
            else:
                root = declaring_file.resolve().parent

        relative = self.value.replace("\\", os.sep) if self.scheme is not None else self.value
        candidate = Path(relative)
        if candidate.is_absolute() or _WINDOWS_DRIVE_RE.match(relative):
            if self.scheme is not None:
                raise PathReferenceError(f"{self.scheme}:// references must be root-relative")
            if external_paths == "deny":
                raise ExternalPathError("Absolute paths are denied by project policy")
            result = candidate.resolve(strict=False)
            if cloud:
                raise ExternalPathError("Absolute paths require a declared cloud transport")
        else:
            result = (root / candidate).resolve(strict=False)

        external = not (
            _contained(result, context.project_root)
            or _contained(result, context.engine_root)
            or any(_contained(result, writable) for writable in context.writable_roots)
        )
        if self.scheme is None and external:
            if external_paths == "deny":
                raise ExternalPathError("External paths are denied by project policy")
            if cloud:
                raise ExternalPathError("External paths require a declared cloud transport")

        if self.scheme is not None and not _contained(result, root):
            raise PathEscapeError(
                f"{self.raw!r} escapes its {self.scheme} root",
                details={"root": str(root), "resolved": str(result)},
            )

        if access == "write":
            if any(_contained(result, writable) for writable in context.writable_roots):
                return result
            raise WriteAccessError(
                f"Output path {self.raw!r} is outside project writable roots",
                details={"resolved": str(result)},
            )
        return result


def resolve_path(
    value: str | os.PathLike[str],
    context: ProjectContext,
    **kwargs: object,
) -> Path:
    return PathRef.parse(value).resolve(context, **kwargs)  # type: ignore[arg-type]
