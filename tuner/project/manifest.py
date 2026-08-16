"""Loader and validator for the synaptic-project/v1 host manifest."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from .context import ProjectContext
from .errors import (
    ManifestNotFoundError,
    ManifestValidationError,
    ManifestVersionError,
    SecretReferenceError,
)
from .path_refs import PathRef
from .secrets import reject_literal_secrets

SCHEMA_VERSION = "synaptic-project/v1"
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "project",
    "engine",
    "paths",
    "configuration",
    "plugins",
    "policies",
}


@dataclass(frozen=True)
class ProjectManifest:
    path: Path
    data: Mapping[str, Any]

    @property
    def schema_version(self) -> str:
        return str(self.data["schema_version"])

    @property
    def project_id(self) -> str:
        return str(self.data["project"]["id"])

    @property
    def engine_requires(self) -> str:
        return str(self.data["engine"]["requires"])

    @property
    def policies(self) -> Mapping[str, Any]:
        return self.data.get("policies", {})

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    def create_context(
        self, *, engine_root: Path, invocation_cwd: Path | None = None
    ) -> ProjectContext:
        project_root = self.path.parent.resolve()
        base = ProjectContext.host(
            engine_root=engine_root,
            project_root=project_root,
            invocation_cwd=invocation_cwd,
            manifest_path=self.path,
        )
        declared = self.data.get("paths", {})

        def root(name: str, default: Path) -> Path:
            raw = declared.get(name)
            if raw is None:
                return default
            return PathRef.parse(str(raw)).resolve(base, declaring_file=self.path)

        context = ProjectContext.host(
            engine_root=engine_root,
            project_root=project_root,
            invocation_cwd=invocation_cwd,
            manifest_path=self.path,
            config_root=root("configs", base.config_root),
            artifact_root=root("artifacts", base.artifact_root),
            state_root=root("state", base.state_root),
            tracking_root=root("tracking", base.tracking_root),
            cache_root=root("cache", base.cache_root),
            tmp_root=root("tmp", base.tmp_root),
        )
        _validate_context_roots(context)
        return context


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / "synaptic-project-v1.schema.json"


def load_project_manifest(
    path: Path | str, *, schema_path: Path | None = None
) -> ProjectManifest:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ManifestNotFoundError(
            f"Project manifest not found: {manifest_path}",
            details={"path": str(manifest_path)},
        )
    try:
        parsed = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestValidationError(f"Could not parse project manifest: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ManifestValidationError("Project manifest must be a YAML mapping")
    try:
        reject_literal_secrets(parsed)
    except SecretReferenceError as exc:
        raise ManifestValidationError(
            "Project manifest contains a literal value in a secret-bearing field",
            details=exc.details,
        ) from exc
    version = parsed.get("schema_version")
    if not isinstance(version, str) or not version.startswith("synaptic-project/"):
        raise ManifestVersionError("Project manifest must declare synaptic-project/v1")
    if version.split("/", 1)[1].split(".", 1)[0] != "v1":
        raise ManifestVersionError(f"Unsupported project schema: {version}")

    selected_schema = schema_path or _schema_path()
    try:
        import json

        schema = json.loads(selected_schema.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestValidationError(f"Could not load project manifest schema: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(parsed), key=lambda e: list(e.path))
    if errors:
        details = [
            {"path": ".".join(map(str, error.path)), "message": error.message}
            for error in errors
        ]
        raise ManifestValidationError(
            "Project manifest failed schema validation", details={"errors": details}
        )
    unknown = sorted(set(parsed) - _TOP_LEVEL_FIELDS)
    if unknown:
        warnings.warn(
            f"Unknown optional project manifest fields retained: {', '.join(unknown)}",
            UserWarning,
            stacklevel=2,
        )
    return ProjectManifest(path=manifest_path, data=parsed)


def _validate_context_roots(context: ProjectContext) -> None:
    if context.mode != "host":
        return
    engine = context.engine_root.resolve(strict=False)
    mutable_root = (context.project_root / ".synaptic").resolve(strict=False)
    for name, root in (
        ("artifacts", context.artifact_root),
        ("state", context.state_root),
        ("tracking", context.tracking_root),
        ("cache", context.cache_root),
        ("tmp", context.tmp_root),
    ):
        try:
            inside_engine = root.resolve(strict=False).is_relative_to(engine)
        except (OSError, ValueError):
            inside_engine = False
        if inside_engine:
            raise ManifestValidationError(
                f"Writable root {name!r} cannot be inside the engine checkout",
                details={"root": str(root), "engine_root": str(engine)},
            )
        try:
            inside_mutable_root = root.resolve(strict=False).is_relative_to(mutable_root)
        except (OSError, ValueError):
            inside_mutable_root = False
        if not inside_mutable_root or root.resolve(strict=False) == mutable_root:
            raise ManifestValidationError(
                f"Writable root {name!r} must be below the host .synaptic directory",
                details={"root": str(root), "mutable_root": str(mutable_root)},
            )
