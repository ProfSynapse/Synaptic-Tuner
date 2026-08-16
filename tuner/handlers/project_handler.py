"""Side-effect-free inspection and validation of host-project contracts."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from tuner import __version__
from tuner.handlers.base import BaseHandler
from tuner.project import ProjectContext, load_project_manifest
from tuner.project.errors import ProjectError


def _context_dict(context: ProjectContext) -> dict[str, Any]:
    return {
        "mode": context.mode,
        "path_mode": context.path_mode,
        "engine_root": str(context.engine_root),
        "project_root": str(context.project_root),
        "config_root": str(context.config_root),
        "artifact_root": str(context.artifact_root),
        "state_root": str(context.state_root),
        "tracking_root": str(context.tracking_root),
        "cache_root": str(context.cache_root),
        "tmp_root": str(context.tmp_root),
        "invocation_cwd": str(context.invocation_cwd),
        "manifest_path": str(context.manifest_path) if context.manifest_path else None,
    }


class ProjectHandler(BaseHandler):
    """Handle ``project inspect|validate|migrate-dry-run``."""

    def __init__(self, args: Namespace, context: ProjectContext) -> None:
        super().__init__(args=args, context=context)

    @property
    def name(self) -> str:
        return "project"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _manifest_state(self) -> dict[str, Any]:
        path = self.context.manifest_path
        if path is None:
            return {"exists": False, "valid": self.context.mode == "standalone"}
        state: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            state["valid"] = False
            return state
        try:
            manifest = load_project_manifest(path)
            resolved = manifest.create_context(
                engine_root=self.engine_root,
                invocation_cwd=self.context.invocation_cwd,
            )
        except ProjectError as exc:
            state.update({"valid": False, "error": exc.to_dict()})
            return state
        state.update(
            {
                "valid": True,
                "schema_version": manifest.schema_version,
                "project_id": manifest.project_id,
                "engine_requires": manifest.engine_requires,
                "resolved_context": _context_dict(resolved),
            }
        )
        return state

    def _inspect(self) -> int:
        manifest = self._manifest_state()
        data = {
            "schema_version": "synaptic-project-inspection/v1",
            "engine_version": __version__,
            "context": _context_dict(self.context),
            "manifest": manifest,
        }
        self.output(
            data,
            human_readable=(
                f"Project mode: {self.context.mode}\n"
                f"Project root: {self.project_root}\n"
                f"Engine root: {self.engine_root}\n"
                f"Manifest: {manifest.get('path') or 'not selected'}"
            ),
        )
        return 0

    def _validate(self) -> int:
        manifest = self._manifest_state()
        valid = bool(manifest.get("valid"))
        data = {
            "schema_version": "synaptic-project-validation/v1",
            "valid": valid,
            "mode": self.context.mode,
            "manifest": manifest,
        }
        if valid:
            self.output(data, "Project contract is valid.")
            return 0
        error = manifest.get("error")
        details = error if isinstance(error, dict) else {"manifest": manifest}
        self.output_error(
            "Project contract is invalid or its manifest is missing.",
            code=str(error.get("code")) if isinstance(error, dict) else "PROJECT_MANIFEST_NOT_FOUND",
            details=details,
        )
        return 1

    def _migrate_dry_run(self) -> int:
        manifest = self._manifest_state()
        changes: list[dict[str, Any]] = []
        if not manifest.get("exists"):
            target = self.project_root / "synaptic.yaml"
            changes.append(
                {
                    "action": "create",
                    "path": str(target),
                    "schema_version": "synaptic-project/v1",
                }
            )
            changes.append(
                {
                    "action": "create_runtime_roots",
                    "paths": [
                        str(self.project_root / ".synaptic" / name)
                        for name in ("artifacts", "state", "tracking", "cache", "tmp")
                    ],
                }
            )
        elif not manifest.get("valid"):
            changes.append(
                {
                    "action": "repair_manifest",
                    "path": str(self.context.manifest_path),
                    "reason": manifest.get("error", {}).get("code", "invalid"),
                }
            )
        data = {
            "schema_version": "synaptic-project-migration-plan/v1",
            "dry_run": True,
            "writes_performed": False,
            "changes": changes,
        }
        summary = "No migration changes required." if not changes else f"Would perform {len(changes)} migration change(s)."
        self.output(data, summary)
        return 0

    def handle(self) -> int:
        subcommand = getattr(self.args, "subcommand", None) or "inspect"
        if subcommand == "inspect":
            return self._inspect()
        if subcommand == "validate":
            return self._validate()
        if subcommand == "migrate-dry-run":
            return self._migrate_dry_run()
        self.output_error(
            f"Unknown project subcommand: {subcommand}",
            code="PROJECT_SUBCOMMAND_INVALID",
            details={"allowed": ["inspect", "validate", "migrate-dry-run"]},
        )
        return 2
