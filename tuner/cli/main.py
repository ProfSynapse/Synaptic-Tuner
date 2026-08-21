"""
Main CLI entry point.

Location: tuner/cli/main.py
Purpose: Main entry point for Synaptic Tuner CLI
Used by: tuner.py wrapper, python -m tuner
"""

import json
import os
import sys
from pathlib import Path

from shared.utilities.env import load_env_file
from tuner import __version__
from tuner.project import (
    ProjectContext,
    discover_project_context,
    load_project_manifest,
    validate_engine_requirement,
)
from tuner.project.errors import (
    ManifestNotFoundError,
    ProjectError,
    ProjectRootAmbiguousError,
)
from .parser import create_parser
from .router import route_command


class EnvironmentFileError(ProjectError):
    """Stable CLI bootstrap error for an explicitly selected dotenv file."""

    code = "ENV_FILE_NOT_FOUND"


class EnvironmentFileSupportError(ProjectError):
    """Explicit dotenv selection cannot be honored by this installation."""

    code = "ENV_FILE_SUPPORT_UNAVAILABLE"


def _primary_config_path(args, invocation_cwd: Path) -> Path | None:
    """Return the first command-specific config useful for project discovery."""

    for name in (
        "job_config",
        "experiment_spec",
        "ml_config",
        "experiment_loop_config",
        "prompt_opt_config",
        "surgery_config",
        "mechinterp_config",
        "pipeline_config",
        "gates_config",
        "flywheel_config",
        "export_config",
    ):
        raw = getattr(args, name, None)
        if not raw or "://" in str(raw):
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = invocation_cwd / path
        path = path.resolve()
        # Discovery walks directories. For a not-yet-created config, begin at
        # its declaring directory rather than treating its filename as a root.
        return path if path.is_file() else path.parent
    return None


def build_project_context(args, *, engine_root: Path | None = None) -> ProjectContext:
    """Resolve and validate the invocation context without loading dotenv."""

    cwd = Path.cwd().resolve()
    process_engine_root = os.environ.get("SYNAPTIC_ENGINE_ROOT", "").strip()
    selected_engine_root = (
        engine_root
        if engine_root is not None
        else Path(process_engine_root)
        if process_engine_root
        else Path(__file__).parents[2]
    )
    engine = Path(selected_engine_root).expanduser().resolve()
    explicit_root = Path(args.project_root) if getattr(args, "project_root", None) else None
    manifest_arg = getattr(args, "manifest", None)
    explicit_manifest = None
    if manifest_arg:
        explicit_manifest = Path(manifest_arg)
        if not explicit_manifest.is_absolute():
            explicit_manifest = cwd / explicit_manifest
        explicit_manifest = explicit_manifest.resolve()
        manifest_root = explicit_manifest.parent
        if explicit_root is not None and explicit_root.resolve() != manifest_root:
            raise ProjectRootAmbiguousError(
                "--project-root and --manifest select different projects",
                details={
                    "project_root": str(explicit_root.resolve()),
                    "manifest": str(explicit_manifest),
                },
            )
        explicit_root = manifest_root

    context = discover_project_context(
        engine_root=engine,
        explicit_project_root=explicit_root,
        primary_config=_primary_config_path(args, cwd),
        invocation_cwd=cwd,
    )
    if explicit_manifest is not None:
        context = ProjectContext.host(
            engine_root=engine,
            project_root=explicit_manifest.parent,
            invocation_cwd=cwd,
            manifest_path=explicit_manifest,
        )

    if context.mode == "host" and context.manifest_path is not None:
        if context.manifest_path.is_file():
            try:
                manifest = load_project_manifest(context.manifest_path)
            except ProjectError:
                # Project inspection owns structured reporting for malformed
                # manifests. All execution commands fail closed immediately.
                if getattr(args, "command", None) == "project":
                    return context
                raise
            validate_engine_requirement(manifest, __version__)
            if getattr(args, "command", None) != "project":
                context = manifest.create_context(engine_root=engine, invocation_cwd=cwd)
        elif getattr(args, "command", None) != "project":
            raise ManifestNotFoundError(
                f"Project manifest not found: {context.manifest_path}",
                details={"path": str(context.manifest_path)},
            )
    return context


def _explicit_env_path(args, invocation_cwd: Path) -> Path | None:
    raw = getattr(args, "env_file", None)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = invocation_cwd / path
    path = path.resolve()
    if not path.is_file():
        raise EnvironmentFileError(
            f"Explicit env file not found: {path}",
            details={"path": str(path)},
        )
    return path


def _print_project_error(error: ProjectError, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps({"success": False, "error": error.to_dict()}, indent=2))
    else:
        print(f"Error [{error.code}]: {error}", file=sys.stderr)


def main(argv=None):
    """
    Main CLI entry point.

    Parses command-line arguments, routes to appropriate handler,
    and handles top-level errors gracefully.

    Exit Codes:
        0: Success
        1: General error
        130: Interrupted by user (Ctrl+C)

    Example:
        >>> if __name__ == "__main__":
        ...     main()
    """
    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        # Project discovery intentionally precedes dotenv loading. Existing
        # process environment participates in discovery and always wins over
        # values loaded from the selected project/engine env file.
        context = build_project_context(args)
        explicit_env = _explicit_env_path(args, context.invocation_cwd)
        protected_hf_command = getattr(args, "command", None) in {"hf-source", "hf-smoke", "hf-training-smoke"}
        if protected_hf_command:
            # These handlers own an effect-aware authorization boundary. Keep
            # the selected file metadata available, but do not place secrets
            # in process state before their explicit claim/provisioning gate.
            args._env_loaded = False
            args._env_loading_deferred = True
            args._explicit_env_path = explicit_env
        else:
            args._env_loaded = load_env_file(
                context=context,
                explicit_path=explicit_env,
            )
            if explicit_env is not None and not args._env_loaded:
                raise EnvironmentFileSupportError(
                    "Explicit env file requires python-dotenv support",
                    details={
                        "path": str(explicit_env),
                        "dependency": "python-dotenv",
                    },
                )

        # Route to handler and exit with its code
        exit_code = route_command(args, context=context)
        sys.exit(exit_code)

    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        print("\n\nInterrupted by user")
        sys.exit(130)

    except ProjectError as e:
        _print_project_error(e, json_mode=bool(getattr(args, "json", False)))
        sys.exit(1)

    except Exception as e:
        # Catch-all for unexpected errors
        if bool(getattr(args, "json", False)):
            print(json.dumps({
                "success": False,
                "error": {
                    "code": "UNEXPECTED_CLI_ERROR",
                    "message": "Unexpected CLI error",
                    "details": {"type": type(e).__name__},
                },
            }, indent=2))
            sys.exit(1)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
