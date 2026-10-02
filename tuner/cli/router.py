"""
Command router.

Location: tuner/cli/router.py
Purpose: Route CLI commands to appropriate handlers
Used by: Main entry point (cli/main.py)

Every command the parser accepts has an entry in ``COMMAND_ROUTES``. A route
names its handler as ``"module:attribute"`` and a runner that knows how to
construct and invoke it. Handler modules are imported only when their command
runs, so light commands (status, doctor, list, project, capabilities, ...) never
pull in torch or provider stacks, and a missing optional dependency is reported
for the one command that needs it instead of breaking unrelated commands.

Routing rules:
  - A command with a route runs its handler.
  - A command the parser accepts but that has no route is refused with a
    ``COMMAND_NOT_ROUTED`` error and a nonzero exit code. It never falls
    through to the interactive menu.
  - Only an invocation with no command at all opens the interactive menu
    (MainMenuHandler); in --json mode that is an error because the menu needs
    input.

Args are passed to handlers to support global flags like --json.
"""

import importlib
import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple

from tuner.project import ProjectContext


def _default_context() -> ProjectContext:
    return ProjectContext.standalone(engine_root=Path(__file__).parents[2])


def _bind_context(handler, context: ProjectContext):
    binder = getattr(handler, "bind_context", None)
    return binder(context) if callable(binder) else handler


def _emit_error(json_mode: bool, message: str, code: str) -> None:
    if json_mode:
        output = {
            "success": False,
            "error": {"message": message, "code": code},
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Error: {message}")


# ---------------------------------------------------------------------------
# Runners: construct and invoke a resolved handler (class or function).
# Signature: (target, args, context, json_mode) -> exit code
# ---------------------------------------------------------------------------

def _run_with_context(handler_cls, args, context, json_mode) -> int:
    return handler_cls(args=args, context=context).handle()


def _run_bound(handler_cls, args, context, json_mode) -> int:
    return _bind_context(handler_cls(args=args), context).handle()


def _run_status(handler_cls, args, context, json_mode) -> int:
    return handler_cls(json_output=json_mode, context=context).handle()


def _run_doctor(handler_cls, args, context, json_mode) -> int:
    subcommand = getattr(args, "subcommand", None)
    if subcommand is not None:
        # `doctor <subcommand>` runs its own lazily imported handler. An unknown
        # subcommand is refused instead of silently running system diagnostics.
        route = DOCTOR_SUBCOMMAND_ROUTES.get(subcommand)
        if route is None:
            _emit_error(
                json_mode,
                f"Unknown doctor subcommand {subcommand!r}. Available: "
                + ", ".join(DOCTOR_SUBCOMMAND_ROUTES)
                + " (or none for system diagnostics).",
                "UNKNOWN_SUBCOMMAND",
            )
            return 2
        return _run_target(route.target, route.run, args, context, json_mode)
    return handler_cls(
        json_output=json_mode,
        auto_fix=getattr(args, "doctor_fix", False),
        context=context,
    ).handle()


def _run_list(handler_cls, args, context, json_mode) -> int:
    handler = _bind_context(
        handler_cls(subcommand=getattr(args, "subcommand", None), output_json=json_mode),
        context,
    )
    return handler.handle()


def _run_ml(handler_cls, args, context, json_mode) -> int:
    # The shared positional subcommand is the ML subcommand for this handler.
    args.ml_subcommand = getattr(args, "subcommand", None)
    return _bind_context(handler_cls(args=args), context).handle()


def _run_function(func, args, context, json_mode) -> int:
    return func(args, json_mode)


def _run_tool_script(func, args, context, json_mode) -> int:
    return func(args, context)


class Route(NamedTuple):
    """How one CLI command reaches its handler."""

    target: str  # "module:attribute", imported lazily when the command runs
    run: Callable[[Any, Namespace, ProjectContext, bool], int]


def _handle_compare_runs(args: Namespace, context: ProjectContext) -> int:
    """Run Tools/compare_runs.py from the engine checkout."""
    script = context.engine_root / "Tools" / "compare_runs.py"
    cmd = [sys.executable, str(script)]
    if getattr(args, "experiment_id", None):
        cmd.extend(["--experiment-id", args.experiment_id])
    if getattr(args, "base_dir", None):
        cmd.extend(["--base-dir", args.base_dir])
    return subprocess.run(cmd).returncode


def _handle_create_experiment(args: Namespace, json_mode: bool) -> int:
    """Create an experiment record in the tracking directory."""
    from shared.experiment_tracking import create_experiment

    if not getattr(args, "name", None):
        args.name = f"experiment_{datetime.now().strftime('%Y%m%d')}"
    exp = create_experiment(
        name=getattr(args, "name", "Experiment"),
        dataset_path=getattr(args, "dataset_path", ""),
        dataset_hash=getattr(args, "dataset_hash", ""),
        base_model_name=getattr(args, "base_model_name", "unsloth/phi-4"),
        base_dir=getattr(args, "base_dir", ".tracking"),
    )
    print(f"Created experiment: {exp.experiment_id}")
    return 0


_H = "tuner.handlers."
_R = "tuner.cli.router:"

# Every command in the parser's choices must appear here (enforced by
# tests/cli/test_router_coverage.py).
COMMAND_ROUTES: dict[str, Route] = {
    # Capability discovery and protected/provider commands. These stay free of
    # ML and provider imports.
    "capabilities": Route(_H + "capabilities_handler:CapabilitiesHandler", _run_with_context),
    "hf-source": Route(_H + "hf_source_handler:HFSourceHandler", _run_with_context),
    "hf-smoke": Route(_H + "hf_smoke_handler:HFSmokeHandler", _run_with_context),
    "hf-training-smoke": Route(
        _H + "hf_training_smoke_handler:HFTrainingSmokeHandler", _run_with_context
    ),
    "modal-runtime-release": Route(
        _H + "modal_runtime_release_handler:ModalRuntimeReleaseHandler", _run_with_context
    ),
    "ingest": Route(_H + "ingestion_handler:IngestionHandler", _run_with_context),
    "prepare-dataset": Route(_H + "dataset_prepare_handler:DatasetPrepareHandler", _run_with_context),
    "check-contamination": Route(
        _H + "contamination_handler:ContaminationHandler", _run_with_context
    ),
    "batch-generate": Route(_H + "batch_generate_handler:BatchGenerateHandler", _run_bound),
    "batch-capture": Route(_H + "batch_capture_handler:BatchCaptureHandler", _run_bound),
    "local-run": Route(_H + "local_run_handler:LocalRunHandler", _run_bound),
    # Project introspection: cheap and side-effect free.
    "project": Route(_H + "project_handler:ProjectHandler", _run_with_context),
    "status": Route(_H + "status_handler:StatusHandler", _run_status),
    "doctor": Route(_H + "doctor_handler:DoctorHandler", _run_doctor),
    "list": Route(_H + "list_handler:ListHandler", _run_list),
    "list-runs": Route(_R + "_handle_list_runs", _run_function),
    # Training, cloud and evaluation.
    "train": Route(_H + "train_handler:TrainHandler", _run_bound),
    "cloud": Route(_H + "cloud_train_handler:CloudTrainHandler", _run_bound),
    "cloud-run": Route(_H + "cloud_run_handler:CloudRunHandler", _run_bound),
    "cloud-jobs": Route(_H + "cloud_jobs_handler:CloudJobsHandler", _run_bound),
    "plan-hardware": Route(_H + "hardware_plan_handler:HardwarePlanHandler", _run_bound),
    "cloud-pipeline": Route(_H + "cloud_pipeline_handler:CloudPipelineHandler", _run_bound),
    "cloud-eval": Route(_H + "cloud_eval_handler:CloudEvalHandler", _run_bound),
    "cloud-gym": Route(_H + "cloud_gym_handler:CloudGymHandler", _run_bound),
    "cloud-inspect": Route(_H + "cloud_inspect_handler:CloudInspectHandler", _run_bound),
    "cloud-extract": Route(_H + "cloud_extract_handler:CloudExtractHandler", _run_bound),
    "bucket": Route(_H + "bucket_handler:BucketHandler", _run_bound),
    "eval": Route(_H + "eval_handler:EvalHandler", _run_bound),
    # Experiments.
    "run-experiment": Route(_H + "experiment_handler:ExperimentHandler", _run_bound),
    "analyze-experiment": Route(
        _H + "experiment_analysis_handler:ExperimentAnalysisHandler", _run_bound
    ),
    "experiment-loop": Route(_R + "_handle_experiment_loop", _run_function),
    "compare-runs": Route(_R + "_handle_compare_runs", _run_tool_script),
    "create-experiment": Route(_R + "_handle_create_experiment", _run_function),
    # Data, models and research tooling.
    "synthchat": Route(_H + "synthchat_handler:SynthChatHandler", _run_bound),
    "modelops": Route(_H + "modelops_handler:ModelOpsHandler", _run_bound),
    "ml": Route(_H + "ml_handler:MLHandler", _run_ml),
    "mechinterp": Route(_H + "mechinterp_handler:MechInterpHandler", _run_bound),
    "flywheel": Route(_H + "flywheel_handler:FlywheelHandler", _run_bound),
    "prompt-optimize": Route(_H + "prompt_optimize_handler:PromptOptimizeHandler", _run_bound),
    "surgery": Route(_H + "surgery_handler:SurgeryHandler", _run_bound),
}

# `doctor <subcommand>` routes (dispatched by _run_doctor).
DOCTOR_SUBCOMMAND_ROUTES: dict[str, Route] = {
    "sft-mask": Route(_H + "sft_mask_doctor_handler:SFTMaskDoctorHandler", _run_with_context),
}

# `train --job-config` is a managed-provider job, not the interactive trainer.
TRAIN_JOB_CONFIG_ROUTE = Route(
    _H + "modal_job_config_handler:ModalJobConfigHandler", _run_with_context
)

MAIN_MENU_TARGET = _H + "main_menu_handler:MainMenuHandler"


def iter_route_targets():
    """Yield (label, target) for every route, including the train variant and the menu."""
    for command, route in COMMAND_ROUTES.items():
        yield command, route.target
    for subcommand, route in DOCTOR_SUBCOMMAND_ROUTES.items():
        yield f"doctor {subcommand}", route.target
    yield "train --job-config", TRAIN_JOB_CONFIG_ROUTE.target
    yield "(no command)", MAIN_MENU_TARGET


def _resolve_target(target: str):
    """Import ``module:attribute`` at call time (so patched attributes are seen)."""
    module_name, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


def _run_target(target: str, run, args, context, json_mode: bool) -> int:
    try:
        resolved = _resolve_target(target)
    except ImportError as exc:
        _emit_error(
            json_mode,
            f"Handler import failed for {target}: {exc}",
            "HANDLER_IMPORT_ERROR",
        )
        return 1
    return run(resolved, args, context, json_mode)


def route_command(args: Namespace, context: ProjectContext | None = None) -> int:
    """
    Route a parsed command to its handler.

    Args:
        args: Parsed command-line arguments
        context: Project context (defaults to the standalone engine checkout)

    Returns:
        int: Exit code (0 = success, non-zero = error). A command that the
        parser accepts but that has no route returns 2 without opening the
        interactive menu. Only an invocation with no command opens the menu.

    Example:
        >>> exit_code = route_command(parser.parse_args(['status', '--json']))
        >>> exit_code = route_command(parser.parse_args(['doctor', '--fix']))
        >>> exit_code = route_command(parser.parse_args(['list', 'datasets']))
    """
    context = context or _default_context()

    json_mode = getattr(args, 'json', False)
    command = getattr(args, 'command', None)

    if not command:
        # JSON mode without a command is an error (the interactive menu needs input).
        if json_mode:
            _emit_error(
                True,
                "JSON mode requires a command (" + ", ".join(COMMAND_ROUTES) + ")",
                "COMMAND_REQUIRED",
            )
            return 1
        return _run_target(MAIN_MENU_TARGET, _run_with_context, args, context, json_mode)

    route = COMMAND_ROUTES.get(command)
    if route is None:
        _emit_error(
            json_mode,
            f"Command '{command}' is accepted by the parser but has no handler. "
            "This is a CLI bug; the command cannot be run.",
            "COMMAND_NOT_ROUTED",
        )
        return 2

    if command == "train" and getattr(args, "job_config", None):
        route = TRAIN_JOB_CONFIG_ROUTE

    return _run_target(route.target, route.run, args, context, json_mode)


def _handle_experiment_loop(args: Namespace, json_mode: bool) -> int:
    """Run the autonomous experiment loop."""
    try:
        from shared.flywheel.experiment_config import load_experiment_config
        from shared.flywheel.experiment_loop import ExperimentLoop
    except ImportError as exc:
        if json_mode:
            print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        else:
            print(f"Error: experiment loop module unavailable: {exc}")
        return 1

    config_path = getattr(args, "experiment_loop_config", None)
    config = load_experiment_config(config_path)

    # Apply CLI overrides
    max_exp = getattr(args, "max_experiments", None)
    if max_exp is not None:
        config.max_experiments = max_exp

    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path:
        config.dataset_path = dataset_path

    issues = config.validate()
    if issues:
        msg = "Config validation failed:\n  " + "\n  ".join(issues)
        if json_mode:
            print(json.dumps({"success": False, "error": msg}, indent=2))
        else:
            print(f"Error: {msg}")
        return 1

    if not json_mode:
        print(f"Starting experiment loop: {config.max_experiments} experiments")
        print(f"  Strategy: {config.search_strategy}")
        print(f"  Trainer: {config.trainer_type}")
        print(f"  Output: {config.output_dir}")

    try:
        loop = ExperimentLoop(config)
        results = loop.run()
    except Exception as exc:
        if json_mode:
            print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        else:
            print(f"Error during experiment loop: {exc}")
            import traceback
            traceback.print_exc()
        return 1

    completed = [r for r in results if r.status == "completed"]
    if json_mode:
        from dataclasses import asdict
        output = {
            "success": True,
            "total_experiments": len(results),
            "completed": len(completed),
            "best_score": loop.best_score,
            "best_config": loop.best_config,
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nExperiment loop complete.")
        print(f"  Total: {len(results)}, Completed: {len(completed)}")
        print(f"  Best score: {loop.best_score:.4f}")
        if loop.best_config:
            print(f"  Best config: {loop.best_config}")

    return 0


def _handle_list_runs(args: Namespace, json_mode: bool) -> int:
    """Query the unified experiment tracking registry and display results.

    Supports --json for structured output, and --run-type / --since / --until-date
    for filtering.
    """
    try:
        from shared.experiment_tracking import RunFilter, RunRegistry
    except ImportError as exc:
        if json_mode:
            print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        else:
            print(f"Error: experiment tracking module unavailable: {exc}")
        return 1

    registry = RunRegistry()

    # Build filter from CLI args
    run_type = getattr(args, "run_type", None)
    since = getattr(args, "since", None)
    until_date = getattr(args, "until_date", None)

    run_filter = None
    if run_type or since or until_date:
        run_filter = RunFilter(
            run_type=run_type,
            since=since,
            until=until_date,
        )

    records = registry.find_runs(run_filter)

    if json_mode:
        from dataclasses import asdict
        output = {
            "success": True,
            "runs": [asdict(r) for r in records],
            "count": len(records),
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(output, indent=2))
        return 0

    if not records:
        print("No tracked runs found in registry.")
        print("Runs are registered automatically after SFT, KTO, ML training, and evaluations.")
        return 0

    # Table display
    print(f"\nTracked Runs ({len(records)} total):")
    print("-" * 100)
    print(f"{'Type':<12} {'Name':<35} {'Status':<10} {'Model':<25} {'Metric':<15}")
    print("-" * 100)

    for record in records:
        metric_str = ""
        if record.primary_metric is not None:
            try:
                metric_str = f"{record.primary_metric_name}={float(record.primary_metric):.4f}"
            except (TypeError, ValueError):
                metric_str = f"{record.primary_metric_name}={record.primary_metric}"

        name_display = record.name[:33] + ".." if len(record.name) > 35 else record.name
        model_display = (record.model_name or "")[:23]
        if len(record.model_name or "") > 25:
            model_display += ".."

        print(f"{record.run_type:<12} {name_display:<35} {record.status:<10} {model_display:<25} {metric_str:<15}")

    print("-" * 100)
    return 0
