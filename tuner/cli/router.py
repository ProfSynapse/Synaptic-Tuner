"""
Command router.

Location: tuner/cli/router.py
Purpose: Route CLI commands to appropriate handlers
Used by: Main entry point (cli/main.py)

Routes top-level commands to their handlers:
  - train: TrainHandler (SFT, KTO, GRPO workflows)
  - cloud: CloudTrainHandler (cloud GPU training via HF Jobs, Modal, RunPod)
  - eval: EvalHandler (model evaluation)
  - synthchat: SynthChatHandler (data generation and improvement)
  - modelops: ModelOpsHandler (run, merge, convert, upload)
  - ml: MLHandler (traditional ML training - LightGBM, XGBoost, sklearn)
  - status: StatusHandler (system status overview)
  - doctor: DoctorHandler (system diagnostics with recommendations)
  - list: ListHandler (resource discovery)
  - (none): MainMenuHandler (interactive menu)

Args are passed to handlers to support global flags like --json.
"""

import json
from argparse import Namespace
from datetime import datetime


def route_command(args: Namespace) -> int:
    """
    Route command to appropriate handler.

    Maps command strings to handler classes and executes them.
    If no command is provided, shows the interactive main menu.

    Args are passed to handlers to support global flags like --json
    for AI-parseable output.

    Args:
        args: Parsed command-line arguments

    Returns:
        int: Exit code (0 = success, non-zero = error)

    Command Mapping:
        train     -> TrainHandler (SFT, KTO, GRPO training)
        eval      -> EvalHandler (model evaluation)
        synthchat -> SynthChatHandler (data generation/improvement)
        modelops  -> ModelOpsHandler (run, merge, convert, upload)
        ml        -> MLHandler (traditional ML training)
        status    -> StatusHandler (system status overview)
        doctor    -> DoctorHandler (system diagnostics)
        list      -> ListHandler (resource discovery)
        (none)    -> MainMenuHandler (interactive menu)

    Example:
        >>> args = parser.parse_args(['train'])
        >>> exit_code = route_command(args)
        >>> sys.exit(exit_code)

        >>> args = parser.parse_args(['eval', '--json'])
        >>> exit_code = route_command(args)  # JSON output mode

        >>> args = parser.parse_args(['doctor', '--fix'])
        >>> exit_code = route_command(args)  # Auto-fix mode

        >>> args = parser.parse_args(['list', 'datasets'])
        >>> exit_code = route_command(args)  # List datasets
    """
    # Check for JSON mode - affects error output
    json_mode = getattr(args, 'json', False)

    # Import handlers (deferred to avoid circular imports)
    try:
        from tuner.handlers.train_handler import TrainHandler
        from tuner.handlers.eval_handler import EvalHandler
        from tuner.handlers.cloud_pipeline_handler import CloudPipelineHandler
        from tuner.handlers.cloud_eval_handler import CloudEvalHandler
        from tuner.handlers.cloud_inspect_handler import CloudInspectHandler
        from tuner.handlers.cloud_jobs_handler import CloudJobsHandler
        from tuner.handlers.cloud_gym_handler import CloudGymHandler
        from tuner.handlers.cloud_run_handler import CloudRunHandler
        from tuner.handlers.experiment_handler import ExperimentHandler
        from tuner.handlers.experiment_analysis_handler import ExperimentAnalysisHandler
        from tuner.handlers.synthchat_handler import SynthChatHandler
        from tuner.handlers.modelops_handler import ModelOpsHandler
        from tuner.handlers.ml_handler import MLHandler
        from tuner.handlers.status_handler import StatusHandler
        from tuner.handlers.doctor_handler import DoctorHandler
        from tuner.handlers.list_handler import ListHandler
        from tuner.handlers.main_menu_handler import MainMenuHandler
        from tuner.handlers.flywheel_handler import FlywheelHandler
        from tuner.handlers.surgery_handler import SurgeryHandler
    except ImportError as e:
        # Graceful degradation if handlers not yet implemented
        error_msg = f"Handlers not yet implemented: {e}"
        if json_mode:
            output = {
                "success": False,
                "error": {
                    "message": error_msg,
                    "code": "HANDLER_IMPORT_ERROR",
                },
                "timestamp": datetime.now().isoformat()
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Error: {error_msg}")
            print("This is expected during migration. Please use tuner_legacy.py instead.")
        return 1

    # Get command from args
    command = getattr(args, 'command', None)

    # JSON mode without command is an error (interactive menu needs input)
    # Exception: status, doctor, and list commands work in JSON mode
    if json_mode and not command:
        output = {
            "success": False,
            "error": {
                "message": "JSON mode requires a command (train, cloud, cloud-run, cloud-jobs, cloud-pipeline, cloud-eval, cloud-gym, cloud-inspect, run-experiment, analyze-experiment, eval, synthchat, modelops, ml, flywheel, surgery, status, doctor, list)",
                "code": "COMMAND_REQUIRED",
            },
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(output, indent=2))
        return 1

    # Special handling for status command (uses json_output parameter)
    if command == 'status':
        handler = StatusHandler(json_output=json_mode)
        return handler.handle()

    # Special handling for doctor command (has json_output and auto_fix parameters)
    if command == 'doctor':
        doctor_fix = getattr(args, 'doctor_fix', False)
        handler = DoctorHandler(json_output=json_mode, auto_fix=doctor_fix)
        return handler.handle()

    # Special handling for list command (has subcommand and json_output)
    if command == 'list':
        list_subcommand = getattr(args, 'subcommand', None)
        handler = ListHandler(subcommand=list_subcommand, output_json=json_mode)
        return handler.handle()

    # list-runs: query unified experiment tracking registry
    if command == 'list-runs':
        return _handle_list_runs(args, json_mode)

    # Special handling for ml command (has subcommand and --config)
    if command == 'ml':
        ml_sub = getattr(args, 'subcommand', None)
        # Map the generic subcommand to ml_subcommand for the handler
        if args is not None:
            args.ml_subcommand = ml_sub
        handler = MLHandler(args=args)
        return handler.handle()

    # Special handling for flywheel command (has subcommand)
    if command == 'flywheel':
        handler = FlywheelHandler(args=args)
        return handler.handle()

    # Autonomous experiment loop
    if command == 'experiment-loop':
        return _handle_experiment_loop(args, json_mode)

    # Surgery command
    if command == 'surgery':
        handler = SurgeryHandler(args=args)
        return handler.handle()

    # Experiment pipeline
    if command == 'compare-runs':
        import subprocess
        import sys
        from pathlib import Path
        cmd = [sys.executable, str(Path("Tools/compare_runs.py"))]
        if getattr(args, "experiment_id", None):
            cmd.extend(["--experiment-id", args.experiment_id])
        if getattr(args, "base_dir", None):
            cmd.extend(["--base-dir", args.base_dir])
        return subprocess.run(cmd).returncode
        
    if command == 'create-experiment':
        from shared.experiment_tracking import create_experiment
        if not getattr(args, "name", None):
            args.name = f"experiment_{datetime.now().strftime('%Y%m%d')}"
        exp = create_experiment(
            name=getattr(args, "name", "Experiment"),
            dataset_path=getattr(args, "dataset_path", ""),
            dataset_hash=getattr(args, "dataset_hash", ""),
            base_model_name=getattr(args, "base_model_name", "unsloth/phi-4"),
            base_dir=getattr(args, "base_dir", ".tracking")
        )
        print(f"Created experiment: {exp.experiment_id}")
        return 0

    # Import cloud handler (conditional - may not have deps)
    try:
        from tuner.handlers.cloud_train_handler import CloudTrainHandler
    except ImportError:
        CloudTrainHandler = None

    # Map commands to handlers
    handlers = {
        'train': TrainHandler,
        'cloud-pipeline': CloudPipelineHandler,
        'cloud-run': CloudRunHandler,
        'cloud-jobs': CloudJobsHandler,
        'eval': EvalHandler,
        'cloud-eval': CloudEvalHandler,
        'cloud-gym': CloudGymHandler,
        'cloud-inspect': CloudInspectHandler,
        'run-experiment': ExperimentHandler,
        'analyze-experiment': ExperimentAnalysisHandler,
        'synthchat': SynthChatHandler,
        'modelops': ModelOpsHandler,
        'ml': MLHandler,
    }
    if CloudTrainHandler is not None:
        handlers['cloud'] = CloudTrainHandler

    # Execute handler with args
    if command and command in handlers:
        handler_class = handlers[command]
        handler = handler_class(args=args)
        return handler.handle()
    else:
        # No command = interactive menu
        handler = MainMenuHandler(args=args)
        return handler.handle()


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
