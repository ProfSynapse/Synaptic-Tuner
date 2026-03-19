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
        from tuner.handlers.cloud_gym_handler import CloudGymHandler
        from tuner.handlers.cloud_run_handler import CloudRunHandler
        from tuner.handlers.synthchat_handler import SynthChatHandler
        from tuner.handlers.modelops_handler import ModelOpsHandler
        from tuner.handlers.ml_handler import MLHandler
        from tuner.handlers.status_handler import StatusHandler
        from tuner.handlers.doctor_handler import DoctorHandler
        from tuner.handlers.list_handler import ListHandler
        from tuner.handlers.main_menu_handler import MainMenuHandler
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
                "message": "JSON mode requires a command (train, cloud, cloud-run, cloud-pipeline, cloud-eval, cloud-gym, cloud-inspect, eval, synthchat, modelops, ml, status, doctor, list)",
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

    # Special handling for ml command (has subcommand and --config)
    if command == 'ml':
        ml_sub = getattr(args, 'subcommand', None)
        # Map the generic subcommand to ml_subcommand for the handler
        if args is not None:
            args.ml_subcommand = ml_sub
        handler = MLHandler(args=args)
        return handler.handle()

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
        'eval': EvalHandler,
        'cloud-eval': CloudEvalHandler,
        'cloud-gym': CloudGymHandler,
        'cloud-inspect': CloudInspectHandler,
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
