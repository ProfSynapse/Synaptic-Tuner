"""
Command router.

Location: tuner/cli/router.py
Purpose: Route CLI commands to appropriate handlers
Used by: Main entry point (cli/main.py)

Routes top-level commands to their handlers:
  - train: TrainHandler (SFT, KTO, GRPO workflows)
  - eval: EvalHandler (model evaluation)
  - synthchat: SynthChatHandler (data generation and improvement)
  - modelops: ModelOpsHandler (run, merge, convert, upload)
  - (none): MainMenuHandler (interactive menu)
"""

from argparse import Namespace


def route_command(args: Namespace) -> int:
    """
    Route command to appropriate handler.

    Maps command strings to handler classes and executes them.
    If no command is provided, shows the interactive main menu.

    Args:
        args: Parsed command-line arguments

    Returns:
        int: Exit code (0 = success, non-zero = error)

    Command Mapping:
        train     -> TrainHandler (SFT, KTO, GRPO training)
        eval      -> EvalHandler (model evaluation)
        synthchat -> SynthChatHandler (data generation/improvement)
        modelops  -> ModelOpsHandler (run, merge, convert, upload)
        (none)    -> MainMenuHandler (interactive menu)

    Example:
        >>> args = parser.parse_args(['train'])
        >>> exit_code = route_command(args)
        >>> sys.exit(exit_code)
    """
    # Import handlers (deferred to avoid circular imports)
    try:
        from tuner.handlers.train_handler import TrainHandler
        from tuner.handlers.eval_handler import EvalHandler
        from tuner.handlers.synthchat_handler import SynthChatHandler
        from tuner.handlers.modelops_handler import ModelOpsHandler
        from tuner.handlers.main_menu_handler import MainMenuHandler
    except ImportError as e:
        # Graceful degradation if handlers not yet implemented
        print(f"Error: Handlers not yet implemented: {e}")
        print("This is expected during migration. Please use tuner_legacy.py instead.")
        return 1

    # Get command from args
    command = getattr(args, 'command', None)

    # Map commands to handlers
    handlers = {
        'train': TrainHandler,
        'eval': EvalHandler,
        'synthchat': SynthChatHandler,
        'modelops': ModelOpsHandler,
    }

    # Execute handler
    if command and command in handlers:
        handler_class = handlers[command]
        handler = handler_class()
        return handler.handle()
    else:
        # No command = interactive menu
        handler = MainMenuHandler()
        return handler.handle()
