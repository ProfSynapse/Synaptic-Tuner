"""
Main menu handler for interactive CLI mode.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/main_menu_handler.py
Purpose: Present main interactive menu and dispatch to appropriate handlers
Used by: Router when no direct command is specified

This handler implements the main menu workflow:
1. Detect environment and load .env file
2. Build status info (environment, repo, config)
3. Show animated logo menu on first run
4. Display menu options (train, eval, synthchat, modelops)
5. Loop: show menu, dispatch to handler, repeat
6. Exit gracefully when user selects exit/back

Note: This handler does NOT support JSON mode - interactive menus require
user input. If --json flag is passed with no command, the router will
return an error.
"""

from argparse import Namespace
from typing import Dict, Optional

from tuner.handlers.base import BaseHandler
from tuner.utils import detect_environment, load_env_file

# Import shared UI components (delegates to Trainers/shared/ui/)
from shared.ui import (
    print_menu,
    animated_menu,
    console,
    RICH_AVAILABLE,
    COLORS,
    BOX,
)


class MainMenuHandler(BaseHandler):
    """
    Handler for interactive main menu.

    Presents a looping menu of options and dispatches to the appropriate
    handler based on user selection. Shows an animated logo on first run,
    then a static menu for subsequent iterations.

    This handler does NOT support direct CLI invocation - it's only used
    in interactive mode when no command is specified.

    Note: This handler does NOT support JSON mode - interactive menus
    require user input. The router handles this by returning an error
    if --json is passed without a command.

    Example:
        handler = MainMenuHandler()
        exit_code = handler.handle()
        # Shows menu, user selects option, dispatches to handler, repeats
        # Returns 0 when user exits gracefully
    """

    def __init__(self, args: Optional[Namespace] = None):
        """Initialize handler with optional args."""
        super().__init__(args=args)

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "main"

    def can_handle_direct_mode(self) -> bool:
        """This handler does NOT support direct CLI invocation."""
        return False

    def _build_status_info(self, env: str, env_loaded: bool) -> Dict[str, str]:
        """
        Build status information for display in menu.

        Args:
            env: Environment name (wsl, linux, windows, darwin)
            env_loaded: Whether .env file was loaded successfully

        Returns:
            Dictionary of status key-value pairs
        """
        status_info = {
            "Environment": env.upper(),
            "Repo": str(self.repo_root),
        }

        if env_loaded:
            status_info["Config"] = ".env loaded"

        return status_info

    def _print_goodbye(self) -> None:
        """Print goodbye message."""
        if RICH_AVAILABLE:
            console.print()
            console.print(f"  [{COLORS['purple']}]Thanks for using Synaptic Tuner![/{COLORS['purple']}]")
            console.print(f"  [{COLORS['cello']}]Goodbye![/{COLORS['cello']}]")
            console.print()
        else:
            print("\n  Thanks for using Synaptic Tuner!")
            print("  Goodbye!\n")

    def handle(self) -> int:
        """
        Execute main menu workflow.

        Returns:
            Exit code (0 = success, always returns 0 for graceful exit)
        """
        # Import handlers here to avoid circular imports
        from tuner.handlers.train_handler import TrainHandler
        from tuner.handlers.eval_handler import EvalHandler
        from tuner.handlers.synthchat_handler import SynthChatHandler
        from tuner.handlers.modelops_handler import ModelOpsHandler
        from tuner.handlers.ml_handler import MLHandler

        # Step 1: Detect environment and load .env
        env = detect_environment()
        env_loaded = load_env_file(self.repo_root / ".env")

        # Step 2: Build status info
        status_info = self._build_status_info(env, env_loaded)

        # Step 3: Define menu options (4 top-level categories)
        menu_options = [
            ("train", f"{BOX['star']} Training - Train models (SFT, KTO, GRPO)"),
            ("eval", f"{BOX['bullet']} Evaluation - Run benchmarks against a model"),
            ("synthchat", f"{BOX['bullet']} SynthChat - Generate + improve training data"),
            ("modelops", f"{BOX['bullet']} Model Ops - Run, merge, convert, upload"),
            ("ml", f"{BOX['bullet']} ML Training - Traditional ML (LightGBM, XGBoost, sklearn)"),
        ]

        # Step 4: Create handler instances (pass args for consistency)
        handlers = {
            "train": TrainHandler(args=self.args),
            "eval": EvalHandler(args=self.args),
            "synthchat": SynthChatHandler(args=self.args),
            "modelops": ModelOpsHandler(args=self.args),
            "ml": MLHandler(args=self.args),
        }

        # Step 5: Main menu loop
        first_run = True

        while True:
            if first_run:
                # Animated menu with bubbling test tube
                choice = animated_menu(
                    menu_options,
                    "What would you like to do?",
                    status_info
                )
                first_run = False
            else:
                # Static menu after first choice
                print()
                choice = print_menu(menu_options, "What would you like to do?")

            # Step 6: Handle user choice
            if not choice:
                # User selected exit/back
                self._print_goodbye()
                return 0

            # Dispatch to appropriate handler
            handler = handlers.get(choice)
            if handler:
                exit_code = handler.handle()
                # Continue to next iteration regardless of exit code
                # This allows user to try again after errors
            else:
                # Should never happen, but handle gracefully
                if RICH_AVAILABLE:
                    console.print(f"  [{COLORS['orange']}]Unknown option: {choice}[/{COLORS['orange']}]")
                else:
                    print(f"  Unknown option: {choice}")
