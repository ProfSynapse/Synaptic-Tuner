"""
Main menu handler for interactive CLI mode.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/main_menu_handler.py
Purpose: Present main interactive menu and dispatch to appropriate handlers
Used by: Router when no direct command is specified

This handler implements the main menu workflow:
1. Detect environment and load .env file
2. Build status info (environment, repo, config)
3. Show animated logo menu on first run
4. Display menu options (train, upload, eval, pipeline)
5. Loop: show menu, dispatch to handler, repeat
6. Exit gracefully when user selects exit/back
"""

from typing import Dict

from tuner.handlers.base import BaseHandler
from tuner.utils import detect_environment, load_env_file

# Import shared UI components (delegates to Trainers/shared/ui/)
from Trainers.shared.ui import (
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

    Example:
        handler = MainMenuHandler()
        exit_code = handler.handle()
        # Shows menu, user selects option, dispatches to handler, repeats
        # Returns 0 when user exits gracefully
    """

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
        from tuner.handlers.upload_handler import UploadHandler
        from tuner.handlers.eval_handler import EvalHandler
        from tuner.handlers.pipeline_handler import PipelineHandler

        # Step 1: Detect environment and load .env
        env = detect_environment()
        env_loaded = load_env_file(self.repo_root)

        # Step 2: Build status info
        status_info = self._build_status_info(env, env_loaded)

        # Step 3: Define menu options
        menu_options = [
            ("train", f"{BOX['star']} Train a model (SFT, KTO, or MLX)"),
            ("upload", f"{BOX['bullet']} Upload model to HuggingFace"),
            ("eval", f"{BOX['bullet']} Evaluate a model"),
            ("pipeline", f"{BOX['bullet']} Full pipeline (Train -> Upload -> Eval)"),
        ]

        # Step 4: Create handler instances
        handlers = {
            "train": TrainHandler(),
            "upload": UploadHandler(),
            "eval": EvalHandler(),
            "pipeline": PipelineHandler(),
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
