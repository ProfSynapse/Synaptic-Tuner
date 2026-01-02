"""
Model Operations handler.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/modelops_handler.py
Purpose: Submenu for model operations (run, merge, convert, upload)
Used by: Main menu when 'modelops' is selected

This handler provides a consolidated submenu for all model-related operations:
- Running models (chat interface)
- Merging LoRA adapters with base models
- Converting models to GGUF format
- Converting models for WebLLM (browser deployment)
- Uploading models to HuggingFace

The submenu loops until the user selects Back, allowing multiple operations
to be performed in sequence without returning to the main menu.
"""

from tuner.handlers.base import BaseHandler
from tuner.ui import (
    print_header,
    print_menu,
    print_info,
    BOX,
    RICH_AVAILABLE,
    console,
    COLORS,
)


class ModelOpsHandler(BaseHandler):
    """
    Handler for model operations submenu.

    Provides a consolidated interface for all model-related post-training
    operations. Users can perform multiple operations in sequence before
    returning to the main menu.

    Supported Operations:
        - Run Model: Interactive chat with trained models
        - Merge LoRA: Combine LoRA adapters with base model
        - Convert GGUF: Quantize to GGUF format for llama.cpp/Ollama
        - Convert WebLLM: Prepare for browser deployment
        - Upload to HF: Push models to HuggingFace Hub

    Example:
        handler = ModelOpsHandler()
        exit_code = handler.handle()
        # Shows submenu, user selects operation, dispatches to handler
        # Returns 0 when user selects Back
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "modelops"

    def can_handle_direct_mode(self) -> bool:
        """
        This handler supports direct CLI invocation.

        When invoked directly (e.g., `python tuner.py modelops`), it will
        display the model operations submenu.

        Returns:
            True - this handler can be invoked directly
        """
        return True

    def _print_return_message(self) -> None:
        """Print message when returning to main menu."""
        if RICH_AVAILABLE:
            console.print()
            console.print(f"  [{COLORS['cello']}]Returning to main menu...[/{COLORS['cello']}]")
            console.print()
        else:
            print("\n  Returning to main menu...\n")

    def handle(self) -> int:
        """
        Execute model operations submenu workflow.

        Displays a submenu of model operations and dispatches to the
        appropriate handler based on user selection. Loops until user
        selects Back.

        Returns:
            int: Exit code (0 = success, always returns 0 for graceful exit)
        """
        # Import handlers here to avoid circular imports
        from tuner.handlers.inference_handler import InferenceHandler
        from tuner.handlers.merge_handler import MergeHandler
        from tuner.handlers.convert_handler import ConvertHandler
        from tuner.handlers.webllm_handler import WebLLMHandler
        from tuner.handlers.upload_handler import UploadHandler

        # Define menu options with descriptions
        menu_options = [
            ("run", f"{BOX['star']} Run Model - Chat with your trained model"),
            ("merge", f"{BOX['bullet']} Merge LoRA - Combine adapters with base model"),
            ("gguf", f"{BOX['bullet']} Convert GGUF - Quantize to GGUF format"),
            ("webllm", f"{BOX['bullet']} Convert WebLLM - Prepare for browser"),
            ("upload", f"{BOX['bullet']} Upload to HF - Push to HuggingFace"),
        ]

        # Create handler instances (lazy initialization)
        handlers = {
            "run": InferenceHandler,
            "merge": MergeHandler,
            "gguf": ConvertHandler,
            "webllm": WebLLMHandler,
            "upload": UploadHandler,
        }

        # Main submenu loop
        while True:
            # Display header
            print_header("MODEL OPERATIONS", "Post-training model management")

            # Show menu and get selection
            # print_menu always shows Exit/Back as the last option
            choice = print_menu(menu_options, "Select operation:")

            # Handle user choice
            if not choice:
                # User selected Back or pressed Escape
                self._print_return_message()
                return 0

            # Get handler class and instantiate
            handler_class = handlers.get(choice)
            if handler_class:
                try:
                    # Instantiate and run handler
                    handler = handler_class()
                    exit_code = handler.handle()

                    # Show brief status after operation completes
                    if exit_code == 0:
                        print_info("Operation completed successfully")
                    elif exit_code == 130:
                        # User cancelled (Ctrl+C)
                        print_info("Operation cancelled")
                    # Continue to show submenu again regardless of exit code
                    # This allows user to perform multiple operations

                except KeyboardInterrupt:
                    print_info("Operation cancelled")
                    # Continue to show submenu
                except Exception as e:
                    if RICH_AVAILABLE:
                        console.print(f"  [{COLORS['orange']}]Error: {e}[/{COLORS['orange']}]")
                    else:
                        print(f"  Error: {e}")
                    # Continue to show submenu despite error
            else:
                # Should never happen, but handle gracefully
                if RICH_AVAILABLE:
                    console.print(f"  [{COLORS['orange']}]Unknown operation: {choice}[/{COLORS['orange']}]")
                else:
                    print(f"  Unknown operation: {choice}")
