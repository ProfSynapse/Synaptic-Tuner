"""
SynthChat handler - unified data generation and improvement.

Location: tuner/handlers/synthchat_handler.py
Purpose: Submenu for synthetic data operations (generate + improve)
Used by: Main menu when 'synthchat' is selected

This handler provides a unified entry point for SynthChat operations:
1. Generate New - Create examples from scenarios using LLM
2. Improve Existing - Refine datasets with rubrics

It loads defaults from SynthChat/config/settings.yaml and allows users
to either use defaults or customize settings before proceeding.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

from tuner.handlers.base import BaseHandler
from shared.ui import (
    print_menu,
    print_header,
    print_config,
    print_info,
    print_error,
    confirm,
    BOX,
    COLORS,
    console,
    RICH_AVAILABLE,
)


class SynthChatHandler(BaseHandler):
    """
    Handler for SynthChat submenu (generate + improve).

    Provides a unified interface for synthetic data operations with
    sensible defaults loaded from configuration files.

    Example:
        handler = SynthChatHandler()
        exit_code = handler.handle()
        # Shows submenu, user selects generate or improve
        # Returns 0 on success
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "synthchat"

    def can_handle_direct_mode(self) -> bool:
        """This handler supports direct CLI invocation."""
        return True

    def _load_defaults(self) -> Dict[str, Any]:
        """
        Load defaults from SynthChat/config/settings.yaml.

        Returns:
            Dictionary with configuration defaults, or empty dict if not found
        """
        settings_path = self.repo_root / "SynthChat" / "config" / "settings.yaml"
        if settings_path.exists():
            try:
                import yaml
                with open(settings_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print_error(f"Error loading settings: {e}")
                return {}
        return {}

    def _show_generate_defaults(self, config: Dict[str, Any]) -> None:
        """
        Display generate defaults in a config panel.

        Args:
            config: Full configuration dictionary
        """
        # Extract generation settings
        llm_config = config.get("llm", {}).get("generation", {})
        gen_config = config.get("generation", {})
        defaults = config.get("defaults", {})
        output_config = config.get("output", {})

        # Build display config
        display = {
            "Provider": llm_config.get("provider", "lmstudio"),
            "Model": llm_config.get("model", "local-model"),
            "Temperature": str(llm_config.get("temperature", 0.7)),
            "Max Tokens": str(llm_config.get("max_tokens", 4096)),
            "Stage Validation": "Yes" if gen_config.get("stage_validation", True) else "No",
            "Output Dir": output_config.get("default_dir", "Datasets/synthchat/"),
        }

        # Count default targets
        targets = defaults.get("targets", {})
        if targets:
            total_examples = sum(targets.values())
            display["Targets"] = f"{len(targets)} scenarios ({total_examples} examples)"

        print_config(display, "Generate Configuration")

    def _show_improve_defaults(self, config: Dict[str, Any]) -> None:
        """
        Display improve defaults in a config panel.

        Args:
            config: Full configuration dictionary
        """
        # Extract improvement settings
        llm_config = config.get("llm", {}).get("improvement", {})
        improve_config = config.get("improvement", {})
        logging_config = config.get("logging", {})

        # Build display config
        display = {
            "Provider": llm_config.get("provider", "openrouter"),
            "Model": llm_config.get("model", "openai/gpt-4o"),
            "Temperature": str(llm_config.get("temperature", 0.1)),
            "Max Tokens": str(llm_config.get("max_tokens", 2048)),
            "Max Iterations": str(improve_config.get("max_iterations", 3)),
            "On Max Iterations": improve_config.get("on_max_iterations", "skip"),
            "Save Interactions": "Yes" if logging_config.get("save_interactions", True) else "No",
        }

        # Show default rubrics
        rubrics = improve_config.get("default_rubrics", [])
        if rubrics:
            display["Default Rubrics"] = ", ".join(rubrics[:3])
            if len(rubrics) > 3:
                display["Default Rubrics"] += f" (+{len(rubrics) - 3} more)"

        print_config(display, "Improve Configuration")

    def _show_action_menu(self, mode: str, config: Dict[str, Any]) -> Optional[str]:
        """
        Show action menu (Use defaults / Customize / Back).

        Args:
            mode: Either 'generate' or 'improve'
            config: Configuration dictionary

        Returns:
            Selected action or None if back
        """
        # Show appropriate defaults panel
        if mode == "generate":
            self._show_generate_defaults(config)
        else:
            self._show_improve_defaults(config)

        print()

        # Action menu
        choice = print_menu([
            ("defaults", f"{BOX['star']} Use defaults - Start with settings above"),
            ("customize", f"{BOX['bullet']} Customize - Modify settings before running"),
        ], "How would you like to proceed?")

        return choice

    def _handle_generate(self, config: Dict[str, Any]) -> int:
        """
        Handle generate flow with defaults.

        Args:
            config: Configuration dictionary

        Returns:
            Exit code (0 = success)
        """
        print_header("SYNTHCHAT GENERATE", "Create new training examples")

        action = self._show_action_menu("generate", config)

        if not action:
            return 0

        if action == "defaults":
            # Delegate to existing GenerateHandler with defaults
            print_info("Starting generation with default configuration...")
            print()

            # Import here to avoid circular imports
            from tuner.handlers.generate_handler import GenerateHandler
            handler = GenerateHandler()
            return handler.handle()

        elif action == "customize":
            # Placeholder for customization flow
            print()
            print_info("Customization options:")
            print()

            if RICH_AVAILABLE:
                console.print(f"  [{COLORS['cello']}]Coming soon:[/{COLORS['cello']}]")
                console.print(f"    {BOX['bullet']} Provider selection (lmstudio, ollama, openrouter)")
                console.print(f"    {BOX['bullet']} Model selection from available models")
                console.print(f"    {BOX['bullet']} Temperature and token limit adjustment")
                console.print(f"    {BOX['bullet']} Scenario checkboxes for targeted generation")
                console.print(f"    {BOX['bullet']} Output path customization")
            else:
                print("  Coming soon:")
                print("    - Provider selection (lmstudio, ollama, openrouter)")
                print("    - Model selection from available models")
                print("    - Temperature and token limit adjustment")
                print("    - Scenario checkboxes for targeted generation")
                print("    - Output path customization")

            print()

            # For now, fall back to the existing generate handler which has its own prompts
            if confirm("Continue with interactive configuration?"):
                from tuner.handlers.generate_handler import GenerateHandler
                handler = GenerateHandler()
                return handler.handle()

            return 0

        return 0

    def _handle_improve(self, config: Dict[str, Any]) -> int:
        """
        Handle improve flow with defaults.

        Args:
            config: Configuration dictionary

        Returns:
            Exit code (0 = success)
        """
        print_header("SYNTHCHAT IMPROVE", "Refine datasets with rubrics")

        action = self._show_action_menu("improve", config)

        if not action:
            return 0

        if action == "defaults":
            # Delegate to existing handle_improve
            print_info("Starting improvement with default configuration...")
            print()

            # Import here to avoid circular imports
            from tuner.handlers.improve_handler import handle_improve
            handle_improve()
            return 0

        elif action == "customize":
            # Placeholder for customization flow
            print()
            print_info("Customization options:")
            print()

            if RICH_AVAILABLE:
                console.print(f"  [{COLORS['cello']}]Coming soon:[/{COLORS['cello']}]")
                console.print(f"    {BOX['bullet']} Provider selection (openrouter, lmstudio, ollama)")
                console.print(f"    {BOX['bullet']} Model selection from available models")
                console.print(f"    {BOX['bullet']} Rubric checkboxes for multi-select")
                console.print(f"    {BOX['bullet']} Max iterations adjustment")
                console.print(f"    {BOX['bullet']} Line range selection with preview")
            else:
                print("  Coming soon:")
                print("    - Provider selection (openrouter, lmstudio, ollama)")
                print("    - Model selection from available models")
                print("    - Rubric checkboxes for multi-select")
                print("    - Max iterations adjustment")
                print("    - Line range selection with preview")

            print()

            # For now, fall back to the existing improve handler which has its own prompts
            if confirm("Continue with interactive configuration?"):
                from tuner.handlers.improve_handler import handle_improve
                handle_improve()
                return 0

            return 0

        return 0

    def handle(self) -> int:
        """
        Main submenu loop.

        Returns:
            Exit code (0 = success)
        """
        # Load configuration once
        config = self._load_defaults()

        while True:
            print_header("SYNTHCHAT", "Data generation and improvement")

            # Build menu with descriptions
            menu_options = [
                ("generate", f"{BOX['star']} Generate New - Create examples from scenarios"),
                ("improve", f"{BOX['bullet']} Improve Existing - Refine datasets with rubrics"),
            ]

            choice = print_menu(menu_options, "What would you like to do?")

            if not choice:
                # User selected back/exit
                return 0

            if choice == "generate":
                result = self._handle_generate(config)
                # Continue loop after generate completes
                if result != 0:
                    print_info("Generation finished with warnings or errors")
                print()

            elif choice == "improve":
                result = self._handle_improve(config)
                # Continue loop after improve completes
                if result != 0:
                    print_info("Improvement finished with warnings or errors")
                print()

        return 0
