"""
Evaluation handler for testing model performance.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/eval_handler.py
Purpose: Orchestrate evaluation workflow - select backend, model, and prompt set
Used by: Router when handling 'eval' command or main menu selection

This handler implements the evaluation workflow:
1. Select evaluation backend (Ollama or LM Studio)
2. List available models from backend
3. Select model to evaluate
4. List available prompt sets
5. Select prompt set
6. Display configuration
7. Execute evaluation via Evaluator.cli
"""

import subprocess
from pathlib import Path
from typing import List, Tuple

from tuner.handlers.base import BaseHandler
from tuner.backends.registry import EvaluationBackendRegistry

# Import shared UI components (delegates to Trainers/shared/ui/)
from Trainers.shared.ui import (
    print_header,
    print_menu,
    print_config,
    print_info,
    print_error,
    confirm,
    prompt,
    console,
    RICH_AVAILABLE,
    COLORS,
    BOX,
)


class EvalHandler(BaseHandler):
    """
    Handler for evaluation workflow.

    Coordinates backend selection, model discovery, prompt set selection,
    and execution of the Evaluator CLI to test model performance.

    Example:
        handler = EvalHandler()
        exit_code = handler.handle()
        # User interacts with menus, selects backend/model/prompts
        # Returns 0 on success, non-zero on failure
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "eval"

    def can_handle_direct_mode(self) -> bool:
        """This handler supports direct CLI invocation."""
        return True

    def _list_prompt_sets(self) -> List[Tuple[str, str, int]]:
        """
        List available prompt sets with their prompt counts.

        Returns:
            List of (name, description, count) tuples
        """
        from tuner.discovery import PromptSetDiscovery
        discovery = PromptSetDiscovery(repo_root=self.repo_root)
        return discovery.discover()

    def _display_models_table(self, backend: str, models: List[str]) -> None:
        """
        Display available models in a table.

        Args:
            backend: Backend name for table title
            models: List of model names
        """
        if RICH_AVAILABLE:
            from rich.table import Table
            from rich import box as rich_box

            table = Table(
                title=f"Available {backend.title()} Models",
                box=rich_box.ROUNDED,
                border_style=COLORS["cello"],
            )
            table.add_column("#", style=COLORS["orange"], width=4, justify="center")
            table.add_column("Model", style="white")

            for i, m in enumerate(models, 1):
                table.add_row(str(i), m)

            console.print()
            console.print(table)
            console.print()
        else:
            print(f"\nAvailable {backend} models:")
            for i, m in enumerate(models, 1):
                print(f"  [{i}] {m}")
            print()

    def _display_prompt_sets_table(self, prompt_sets: List[Tuple[str, str, int]]) -> None:
        """
        Display available prompt sets in a table.

        Args:
            prompt_sets: List of (name, description, count) tuples
        """
        if RICH_AVAILABLE:
            from rich.table import Table
            from rich import box as rich_box

            table = Table(
                title="Available Prompt Sets",
                box=rich_box.ROUNDED,
                border_style=COLORS["cello"],
            )
            table.add_column("#", style=COLORS["orange"], width=4, justify="center")
            table.add_column("Name", style="white")
            table.add_column("Description", style="dim")
            table.add_column("Tests", style=COLORS["aqua"], justify="right")

            for i, (name, desc, count) in enumerate(prompt_sets, 1):
                table.add_row(str(i), name, desc, str(count))

            console.print()
            console.print(table)
            console.print()
        else:
            print("\nAvailable prompt sets:")
            for i, (name, desc, count) in enumerate(prompt_sets, 1):
                print(f"  [{i}] {name} ({count} tests) - {desc}")
            print()

    def handle(self) -> int:
        """
        Execute evaluation workflow.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        print_header("EVALUATION", "Test your model's performance")

        # Step 1: Select backend
        backend_choice = print_menu([
            ("ollama", f"{BOX['bullet']} Ollama (local)"),
            ("lmstudio", f"{BOX['bullet']} LM Studio (local)"),
        ], "Select backend:")

        if not backend_choice:
            return 0

        # Step 2: Get backend and list models
        print_info(f"Fetching models from {backend_choice}...")

        try:
            backend = EvaluationBackendRegistry.get(backend_choice)
        except ValueError as e:
            print_error(str(e))
            return 1

        # Validate connection
        is_connected, error = backend.validate_connection()
        if not is_connected:
            print_error(f"Cannot connect to {backend_choice}: {error}")
            if backend_choice == "lmstudio":
                print_info("Make sure LM Studio server is running on http://localhost:1234")
            return 1

        # List models
        models = backend.list_models()
        if not models:
            print_error(f"No models found. Is {backend_choice} running?")
            if backend_choice == "lmstudio":
                print_info("Make sure LM Studio server is running on http://localhost:1234")
            return 1

        # Step 3: Display models and select
        self._display_models_table(backend_choice, models)

        while True:
            try:
                sel = prompt(f"Select model (1-{len(models)})", "1")
                idx = int(sel) - 1
                if 0 <= idx < len(models):
                    model = models[idx]
                    break
            except ValueError:
                pass
            print_error("Invalid selection.")

        # Step 4: List prompt sets
        prompt_sets = self._list_prompt_sets()

        if not prompt_sets:
            print_error("No prompt sets found in Evaluator/prompts/")
            return 1

        # Step 5: Display prompt sets and select
        self._display_prompt_sets_table(prompt_sets)

        while True:
            try:
                sel = prompt(f"Select prompt set (1-{len(prompt_sets)})", "1")
                idx = int(sel) - 1
                if 0 <= idx < len(prompt_sets):
                    prompt_set = prompt_sets[idx][0]
                    prompt_count = prompt_sets[idx][2]
                    break
            except ValueError:
                pass
            print_error("Invalid selection.")

        prompt_file = self.repo_root / "Evaluator" / "prompts" / f"{prompt_set}.json"

        # Step 6: Display configuration and confirm
        print_config({
            "Backend": backend_choice,
            "Model": model,
            "Prompts": f"{prompt_file.name} ({prompt_count} tests)",
        }, "Evaluation Configuration")

        if not confirm("Start evaluation?"):
            print_info("Evaluation cancelled.")
            return 0

        # Step 7: Execute evaluation
        python = self.get_conda_python()
        cmd = [
            python, "-m", "Evaluator.cli",
            "--backend", backend_choice,
            "--model", model,
            "--prompt-set", str(prompt_file)
        ]

        print_info(f"Running: {' '.join(cmd)}")
        print()

        result = subprocess.run(cmd, cwd=str(self.repo_root))
        return result.returncode
