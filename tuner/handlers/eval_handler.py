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
from shared.ui import (
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

    def _list_scenarios(self):
        """
        List available YAML scenarios.

        Returns:
            List of PromptSetInfo objects
        """
        from tuner.discovery import PromptSetDiscovery
        discovery = PromptSetDiscovery(repo_root=self.repo_root)
        return discovery.discover_all()

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

    def _display_gguf_models_table(self, backend, models: List[str]) -> None:
        """
        Display available GGUF models with detailed info.

        Args:
            backend: LlamaCppBackend instance (for get_model_info)
            models: List of GGUF file paths
        """
        from pathlib import Path

        if RICH_AVAILABLE:
            from rich.table import Table
            from rich import box as rich_box

            table = Table(
                title="Available GGUF Models",
                box=rich_box.ROUNDED,
                border_style=COLORS["cello"],
            )
            table.add_column("#", style=COLORS["orange"], width=4, justify="center")
            table.add_column("Name", style="white")
            table.add_column("Quant", style=COLORS["purple"])
            table.add_column("Size", style="dim", justify="right")
            table.add_column("Type", style=COLORS["aqua"])

            for i, model_path in enumerate(models, 1):
                info = backend.get_model_info(model_path)
                name = info.get("name", Path(model_path).stem)
                quant = info.get("quantization") or "-"
                size = f"{info.get('size_gb', 0):.1f}GB"
                trainer = info.get("trainer_type", "-").upper()

                table.add_row(str(i), name, quant, size, trainer)

            console.print()
            console.print(table)
            console.print()
        else:
            print("\nAvailable GGUF models:")
            for i, model_path in enumerate(models, 1):
                info = backend.get_model_info(model_path)
                name = info.get("name", Path(model_path).stem)
                quant = info.get("quantization") or ""
                size = info.get("size_gb", 0)
                quant_str = f" ({quant})" if quant else ""
                print(f"  [{i}] {name}{quant_str} [{size:.1f}GB]")
            print()

    def _display_lora_models_table(self, backend, models: List[str]) -> None:
        """
        Display available LoRA adapters with detailed info.

        Args:
            backend: UnslothBackend instance (for get_model_info)
            models: List of adapter directory paths
        """
        from pathlib import Path

        if RICH_AVAILABLE:
            from rich.table import Table
            from rich import box as rich_box

            table = Table(
                title="Available LoRA Adapters",
                box=rich_box.ROUNDED,
                border_style=COLORS["cello"],
            )
            table.add_column("#", style=COLORS["orange"], width=4, justify="center")
            table.add_column("Run", style="white")
            table.add_column("Base Model", style=COLORS["aqua"])
            table.add_column("Type", style=COLORS["purple"])
            table.add_column("Size", style="dim", justify="right")

            for i, model_path in enumerate(models, 1):
                info = backend.get_model_info(model_path)
                timestamp = info.get("timestamp", "unknown")
                base = info.get("base_model_short", "unknown")
                trainer = info.get("trainer_type", "-").upper()
                size = f"{info.get('size_mb', 0):.0f}MB" if info.get('size_mb') else "-"

                table.add_row(str(i), timestamp, base, trainer, size)

            console.print()
            console.print(table)
            console.print()
        else:
            print("\nAvailable LoRA adapters:")
            for i, model_path in enumerate(models, 1):
                info = backend.get_model_info(model_path)
                timestamp = info.get("timestamp", "unknown")
                base = info.get("base_model_short", "unknown")
                trainer = info.get("trainer_type", "-").upper()
                print(f"  [{i}] {timestamp} ({base}) [{trainer}]")
            print()

    def _display_scenarios_table(self, scenarios) -> None:
        """
        Display available YAML scenarios in a table.

        Args:
            scenarios: List of PromptSetInfo objects
        """
        if RICH_AVAILABLE:
            from rich.table import Table
            from rich import box as rich_box

            table = Table(
                title="Available Test Scenarios",
                box=rich_box.ROUNDED,
                border_style=COLORS["cello"],
            )
            table.add_column("#", style=COLORS["orange"], width=4, justify="center")
            table.add_column("Name", style="white")
            table.add_column("Description", style="dim")
            table.add_column("Tests", style=COLORS["aqua"], justify="right")

            for i, info in enumerate(scenarios, 1):
                table.add_row(str(i), info.name, info.description, str(info.count))

            console.print()
            console.print(table)
            console.print()
        else:
            print("\nAvailable test scenarios:")
            for i, info in enumerate(scenarios, 1):
                print(f"  [{i}] {info.name} ({info.count} tests) - {info.description}")
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
            ("unsloth", f"{BOX['star']} Unsloth (LoRA - direct, no conversion)"),
            ("llamacpp", f"{BOX['bullet']} llama.cpp (GGUF - fast, portable)"),
            ("ollama", f"{BOX['bullet']} Ollama (local server)"),
            ("lmstudio", f"{BOX['bullet']} LM Studio (local server)"),
        ], "Select backend:")

        if not backend_choice:
            return 0

        # Step 2: Get backend and list models
        print_info(f"Fetching models from {backend_choice}...")

        try:
            # Pass repo_root for backends that discover local models
            if backend_choice in ("llamacpp", "unsloth"):
                backend = EvaluationBackendRegistry.get(backend_choice, repo_root=self.repo_root)
            else:
                backend = EvaluationBackendRegistry.get(backend_choice)
        except ValueError as e:
            print_error(str(e))
            return 1

        # Validate connection
        is_connected, error = backend.validate_connection()
        if not is_connected:
            print_error(f"Cannot connect to {backend_choice}:")
            print_info(error)
            if backend_choice == "lmstudio":
                print_info("Make sure LM Studio server is running on http://localhost:1234")
            return 1

        # List models
        models = backend.list_models()
        if not models:
            print_error(f"No models found.")
            if backend_choice == "lmstudio":
                print_info("Make sure LM Studio server is running on http://localhost:1234")
            elif backend_choice == "llamacpp":
                print_info("No GGUF models found in training outputs.")
                print_info("Train a model first, then convert to GGUF.")
            elif backend_choice == "unsloth":
                print_info("No LoRA adapters found in training outputs.")
                print_info("Train a model first - the final_model/ directory will appear.")
            elif backend_choice == "ollama":
                print_info("Is Ollama running? Check with: ollama list")
            return 1

        # Step 3: Display models and select
        if backend_choice == "llamacpp":
            self._display_gguf_models_table(backend, models)
        elif backend_choice == "unsloth":
            self._display_lora_models_table(backend, models)
        else:
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

        # Step 4: List YAML scenarios
        scenarios = self._list_scenarios()

        if not scenarios:
            print_error("No test scenarios found in Evaluator/config/scenarios/")
            return 1

        # Step 5: Display scenarios and select
        self._display_scenarios_table(scenarios)

        while True:
            try:
                sel = prompt(f"Select test scenario (1-{len(scenarios)})", "1")
                idx = int(sel) - 1
                if 0 <= idx < len(scenarios):
                    selected = scenarios[idx]
                    break
            except ValueError:
                pass
            print_error("Invalid selection.")

        # Step 6: Display configuration and confirm
        print_config({
            "Backend": backend_choice,
            "Model": model,
            "Scenario": f"{selected.name} ({selected.count} tests)",
        }, "Evaluation Configuration")

        if not confirm("Start evaluation?"):
            print_info("Evaluation cancelled.")
            return 0

        # Step 7: Execute evaluation
        python = self.get_conda_python()

        # Generate timestamped output paths
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.repo_root / "Evaluator" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        output_json = results_dir / f"run_{timestamp}.json"
        output_md = results_dir / f"run_{timestamp}.md"

        cmd = [
            python, "-m", "Evaluator.cli",
            "--backend", backend_choice,
            "--model", model,
            "--scenario", selected.path.name,
            "--output", str(output_json),
            "--markdown", str(output_md)
        ]

        print_info(f"Running: {' '.join(cmd)}")
        print()

        result = subprocess.run(cmd, cwd=str(self.repo_root))

        if result.returncode == 0:
            print()
            print_info(f"Results saved to: {output_json.relative_to(self.repo_root)}")
            print_info(f"Markdown report: {output_md.relative_to(self.repo_root)}")

        return result.returncode
