"""
List handler for discovering and displaying available resources.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/list_handler.py
Purpose: Handle 'list' subcommands to discover and display available resources
Used by: Router when handling 'list' command with subcommands (datasets, models, runs, rubrics, scenarios)

This handler implements the list workflow for AI assistants and users to discover:
- datasets: Available JSONL datasets with metadata
- models: Base models (HuggingFace) and fine-tuned models (local)
- runs: Training run directories with status
- rubrics: Available rubric files for data improvement
- scenarios: Evaluation scenarios with test counts

All commands support --json flag for machine-readable output.
"""

import json
from pathlib import Path
from typing import List, Optional

from tuner.handlers.base import BaseHandler
from tuner.discovery import (
    DatasetDiscovery,
    DatasetInfo,
    TrainingRunDiscovery,
    CheckpointDiscovery,
    RubricDiscovery,
    RubricInfo,
    PromptSetDiscovery,
    BaseModelDiscovery,
)

# Import shared UI components
try:
    from shared.ui import (
        print_header,
        print_info,
        print_error,
        console,
        RICH_AVAILABLE,
        COLORS,
    )
except ImportError:
    RICH_AVAILABLE = False
    console = None
    COLORS = {}

    def print_header(title, subtitle=""): print(f"\n=== {title} ===\n{subtitle}")
    def print_info(msg): print(f"[INFO] {msg}")
    def print_error(msg): print(f"[ERROR] {msg}")


class ListHandler(BaseHandler):
    """
    Handler for list subcommands.

    Coordinates resource discovery and displays results in formatted tables
    or JSON output for machine consumption.

    Example:
        handler = ListHandler(subcommand='datasets', output_json=False)
        exit_code = handler.handle()
    """

    def __init__(self, subcommand: str = None, output_json: bool = False):
        """
        Initialize the list handler.

        Args:
            subcommand: Which resource to list (datasets, models, runs, rubrics, scenarios)
            output_json: If True, output JSON instead of formatted tables
        """
        super().__init__()
        self.subcommand = subcommand
        self.output_json = output_json

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "list"

    def can_handle_direct_mode(self) -> bool:
        """This handler supports direct CLI invocation."""
        return True

    def handle(self) -> int:
        """
        Execute list workflow based on subcommand.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        if not self.subcommand:
            self._show_help()
            return 0

        # Route to appropriate list method
        handlers = {
            'datasets': self._list_datasets,
            'models': self._list_models,
            'runs': self._list_runs,
            'rubrics': self._list_rubrics,
            'scenarios': self._list_scenarios,
        }

        handler = handlers.get(self.subcommand)
        if not handler:
            print_error(f"Unknown subcommand: {self.subcommand}")
            print_info("Valid subcommands: datasets, models, runs, rubrics, scenarios")
            return 1

        return handler()

    def _show_help(self):
        """Show help for list subcommands."""
        if not self.output_json:
            print_header("LIST RESOURCES", "Discover available resources for training and evaluation")
            print()
            print("Usage: ./run.sh list <subcommand> [--json]")
            print()
            print("Subcommands:")
            print("  datasets   List available JSONL datasets")
            print("  models     List base models and fine-tuned models")
            print("  runs       List training runs with status")
            print("  rubrics    List available rubrics for data improvement")
            print("  scenarios  List evaluation scenarios with test counts")
            print()
            print("Options:")
            print("  --json     Output as JSON for machine consumption")
            print()

    def _list_datasets(self) -> int:
        """List available datasets."""
        discovery = DatasetDiscovery(repo_root=self.repo_root)
        datasets = discovery.discover_all()

        if self.output_json:
            output = {
                "datasets": [
                    {
                        "path": ds.relative_path,
                        "name": ds.name,
                        "type": ds.dataset_type,
                        "examples": ds.example_count,
                        "size": ds.size_human,
                        "size_bytes": ds.size_bytes,
                    }
                    for ds in datasets
                ]
            }
            print(json.dumps(output, indent=2))
            return 0

        if not datasets:
            print_info("No datasets found in Datasets/ folder.")
            return 0

        if RICH_AVAILABLE:
            self._display_datasets_table(datasets)
        else:
            self._display_datasets_plain(datasets)

        return 0

    def _display_datasets_table(self, datasets: List[DatasetInfo]):
        """Display datasets in a rich table."""
        from rich.table import Table
        from rich import box as rich_box

        print_header("AVAILABLE DATASETS", f"Found {len(datasets)} datasets")

        table = Table(
            box=rich_box.ROUNDED,
            border_style=COLORS.get("cello", "blue"),
        )
        table.add_column("Path", style="white", max_width=55)
        table.add_column("Type", style=COLORS.get("purple", "magenta"), justify="center")
        table.add_column("Examples", style=COLORS.get("aqua", "cyan"), justify="right")
        table.add_column("Size", style="dim", justify="right")

        for ds in datasets:
            table.add_row(
                ds.relative_path,
                ds.dataset_type,
                f"{ds.example_count:,}",
                ds.size_human,
            )

        console.print()
        console.print(table)
        console.print()

    def _display_datasets_plain(self, datasets: List[DatasetInfo]):
        """Display datasets in plain text."""
        print("\nAvailable Datasets:")
        print("-" * 80)
        for ds in datasets:
            print(f"  {ds.relative_path}")
            print(f"    Type: {ds.dataset_type}, Examples: {ds.example_count:,}, Size: {ds.size_human}")
        print()

    def _list_models(self) -> int:
        """List available models."""
        discovery = BaseModelDiscovery(repo_root=self.repo_root)
        base_models, finetuned_models = discovery.discover_all()

        if self.output_json:
            output = {
                "base_models": [
                    {
                        "name": m.name,
                        "source": m.source,
                    }
                    for m in base_models
                ],
                "finetuned_models": [
                    {
                        "name": m.name,
                        "source": m.source,
                        "path": m.path,
                    }
                    for m in finetuned_models
                ]
            }
            print(json.dumps(output, indent=2))
            return 0

        if RICH_AVAILABLE:
            self._display_models_rich(base_models, finetuned_models)
        else:
            self._display_models_plain(base_models, finetuned_models)

        return 0

    def _display_models_rich(self, base_models, finetuned_models):
        """Display models using rich formatting."""
        from rich.panel import Panel
        from rich.text import Text

        print_header("AVAILABLE MODELS", "Base models and fine-tuned adapters")

        # Base models section
        if base_models:
            console.print()
            console.print("[bold]Base Models (HuggingFace):[/bold]")
            for m in base_models:
                console.print(f"  - {m.name}")

        # Fine-tuned models section
        if finetuned_models:
            console.print()
            console.print("[bold]Fine-tuned Models (Local):[/bold]")
            for m in finetuned_models:
                console.print(f"  - {m.path} [dim]({m.source})[/dim]")

        if not base_models and not finetuned_models:
            print_info("No models found.")

        console.print()

    def _display_models_plain(self, base_models, finetuned_models):
        """Display models in plain text."""
        print("\nAvailable Models:")
        print("-" * 60)

        if base_models:
            print("\nBase Models (HuggingFace):")
            for m in base_models:
                print(f"  - {m.name}")

        if finetuned_models:
            print("\nFine-tuned Models (Local):")
            for m in finetuned_models:
                print(f"  - {m.path} ({m.source})")

        print()

    def _list_runs(self) -> int:
        """List training runs."""
        discovery = TrainingRunDiscovery(repo_root=self.repo_root)

        # Discover runs for each trainer type
        sft_runs = discovery.discover('sft', limit=None)
        kto_runs = discovery.discover('kto', limit=None)
        grpo_runs = discovery.discover('grpo', limit=None)

        if self.output_json:
            output = {
                "runs": []
            }

            for trainer_type, runs in [('SFT', sft_runs), ('KTO', kto_runs), ('GRPO', grpo_runs)]:
                for run_path in runs:
                    run_info = self._get_run_info(run_path, trainer_type)
                    output["runs"].append(run_info)

            print(json.dumps(output, indent=2))
            return 0

        all_runs = []
        for trainer_type, runs in [('SFT', sft_runs), ('KTO', kto_runs), ('GRPO', grpo_runs)]:
            for run_path in runs:
                run_info = self._get_run_info(run_path, trainer_type)
                all_runs.append(run_info)

        if not all_runs:
            print_info("No training runs found.")
            print_info("Train a model first - runs will appear in Trainers/*/output*/")
            return 0

        if RICH_AVAILABLE:
            self._display_runs_table(all_runs)
        else:
            self._display_runs_plain(all_runs)

        return 0

    def _get_run_info(self, run_path: Path, trainer_type: str) -> dict:
        """Extract information about a training run."""
        timestamp = run_path.name
        has_final = (run_path / "final_model").exists()

        # Count checkpoints
        checkpoints_dir = run_path / "checkpoints"
        checkpoint_count = 0
        if checkpoints_dir.exists():
            checkpoint_count = len(list(checkpoints_dir.glob("checkpoint-*")))

        # Determine status
        if has_final:
            status = "Done"
        elif checkpoint_count > 0:
            status = "Partial"
        else:
            status = "Started"

        # Try to get model name from config or logs
        model_name = self._get_run_model_name(run_path)

        # Try to get dataset name
        dataset_name = self._get_run_dataset(run_path)

        # Calculate relative path
        try:
            relative_path = str(run_path.relative_to(self.repo_root))
        except ValueError:
            relative_path = str(run_path)

        return {
            "run_id": timestamp,
            "type": trainer_type,
            "status": status,
            "model": model_name or "unknown",
            "dataset": dataset_name or "unknown",
            "has_final": has_final,
            "checkpoints": checkpoint_count,
            "path": relative_path,
        }

    def _get_run_model_name(self, run_path: Path) -> Optional[str]:
        """Try to extract model name from run directory."""
        # Check for adapter_config.json in final_model
        adapter_config = run_path / "final_model" / "adapter_config.json"
        if adapter_config.exists():
            try:
                with open(adapter_config, 'r') as f:
                    config = json.load(f)
                base_model = config.get('base_model_name_or_path', '')
                # Extract short name
                if '/' in base_model:
                    return base_model.split('/')[-1]
                return base_model
            except Exception:
                pass

        return None

    def _get_run_dataset(self, run_path: Path) -> Optional[str]:
        """Try to extract dataset name from run logs."""
        # This would require parsing logs - return None for now
        return None

    def _display_runs_table(self, runs: List[dict]):
        """Display runs in a rich table."""
        from rich.table import Table
        from rich import box as rich_box

        print_header("TRAINING RUNS", f"Found {len(runs)} runs")

        table = Table(
            box=rich_box.ROUNDED,
            border_style=COLORS.get("cello", "blue"),
        )
        table.add_column("Run ID", style="white")
        table.add_column("Type", style=COLORS.get("purple", "magenta"), justify="center")
        table.add_column("Status", style=COLORS.get("aqua", "cyan"), justify="center")
        table.add_column("Model", style="dim")
        table.add_column("Checkpoints", style="dim", justify="right")

        for run in runs:
            status_style = ""
            if run["status"] == "Done":
                status_style = "green"
            elif run["status"] == "Partial":
                status_style = "yellow"

            table.add_row(
                run["run_id"],
                run["type"],
                f"[{status_style}]{run['status']}[/{status_style}]" if status_style else run["status"],
                run["model"][:30] if run["model"] else "-",
                str(run["checkpoints"]),
            )

        console.print()
        console.print(table)
        console.print()

    def _display_runs_plain(self, runs: List[dict]):
        """Display runs in plain text."""
        print("\nTraining Runs:")
        print("-" * 80)
        for run in runs:
            status_char = "+" if run["status"] == "Done" else "~" if run["status"] == "Partial" else "-"
            print(f"  [{status_char}] {run['run_id']} ({run['type']}) - {run['model']}")
        print()

    def _list_rubrics(self) -> int:
        """List available rubrics."""
        discovery = RubricDiscovery(repo_root=self.repo_root)
        rubrics = discovery.discover_all()

        if self.output_json:
            output = {
                "rubrics": [
                    {
                        "name": r.name,
                        "description": r.description,
                        "scope": r.scope,
                        "source": r.source,
                    }
                    for r in rubrics
                ]
            }
            print(json.dumps(output, indent=2))
            return 0

        if not rubrics:
            print_info("No rubrics found.")
            return 0

        if RICH_AVAILABLE:
            self._display_rubrics_rich(rubrics)
        else:
            self._display_rubrics_plain(rubrics)

        return 0

    def _display_rubrics_rich(self, rubrics: List[RubricInfo]):
        """Display rubrics using rich formatting."""
        print_header("AVAILABLE RUBRICS", f"Found {len(rubrics)} rubrics")

        console.print()
        for r in rubrics:
            scope_str = f" [dim]({r.scope})[/dim]" if r.scope else ""
            source_str = f" [dim][{r.source}][/dim]"
            console.print(f"  - [bold]{r.name}[/bold]{scope_str}{source_str}")
            if r.description and r.description != r.name:
                console.print(f"    {r.description[:70]}...")
        console.print()

    def _display_rubrics_plain(self, rubrics: List[RubricInfo]):
        """Display rubrics in plain text."""
        print("\nAvailable Rubrics:")
        print("-" * 60)
        for r in rubrics:
            scope_str = f" ({r.scope})" if r.scope else ""
            print(f"  - {r.name}{scope_str} [{r.source}]")
        print()

    def _list_scenarios(self) -> int:
        """List evaluation scenarios."""
        discovery = PromptSetDiscovery(repo_root=self.repo_root)
        scenarios = discovery.discover_all()

        if self.output_json:
            output = {
                "scenarios": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "tests": s.count,
                        "path": str(s.path.relative_to(self.repo_root)) if s.path else None,
                    }
                    for s in scenarios
                ]
            }
            print(json.dumps(output, indent=2))
            return 0

        if not scenarios:
            print_info("No evaluation scenarios found in Evaluator/config/scenarios/")
            return 0

        if RICH_AVAILABLE:
            self._display_scenarios_rich(scenarios)
        else:
            self._display_scenarios_plain(scenarios)

        return 0

    def _display_scenarios_rich(self, scenarios):
        """Display scenarios using rich formatting."""
        from rich.table import Table
        from rich import box as rich_box

        total_tests = sum(s.count for s in scenarios)
        print_header("EVALUATION SCENARIOS", f"Found {len(scenarios)} scenarios ({total_tests} total tests)")

        table = Table(
            box=rich_box.ROUNDED,
            border_style=COLORS.get("cello", "blue"),
        )
        table.add_column("Name", style="white")
        table.add_column("Description", style="dim", max_width=50)
        table.add_column("Tests", style=COLORS.get("aqua", "cyan"), justify="right")

        for s in scenarios:
            table.add_row(
                f"{s.name}.yaml",
                s.description[:50] if s.description else "-",
                str(s.count),
            )

        console.print()
        console.print(table)
        console.print()

    def _display_scenarios_plain(self, scenarios):
        """Display scenarios in plain text."""
        print("\nEvaluation Scenarios:")
        print("-" * 60)
        for s in scenarios:
            print(f"  - {s.name}.yaml ({s.count} tests)")
            if s.description:
                print(f"    {s.description}")
        print()
