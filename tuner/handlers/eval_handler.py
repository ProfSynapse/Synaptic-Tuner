"""
Evaluation handler for testing model performance.

Location: tuner/handlers/eval_handler.py
Purpose: Orchestrate evaluation workflow - select backend, model, and prompt set
Used by: Router when handling 'eval' command or main menu selection

This handler implements the evaluation workflow:
1. Select evaluation backend (Ollama or LM Studio)
2. List available models from backend
3. Select model to evaluate
4. List available prompt sets
5. Select prompt set
6. Display configuration
7. Execute evaluation with live dashboard

Supports --json flag for AI-parseable output. In JSON mode:
- Returns available backends, models, and scenarios
- All output is JSON formatted for programmatic parsing
"""

from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from tuner.handlers.base import BaseHandler
from tuner.backends.registry import EvaluationBackendRegistry
from tuner.discovery import TrainingRunDiscovery, CheckpointDiscovery

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
from shared.cloud_eval_progress import extract_record_progress

# Import evaluation dashboard and UI
try:
    from shared.ui import LiveEvaluationDashboard
    from Evaluator.ui import rich_summary, rich_failure_details, print_evaluation_header
    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False

# Import evaluation core components
try:
    from Evaluator.config_loader import load_yaml_scenarios
    from Evaluator.client_factory import create_client, create_settings
    from Evaluator.runner import evaluate_cases
    from Evaluator.reporting import build_run_payload, write_json, render_markdown, console_summary
    _EVALUATOR_AVAILABLE = True
except ImportError:
    _EVALUATOR_AVAILABLE = False


class EvalHandler(BaseHandler):
    """
    Handler for evaluation workflow.

    Coordinates backend selection, model discovery, prompt set selection,
    and execution of the Evaluator CLI to test model performance.

    JSON Mode (--json flag):
        In JSON mode, returns structured status about available backends,
        models, and test scenarios. Does not execute interactive menus.

    Example:
        handler = EvalHandler()
        exit_code = handler.handle()
        # User interacts with menus, selects backend/model/prompts
        # Returns 0 on success, non-zero on failure

        # With JSON mode
        handler = EvalHandler(args=args)  # args.json = True
        exit_code = handler.handle()  # Returns JSON status
    """

    def __init__(self, args: Optional[Namespace] = None):
        """Initialize handler with optional args."""
        super().__init__(args=args)

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "eval"

    def can_handle_direct_mode(self) -> bool:
        """This handler supports direct CLI invocation."""
        return True

    def _get_eval_status(self) -> dict:
        """
        Get evaluation status for JSON output.

        Returns dict with available backends, models, and scenarios.
        """
        # List available backends
        backends = []

        # Check each backend
        backend_configs = [
            ("unsloth", "Unsloth (LoRA - direct)"),
            ("llamacpp", "llama.cpp (GGUF)"),
            ("mlc", "MLC/WebLLM (WebGPU)"),
            ("ollama", "Ollama (local server)"),
            ("lmstudio", "LM Studio (local server)"),
        ]

        for backend_id, backend_name in backend_configs:
            backend_info = {
                "id": backend_id,
                "name": backend_name,
                "available": False,
                "models": [],
            }

            try:
                if backend_id in ("llamacpp", "mlc", "unsloth"):
                    backend = EvaluationBackendRegistry.get(backend_id, repo_root=self.repo_root)
                else:
                    backend = EvaluationBackendRegistry.get(backend_id)

                is_connected, _ = backend.validate_connection()
                backend_info["available"] = is_connected

                if is_connected:
                    models = backend.list_models()
                    backend_info["models"] = models[:20] if models else []  # Limit for brevity
                    backend_info["model_count"] = len(models) if models else 0

            except (ValueError, Exception):
                pass

            backends.append(backend_info)

        # List available scenarios
        scenarios = []
        try:
            scenario_infos = self._list_scenarios()
            for info in scenario_infos:
                scenarios.append({
                    "name": info.name,
                    "description": info.description,
                    "test_count": info.count,
                    "path": str(info.path),
                })
        except Exception:
            pass

        return {
            "command": "eval",
            "status": "ready",
            "backends": backends,
            "scenarios": scenarios,
        }

    def _list_scenarios(self):
        """
        List available YAML scenarios.

        Returns:
            List of PromptSetInfo objects
        """
        from tuner.discovery import PromptSetDiscovery
        discovery = PromptSetDiscovery(repo_root=self.repo_root)
        return discovery.discover_all()

    # -- Generic table display infrastructure ----------------------------------

    @dataclass
    class _ColumnSpec:
        """Specification for a single table column."""
        header: str
        style: str = "white"
        width: Optional[int] = None
        justify: str = "left"

    @dataclass
    class _TableSpec:
        """Specification for a display table."""
        title: str
        columns: List[Any] = field(default_factory=list)   # List[_ColumnSpec]
        row_extractor: Callable[[int, Any], List[str]] = lambda i, x: [str(x)]
        plain_formatter: Callable[[int, Any], str] = lambda i, x: str(x)

    def _display_table(self, items: List[Any], spec: "_TableSpec") -> None:
        """
        Generic table renderer that handles both Rich and plain-text output.

        Renders *items* according to *spec*. In Rich mode, produces a bordered
        table with an auto-numbered ``#`` column followed by the columns listed
        in the spec.  In plain-text mode, prints a numbered list using the
        spec's plain_formatter.
        """
        if RICH_AVAILABLE:
            from rich.table import Table
            from rich import box as rich_box

            table = Table(
                title=spec.title,
                box=rich_box.ROUNDED,
                border_style=COLORS["cello"],
            )
            table.add_column("#", style=COLORS["orange"], width=4, justify="center")
            for col in spec.columns:
                table.add_column(
                    col.header, style=col.style,
                    width=col.width, justify=col.justify,
                )

            for i, item in enumerate(items, 1):
                row = spec.row_extractor(i, item)
                table.add_row(str(i), *row)

            console.print()
            console.print(table)
            console.print()
        else:
            print(f"\n{spec.title}:")
            for i, item in enumerate(items, 1):
                print(f"  [{i}] {spec.plain_formatter(i, item)}")
            print()

    # -- Thin wrappers that build a TableSpec and delegate ---------------------

    def _display_models_table(self, backend: str, models: List[str]) -> None:
        """Display available models in a table."""
        spec = self._TableSpec(
            title=f"Available {backend.title()} Models",
            columns=[self._ColumnSpec("Model")],
            row_extractor=lambda i, m: [m],
            plain_formatter=lambda i, m: m,
        )
        self._display_table(models, spec)

    def _display_gguf_models_table(self, backend, models: List[str]) -> None:
        """Display available GGUF models with detailed info."""
        C = self._ColumnSpec
        spec = self._TableSpec(
            title="Available GGUF Models",
            columns=[
                C("Name"), C("Quant", style=COLORS["purple"]),
                C("Size", style="dim", justify="right"), C("Type", style=COLORS["aqua"]),
            ],
            row_extractor=lambda i, mp: self._gguf_row(backend, mp),
            plain_formatter=lambda i, mp: self._gguf_plain(backend, mp),
        )
        self._display_table(models, spec)

    @staticmethod
    def _gguf_row(backend, model_path: str) -> List[str]:
        info = backend.get_model_info(model_path)
        return [
            info.get("name", Path(model_path).stem),
            info.get("quantization") or "-",
            f"{info.get('size_gb', 0):.1f}GB",
            info.get("trainer_type", "-").upper(),
        ]

    @staticmethod
    def _gguf_plain(backend, model_path: str) -> str:
        info = backend.get_model_info(model_path)
        name = info.get("name", Path(model_path).stem)
        quant = info.get("quantization") or ""
        size = info.get("size_gb", 0)
        quant_str = f" ({quant})" if quant else ""
        return f"{name}{quant_str} [{size:.1f}GB]"

    def _display_lora_models_table(self, backend, models: List[str]) -> None:
        """Display available LoRA adapters with detailed info."""
        C = self._ColumnSpec
        spec = self._TableSpec(
            title="Available LoRA Adapters",
            columns=[
                C("Run"), C("Base Model", style=COLORS["aqua"]),
                C("Type", style=COLORS["purple"]),
                C("Size", style="dim", justify="right"),
            ],
            row_extractor=lambda i, mp: self._lora_row(backend, mp),
            plain_formatter=lambda i, mp: self._lora_plain(backend, mp),
        )
        self._display_table(models, spec)

    @staticmethod
    def _lora_row(backend, model_path: str) -> List[str]:
        info = backend.get_model_info(model_path)
        return [
            info.get("timestamp", "unknown"),
            info.get("base_model_short", "unknown"),
            info.get("trainer_type", "-").upper(),
            f"{info.get('size_mb', 0):.0f}MB" if info.get("size_mb") else "-",
        ]

    @staticmethod
    def _lora_plain(backend, model_path: str) -> str:
        info = backend.get_model_info(model_path)
        return (
            f"{info.get('timestamp', 'unknown')} "
            f"({info.get('base_model_short', 'unknown')}) "
            f"[{info.get('trainer_type', '-').upper()}]"
        )

    def _display_mlc_models_table(self, backend, models: List[str]) -> None:
        """Display available MLC/WebGPU models with detailed info."""
        C = self._ColumnSpec
        spec = self._TableSpec(
            title="Available MLC/WebGPU Models",
            columns=[
                C("Name"), C("Arch", style=COLORS["aqua"]),
                C("Quant", style=COLORS["purple"]), C("Type", style="dim"),
            ],
            row_extractor=lambda i, mp: self._mlc_row(backend, mp),
            plain_formatter=lambda i, mp: self._mlc_plain(backend, mp),
        )
        self._display_table(models, spec)

    @staticmethod
    def _mlc_row(backend, model_path: str) -> List[str]:
        info = backend.get_model_info(model_path)
        return [
            info.get("name", Path(model_path).name),
            info.get("architecture") or "-",
            info.get("quantization") or "-",
            info.get("trainer_type", "-").upper(),
        ]

    @staticmethod
    def _mlc_plain(backend, model_path: str) -> str:
        info = backend.get_model_info(model_path)
        name = info.get("name", Path(model_path).name)
        quant = info.get("quantization") or ""
        quant_str = f" ({quant})" if quant else ""
        return f"{name}{quant_str}"

    def _display_training_runs_table(self, runs: List[Path], trainer_type: str) -> None:
        """Display available training runs in a table."""
        C = self._ColumnSpec
        spec = self._TableSpec(
            title=f"Available {trainer_type.upper()} Training Runs",
            columns=[
                C("Run"), C("Has Final", style=COLORS["aqua"], justify="center"),
                C("Checkpoints", style=COLORS["purple"], justify="right"),
            ],
            row_extractor=lambda i, rp: self._training_run_row(rp),
            plain_formatter=lambda i, rp: self._training_run_plain(rp),
        )
        self._display_table(runs, spec)

    @staticmethod
    def _training_run_row(run_path: Path) -> List[str]:
        has_final = "\u2713" if (run_path / "final_model").exists() else "-"
        checkpoints_dir = run_path / "checkpoints"
        cp_count = 0
        if checkpoints_dir.exists():
            cp_count = len(list(checkpoints_dir.glob("checkpoint-*")))
        return [run_path.name, has_final, str(cp_count)]

    @staticmethod
    def _training_run_plain(run_path: Path) -> str:
        has_final = "(final)" if (run_path / "final_model").exists() else ""
        return f"{run_path.name} {has_final}"

    def _display_checkpoints_table(self, checkpoints: List, trainer_type: str) -> None:
        """Display available checkpoints with metrics in a table."""
        C = self._ColumnSpec
        columns = [
            C("Checkpoint"), C("Step", style=COLORS["aqua"], justify="right"),
            C("Loss", style=COLORS["purple"], justify="right"),
        ]
        if trainer_type == "kto":
            columns += [
                C("KL", style="dim", justify="right"),
                C("Margin", style="dim", justify="right"),
            ]
        elif trainer_type == "grpo":
            columns.append(C("Reward", style="dim", justify="right"))
        columns.append(C("Epoch", style="dim", justify="right"))

        def row_extractor(_i: int, cp: Any) -> List[str]:
            return self._checkpoint_row(cp, trainer_type)

        spec = self._TableSpec(
            title="Available Checkpoints",
            columns=columns,
            row_extractor=row_extractor,
            plain_formatter=lambda i, cp: self._checkpoint_plain(cp),
        )
        self._display_table(checkpoints, spec)

    @staticmethod
    def _checkpoint_row(cp: Any, trainer_type: str) -> List[str]:
        name = "final_model \u2605" if cp.is_final else f"checkpoint-{cp.step}"
        step_str = "-" if cp.is_final else str(cp.step)
        loss = cp.metrics.get("loss")
        loss_str = f"{loss:.4f}" if loss is not None else "-"
        epoch = cp.metrics.get("epoch")
        epoch_str = f"{epoch:.2f}" if epoch is not None else "-"

        row = [name, step_str, loss_str]
        if trainer_type == "kto":
            kl = cp.metrics.get("kl")
            margin = cp.metrics.get("rewards/margins")
            row += [
                f"{kl:.4f}" if kl is not None else "-",
                f"{margin:.4f}" if margin is not None else "-",
            ]
        elif trainer_type == "grpo":
            reward = cp.metrics.get("reward") or cp.metrics.get("rewards/mean")
            row.append(f"{reward:.4f}" if reward is not None else "-")
        row.append(epoch_str)
        return row

    @staticmethod
    def _checkpoint_plain(cp: Any) -> str:
        name = "final_model" if cp.is_final else f"checkpoint-{cp.step}"
        loss = cp.metrics.get("loss", "N/A")
        loss_str = f"loss={loss:.4f}" if isinstance(loss, (int, float)) else f"loss={loss}"
        return f"{name} ({loss_str})"

    def _select_unsloth_model(self) -> Tuple[str, str]:
        """
        Two-step model selection for Unsloth backend.

        Step 1: Select training run
        Step 2: Select checkpoint (final or intermediate)

        Returns:
            Tuple of (model_path, trainer_type) or (None, None) if cancelled
        """
        # Discover training runs for SFT, KTO, and GRPO
        discovery = TrainingRunDiscovery(repo_root=self.repo_root)

        sft_runs = discovery.discover("sft", limit=10)
        kto_runs = discovery.discover("kto", limit=10)
        grpo_runs = discovery.discover("grpo", limit=10)

        if not sft_runs and not kto_runs and not grpo_runs:
            print_error("No training runs found.")
            print_info("Train a model first - the final_model/ directory will appear.")
            return None, None

        # Let user choose trainer type first
        trainer_options = []
        if sft_runs:
            trainer_options.append(("sft", f"SFT ({len(sft_runs)} runs)"))
        if kto_runs:
            trainer_options.append(("kto", f"KTO ({len(kto_runs)} runs)"))
        if grpo_runs:
            trainer_options.append(("grpo", f"GRPO/GSPO ({len(grpo_runs)} runs)"))

        trainer_type = print_menu(trainer_options, "Select training type:")
        if not trainer_type:
            return None, None

        if trainer_type == "sft":
            runs = sft_runs
        elif trainer_type == "kto":
            runs = kto_runs
        else:
            runs = grpo_runs

        # Display and select training run
        self._display_training_runs_table(runs, trainer_type)

        while True:
            try:
                sel = prompt(f"Select training run (1-{len(runs)})", "1")
                idx = int(sel) - 1
                if 0 <= idx < len(runs):
                    selected_run = runs[idx]
                    break
            except ValueError:
                pass
            print_error("Invalid selection.")

        # Discover checkpoints for selected run
        checkpoints = CheckpointDiscovery.discover(selected_run)

        if not checkpoints:
            print_error("No checkpoints found in selected run.")
            return None, None

        # Display and select checkpoint
        self._display_checkpoints_table(checkpoints, trainer_type)

        while True:
            try:
                sel = prompt(f"Select checkpoint (1-{len(checkpoints)})", "1")
                idx = int(sel) - 1
                if 0 <= idx < len(checkpoints):
                    selected_checkpoint = checkpoints[idx]
                    break
            except ValueError:
                pass
            print_error("Invalid selection.")

        return str(selected_checkpoint.path), trainer_type

    def _display_scenarios_table(self, scenarios) -> None:
        """Display available YAML scenarios in a table."""
        C = self._ColumnSpec
        spec = self._TableSpec(
            title="Available Test Scenarios",
            columns=[
                C("Name"), C("Description", style="dim"),
                C("Tests", style=COLORS["aqua"], justify="right"),
            ],
            row_extractor=lambda i, info: [info.name, info.description, str(info.count)],
            plain_formatter=lambda i, info: (
                f"{info.name} ({info.count} tests) - {info.description}"
            ),
        )
        self._display_table(scenarios, spec)

    def handle(self) -> int:
        """
        Execute evaluation workflow.

        In JSON mode, returns evaluation status without interactive prompts.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        # JSON mode: return status information
        if self.json_mode:
            status = self._get_eval_status()
            self.output(status)
            return 0

        print_header("EVALUATION", "Test your model's performance")

        # Step 1: Select backend
        backend_choice = print_menu([
            ("unsloth", f"{BOX['star']} Unsloth (LoRA - direct, no conversion)"),
            ("llamacpp", f"{BOX['bullet']} llama.cpp (GGUF - fast, portable)"),
            ("mlc", f"{BOX['bullet']} MLC/WebLLM (WebGPU - browser-based)"),
            ("ollama", f"{BOX['bullet']} Ollama (local server)"),
            ("lmstudio", f"{BOX['bullet']} LM Studio (local server)"),
        ], "Select backend:")

        if not backend_choice:
            return 0

        # Step 2: Get backend and list models
        # Unsloth uses special two-step flow (training run → checkpoint)
        if backend_choice == "unsloth":
            # Validate Unsloth is available first
            try:
                backend = EvaluationBackendRegistry.get(backend_choice, repo_root=self.repo_root)
            except ValueError as e:
                print_error(str(e))
                return 1

            is_connected, error = backend.validate_connection()
            if not is_connected:
                print_error(f"Cannot connect to {backend_choice}:")
                print_info(error)
                return 1

            # Use two-step selection: training run → checkpoint
            model, trainer_type = self._select_unsloth_model()
            if not model:
                return 0  # User cancelled
        else:
            print_info(f"Fetching models from {backend_choice}...")

            try:
                # Pass repo_root for backends that discover local models
                if backend_choice in ("llamacpp", "mlc"):
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
                elif backend_choice == "mlc":
                    print_info("No MLC/WebGPU models found in training outputs.")
                    print_info("Train a model first, then use Upload -> WebGPU/MLC conversion.")
                elif backend_choice == "ollama":
                    print_info("Is Ollama running? Check with: ollama list")
                return 1

            # Step 3: Display models and select
            if backend_choice == "llamacpp":
                self._display_gguf_models_table(backend, models)
            elif backend_choice == "mlc":
                self._display_mlc_models_table(backend, models)
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

        # MLC: Skip to browser-based evaluation (tests selected in browser)
        if backend_choice == "mlc":
            return self._run_mlc_evaluation(model, None)

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
        if backend_choice == "mlc":
            # MLC uses browser-based WebLLM evaluation
            return self._run_mlc_evaluation(model, selected)

        # Run inline with dashboard if available
        if _EVALUATOR_AVAILABLE and _DASHBOARD_AVAILABLE and RICH_AVAILABLE:
            return self._run_inline_evaluation(backend_choice, model, selected)

        # Fallback to subprocess for missing dependencies
        return self._run_subprocess_evaluation(backend_choice, model, selected)

    def _run_inline_evaluation(self, backend: str, model: str, scenario) -> int:
        """
        Run evaluation inline with live dashboard.

        Args:
            backend: Backend name (ollama, lmstudio, llamacpp, unsloth)
            model: Model name or path
            scenario: PromptSetInfo object with path and count

        Returns:
            Exit code (0 = success)
        """
        config_dir = self.repo_root / "Evaluator" / "config"

        # Generate timestamped output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.repo_root / "Evaluator" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_json = results_dir / f"run_{timestamp}.json"
        output_md = results_dir / f"run_{timestamp}.md"

        # Load test cases from scenario
        try:
            cases = load_yaml_scenarios(
                config_dir=config_dir,
                scenario_files=[scenario.path.name],
            )
        except Exception as e:
            print_error(f"Failed to load scenario: {e}")
            return 1

        if not cases:
            print_error("No test cases found in scenario.")
            return 1

        # Create settings and client
        try:
            settings = create_settings(
                backend=backend,
                model=model,
                temperature=0.7,
                max_tokens=4096,
            )
            client = create_client(
                backend=backend,
                settings=settings,
                timeout=60.0,
                retries=2,
            )
        except Exception as e:
            print_error(f"Failed to create client: {e}")
            return 1

        # Print evaluation header
        print_evaluation_header(
            model_name=model,
            backend=backend,
            total_tests=len(cases),
            scenario_file=scenario.path.name,
        )

        # Create dashboard
        dashboard = LiveEvaluationDashboard(
            title="Model Evaluation",
            total_tests=len(cases),
            log_lines=5,
        )

        def on_record_dashboard(record):
            """Update dashboard with evaluation result."""
            dashboard.update(**extract_record_progress(record))

        # Run evaluation with dashboard
        try:
            with dashboard:
                records = evaluate_cases(
                    cases,
                    client=client,
                    on_record=on_record_dashboard,
                )
        except Exception as e:
            print_error(f"Evaluation failed: {e}")
            return 1

        print()  # Blank line before summary

        # Display rich summary
        rich_summary(records)

        # Show failure details
        failed_count = sum(1 for r in records if not r.passed)
        if failed_count > 0:
            rich_failure_details(records, max_display=10)

        # Save results
        try:
            # Build metadata for the evaluation run
            metadata = {
                "backend": backend,
                "model": settings.model,
                "host": getattr(settings, 'host', 'localhost'),
                "port": getattr(settings, 'port', 0),
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "total_prompts": len(cases),
                "selected_prompts": len(cases),
                "scenario": scenario.path.name,
            }
            payload = build_run_payload(records, metadata=metadata)
            write_json(output_json, payload)
            output_md.write_text(
                render_markdown(records, model, scenario.path.name),
                encoding="utf-8",
            )
        except Exception as e:
            print_error(f"Failed to save results: {e}")
            return 1

        print_info(f"Results saved to: {output_json.relative_to(self.repo_root)}")
        print_info(f"Markdown report: {output_md.relative_to(self.repo_root)}")

        # Return appropriate exit code
        passed = sum(1 for r in records if r.passed)
        return 0 if passed == len(records) else 1

    def _run_subprocess_evaluation(self, backend: str, model: str, scenario) -> int:
        """
        Fallback: Run evaluation via subprocess.

        Args:
            backend: Backend name
            model: Model name or path
            scenario: PromptSetInfo object

        Returns:
            Exit code from subprocess
        """
        import subprocess

        python = self.get_conda_python()

        # Generate timestamped output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.repo_root / "Evaluator" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        output_json = results_dir / f"run_{timestamp}.json"
        output_md = results_dir / f"run_{timestamp}.md"

        cmd = [
            python, "-m", "Evaluator.cli",
            "--backend", backend,
            "--model", model,
            "--scenario", scenario.path.name,
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

    def _run_mlc_evaluation(self, model_path: str, scenario=None) -> int:
        """
        Run MLC/WebLLM browser-based evaluation.

        Args:
            model_path: Path to MLC model directory
            scenario: Selected scenario info (optional, tests selected in browser)

        Returns:
            Exit code (0 = success)
        """
        from Evaluator.mlc_eval_handler import run_mlc_evaluation

        print_info("Launching WebLLM browser-based evaluation...")
        print_info(f"Model: {Path(model_path).name}")
        print_info("Select tests in browser, then click 'Run Evaluation'")
        print()

        config_dir = self.repo_root / "Evaluator" / "config"

        return run_mlc_evaluation(
            model_path=model_path,
            config_dir=str(config_dir),
            port=8080,
            open_browser=True,
        )
