"""
SynthChat handler - unified data generation and improvement.

Location: tuner/handlers/synthchat_handler.py
Purpose: Submenu for synthetic data operations (generate + improve)
Used by: Main menu when 'synthchat' is selected

This handler provides a unified entry point for SynthChat operations:
1. Generate New - Create examples from scenarios using LLM
2. Improve Existing - Refine datasets with rubrics

It loads defaults from SynthChat/config/defaults.yaml and allows users
to either use defaults or customize settings before proceeding.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from tuner.handlers.base import BaseHandler
from shared.ui import (
    print_menu,
    print_header,
    print_config,
    print_info,
    print_error,
    print_success,
    print_checkbox_menu,
    confirm,
    prompt,
    spinner,
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
        Load defaults from SynthChat/config/defaults.yaml.

        Returns:
            Dictionary with configuration defaults, or empty dict if not found
        """
        defaults_path = self.repo_root / "SynthChat" / "config" / "defaults.yaml"
        if defaults_path.exists():
            try:
                import yaml
                with open(defaults_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print_error(f"Error loading defaults: {e}")
                return {}
        return {}

    def _list_scenarios(self) -> List[str]:
        """List available scenarios from SynthChat/scenarios/."""
        scenarios_dir = self.repo_root / "SynthChat" / "scenarios"
        if not scenarios_dir.exists():
            return []
        return [f.stem for f in scenarios_dir.glob("*.yaml") if not f.stem.startswith("_")]

    def _list_rubrics(self) -> List[str]:
        """List available rubrics from SynthChat/rubrics/."""
        rubrics_dir = self.repo_root / "SynthChat" / "rubrics"
        if not rubrics_dir.exists():
            return []
        return [
            f.stem for f in rubrics_dir.glob("*.yaml")
            if not f.stem.startswith("_") and f.stem not in ("rubric", "README")
            and "example" not in f.stem
        ]

    def _select_backend(self, default: str = "lmstudio") -> Optional[str]:
        """Select LLM backend."""
        return print_menu([
            ("lmstudio", f"{BOX['bullet']} LM Studio (local)"),
            ("ollama", f"{BOX['bullet']} Ollama (local)"),
            ("openrouter", f"{BOX['bullet']} OpenRouter (cloud)"),
        ], f"Select backend (default: {default})") or default

    def _select_model(self, backend: str) -> Optional[str]:
        """Select model from available models on the backend."""
        try:
            from shared.llm import create_client

            with spinner(f"Connecting to {backend}..."):
                client = create_client(provider=backend, model="temp")
                models = client.list_models()

            if not models:
                print_info(f"No models found on {backend}")
                return None

            # Build menu options
            options = [(m, f"{BOX['bullet']} {m}") for m in models[:20]]  # Limit to 20
            if len(models) > 20:
                print_info(f"Showing first 20 of {len(models)} models")

            return print_menu(options, "Select model:")

        except Exception as e:
            print_error(f"Could not list models: {e}")
            return None

    def _show_generate_defaults(self, config: Dict[str, Any]) -> None:
        """Display generate defaults in a config panel."""
        gen = config.get("generate", {})
        shared = config.get("shared", {})

        display = {
            "Backend": gen.get("backend", "lmstudio"),
            "Model": gen.get("model") or "(auto-detect)",
            "Temperature": str(gen.get("temperature", 0.7)),
            "Scenarios": ", ".join(gen.get("scenarios", ["behaviors", "tools"])),
            "Rubrics": ", ".join(gen.get("rubrics", ["context_alignment", "response_quality"])),
            "Examples/Scenario": str(gen.get("examples_per_scenario", 10)),
            "Max Iterations": str(gen.get("max_iterations", 3)),
            "Output Dir": gen.get("output_dir", "SynthChat/output"),
        }

        print_config(display, "Generate Configuration")

    def _show_improve_defaults(self, config: Dict[str, Any]) -> None:
        """Display improve defaults in a config panel."""
        imp = config.get("improve", {})
        shared = config.get("shared", {})

        display = {
            "Backend": imp.get("backend", "lmstudio"),
            "Model": imp.get("model") or "(auto-detect)",
            "Temperature": str(imp.get("temperature", 0.3)),
            "Rubric Mode": imp.get("rubric_mode", "auto"),
            "Fallback Rubrics": ", ".join(imp.get("fallback_rubrics", ["response_quality"])),
            "Max Iterations": str(imp.get("max_iterations", 3)),
            "Batch Size": str(imp.get("batch_size", 1)),
            "Log Interactions": "Yes" if shared.get("log_interactions", True) else "No",
        }

        print_config(display, "Improve Configuration")

    def _run_generation(
        self,
        backend: str,
        model: Optional[str],
        scenarios: List[str],
        rubrics: List[str],
        examples_per_scenario: int,
        max_iterations: int,
        output_dir: str,
    ) -> int:
        """Run the actual generation with the given configuration."""
        try:
            from shared.llm import create_client
            from SynthChat.generator import SynthChatGenerator
            from SynthChat.engine import ImprovementEngine
            from shared.ui import LiveSynthChatDashboard
            import json

            # Create output path
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

            # Connect to backend
            print_info(f"Connecting to {backend}...")
            client = create_client(provider=backend, model=model or "auto")

            # Setup paths
            config_dir = self.repo_root / "SynthChat" / "config"
            scenarios_dir = self.repo_root / "SynthChat" / "scenarios"
            rubrics_dir = self.repo_root / "SynthChat" / "rubrics"

            # Create improvement engine for validation
            engine = ImprovementEngine(
                llm_client=client,
                rubrics_dir=rubrics_dir,
                enable_interactions=True
            )

            # Create generator
            generator = SynthChatGenerator(
                config_dir=config_dir,
                scenarios_dir=scenarios_dir,
                rubrics_dir=rubrics_dir,
                llm_client=client,
                engine=engine,
                enable_stage_validation=True
            )

            # Build targets
            targets = {s: examples_per_scenario for s in scenarios}
            total_examples = sum(targets.values())

            print_info(f"Generating {total_examples} examples across {len(scenarios)} scenarios...")
            print()

            # Generate with dashboard
            valid_count = 0
            invalid_count = 0

            with LiveSynthChatDashboard(
                title="SynthChat Generation",
                total_examples=total_examples,
                log_lines=5,
            ) as dashboard:
                for scenario_key, count in targets.items():
                    scenario = generator.scenario_loader.get_scenario(scenario_key)
                    if not scenario:
                        print_error(f"Scenario not found: {scenario_key}")
                        continue

                    for i in range(count):
                        dashboard.update(category=scenario_key, is_current=True)

                        result = generator._generate_single(
                            scenario_key=scenario_key,
                            scenario=scenario,
                            max_iterations=max_iterations,
                            randomize_params=True,
                            doc_context=None
                        )

                        if result.success:
                            valid_count += 1
                            dashboard.update(status="valid", category=scenario_key)

                            with open(output_file, "a") as f:
                                f.write(json.dumps(result.example) + "\n")
                        else:
                            invalid_count += 1
                            dashboard.update(
                                status="invalid",
                                category=scenario_key,
                                reason=", ".join(result.stage_failures) if result.stage_failures else "validation failed"
                            )

            # Summary
            success_rate = (valid_count / total_examples * 100) if total_examples > 0 else 0

            print()
            print_success("Generation complete!")
            print_info(f"Valid: {valid_count} | Invalid: {invalid_count} | Rate: {success_rate:.1f}%")
            print_info(f"Output: {output_file}")

            return 0

        except KeyboardInterrupt:
            print_error("\nInterrupted by user")
            return 1
        except Exception as e:
            print_error(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _run_improvement(
        self,
        backend: str,
        model: Optional[str],
        input_file: Path,
        output_file: Path,
        rubrics: List[str],
        max_iterations: int,
        start_line: int,
        end_line: int,
    ) -> int:
        """Run the actual improvement with the given configuration."""
        try:
            from SynthChat.services.rubric_runner import RubricRunner

            rubrics_dir = self.repo_root / "SynthChat" / "rubrics"

            runner = RubricRunner(
                rubrics_dir=rubrics_dir,
                backend=backend,
                host=None,
                port=None
            )

            print_info(f"Improving {input_file.name} with {len(rubrics)} rubrics...")
            print()

            runner.run_on_file(
                file_path=input_file,
                output_path=output_file,
                rubric_keys=rubrics,
                start_line=start_line,
                end_line=end_line,
                max_iterations=max_iterations
            )

            print()
            print_success(f"Improvement complete! Output: {output_file}")
            return 0

        except KeyboardInterrupt:
            print_error("\nInterrupted by user")
            return 1
        except Exception as e:
            print_error(f"Improvement error: {e}")
            return 1

    def _handle_generate(self, config: Dict[str, Any]) -> int:
        """Handle generate flow with defaults or customization."""
        gen_config = config.get("generate", {})

        print_header("SYNTHCHAT GENERATE", "Create new training examples")
        self._show_generate_defaults(config)
        print()

        choice = print_menu([
            ("defaults", f"{BOX['star']} Use defaults - Start with settings above"),
            ("customize", f"{BOX['bullet']} Customize - Select scenarios, rubrics, backend"),
        ], "How would you like to proceed?")

        if not choice:
            return 0

        # Get configuration (from defaults or customization)
        if choice == "defaults":
            backend = gen_config.get("backend", "lmstudio")
            model = gen_config.get("model")
            scenarios = gen_config.get("scenarios", ["behaviors", "tools"])
            rubrics = gen_config.get("rubrics", ["context_alignment", "response_quality"])
            examples_per_scenario = gen_config.get("examples_per_scenario", 10)
            max_iterations = gen_config.get("max_iterations", 3)
            output_dir = gen_config.get("output_dir", "SynthChat/output")

        else:  # customize
            print()

            # 1. Select backend
            backend = self._select_backend(gen_config.get("backend", "lmstudio"))
            if not backend:
                return 0

            # 2. Select model
            model = self._select_model(backend)
            # model can be None (auto-detect)

            # 3. Select scenarios (checkbox)
            available_scenarios = self._list_scenarios()
            default_scenarios = gen_config.get("scenarios", ["behaviors", "tools"])
            scenario_options = [
                (s, s, s in default_scenarios) for s in available_scenarios
            ]

            print()
            scenarios = print_checkbox_menu(
                scenario_options,
                title="Select scenarios to generate from:",
                min_select=1
            )
            if not scenarios:
                print_error("No scenarios selected")
                return 0

            # 4. Select rubrics (checkbox)
            available_rubrics = self._list_rubrics()
            default_rubrics = gen_config.get("rubrics", ["context_alignment", "response_quality"])
            rubric_options = [
                (r, r, r in default_rubrics) for r in available_rubrics
            ]

            print()
            rubrics = print_checkbox_menu(
                rubric_options,
                title="Select validation rubrics:",
                min_select=0  # Can skip validation
            )

            # 5. Examples per scenario
            print()
            default_examples = gen_config.get("examples_per_scenario", 10)
            examples_str = prompt(f"Examples per scenario (default: {default_examples})")
            examples_per_scenario = int(examples_str) if examples_str.strip() else default_examples

            # 6. Max iterations
            default_iters = gen_config.get("max_iterations", 3)
            iters_str = prompt(f"Max improvement iterations (default: {default_iters})")
            max_iterations = int(iters_str) if iters_str.strip() else default_iters

            # 7. Output directory
            output_dir = gen_config.get("output_dir", "SynthChat/output")

        # Show final config and confirm
        print()
        final_config = {
            "Backend": backend,
            "Model": model or "(auto-detect)",
            "Scenarios": ", ".join(scenarios),
            "Rubrics": ", ".join(rubrics) if rubrics else "(none)",
            "Examples/Scenario": str(examples_per_scenario),
            "Total Examples": str(len(scenarios) * examples_per_scenario),
            "Max Iterations": str(max_iterations),
            "Output": output_dir,
        }
        print_config(final_config, "Final Configuration")

        if not confirm("\nStart generation?"):
            print_info("Cancelled")
            return 0

        print()
        return self._run_generation(
            backend=backend,
            model=model,
            scenarios=scenarios,
            rubrics=rubrics,
            examples_per_scenario=examples_per_scenario,
            max_iterations=max_iterations,
            output_dir=output_dir,
        )

    def _handle_improve(self, config: Dict[str, Any]) -> int:
        """Handle improve flow with defaults or customization."""
        imp_config = config.get("improve", {})

        print_header("SYNTHCHAT IMPROVE", "Refine datasets with rubrics")
        self._show_improve_defaults(config)
        print()

        choice = print_menu([
            ("defaults", f"{BOX['star']} Use defaults - Start with settings above"),
            ("customize", f"{BOX['bullet']} Customize - Select dataset, rubrics, backend"),
        ], "How would you like to proceed?")

        if not choice:
            return 0

        # First, select the dataset (required for both flows)
        print()
        print_info("Step 1: Select dataset to improve")

        try:
            from SynthChat.utils.dataset_scanner import DatasetScanner
            scanner = DatasetScanner()

            with spinner("Scanning datasets..."):
                datasets = scanner.scan()

            if not datasets:
                print_error("No datasets found")
                return 0

            # Select category
            categories = list(datasets.keys())
            cat_options = [(c, f"{BOX['bullet']} {c} ({len(datasets[c])} datasets)") for c in categories]
            category = print_menu(cat_options, "Select category:")
            if not category:
                return 0

            # Select dataset
            agents = sorted(datasets[category].keys())
            agent_options = [(a, f"{BOX['bullet']} {a} ({len(datasets[category][a])} versions)") for a in agents]
            agent = print_menu(agent_options, "Select dataset:")
            if not agent:
                return 0

            # Select version
            versions = datasets[category][agent]
            version_options = []
            for v in versions:
                file_path = scanner.get_file_path(category, agent, v)
                count = scanner.count_examples(file_path)
                version_options.append((str(v), f"{BOX['bullet']} v{v} ({count} examples)"))
            version = print_menu(version_options, "Select version:")
            if not version:
                return 0

            input_file = scanner.get_file_path(category, agent, version)
            next_version = scanner.get_next_version(version)
            output_file = scanner.get_file_path(category, agent, next_version)
            total_lines = scanner.count_examples(input_file)

        except Exception as e:
            print_error(f"Error scanning datasets: {e}")
            return 0

        # Get configuration based on choice
        if choice == "defaults":
            backend = imp_config.get("backend", "lmstudio")
            model = imp_config.get("model")
            max_iterations = imp_config.get("max_iterations", 3)
            start_line = 1
            end_line = total_lines

            # Auto-detect rubrics based on dataset name
            rubric_mode = imp_config.get("rubric_mode", "auto")
            if rubric_mode == "auto":
                # Try to match rubric to agent name
                available_rubrics = self._list_rubrics()
                matching = [r for r in available_rubrics if agent.lower() in r.lower()]
                if matching:
                    rubrics = matching
                else:
                    rubrics = imp_config.get("fallback_rubrics", ["response_quality", "context_alignment"])
            else:
                rubrics = imp_config.get("fallback_rubrics", ["response_quality"])

        else:  # customize
            print()
            print_info("Step 2: Configure improvement settings")

            # 1. Select backend
            backend = self._select_backend(imp_config.get("backend", "lmstudio"))
            if not backend:
                return 0

            # 2. Select model
            model = self._select_model(backend)

            # 3. Select rubrics (checkbox)
            available_rubrics = self._list_rubrics()

            # Pre-select rubrics matching dataset name
            matching = [r for r in available_rubrics if agent.lower() in r.lower()]
            fallback = imp_config.get("fallback_rubrics", ["response_quality", "context_alignment"])
            defaults = matching if matching else fallback

            rubric_options = [
                (r, r, r in defaults) for r in available_rubrics
            ]

            print()
            rubrics = print_checkbox_menu(
                rubric_options,
                title="Select rubrics to apply:",
                min_select=1
            )
            if not rubrics:
                print_error("No rubrics selected")
                return 0

            # 4. Max iterations
            print()
            default_iters = imp_config.get("max_iterations", 3)
            iters_str = prompt(f"Max iterations per example (default: {default_iters})")
            max_iterations = int(iters_str) if iters_str.strip() else default_iters

            # 5. Line range
            start_str = prompt(f"Start line (default: 1)")
            start_line = int(start_str) if start_str.strip() else 1

            end_str = prompt(f"End line (default: {total_lines})")
            end_line = int(end_str) if end_str.strip() else total_lines

        # Show final config and confirm
        print()
        final_config = {
            "Input": f"{category}/{agent} v{version} ({total_lines} examples)",
            "Output": f"{category}/{agent} v{next_version}",
            "Backend": backend,
            "Model": model or "(auto-detect)",
            "Rubrics": ", ".join(rubrics),
            "Max Iterations": str(max_iterations),
            "Range": f"Lines {start_line}-{end_line}",
        }
        print_config(final_config, "Final Configuration")

        if not confirm("\nStart improvement?"):
            print_info("Cancelled")
            return 0

        print()
        return self._run_improvement(
            backend=backend,
            model=model,
            input_file=input_file,
            output_file=output_file,
            rubrics=rubrics,
            max_iterations=max_iterations,
            start_line=start_line,
            end_line=end_line,
        )

    def handle(self) -> int:
        """Main submenu loop."""
        config = self._load_defaults()

        while True:
            print_header("SYNTHCHAT", "Data generation and improvement")

            menu_options = [
                ("generate", f"{BOX['star']} Generate New - Create examples from scenarios"),
                ("improve", f"{BOX['bullet']} Improve Existing - Refine datasets with rubrics"),
            ]

            choice = print_menu(menu_options, "What would you like to do?")

            if not choice:
                return 0

            if choice == "generate":
                self._handle_generate(config)
                print()

            elif choice == "improve":
                self._handle_improve(config)
                print()

        return 0
