"""
Generate handler for synthetic data generation (SynthChat).

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/generate_handler.py
Purpose: Orchestrate SynthChat generation workflow
Used by: Router when handling 'generate' command or main menu selection

This handler implements the generation workflow:
1. Check LM Studio connection
2. Select generation mode (random, targeted, or custom)
3. For targeted mode: select specific scenarios and quantities
4. Configure output settings
5. Execute generation with live dashboard
6. Optionally split into individual dataset files
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from tuner.handlers.base import BaseHandler

# Import shared UI components
from shared.ui import (
    print_header,
    print_menu,
    print_config,
    print_info,
    print_error,
    print_success,
    confirm,
    prompt,
    console,
    RICH_AVAILABLE,
    COLORS,
    spinner,
    BOX,
    LiveSynthChatDashboard,
)

# Add SynthChat to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class GenerateHandler(BaseHandler):
    """
    Handler for synthetic data generation workflow.

    Coordinates LM Studio connection, generation configuration,
    and execution of SynthChat pipeline.

    Example:
        handler = GenerateHandler()
        exit_code = handler.handle()
        # User interacts with menus, configures generation
        # Returns 0 on success, non-zero on failure
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "generate"

    def can_handle_direct_mode(self) -> bool:
        """This handler supports direct CLI invocation."""
        return True

    def _check_lmstudio_connection(self) -> Optional[object]:
        """
        Check if LM Studio is accessible.

        Returns:
            LLM client if successful, None otherwise
        """
        try:
            from shared.llm import create_client

            print_info("Checking LM Studio connection...")

            # Create LM Studio client using shared.llm
            client = create_client(config_defaults={
                "provider": "lmstudio",
                "model": os.getenv("LMSTUDIO_MODEL", "local-model"),
            })

            # Test connection with a simple request
            try:
                test_response = client.chat(
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                print_info("✓ LM Studio connected!")
                return client
            except Exception as e:
                print_error(f"LM Studio server not accessible: {e}")
                print("\nMake sure:")
                print("  1. LM Studio is running")
                print("  2. A model is loaded")
                print("  3. Server is started (click 'Start Server')")
                print(f"  4. Server is accessible at localhost:1234")
                return None

        except ImportError as e:
            print_error(f"Missing dependencies: {e}")
            return None
        except Exception as e:
            print_error(f"Connection error: {e}")
            return None

    def _list_all_scenarios(self) -> Dict[str, Dict]:
        """
        List all available scenarios from SynthChat.

        Returns:
            Dict mapping scenario keys to their configs
        """
        try:
            from SynthChat.generator import ScenarioLoader

            scenarios_dir = self.repo_root / "SynthChat" / "scenarios"
            loader = ScenarioLoader(scenarios_dir)

            return {key: loader.get_scenario(key) for key in loader.list_scenarios()}
        except Exception as e:
            print_error(f"Error loading scenarios: {e}")
            return {}

    def _configure_targeted_generation(self, scenarios: Dict[str, Dict]) -> Dict[str, int]:
        """
        Interactive configuration for targeted generation.

        Args:
            scenarios: Dict mapping scenario keys to configs

        Returns:
            Dict mapping scenario key to count
        """
        targets = {}

        print_header("TARGETED GENERATION", "Select specific scenarios")

        # Group scenarios by type
        by_type = {}
        for key, scenario in scenarios.items():
            stype = scenario.get("type", "other") if scenario else "other"
            if stype not in by_type:
                by_type[stype] = []
            by_type[stype].append(key)

        # Ask what type to generate
        type_choices = list(by_type.keys())
        print("\nAvailable scenario types:")
        for i, stype in enumerate(type_choices, 1):
            print(f"  {i}. {stype} ({len(by_type[stype])} scenarios)")

        type_input = prompt("\nSelect type(s) (comma-separated numbers or 'all')").strip()

        if type_input.lower() == "all":
            selected_types = type_choices
        else:
            indices = [int(i.strip()) - 1 for i in type_input.split(",") if i.strip().isdigit()]
            selected_types = [type_choices[i] for i in indices if 0 <= i < len(type_choices)]

        # For each selected type, choose scenarios
        for stype in selected_types:
            print(f"\n=== {stype.upper()} SCENARIOS ===")
            available = sorted(by_type[stype])

            for i, key in enumerate(available, 1):
                print(f"  {i}. {key}")

            scenario_input = prompt(f"\nSelect scenarios (comma-separated numbers or 'all')").strip()

            if scenario_input.lower() == "all":
                selected = available
            else:
                indices = [int(i.strip()) - 1 for i in scenario_input.split(",") if i.strip().isdigit()]
                selected = [available[i] for i in indices if 0 <= i < len(available)]

            if selected:
                count_str = prompt(f"Examples per scenario (default: 10)")
                count = int(count_str) if count_str.strip() else 10

                for key in selected:
                    targets[key] = count

        return targets

    def handle(self) -> int:
        """
        Execute generation workflow.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        print_header("SYNTHETIC DATA GENERATION", "SynthChat pipeline")

        # Step 1: Check LM Studio connection
        client = self._check_lmstudio_connection()
        if not client:
            return 1

        # Step 2: Load available scenarios
        scenarios = self._list_all_scenarios()
        if not scenarios:
            print_error("No scenarios found in SynthChat/scenarios/")
            return 1

        print_info(f"Found {len(scenarios)} scenarios")

        # Step 3: Select generation mode
        mode_choice = print_menu([
            ("Quick test", "Generate 10 random examples (test run)"),
            ("Random batch", "Generate N random examples"),
            ("Targeted generation", "Generate specific numbers for selected scenarios"),
        ], prompt="Select generation mode")

        # Determine output file
        output_base = self.repo_root / "SynthChat" / "output" / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        output_file = prompt(f"\nOutput file (default: {output_base})").strip()
        if not output_file:
            output_file = str(output_base)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure based on mode
        if mode_choice == 1:  # Quick test
            # Random selection of scenarios
            import random
            scenario_keys = list(scenarios.keys())
            selected = random.sample(scenario_keys, min(5, len(scenario_keys)))
            targets = {key: 2 for key in selected}
            num_examples = sum(targets.values())

        elif mode_choice == 2:  # Random batch
            import random
            num_str = prompt("\nNumber of examples (default: 100)").strip()
            total = int(num_str) if num_str else 100

            # Distribute across all scenarios
            scenario_keys = list(scenarios.keys())
            per_scenario = max(1, total // len(scenario_keys))
            targets = {key: per_scenario for key in scenario_keys}

            # Adjust to hit total
            current = sum(targets.values())
            if current < total:
                for key in scenario_keys[:total - current]:
                    targets[key] += 1

            num_examples = sum(targets.values())

        else:  # Targeted generation
            targets = self._configure_targeted_generation(scenarios)

            if not targets:
                print_error("No targets selected!")
                return 1

            num_examples = sum(targets.values())

        # Display configuration
        config_dict = {
            "LM Studio": "Connected",
            "Output file": str(output_path),
            "Mode": "Targeted" if mode_choice == 3 else "Random",
            "Scenarios": str(len(targets)),
            "Total examples": str(num_examples),
        }

        print_config(config_dict, "Generation Configuration")

        # Confirm
        if not confirm("\nProceed with generation?"):
            print_info("Cancelled")
            return 0

        # Execute generation with dashboard
        try:
            from SynthChat.generator import SynthChatGenerator
            from SynthChat.engine import ImprovementEngine
            from shared.llm import create_client

            # Setup paths
            config_dir = self.repo_root / "SynthChat" / "config"
            scenarios_dir = self.repo_root / "SynthChat" / "scenarios"
            rubrics_dir = self.repo_root / "SynthChat" / "rubrics"

            print_info(f"Starting generation of {num_examples} examples...")
            print()

            # Create improvement engine (for stage validation)
            validation_config = config_dir / "validation.yaml"
            engine = ImprovementEngine(
                llm_client=client,
                rubrics_dir=rubrics_dir,
                config_path=validation_config if validation_config.exists() else None,
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

            # Clear output file if exists
            if output_path.exists():
                output_path.unlink()

            # Generate with live dashboard
            valid_count = 0
            invalid_count = 0

            with LiveSynthChatDashboard(
                title="SynthChat Generation",
                total_examples=num_examples,
                log_lines=5,
            ) as dashboard:
                for scenario_key, count in targets.items():
                    scenario = generator.scenario_loader.get_scenario(scenario_key)
                    if not scenario:
                        continue

                    for i in range(count):
                        # Update dashboard with current generation
                        dashboard.update(
                            category=scenario_key,
                            is_current=True
                        )

                        # Generate single example
                        result = generator._generate_single(
                            scenario_key=scenario_key,
                            scenario=scenario,
                            max_iterations=3,
                            randomize_params=True,
                            doc_context=None
                        )

                        # Update dashboard with result
                        if result.success:
                            valid_count += 1
                            dashboard.update(
                                status="valid",
                                category=scenario_key
                            )

                            # Save to output file
                            with open(output_path, "a") as f:
                                f.write(json.dumps(result.example) + "\n")
                        else:
                            invalid_count += 1
                            dashboard.update(
                                status="invalid",
                                category=scenario_key,
                                reason=", ".join(result.stage_failures) if result.stage_failures else "validation failed"
                            )

            # Summary
            success_rate = (valid_count / num_examples * 100) if num_examples > 0 else 0

            print()
            print_success("Generation complete!")
            print_info(f"Valid examples: {valid_count}")
            print_info(f"Invalid examples: {invalid_count}")
            print_info(f"Success rate: {success_rate:.1f}%")
            print_info(f"Output: {output_path}")

            # Ask about splitting
            if confirm("\nSplit into individual dataset files?"):
                split_script = self.repo_root / "Tools" / "split_synthchat_dataset.py"

                if split_script.exists():
                    import subprocess

                    datasets_dir = self.repo_root / "Datasets"

                    with spinner("Splitting into dataset folders..."):
                        result = subprocess.run([
                            sys.executable,
                            str(split_script),
                            str(output_path),
                            "--datasets-dir", str(datasets_dir)
                        ], capture_output=True, text=True)

                    if result.returncode == 0:
                        print_success("Split complete!")
                        print_info(f"Tools: {datasets_dir / 'tools_datasets'}")
                        print_info(f"Behaviors: {datasets_dir / 'behavior_datasets'}")
                    else:
                        print_error(f"Split failed: {result.stderr}")
                else:
                    print_info("Split script not found - output saved as single file")

            return 0

        except KeyboardInterrupt:
            print_error("\n\nInterrupted by user")
            return 1
        except Exception as e:
            print_error(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return 1
