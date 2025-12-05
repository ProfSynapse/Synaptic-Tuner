"""
Generate handler for synthetic data generation (SelfPlay).

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/generate_handler.py
Purpose: Orchestrate SelfPlay generation workflow
Used by: Router when handling 'generate' command or main menu selection

This handler implements the generation workflow:
1. Check LM Studio connection
2. Select generation mode (random, targeted, or custom)
3. For targeted mode: select specific tools/behaviors and quantities
4. Configure output settings
5. Execute generation
6. Optionally split into individual dataset files
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

from tuner.handlers.base import BaseHandler

# Import shared UI components
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
)

# Add SelfPlay to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "SelfPlay"))


class GenerateHandler(BaseHandler):
    """
    Handler for synthetic data generation workflow.

    Coordinates LM Studio connection, generation configuration,
    and execution of SelfPlay pipeline.

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
            LMStudioClient if successful, None otherwise
        """
        try:
            from Evaluator.lmstudio_client import LMStudioClient
            from Evaluator.config import LMStudioSettings

            print_info("Checking LM Studio connection...")

            settings = LMStudioSettings(model="local-model")
            client = LMStudioClient(settings=settings)

            if not client.is_server_running():
                print_error("LM Studio server not accessible!")
                print("\nMake sure:")
                print("  1. LM Studio is running")
                print("  2. A model is loaded")
                print("  3. Server is started (click 'Start Server')")
                print(f"  4. Server is accessible at {settings.base_url()}")
                return None

            print_info("✓ LM Studio connected!")

            # Try to list models
            try:
                models = client.list_models()
                if models:
                    print(f"  Available models: {', '.join(models[:3])}")
            except:
                pass

            return client

        except ImportError as e:
            print_error(f"Missing dependencies: {e}")
            return None
        except Exception as e:
            print_error(f"Connection error: {e}")
            return None

    def _list_all_categories(self) -> Dict[str, List[str]]:
        """
        List all available categories (tools and behaviors).

        Returns:
            Dict with 'tools' and 'behaviors' lists
        """
        try:
            from SelfPlay.generator import SelfPlayGenerator

            generator = SelfPlayGenerator()

            # Collect all tools
            tools = []
            for agent_name, agent_config in generator.agents_config["agents"].items():
                for tool in agent_config["tools"]:
                    tools.append(tool)

            # Collect all behaviors
            behaviors = list(generator.behaviors_config["behaviors"].keys())

            return {
                "tools": sorted(tools),
                "behaviors": sorted(behaviors)
            }
        except Exception as e:
            print_error(f"Error loading categories: {e}")
            return {"tools": [], "behaviors": []}

    def _configure_targeted_generation(self, categories: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Interactive configuration for targeted generation.

        Args:
            categories: Dict with 'tools' and 'behaviors' lists

        Returns:
            Dict mapping category to count
        """
        targets = {}

        print_header("TARGETED GENERATION", "Select specific tools/behaviors")

        # Ask what type to generate
        type_choice = print_menu([
            ("Tools only", "Generate for specific tools"),
            ("Behaviors only", "Generate for specific behaviors"),
            ("Mixed", "Generate for both tools and behaviors")
        ], prompt="What do you want to generate?")

        if type_choice in [1, 3]:  # Tools or Mixed
            print("\n" + "=" * 70)
            print("TOOL SELECTION")
            print("=" * 70)
            print("\nEnter tool names (comma-separated) or type 'all' for all tools:")
            print("Available tools:")
            for i, tool in enumerate(categories["tools"], 1):
                print(f"  {tool}")
                if i % 10 == 0:
                    print()

            tool_input = prompt("\nTools").strip()

            if tool_input.lower() == "all":
                selected_tools = categories["tools"]
            else:
                selected_tools = [t.strip() for t in tool_input.split(",") if t.strip()]

            if selected_tools:
                count_str = prompt(f"\nExamples per tool (default: 10)")
                count = int(count_str) if count_str.strip() else 10

                for tool in selected_tools:
                    if tool in categories["tools"]:
                        targets[tool] = count
                    else:
                        print_error(f"Unknown tool: {tool}")

        if type_choice in [2, 3]:  # Behaviors or Mixed
            print("\n" + "=" * 70)
            print("BEHAVIOR SELECTION")
            print("=" * 70)
            print("\nAvailable behaviors:")
            for i, behavior in enumerate(categories["behaviors"], 1):
                print(f"  {i}. {behavior}")

            behavior_input = prompt("\nBehaviors (comma-separated numbers or 'all')").strip()

            if behavior_input.lower() == "all":
                selected_behaviors = categories["behaviors"]
            else:
                indices = [int(i.strip()) - 1 for i in behavior_input.split(",") if i.strip().isdigit()]
                selected_behaviors = [categories["behaviors"][i] for i in indices
                                     if 0 <= i < len(categories["behaviors"])]

            if selected_behaviors:
                count_str = prompt(f"\nExamples per behavior (default: 10)")
                count = int(count_str) if count_str.strip() else 10

                for behavior in selected_behaviors:
                    targets[behavior] = count

        return targets

    def handle(self) -> int:
        """
        Execute generation workflow.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        print_header("SYNTHETIC DATA GENERATION", "SelfPlay pipeline")

        # Step 1: Check LM Studio connection
        client = self._check_lmstudio_connection()
        if not client:
            return 1

        # Step 2: Select generation mode
        mode_choice = print_menu([
            ("Quick test", "Generate 10 random examples (test run)"),
            ("Random batch", "Generate N random examples"),
            ("Targeted generation", "Generate specific numbers for selected tools/behaviors"),
        ], prompt="Select generation mode")

        # Determine output file
        output_base = self.repo_root / "SelfPlay" / "selfplay_output.jsonl"
        output_file = prompt(f"\nOutput file (default: {output_base})").strip()
        if not output_file:
            output_file = str(output_base)

        output_path = Path(output_file)

        # Configure based on mode
        if mode_choice == 1:  # Quick test
            num_examples = 10
            targets = None

        elif mode_choice == 2:  # Random batch
            num_str = prompt("\nNumber of examples (default: 100)").strip()
            num_examples = int(num_str) if num_str else 100
            targets = None

        else:  # Targeted generation
            categories = self._list_all_categories()
            targets = self._configure_targeted_generation(categories)

            if not targets:
                print_error("No targets selected!")
                return 1

            num_examples = sum(targets.values())

        # Display configuration
        config_items = [
            ("LM Studio", "Connected"),
            ("Output file", str(output_path)),
        ]

        if targets:
            config_items.append(("Mode", "Targeted generation"))
            config_items.append(("Categories", str(len(targets))))
            config_items.append(("Total examples", str(num_examples)))
        else:
            config_items.append(("Mode", "Random generation"))
            config_items.append(("Examples", str(num_examples)))

        print_config(config_items)

        # Confirm
        if not confirm("\nProceed with generation?"):
            print_info("Cancelled")
            return 0

        # Execute generation
        try:
            from SelfPlay.generator import SelfPlayGenerator

            print_header("GENERATING", f"{num_examples} examples")

            generator = SelfPlayGenerator(
                model_client=client,
                randomize_params=True
            )

            # Clear output file if exists
            if output_path.exists():
                output_path.unlink()

            # Generate
            if targets:
                results = generator.generate_targeted_batch(
                    targets=targets,
                    output_file=output_path,
                    validate=True,
                    save_invalid=False
                )
            else:
                results = generator.generate_batch(
                    num_examples=num_examples,
                    output_file=output_path,
                    validate=True,
                    save_invalid=False
                )

            # Summary
            valid_count = len(results["valid"])
            invalid_count = len(results["invalid"])
            success_rate = (valid_count / num_examples * 100) if num_examples > 0 else 0

            print("\n" + "=" * 70)
            print_info("✓ GENERATION COMPLETE!")
            print("=" * 70)
            print(f"\n📊 Results:")
            print(f"   Valid examples: {valid_count}")
            print(f"   Invalid examples: {invalid_count}")
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"\n💾 Output: {output_path}")

            # Ask about splitting
            if targets and confirm("\nSplit into individual dataset files?"):
                split_script = self.repo_root / "Tools" / "split_selfplay_dataset.py"

                if split_script.exists():
                    import subprocess

                    datasets_dir = self.repo_root / "Datasets"

                    print_info(f"Splitting into dataset folders...")

                    result = subprocess.run([
                        sys.executable,
                        str(split_script),
                        str(output_path),
                        "--datasets-dir", str(datasets_dir)
                    ], capture_output=True, text=True)

                    if result.returncode == 0:
                        print_info("✓ Split complete!")
                        print("\nFiles created in:")
                        print(f"  Tools: {datasets_dir / 'tools_datasets'}")
                        print(f"  Behaviors: {datasets_dir / 'behavior_datasets'}")
                    else:
                        print_error(f"Split failed: {result.stderr}")
                else:
                    print_error("Split script not found")

            return 0

        except KeyboardInterrupt:
            print_error("\n\nInterrupted by user")
            return 1
        except Exception as e:
            print_error(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return 1
