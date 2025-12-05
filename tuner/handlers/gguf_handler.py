"""
Model conversion handler.

Location: tuner/handlers/gguf_handler.py
Purpose: Handle model conversion workflow via interactive menu
Used by: CLI router (cli/router.py)

Supports:
- GGUF: llama.cpp/Ollama format for local inference
- WebGPU: MLC-LLM format for browser deployment via WebLLM
"""

import sys
from pathlib import Path
from typing import Optional, List

from .base import BaseHandler


class GGUFHandler(BaseHandler):
    """
    Handler for model conversion operations.

    Provides interactive menu for converting trained models to:
    - GGUF format (llama.cpp/Ollama)
    - WebGPU format (browser deployment via WebLLM)

    Supports both text and Vision-Language models.
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "gguf"

    def can_handle_direct_mode(self) -> bool:
        """This handler supports direct CLI invocation."""
        return True

    def __init__(self):
        super().__init__()

    def handle(self) -> int:
        """
        Execute model conversion workflow.

        Returns:
            int: Exit code (0 = success, non-zero = error)
        """
        try:
            from tuner.ui import console, RICH_AVAILABLE
            from tuner.discovery.training_runs import TrainingRunDiscovery

            if RICH_AVAILABLE:
                from rich.prompt import Prompt, Confirm
                from rich.table import Table
                from rich.panel import Panel
            else:
                Prompt = None
                Confirm = None

            # Header
            print("\n" + "=" * 60)
            print("MODEL CONVERSION")
            print("=" * 60)
            print("Convert trained models to deployment formats")
            print()

            # Select output format
            print("Available formats:")
            print("  1. GGUF - llama.cpp/Ollama (local inference)")
            print("  2. WebGPU - MLC-LLM/WebLLM (browser deployment)")
            print()

            if RICH_AVAILABLE and Prompt:
                format_choice = Prompt.ask(
                    "Select output format",
                    default="1",
                    choices=["1", "2"]
                )
            else:
                format_choice = input("Select output format [1]: ").strip() or "1"

            output_format = "gguf" if format_choice == "1" else "webgpu"
            print()

            # Find training runs using discovery service
            discovery = TrainingRunDiscovery(repo_root=self.repo_root)
            sft_paths = discovery.discover('sft', limit=10)
            kto_paths = discovery.discover('kto', limit=10)

            all_runs = []
            for path in sft_paths:
                all_runs.append({'path': path, 'type': 'SFT', 'timestamp': path.name})
            for path in kto_paths:
                all_runs.append({'path': path, 'type': 'KTO', 'timestamp': path.name})

            # Sort by timestamp (newest first)
            all_runs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            if not all_runs:
                print("No training runs found!")
                print("Train a model first with: ./run.sh train")
                return 1

            # Display available runs
            print("Available training runs:\n")

            if RICH_AVAILABLE:
                table = Table(show_header=True)
                table.add_column("#", style="cyan", width=3)
                table.add_column("Type", style="magenta", width=5)
                table.add_column("Run", style="green")
                table.add_column("Model", style="yellow")
                table.add_column("Has GGUF", style="blue")

                for i, run in enumerate(all_runs[:10], 1):  # Show last 10
                    has_gguf = "Yes" if (run['path'] / "gguf").exists() else "No"
                    # Try to get model name from config or path
                    model_name = run.get('model', run['path'].name)
                    table.add_row(
                        str(i),
                        run['type'],
                        run['path'].name,
                        model_name[:30] if len(model_name) > 30 else model_name,
                        has_gguf
                    )
                console.print(table)
            else:
                for i, run in enumerate(all_runs[:10], 1):
                    has_gguf = "Yes" if (run['path'] / "gguf").exists() else "No"
                    print(f"  {i}. [{run['type']}] {run['path'].name} (GGUF: {has_gguf})")

            # Select run
            print()
            if RICH_AVAILABLE and Prompt:
                selection = Prompt.ask(
                    "Select training run",
                    default="1",
                    choices=[str(i) for i in range(1, min(len(all_runs) + 1, 11))]
                )
            else:
                selection = input("Select training run [1]: ").strip() or "1"

            try:
                run_idx = int(selection) - 1
                selected_run = all_runs[run_idx]
            except (ValueError, IndexError):
                print("Invalid selection")
                return 1

            # Find final_model directory
            model_path = selected_run['path'] / "final_model"
            if not model_path.exists():
                # Try checkpoints
                checkpoints = selected_run['path'] / "checkpoints"
                if checkpoints.exists():
                    checkpoint_dirs = sorted(checkpoints.iterdir(), reverse=True)
                    if checkpoint_dirs:
                        model_path = checkpoint_dirs[0]

            if not model_path.exists():
                print(f"No model found in {selected_run['path']}")
                return 1

            print(f"\nSelected: {model_path}")

            # Get model name
            print()
            default_name = selected_run['path'].name
            format_label = "GGUF" if output_format == "gguf" else "WebGPU"
            if RICH_AVAILABLE and Prompt:
                model_name = Prompt.ask(
                    f"Model name for {format_label} files",
                    default=default_name
                )
            else:
                model_name = input(f"Model name for {format_label} files [{default_name}]: ").strip() or default_name

            # Select quantizations based on format
            print()
            if output_format == "gguf":
                print("Available quantizations (GGUF):")
                print("  1. Standard (Q4_K_M, Q5_K_M, Q8_0) - Recommended")
                print("  2. Minimal (Q4_K_M only) - Fastest")
                print("  3. Full (Q4_K_M, Q5_K_M, Q6_K, Q8_0) - All common quants")
                print("  4. Custom")
                print()

                if RICH_AVAILABLE and Prompt:
                    quant_choice = Prompt.ask(
                        "Select quantization preset",
                        default="1",
                        choices=["1", "2", "3", "4"]
                    )
                else:
                    quant_choice = input("Select quantization preset [1]: ").strip() or "1"

                quant_presets = {
                    "1": ["Q4_K_M", "Q5_K_M", "Q8_0"],
                    "2": ["Q4_K_M"],
                    "3": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
                }

                if quant_choice == "4":
                    if RICH_AVAILABLE and Prompt:
                        custom = Prompt.ask(
                            "Enter quantizations (comma-separated)",
                            default="Q4_K_M,Q5_K_M"
                        )
                    else:
                        custom = input("Enter quantizations (comma-separated) [Q4_K_M,Q5_K_M]: ").strip() or "Q4_K_M,Q5_K_M"
                    quantizations = [q.strip().upper() for q in custom.split(",")]
                else:
                    quantizations = quant_presets.get(quant_choice, quant_presets["1"])
            else:
                # WebGPU quantizations
                print("Available quantizations (WebGPU/MLC):")
                print("  1. q4f16_1 - 4-bit with float16 (recommended)")
                print("  2. q4f32_1 - 4-bit with float32")
                print("  3. q0f16 - No quantization, float16")
                print("  4. Custom")
                print()

                if RICH_AVAILABLE and Prompt:
                    quant_choice = Prompt.ask(
                        "Select quantization",
                        default="1",
                        choices=["1", "2", "3", "4"]
                    )
                else:
                    quant_choice = input("Select quantization [1]: ").strip() or "1"

                webgpu_quants = {
                    "1": ["q4f16_1"],
                    "2": ["q4f32_1"],
                    "3": ["q0f16"],
                }

                if quant_choice == "4":
                    if RICH_AVAILABLE and Prompt:
                        custom = Prompt.ask(
                            "Enter quantization",
                            default="q4f16_1"
                        )
                    else:
                        custom = input("Enter quantization [q4f16_1]: ").strip() or "q4f16_1"
                    quantizations = [custom.strip().lower()]
                else:
                    quantizations = webgpu_quants.get(quant_choice, webgpu_quants["1"])

            # Output directory
            output_dir = selected_run['path'] / model_name

            # Confirm
            print()
            print("=" * 60)
            print("CONVERSION SUMMARY")
            print("=" * 60)
            print(f"Format: {format_label}")
            print(f"Source: {model_path}")
            print(f"Output: {output_dir}")
            print(f"Name: {model_name}")
            print(f"Quantizations: {', '.join(quantizations)}")
            print()

            if RICH_AVAILABLE and Confirm:
                proceed = Confirm.ask("Proceed with conversion?", default=True)
            else:
                response = input("Proceed with conversion? [Y/n]: ").strip().lower()
                proceed = response in ('', 'y', 'yes')

            if not proceed:
                print("Conversion cancelled")
                return 0

            # Run conversion
            print()
            return self._run_conversion(model_path, output_dir, model_name, quantizations, output_format)

        except KeyboardInterrupt:
            print("\nCancelled")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _run_conversion(
        self,
        model_path: Path,
        output_dir: Path,
        model_name: str,
        quantizations: List[str],
        output_format: str = "gguf"
    ) -> int:
        """
        Run the actual model conversion.

        Args:
            model_path: Path to model (LoRA or full)
            output_dir: Output directory
            model_name: Name for output files
            quantizations: List of quantization methods
            output_format: "gguf" or "webgpu"

        Returns:
            int: Exit code
        """
        # Add shared module to path
        sys.path.insert(0, str(self.repo_root / "Trainers"))

        try:
            if output_format == "gguf":
                from shared.upload.converters.gguf_reliable import ReliableGGUFConverter

                converter = ReliableGGUFConverter()
                output_files = converter.convert(
                    model_path=model_path,
                    output_dir=output_dir,
                    quantizations=quantizations,
                    model_name=model_name,
                )

                if output_files:
                    print("\n" + "=" * 60)
                    print("CONVERSION COMPLETE")
                    print("=" * 60)
                    print(f"Created {len(output_files)} GGUF files in:")
                    print(f"  {output_dir / 'gguf'}")
                    print()
                    print("Next steps:")
                    print("  1. Copy to Ollama: ollama create my-model -f Modelfile")
                    print("  2. Or use with llama.cpp: ./llama-cli -m model.gguf")
                    return 0
                else:
                    print("No GGUF files created")
                    return 1

            else:  # webgpu
                from shared.upload.converters.webgpu import WebGPUConverter

                converter = WebGPUConverter()
                output_files = converter.convert(
                    model_path=model_path,
                    output_dir=output_dir,
                    quantizations=quantizations,
                    model_name=model_name,
                )

                if output_files:
                    # WebGPU converter prints its own summary
                    return 0
                else:
                    print("No WebGPU files created")
                    return 1

        except ImportError as e:
            print(f"Import error: {e}")
            print("Make sure you're in the correct conda environment")
            if output_format == "webgpu":
                print("For WebGPU, run: bash setup.sh --with-webgpu")
            return 1
        except Exception as e:
            print(f"Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
