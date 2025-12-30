"""
Model conversion handler.

Location: tuner/handlers/convert_handler.py
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
from tuner.discovery import CheckpointDiscovery
from tuner.ui import (
    print_header,
    print_menu,
    print_config,
    print_success,
    print_error,
    print_info,
    print_table,
    print_checkpoint_table,
    confirm,
    prompt,
    BOX,
)
from shared.ui import spinner


class ConvertHandler(BaseHandler):
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
        return "convert"

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
            from tuner.discovery.training_runs import TrainingRunDiscovery

            print_header("MODEL CONVERSION", "Convert trained models to deployment formats")

            # Select output format
            format_choice = print_menu([
                ("gguf", f"{BOX['star']} GGUF - llama.cpp/Ollama (local inference)"),
                ("webgpu", f"{BOX['bullet']} WebGPU - MLC-LLM/WebLLM (browser deployment)"),
            ], "Select output format:")

            if not format_choice:
                return 0

            output_format = format_choice

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
                print_error("No training runs found!")
                print_info("Train a model first with: ./run.sh train")
                return 1

            # Build table data
            table_data = []
            for i, run in enumerate(all_runs[:10], 1):
                has_final = "✓" if (run['path'] / "final_model").exists() else "-"
                checkpoints_dir = run['path'] / "checkpoints"
                cp_count = len(list(checkpoints_dir.glob("checkpoint-*"))) if checkpoints_dir.exists() else 0
                table_data.append([str(i), run['type'], run['path'].name, has_final, str(cp_count)])

            print_table(table_data, ["#", "Type", "Training Run", "Final", "CPs"],
                       title="Available Training Runs")

            # Build menu options for arrow-key selection
            run_options = []
            for i, run in enumerate(all_runs[:10]):
                has_final = "✓" if (run['path'] / "final_model").exists() else "-"
                checkpoints_dir = run['path'] / "checkpoints"
                cp_count = len(list(checkpoints_dir.glob("checkpoint-*"))) if checkpoints_dir.exists() else 0
                label = f"[{run['type']}] {run['path'].name} (final: {has_final}, checkpoints: {cp_count})"
                run_options.append((str(i), label))

            selected_key = print_menu(run_options, "Select training run:")
            if selected_key is None:
                return 0
            selected_run = all_runs[int(selected_key)]

            # Discover checkpoints with metrics
            checkpoints = CheckpointDiscovery.discover(selected_run['path'])

            if not checkpoints:
                print_error(f"No checkpoints found in {selected_run['path']}")
                return 1

            # Determine training type for metric display
            training_type = selected_run['type'].lower()

            # If only one checkpoint (final_model), use it directly
            if len(checkpoints) == 1 and checkpoints[0].is_final:
                print_info("Using final_model")
                model_path = checkpoints[0].path
            else:
                # Display checkpoint table with loss values
                print()
                print_checkpoint_table(checkpoints, training_type)

                # Build menu options for checkpoint selection
                cp_options = []
                for i, cp in enumerate(checkpoints):
                    if cp.is_final:
                        label = f"final_model (step {cp.step}, loss: {cp.loss:.4f})"
                    else:
                        label = f"{cp.path.name} (step {cp.step}, loss: {cp.loss:.4f})"
                    cp_options.append((str(i), label))

                selected_cp = print_menu(cp_options, "Select checkpoint:")
                if selected_cp is None:
                    return 0
                model_path = checkpoints[int(selected_cp)].path

            print_info(f"Selected: {model_path}")

            # Get model name
            default_name = selected_run['path'].name
            format_label = "GGUF" if output_format == "gguf" else "WebGPU"
            model_name = prompt(f"Model name for {format_label} files", default_name)

            # Select quantizations based on format
            if output_format == "gguf":
                quant_choice = print_menu([
                    ("standard", f"{BOX['star']} Standard (Q4_K_M, Q5_K_M, Q8_0) - Recommended"),
                    ("minimal", f"{BOX['bullet']} Minimal (Q4_K_M only) - Fastest"),
                    ("full", f"{BOX['bullet']} Full (Q4_K_M, Q5_K_M, Q6_K, Q8_0) - All common"),
                    ("custom", f"{BOX['bullet']} Custom"),
                ], "Select quantization preset:")

                if not quant_choice:
                    return 0

                quant_presets = {
                    "standard": ["Q4_K_M", "Q5_K_M", "Q8_0"],
                    "minimal": ["Q4_K_M"],
                    "full": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
                }

                if quant_choice == "custom":
                    custom = prompt("Enter quantizations (comma-separated)", "Q4_K_M,Q5_K_M")
                    quantizations = [q.strip().upper() for q in custom.split(",")]
                else:
                    quantizations = quant_presets[quant_choice]
            else:
                # WebGPU quantizations
                quant_choice = print_menu([
                    ("q4f16_1", f"{BOX['star']} q4f16_1 - 4-bit with float16 (recommended)"),
                    ("q4f32_1", f"{BOX['bullet']} q4f32_1 - 4-bit with float32"),
                    ("q0f16", f"{BOX['bullet']} q0f16 - No quantization, float16"),
                    ("custom", f"{BOX['bullet']} Custom"),
                ], "Select quantization:")

                if not quant_choice:
                    return 0

                if quant_choice == "custom":
                    custom = prompt("Enter quantization", "q4f16_1")
                    quantizations = [custom.strip().lower()]
                else:
                    quantizations = [quant_choice]

            # Output directory
            output_dir = selected_run['path'] / model_name

            # Show configuration summary
            print_config({
                "Format": format_label,
                "Source": str(model_path.relative_to(self.repo_root)),
                "Output": str(output_dir.relative_to(self.repo_root)),
                "Name": model_name,
                "Quantizations": ", ".join(quantizations),
            }, "Conversion Summary")

            if not confirm("Proceed with conversion?"):
                print_info("Conversion cancelled")
                return 0

            # Run conversion with spinner
            print()
            with spinner(f"Converting to {format_label}..."):
                result = self._run_conversion(model_path, output_dir, model_name, quantizations, output_format)
            return result

        except KeyboardInterrupt:
            print_info("Cancelled")
            return 130
        except Exception as e:
            print_error(f"Error: {e}")
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
                    print_success(f"Created {len(output_files)} GGUF files in: {output_dir / 'gguf'}")
                    print_info("Next steps:")
                    print_info("  1. Copy to Ollama: ollama create my-model -f Modelfile")
                    print_info("  2. Or use with llama.cpp: ./llama-cli -m model.gguf")
                    return 0
                else:
                    print_error("No GGUF files created")
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
                    print_success("WebGPU conversion complete")
                    return 0
                else:
                    print_error("No WebGPU files created")
                    return 1

        except ImportError as e:
            print_error(f"Import error: {e}")
            print_info("Make sure you're in the correct conda environment")
            if output_format == "webgpu":
                print_info("For WebGPU, run: bash setup.sh --with-webgpu")
            return 1
        except Exception as e:
            print_error(f"Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
