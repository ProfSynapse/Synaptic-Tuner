"""
LoRA Surgery handler for the CLI.

Location: tuner/handlers/surgery_handler.py
Purpose: Interactive LoRA weight surgery on trained adapters
Used by: Router and main menu when 'surgery' command is selected

Guides user through:
1. Selecting an adapter path
2. Configuring surgery operations
3. Running eval-guided weight surgery
4. Reporting results
"""

import asyncio
import os
from argparse import Namespace
from pathlib import Path
from typing import Optional

from tuner.handlers.base import BaseHandler


class SurgeryHandler(BaseHandler):
    """
    Handler for LoRA weight surgery operations.

    Performs eval-guided post-training weight surgery on LoRA adapters,
    trying various weight operations (scaling, ablation, interpolation,
    compression) and keeping changes that improve eval scores.

    Example:
        handler = SurgeryHandler(args=args)
        exit_code = handler.handle()
    """

    def __init__(self, args: Optional[Namespace] = None):
        super().__init__(args=args)

    @property
    def name(self) -> str:
        return "surgery"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        """Execute LoRA surgery workflow.

        Returns:
            Exit code (0 = success, non-zero = error).
        """
        try:
            from shared.evolutionary.lora_surgery import (
                LoRASurgeon,
                SurgeryConfig,
            )
        except ImportError as exc:
            self.output_error(
                f"LoRA surgery dependencies not available: {exc}. "
                "Install with: pip install torch safetensors pyyaml",
                code="MISSING_DEPS",
            )
            return 1

        # Determine config path
        config_path = getattr(self.args, "surgery_config", None)
        adapter_path = getattr(self.args, "subcommand", None)

        if config_path and os.path.exists(config_path):
            config = SurgeryConfig.from_yaml(config_path)
        else:
            config = SurgeryConfig()

        # Override adapter path from CLI if provided
        if adapter_path and os.path.isdir(adapter_path):
            config.adapter_path = adapter_path

        # Interactive mode if adapter path not set
        if not config.adapter_path or not os.path.isdir(config.adapter_path):
            print("\n  LoRA Weight Surgery")
            print("  " + "-" * 40)
            print("  Performs eval-guided post-training weight")
            print("  surgery on LoRA adapters.\n")

            adapter_input = input("  Adapter path: ").strip()
            if not adapter_input or not os.path.isdir(adapter_input):
                self.output_error(
                    f"Invalid adapter path: {adapter_input}",
                    code="INVALID_PATH",
                )
                return 1
            config.adapter_path = adapter_input

        # Verify adapter has required files
        adapter_config_path = os.path.join(config.adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            self.output_error(
                f"No adapter_config.json found in {config.adapter_path}",
                code="INVALID_ADAPTER",
            )
            return 1

        safetensor_files = [
            f for f in os.listdir(config.adapter_path)
            if f.endswith(".safetensors")
        ]
        if not safetensor_files:
            self.output_error(
                f"No .safetensors files found in {config.adapter_path}",
                code="INVALID_ADAPTER",
            )
            return 1

        # Require eval scenario
        if not config.eval_scenario:
            eval_scenario = getattr(self.args, "eval_scenario", None)
            if eval_scenario:
                config.eval_scenario = eval_scenario
            else:
                self.output_error(
                    "An eval scenario is required for surgery. "
                    "Provide --eval-scenario path/to/scenarios.yaml or set eval_scenario in surgery config.",
                    code="MISSING_EVAL_SCENARIO",
                )
                return 1

        # Show config summary
        print(f"\n  Adapter: {config.adapter_path}")
        print(f"  Eval scenario: {config.eval_scenario}")
        print(f"  Eval backend: {config.eval_backend}")
        print(f"  Operations: {', '.join(config.operations)}")
        print(f"  Min improvement: {config.min_improvement}")
        print(f"  Output: {config.output_dir}\n")

        # Create eval backend and run surgery
        try:
            from shared.eval_backend import create_eval_backend

            eval_backend = create_eval_backend(
                backend_type=config.eval_backend,
                min_vram_gb=config.local_min_vram_gb,
            )

            surgeon = LoRASurgeon(
                adapter_path=config.adapter_path,
                eval_backend=eval_backend,
                eval_scenario=config.eval_scenario,
                config=config,
            )

            print("  Running surgery...")
            result = asyncio.run(surgeon.run_surgery())

            # Report results
            print(f"\n  Baseline score: {result.baseline_score:.4f}")
            print(f"  Final score:    {result.final_score:.4f}")
            print(f"  Improvement:    {result.total_improvement:+.4f}")
            print(f"  Duration:       {result.duration_seconds:.1f}s")
            if result.operations_applied:
                print(f"  Operations:     {len(result.operations_applied)} applied")
                for op in result.operations_applied:
                    print(f"    - {op.operation}: {op.improvement:+.4f} ({op.variants_tried} variants)")
            print(f"  Best adapter:   {result.best_adapter_path}\n")

            self.output(
                {
                    "baseline_score": result.baseline_score,
                    "final_score": result.final_score,
                    "improvement": result.total_improvement,
                    "best_adapter_path": result.best_adapter_path,
                    "operations_applied": len(result.operations_applied),
                    "status": "completed",
                },
                f"Surgery complete. Improvement: {result.total_improvement:+.4f}",
            )
            return 0

        except Exception as exc:
            self.output_error(
                f"Surgery failed: {exc}",
                code="SURGERY_FAILED",
            )
            return 1
