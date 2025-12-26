"""
Merge handler - merge LoRA adapters into base model.

Location: tuner/handlers/merge_handler.py
Purpose: Standalone merge operation for LoRA → merged-16bit
Used by: CLI router, before GRPO training or upload

This creates a composable merge step:
  1. Train SFT → LoRA checkpoint
  2. Merge → merged-16bit (this handler)
  3. Train GRPO → new LoRA on merged base
  4. Upload
"""

from pathlib import Path
from typing import List

from .base import BaseHandler
from tuner.discovery.training_runs import TrainingRunDiscovery

# Import shared merge utilities
from shared.model_loading import (
    is_lora_checkpoint,
    find_merged_for_run,
    get_base_model_name,
    merge_lora_checkpoint,
)

# Import UI from tuner.ui (wraps shared.ui + adds print_table)
from tuner.ui import (
    print_header,
    print_info,
    print_success,
    print_error,
    print_table,
    print_config,
    confirm,
    COLORS,
    BOX,
)


class MergeHandler(BaseHandler):
    """
    Handler for merging LoRA adapters into base model.

    Creates merged-16bit model from LoRA checkpoint.
    """

    @property
    def name(self) -> str:
        return "merge"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _find_lora_checkpoints(self) -> List[dict]:
        """Find all LoRA checkpoints across training outputs."""
        discovery = TrainingRunDiscovery(repo_root=self.repo_root)

        checkpoints = []

        for trainer_type in ['sft', 'kto', 'grpo']:
            runs = discovery.discover(trainer_type, limit=20)

            for run_path in runs:
                # Check for final_model
                final_model = run_path / "final_model"
                if final_model.exists() and is_lora_checkpoint(final_model):
                    # Check if already merged
                    has_merged = find_merged_for_run(run_path) is not None
                    checkpoints.append({
                        'path': final_model,
                        'run': run_path.name,
                        'type': trainer_type.upper(),
                        'checkpoint': 'final_model',
                        'merged': has_merged,
                    })

                # Check for checkpoints
                cp_dir = run_path / "checkpoints"
                if cp_dir.exists():
                    for cp in sorted(cp_dir.iterdir(), reverse=True):
                        if cp.is_dir() and is_lora_checkpoint(cp):
                            has_merged = find_merged_for_run(run_path) is not None
                            checkpoints.append({
                                'path': cp,
                                'run': run_path.name,
                                'type': trainer_type.upper(),
                                'checkpoint': cp.name,
                                'merged': has_merged,
                            })

        return checkpoints

    def _merge_lora(self, lora_path: Path, output_path: Path) -> bool:
        """Merge LoRA adapters with base model using shared utilities."""
        try:
            print_info(f"Merging: {lora_path.name}")
            print_info(f"Output: {output_path}")
            merge_lora_checkpoint(lora_path, output_path)
            return True
        except Exception as e:
            print_error(f"Merge failed: {e}")
            return False

    def handle(self) -> int:
        """Execute merge workflow."""
        print_header("MERGE LORA TO BASE MODEL")
        print_info("Merge LoRA adapters into base model weights")
        print()

        # Find all LoRA checkpoints
        checkpoints = self._find_lora_checkpoints()

        if not checkpoints:
            print_error("No LoRA checkpoints found")
            return 1

        # Display table
        table_data = []
        for i, cp in enumerate(checkpoints, 1):
            status = f"{BOX['check']} merged" if cp['merged'] else "-"
            table_data.append([
                str(i),
                cp['run'],
                cp['type'],
                cp['checkpoint'],
                status,
            ])

        print_table(
            table_data,
            ["#", "Training Run", "Type", "Checkpoint", "Status"],
            title="Available LoRA Checkpoints"
        )
        print()

        # Get user selection
        try:
            choice = input(f"Select checkpoint to merge [1-{len(checkpoints)}]: ").strip()
            if not choice:
                print_info("Cancelled")
                return 0

            idx = int(choice) - 1
            if idx < 0 or idx >= len(checkpoints):
                print_error("Invalid selection")
                return 1
        except ValueError:
            print_error("Invalid input")
            return 1

        selected = checkpoints[idx]
        lora_path = selected['path']
        run_path = lora_path.parent
        if run_path.name in ('checkpoints', 'final_model'):
            run_path = run_path.parent

        # Check if already merged (using shared utility)
        existing_merged = find_merged_for_run(run_path)
        if existing_merged:
            print_info(f"Merged model already exists: {existing_merged}")
            if not confirm("Overwrite existing merged model?"):
                print_info("Cancelled")
                return 0

        # Determine output path using shared utility
        model_name = get_base_model_name(lora_path)
        output_dir = run_path / model_name / "merged-16bit"

        # Show config
        print()
        print_config({
            "LoRA Path": str(lora_path.relative_to(self.repo_root)),
            "Output": str(output_dir.relative_to(self.repo_root)),
            "Type": selected['type'],
        }, "Merge Configuration")

        if not confirm("Proceed with merge?"):
            print_info("Cancelled")
            return 0

        # Execute merge
        print()
        success = self._merge_lora(lora_path, output_dir)

        if success:
            print()
            print_success(f"Merged model saved to: {output_dir}")
            print_info("You can now use this for GRPO training or upload")
            return 0
        else:
            return 1
