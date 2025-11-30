"""
Pipeline handler for full Train -> Upload -> Evaluate workflow.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/pipeline_handler.py
Purpose: Orchestrate complete pipeline from training through evaluation
Used by: Router when handling 'pipeline' command or main menu selection

This handler implements the full pipeline:
1. Display pipeline steps overview
2. Confirm to start pipeline
3. Execute TrainHandler
4. Confirm to continue to upload
5. Execute UploadHandler
6. Confirm to continue to evaluation
7. Execute EvalHandler
8. Display completion message
"""

from tuner.handlers.base import BaseHandler

# Import shared UI components (delegates to Trainers/shared/ui/)
from Trainers.shared.ui import (
    print_header,
    confirm,
    console,
    RICH_AVAILABLE,
    COLORS,
    BOX,
)


class PipelineHandler(BaseHandler):
    """
    Handler for full pipeline orchestration.

    Coordinates the complete workflow: Train -> Upload -> Evaluate
    Each step can be cancelled, allowing partial pipeline execution.

    Example:
        handler = PipelineHandler()
        exit_code = handler.handle()
        # Runs train -> upload -> eval with user confirmations
        # Returns 0 on success, non-zero on failure
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "pipeline"

    def can_handle_direct_mode(self) -> bool:
        """This handler supports direct CLI invocation."""
        return True

    def _display_pipeline_steps(self) -> None:
        """Display pipeline steps overview."""
        if RICH_AVAILABLE:
            from rich.text import Text

            steps = Text()
            steps.append(f"\n  {BOX['dot']} ", style=COLORS["aqua"])
            steps.append("Step 1: Train your model\n", style="white")
            steps.append(f"  {BOX['dot']} ", style=COLORS["purple"])
            steps.append("Step 2: Upload to HuggingFace\n", style="white")
            steps.append(f"  {BOX['dot']} ", style=COLORS["sky"])
            steps.append("Step 3: Evaluate performance\n", style="white")
            console.print(steps)
        else:
            print("  This will run:")
            print("    1. Train your model")
            print("    2. Upload to HuggingFace")
            print("    3. Evaluate performance")
            print()

    def handle(self) -> int:
        """
        Execute full pipeline workflow.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        # Import handlers here to avoid circular imports
        from tuner.handlers.train_handler import TrainHandler
        from tuner.handlers.upload_handler import UploadHandler
        from tuner.handlers.eval_handler import EvalHandler

        # Display header and pipeline overview
        print_header("FULL PIPELINE", "Train -> Upload -> Evaluate")
        self._display_pipeline_steps()

        # Confirm to start pipeline
        if not confirm("Continue with full pipeline?"):
            return 0

        # Step 1: Training
        print_header("STEP 1: TRAINING")
        train_handler = TrainHandler()
        train_exit_code = train_handler.handle()

        if train_exit_code != 0:
            print_header("PIPELINE STOPPED", "Training failed or was cancelled")
            return train_exit_code

        if not confirm("Continue to upload?"):
            return 0

        # Step 2: Upload
        print_header("STEP 2: UPLOAD")
        upload_handler = UploadHandler()
        upload_exit_code = upload_handler.handle()

        if upload_exit_code != 0:
            print_header("PIPELINE STOPPED", "Upload failed or was cancelled")
            return upload_exit_code

        if not confirm("Continue to evaluation?"):
            return 0

        # Step 3: Evaluation
        print_header("STEP 3: EVALUATION")
        eval_handler = EvalHandler()
        eval_exit_code = eval_handler.handle()

        # Pipeline complete
        print_header("PIPELINE COMPLETE", "All steps finished successfully!")

        return eval_exit_code
