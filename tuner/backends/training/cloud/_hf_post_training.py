"""
HF Jobs post-training actions mixin.

Location: tuner/backends/training/cloud/_hf_post_training.py
Purpose: Handle post-training completion workflows (summary, download, convert, eval)
Used by: HFJobsBackend (via mixin inheritance) in hf_jobs_backend.py

After a training job completes, this mixin prints artifact locations and
offers an interactive menu for next-step workflows: downloading the run,
converting to GGUF, running inference, evaluation, or uploading to the Hub.
"""

import sys
from pathlib import Path
from typing import Optional

from tuner.core.config import CloudTrainingConfig
from tuner.ui import BOX, print_config, print_info, print_menu, print_success


class HFPostTrainingMixin:
    """Methods for post-training completion workflows."""

    def _print_completion_summary(
        self,
        *,
        config: CloudTrainingConfig,
        artifact_prefix: str,
        local_run_dir: Optional[Path] = None,
    ) -> None:
        """Print where the completed artifacts live and how they map locally."""
        summary = {
            "Remote artifacts": self._build_remote_run_uri(config, artifact_prefix),
            "Suggested local path": str(self._local_download_run_dir(config, artifact_prefix)),
        }
        if local_run_dir:
            summary["Local run"] = str(local_run_dir)
        print_config(summary, "Cloud Training Artifacts")

    def _handle_post_training_actions(self, *, config: CloudTrainingConfig, artifact_prefix: str) -> None:
        """Offer a post-training menu that reuses the existing local workflows."""
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return

        from tuner.handlers.convert_handler import ConvertHandler
        from tuner.handlers.eval_handler import EvalHandler
        from tuner.handlers.inference_handler import InferenceHandler
        from tuner.handlers.upload_handler import UploadHandler

        local_run_dir: Optional[Path] = None
        menu_options = [
            ("download", f"{BOX['star']} Download completed run locally"),
            ("gguf", f"{BOX['bullet']} Convert model to GGUF"),
            ("run", f"{BOX['bullet']} Run model locally"),
            ("eval", f"{BOX['bullet']} Run evaluation"),
            ("upload", f"{BOX['bullet']} Upload model to Hugging Face"),
        ]

        while True:
            choice = print_menu(menu_options, "Choose your next step:")
            if not choice:
                return

            if local_run_dir is None:
                print_info("Downloading completed cloud run into local training outputs...")
                local_run_dir = self._download_completed_run(
                    config=config,
                    artifact_prefix=artifact_prefix,
                )
                print_success(f"Downloaded run to: {local_run_dir}")

            if choice == "download":
                continue
            if choice == "gguf":
                print_info("Opening conversion workflow. Select the newly downloaded run when prompted.")
                ConvertHandler().handle()
                continue
            if choice == "run":
                print_info("Opening local inference workflow. Select the newly downloaded run when prompted.")
                InferenceHandler().handle()
                continue
            if choice == "eval":
                print_info("Opening evaluation workflow. Select the newly downloaded run when prompted.")
                EvalHandler(args=None).handle()
                continue
            if choice == "upload":
                print_info("Opening upload workflow. Select the newly downloaded run when prompted.")
                UploadHandler(args=None).handle()

    def _finalize_completed_job(
        self,
        *,
        config: CloudTrainingConfig,
        artifact_prefix: str,
        local_run_dir: Optional[Path] = None,
    ) -> int:
        """Print success details and offer post-training actions."""
        self._print_completion_summary(
            config=config,
            artifact_prefix=artifact_prefix,
            local_run_dir=local_run_dir,
        )
        if self.show_post_training_actions:
            self._handle_post_training_actions(config=config, artifact_prefix=artifact_prefix)
        return 0
