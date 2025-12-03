"""
Upload workflow handler.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/upload_handler.py
Purpose: Orchestrate the upload workflow (run selection, checkpoint selection, upload config, execution)
Used by: Router when 'upload' command is invoked
"""

import os
import subprocess
from pathlib import Path
from tuner.handlers.base import BaseHandler
from tuner.discovery import TrainingRunDiscovery, CheckpointDiscovery
from tuner.ui import (
    print_menu,
    print_header,
    print_config,
    print_success,
    print_error,
    print_info,
    print_table,
    print_checkpoint_table,
    confirm,
    prompt,
    BOX,
    RICH_AVAILABLE,
    console,
    COLORS,
)
from tuner.utils.validation import validate_repo_id, load_env_file


class UploadHandler(BaseHandler):
    """
    Handler for upload workflow.

    Orchestrates the complete upload process:
    1. Check HF_TOKEN availability
    2. Select model type (SFT/KTO)
    3. List and select training run
    4. Display checkpoint metrics
    5. Select checkpoint or final model
    6. Configure upload (repo ID, save method, GGUF)
    7. Execute upload via shared upload CLI

    Example:
        handler = UploadHandler()
        exit_code = handler.handle()
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "upload"

    def can_handle_direct_mode(self) -> bool:
        """Can be invoked as 'python -m tuner upload'."""
        return True

    def handle(self) -> int:
        """
        Execute upload workflow.

        Returns:
            int: Exit code (0 = success, non-zero = failure)
        """
        print_header("UPLOAD", "Push your model to HuggingFace")

        # Step 1: Check HF_TOKEN
        env_file = self.repo_root / ".env"
        load_env_file(env_file)

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        if not hf_token:
            print_error("HF_TOKEN not found in .env file")
            print_info("Create .env with: HF_TOKEN=hf_your_token_here")
            return 1

        print_success("HuggingFace token found")

        # Step 2: Select upload mode
        upload_mode = print_menu([
            ("local", f"{BOX['bullet']} Upload local training run"),
            ("gguf_only", f"{BOX['bullet']} Create GGUF from existing HuggingFace repo"),
        ], "Select upload mode:")

        if not upload_mode:
            return 0

        # Handle GGUF-only mode
        if upload_mode == "gguf_only":
            return self._handle_gguf_only(hf_token)

        # Step 3: Select model type (for local upload)
        model_type = print_menu([
            ("sft", f"{BOX['bullet']} SFT model"),
            ("kto", f"{BOX['bullet']} KTO model"),
        ], "Select model type:")

        if not model_type:
            return 0

        # Step 3: List training runs
        discovery = TrainingRunDiscovery(repo_root=self.repo_root)
        runs = discovery.discover(trainer_type=model_type, limit=10)

        if not runs:
            print_error(f"No training runs found for {model_type.upper()}")
            return 1

        # Step 4: Display runs and select
        run_data = []
        for i, run in enumerate(runs, 1):
            run_data.append([str(i), run.name])

        print_table(run_data, ["#", "Training Run"], title=f"Available {model_type.upper()} Training Runs")

        while True:
            try:
                sel = prompt(f"Select run (1-{len(runs)})")
                idx = int(sel) - 1
                if 0 <= idx < len(runs):
                    selected_run = runs[idx]
                    break
            except ValueError:
                pass
            print_error("Invalid selection.")

        # Step 5: Display checkpoints and select
        checkpoint_path = self._select_checkpoint(selected_run, model_type)
        if not checkpoint_path:
            return 0

        # Step 6: Get repo ID
        hf_username = os.environ.get("HF_USERNAME", "")
        if hf_username:
            print_info(f"HuggingFace username: {hf_username}")
            model_name = prompt("Model name", "")
            if not model_name:
                print_error("Model name required")
                return 1
            repo_id = f"{hf_username}/{model_name}"
        else:
            repo_id = prompt("HuggingFace repo ID (username/model-name)")
            if not validate_repo_id(repo_id):
                print_error("Invalid repo ID format. Add HF_USERNAME to .env for easier input.")
                return 1

        # Step 7: Select save method
        save_method = print_menu([
            ("merged_16bit", f"{BOX['star']} Merged 16-bit (~14GB) - Full quality"),
            ("merged_4bit", f"{BOX['bullet']} Merged 4-bit (~3.5GB) - Smaller"),
            ("lora", f"{BOX['bullet']} LoRA adapters only (~320MB) - Fastest"),
        ], "Select save method:")

        if not save_method:
            return 0

        # Step 8: GGUF option
        create_gguf = confirm("Create GGUF versions?")

        # Step 9: Confirmation
        print_config({
            "Model": str(checkpoint_path.relative_to(self.repo_root)),
            "Repository": repo_id,
            "Save Method": save_method,
            "GGUF": "Yes" if create_gguf else "No",
        }, "Upload Configuration")

        if not confirm("Start upload?"):
            print_info("Upload cancelled.")
            return 0

        # Step 10: Execute upload
        # Run as module from Trainers directory to handle relative imports properly
        python = self.get_conda_python()
        trainers_dir = self.repo_root / "Trainers"

        cmd = [
            python,
            "-m", "shared.upload.cli.upload_cli",
            str(checkpoint_path),
            repo_id,
            "--save-method", save_method,
        ]
        if create_gguf:
            cmd.append("--create-gguf")

        print_info(f"Running: {' '.join(cmd)}")
        print()

        # Run from Trainers directory so 'shared' package is found
        exit_code = subprocess.run(cmd, cwd=str(trainers_dir)).returncode

        if exit_code == 0:
            print_success("Upload completed successfully.")
        else:
            print_error(f"Upload failed with exit code: {exit_code}")

        return exit_code

    def _select_checkpoint(self, run_dir: Path, training_type: str) -> Path:
        """
        Select checkpoint from a training run.

        Args:
            run_dir: Training run directory
            training_type: 'sft' or 'kto'

        Returns:
            Path to selected checkpoint, or None if cancelled
        """
        discovery = CheckpointDiscovery()
        checkpoints = discovery.discover(run_dir=run_dir)

        if not checkpoints:
            print_error("No checkpoints found in training run")
            return None

        # If only final_model exists, use it directly
        if len(checkpoints) == 1 and checkpoints[0].is_final:
            print_info("Using final_model")
            return checkpoints[0].path

        # Display checkpoint table
        print_checkpoint_table(checkpoints, training_type)

        # Build selection options
        options = []
        for cp in checkpoints:
            if cp.is_final:
                options.append(("final", cp.path))
            else:
                options.append((cp.path.name, cp.path))

        # Let user choose
        while True:
            try:
                sel = prompt(f"Select checkpoint (1-{len(options)})", "1")
                idx = int(sel) - 1
                if 0 <= idx < len(options):
                    return options[idx][1]
            except ValueError:
                pass
            print_error("Invalid selection.")

    def _handle_gguf_only(self, hf_token: str) -> int:
        """
        Handle GGUF-only mode: create GGUF from existing HuggingFace repo.

        Args:
            hf_token: HuggingFace token

        Returns:
            int: Exit code
        """
        print_header("GGUF FROM HUGGINGFACE", "Create GGUF from existing repo")

        # Get source repo
        hf_username = os.environ.get("HF_USERNAME", "")
        if hf_username:
            print_info(f"HuggingFace username: {hf_username}")
            model_name = prompt("Model name (existing repo)")
            if not model_name:
                print_error("Model name required")
                return 1
            source_repo = f"{hf_username}/{model_name}"
        else:
            source_repo = prompt("HuggingFace repo ID (username/model-name)")
            if not validate_repo_id(source_repo):
                print_error("Invalid repo ID format")
                return 1

        target_repo = f"{source_repo}-GGUF"

        # Select quantizations
        quant_choice = print_menu([
            ("default", f"{BOX['star']} Default (Q4_K_M, Q5_K_M, Q8_0)"),
            ("q4_only", f"{BOX['bullet']} Q4_K_M only (smallest)"),
            ("q8_only", f"{BOX['bullet']} Q8_0 only (best quality)"),
            ("all", f"{BOX['bullet']} All (Q4_K_M, Q5_K_M, Q6_K, Q8_0)"),
        ], "Select quantizations:")

        if not quant_choice:
            return 0

        quant_map = {
            "default": ["Q4_K_M", "Q5_K_M", "Q8_0"],
            "q4_only": ["Q4_K_M"],
            "q8_only": ["Q8_0"],
            "all": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        }
        quantizations = quant_map[quant_choice]

        # Confirmation
        print_config({
            "Source": source_repo,
            "Target": target_repo,
            "Quantizations": ", ".join(quantizations),
        }, "GGUF Configuration")

        if not confirm("Create GGUF files?"):
            print_info("Cancelled.")
            return 0

        # Execute via subprocess to shared upload CLI
        python = self.get_conda_python()
        trainers_dir = self.repo_root / "Trainers"

        cmd = [
            python,
            "-m", "shared.upload.cli.upload_cli",
            "--gguf-only", source_repo,
            "--gguf-quantizations", *quantizations,
        ]

        print_info(f"Running GGUF conversion...")
        print()

        exit_code = subprocess.run(cmd, cwd=str(trainers_dir)).returncode

        if exit_code == 0:
            print_success(f"GGUF files uploaded to: https://huggingface.co/{target_repo}")
        else:
            print_error(f"GGUF conversion failed with exit code: {exit_code}")

        return exit_code
