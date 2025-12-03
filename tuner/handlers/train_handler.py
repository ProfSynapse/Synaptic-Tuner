"""
Training workflow handler.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/train_handler.py
Purpose: Orchestrate the training workflow (platform selection, config loading, execution)
Used by: Router when 'train' command is invoked
"""

import subprocess
from pathlib import Path
from tuner.handlers.base import BaseHandler
from tuner.backends.registry import TrainingBackendRegistry
from tuner.ui import (
    print_menu,
    print_header,
    print_config,
    print_error,
    print_info,
    confirm,
    BOX,
)


def detect_platform() -> str | None:
    """
    Auto-detect the available training platform.

    Returns:
        'rtx' if CUDA is available
        'mac' if MLX is available (Apple Silicon)
        None if neither or both are available (user must choose)
    """
    has_cuda = False
    has_mlx = False

    # Check for CUDA
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    # Check for MLX (Apple Silicon)
    try:
        import mlx.core as mx
        has_mlx = mx.metal.is_available()
    except ImportError:
        pass

    # Auto-select if only one is available
    if has_cuda and not has_mlx:
        return "rtx"
    elif has_mlx and not has_cuda:
        return "mac"
    else:
        return None


class TrainHandler(BaseHandler):
    """
    Handler for training workflow.

    Orchestrates the complete training process:
    1. Platform selection (RTX/Mac)
    2. Method selection (SFT/KTO/MLX)
    3. Configuration loading and display
    4. User confirmation
    5. Training execution

    Example:
        handler = TrainHandler()
        exit_code = handler.handle()
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "train"

    def can_handle_direct_mode(self) -> bool:
        """Can be invoked as 'python -m tuner train'."""
        return True

    def handle(self) -> int:
        """
        Execute training workflow.

        Returns:
            int: Exit code (0 = success, non-zero = failure)
        """
        print_header("TRAINING", "Select your platform and training method")

        # Step 1: Auto-detect or select platform
        platform_choice = detect_platform()

        if platform_choice:
            platform_name = "NVIDIA GPU (CUDA)" if platform_choice == "rtx" else "Apple Silicon (MLX)"
            print_info(f"Auto-detected platform: {platform_name}")
        else:
            platform_choice = print_menu([
                ("rtx", f"{BOX['bullet']} NVIDIA GPU (RTX 3090 / CUDA) - SFT or KTO"),
                ("mac", f"{BOX['bullet']} Apple Silicon (M1/M2/M3) - MLX LoRA"),
            ], "Select platform:")

        if not platform_choice:
            return 0

        # Step 2: Get backend
        try:
            backend = TrainingBackendRegistry.get(platform_choice, repo_root=self.repo_root)
        except ValueError as e:
            print_error(str(e))
            return 1

        # Step 3: Validate environment
        is_valid, error = backend.validate_environment()
        if not is_valid:
            print_error(f"Environment validation failed: {error}")
            return 1

        # Step 4: Select method (if multiple available)
        methods = backend.get_available_methods()
        method_options = [(m, f"{BOX['bullet']} {m.upper()} training") for m in methods]

        if len(methods) > 1:
            method = print_menu(method_options, "Select training method:")
            if not method:
                return 0
        else:
            method = methods[0]
            print_info(f"Using method: {method.upper()}")

        # Step 5: Load configuration
        try:
            config = backend.load_config(method)
        except Exception as e:
            print_error(f"Failed to load configuration: {e}")
            return 1

        # Step 6: Display configuration
        config_display = {
            "Platform": platform_choice.upper(),
            "Method": method.upper(),
            "Model": config.model_name.split('/')[-1] if '/' in config.model_name else config.model_name,
            "Dataset": Path(config.dataset_file).name if config.dataset_file else "Unknown",
            "Epochs": str(config.epochs),
            "Batch Size": str(config.batch_size),
            "Learning Rate": str(config.learning_rate),
            "Config": str(config.config_path.relative_to(self.repo_root)),
        }

        print_config(config_display, "Training Configuration")

        # Step 7: Confirm with user
        if not confirm("Start training with this configuration?"):
            print_info("Training cancelled.")
            return 0

        # Step 8: Execute training
        python = self.get_conda_python()
        print_info(f"Executing training with: {python}")
        print()

        exit_code = backend.execute(config, python)

        if exit_code == 0:
            print_info("Training completed successfully.")
        else:
            print_error(f"Training failed with exit code: {exit_code}")

        return exit_code
