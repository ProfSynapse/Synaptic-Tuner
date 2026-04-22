"""
Training workflow handler.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/train_handler.py
Purpose: Orchestrate the training workflow (platform selection, config loading, execution)
Used by: Router when 'train' command is invoked

Supports --json flag for AI-parseable output. In JSON mode:
- Returns structured training status without interactive menus
- All output is JSON formatted for programmatic parsing
"""

import shutil
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Optional

from tuner.handlers.base import BaseHandler
from tuner.backends.registry import TrainingBackendRegistry
from tuner.utils.docker_runtime import (
    CONTAINER_REPO_ROOT,
    build_docker_run_command,
    container_repo_path,
    ensure_docker_cli,
    resolve_training_image,
)
from tuner.ui import (
    print_menu,
    print_header,
    print_config,
    print_error,
    print_info,
    confirm,
    BOX,
)

# Try to import animations (optional, graceful fallback)
try:
    from shared.ui.animations import (
        play_training_start,
        play_training_complete,
        ASCIIMATICS_AVAILABLE,
    )
except ImportError:
    ASCIIMATICS_AVAILABLE = False
    play_training_start = lambda: None
    play_training_complete = lambda **kwargs: None


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


def detect_docker_platform() -> str | None:
    """Detect Docker-capable NVIDIA hardware without importing host torch."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "rtx"
    except Exception:
        pass
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

    JSON Mode (--json flag):
        In JSON mode, returns structured status about available platforms,
        methods, and configurations. Does not execute interactive menus.

    Example:
        handler = TrainHandler()
        exit_code = handler.handle()

        # With JSON mode
        handler = TrainHandler(args=args)  # args.json = True
        exit_code = handler.handle()  # Returns JSON status
    """

    def __init__(self, args: Optional[Namespace] = None):
        """Initialize handler with optional args."""
        super().__init__(args=args)

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "train"

    def can_handle_direct_mode(self) -> bool:
        """Can be invoked as 'python -m tuner train'."""
        return True

    def _get_training_status(self) -> dict:
        """
        Get training status for JSON output.

        Returns dict with available platforms, methods, and configurations.
        """
        # Detect available platforms
        has_cuda = False
        has_mlx = False

        runtime = getattr(self.args, "runtime", "native") if self.args else "native"

        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            pass
        if runtime == "docker" and not has_cuda:
            has_cuda = detect_docker_platform() == "rtx"

        try:
            import mlx.core as mx
            has_mlx = mx.metal.is_available()
        except ImportError:
            pass

        platforms = []
        if has_cuda:
            platforms.append({
                "id": "rtx",
                "name": "NVIDIA GPU (CUDA)",
                "methods": ["sft", "kto", "grpo"]
            })
        if has_mlx:
            platforms.append({
                "id": "mac",
                "name": "Apple Silicon (MLX)",
                "methods": ["mlx"]
            })

        docker_ok, docker_error = ensure_docker_cli() if runtime == "docker" else (True, "")

        return {
            "command": "train",
            "status": "ready" if platforms else "no_platforms",
            "platforms": platforms,
            "detected_platform": detect_platform() if runtime != "docker" else (detect_platform() or detect_docker_platform()),
            "runtime": runtime,
            "docker_available": docker_ok,
            "docker_error": docker_error or None,
        }

    @staticmethod
    def _script_name_for_config(config) -> str:
        if config.method == "grpo" and config.config_path.name == "env_config.yaml":
            return "train_env_grpo.py"
        return f"train_{config.method}.py"

    def _execute_docker_training(self, config) -> int:
        try:
            image, profile = resolve_training_image(
                self.repo_root,
                explicit_image=getattr(self.args, "docker_image", None),
                requested_profile=getattr(self.args, "docker_profile", None),
            )
        except Exception as exc:
            print_error(f"Failed to resolve Docker training image: {exc}")
            return 1

        script_name = self._script_name_for_config(config)
        trainer_dir = container_repo_path(config.trainer_dir, self.repo_root)
        command = [script_name]
        if script_name == "train_env_grpo.py":
            command.extend(["--config", container_repo_path(config.config_path, self.repo_root)])

        cmd = build_docker_run_command(
            image=image,
            repo_root=self.repo_root,
            workdir=trainer_dir,
            entrypoint="python",
            env={"PYTHONPATH": str(CONTAINER_REPO_ROOT)},
            command=command,
        )

        profile_suffix = f" ({profile})" if profile else ""
        print_info(f"Executing training in Docker with: {image}{profile_suffix}")
        print()

        try:
            process = subprocess.Popen(cmd, cwd=str(self.repo_root))
            return process.wait()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            if "process" in locals():
                process.terminate()
            return 130
        except Exception as exc:
            print_error(f"Docker training execution error: {exc}")
            return 1

    def handle(self) -> int:
        """
        Execute training workflow.

        In JSON mode, returns training status without interactive prompts.

        Returns:
            int: Exit code (0 = success, non-zero = failure)
        """
        # JSON mode: return status information
        if self.json_mode:
            status = self._get_training_status()
            self.output(status)
            return 0

        runtime = getattr(self.args, "runtime", "native") if self.args else "native"
        print_header("TRAINING", "Select your platform and training method")
        if runtime == "docker":
            print_info("Using Docker runtime for local GPU execution.")

        # Step 1: Auto-detect or select platform
        platform_choice = detect_platform()
        if runtime == "docker" and not platform_choice:
            platform_choice = detect_docker_platform()

        if platform_choice:
            platform_name = "NVIDIA GPU (CUDA)" if platform_choice == "rtx" else "Apple Silicon (MLX)"
            print_info(f"Auto-detected platform: {platform_name}")
        else:
            platform_choice = print_menu([
                ("rtx", f"{BOX['bullet']} NVIDIA GPU (RTX 3090 / CUDA) - SFT, KTO, or GRPO"),
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
        if runtime == "docker":
            if platform_choice != "rtx":
                print_error("Docker runtime currently supports NVIDIA/CUDA local training only.")
                return 1
            docker_ok, docker_error = ensure_docker_cli()
            if not docker_ok:
                print_error(docker_error)
                return 1
        else:
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

        # Play training start animation (if available)
        if ASCIIMATICS_AVAILABLE:
            play_training_start(duration_frames=40)

        if runtime == "docker":
            exit_code = self._execute_docker_training(config)
        else:
            if platform_choice == "mac":
                python = shutil.which("python3") or "python3"
            else:
                python = self.get_conda_python()
            print_info(f"Executing training with: {python}")
            print()
            exit_code = backend.execute(config, python)

        if exit_code == 0:
            # Play celebration animation on success
            if ASCIIMATICS_AVAILABLE:
                play_training_complete(simple=True, duration_frames=60)
            print_info("Training completed successfully.")
        else:
            print_error(f"Training failed with exit code: {exit_code}")

        return exit_code
