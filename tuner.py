#!/usr/bin/env python3
"""
Synaptic Tuner - Fine-tuning CLI for the Claudesidian MCP

Run from repo root:
    python tuner.py          # Interactive mode
    python tuner.py train    # Training submenu
    python tuner.py upload   # Upload submenu
    python tuner.py eval     # Evaluation submenu
    python tuner.py pipeline # Full pipeline (train -> upload -> eval)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Ensure we're in the repo root
REPO_ROOT = Path(__file__).parent.resolve()
os.chdir(REPO_ROOT)

# Add paths
sys.path.insert(0, str(REPO_ROOT / "Trainers" / "shared"))

# Import UI components (with graceful fallback)
try:
    from ui import (
        COLORS, BOX, RICH_AVAILABLE, console,
        clear_screen, print_logo, print_header, print_menu,
        print_config, print_success, print_error, print_info,
        confirm, prompt, animated_menu,
    )
except ImportError:
    # Fallback if UI module not available
    RICH_AVAILABLE = False
    console = None
    COLORS = {}
    BOX = {"bullet": "*", "star": "*", "check": "[OK]", "cross": "[X]", "arrow": ">", "dot": "-"}

    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_logo(small=False):
        print("\n" + "=" * 60)
        print("  SYNAPTIC TUNER")
        print("  Fine-tuning for the Claudesidian MCP")
        print("=" * 60 + "\n")

    def print_header(title, subtitle=None):
        print("\n" + "=" * 60)
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print("=" * 60 + "\n")

    def print_menu(options, title="Select"):
        print(title + "\n")
        for i, (k, d) in enumerate(options, 1):
            print(f"  [{i}] {d}")
        print("  [0] Back / Exit\n")
        while True:
            try:
                c = input("Enter choice: ").strip()
                if c == "0":
                    return None
                idx = int(c) - 1
                if 0 <= idx < len(options):
                    return options[idx][0]
            except (ValueError, IndexError):
                pass
            print("Invalid choice.")

    def print_config(config, title="Configuration"):
        print(f"\n  {title}\n  " + "-" * 40)
        for k, v in config.items():
            print(f"    {k}: {v}")
        print()

    def print_success(msg):
        print(f"  [OK] {msg}")

    def print_error(msg):
        print(f"  [ERROR] {msg}")

    def print_info(msg):
        print(f"  [INFO] {msg}")

    def confirm(msg):
        return input(f"  {msg} (y/N): ").strip().lower() == "y"

    def prompt(msg, default=""):
        if default:
            r = input(f"  {msg} [{default}]: ").strip()
            return r if r else default
        return input(f"  {msg}: ").strip()

    def animated_menu(options, title="Select", status_info=None):
        print_logo()
        if status_info:
            for k, v in status_info.items():
                print(f"  {k}: {v}")
        return print_menu(options, title)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_environment() -> str:
    """Detect if running in WSL, native Linux, or Windows."""
    if sys.platform == "win32":
        return "windows"
    elif Path("/mnt/c").exists():
        return "wsl"
    else:
        return "linux"


def get_conda_python() -> str:
    """Find Python from conda environment."""
    env = detect_environment()

    if env == "windows":
        paths = [
            Path(os.environ.get("USERPROFILE", "")) / "miniconda3" / "envs" / "unsloth_env" / "python.exe",
            Path(os.environ.get("USERPROFILE", "")) / "anaconda3" / "envs" / "unsloth_env" / "python.exe",
            Path("C:/ProgramData/miniconda3/envs/unsloth_env/python.exe"),
            Path("C:/ProgramData/anaconda3/envs/unsloth_env/python.exe"),
        ]
    else:
        paths = [
            Path.home() / "miniconda3" / "envs" / "unsloth_env" / "bin" / "python",
            Path.home() / "anaconda3" / "envs" / "unsloth_env" / "bin" / "python",
            Path("/opt/conda/envs/unsloth_env/bin/python"),
        ]

    for p in paths:
        if p.exists():
            return str(p)

    return sys.executable


def load_env() -> bool:
    """Load .env file if exists."""
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        return True
    return False


def list_training_runs(trainer_type: str) -> list:
    """List available training runs for a trainer type."""
    output_dir = REPO_ROOT / "Trainers" / f"rtx3090_{trainer_type}" / f"{trainer_type}_output_rtx3090"
    if not output_dir.exists():
        return []

    runs = []
    for d in sorted(output_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "final_model").exists():
            runs.append(d)
    return runs[:10]


# =============================================================================
# TRAINING MENUS
# =============================================================================

def train_menu():
    """Training submenu - select platform and training method."""
    print_header("TRAINING", "Select your platform and training method")

    platform = print_menu([
        ("rtx", f"{BOX['bullet']} NVIDIA GPU (RTX 3090 / CUDA) - SFT or KTO"),
        ("mac", f"{BOX['bullet']} Apple Silicon (M1/M2/M3) - MLX LoRA"),
    ], "Select platform:")

    if not platform:
        return

    if platform == "mac":
        _train_mac()
        return

    _train_rtx()


def _train_rtx():
    """RTX/CUDA training flow - pulls config directly from config.yaml."""
    method = print_menu([
        ("sft", f"{BOX['bullet']} SFT (Supervised Fine-Tuning) - Teach from scratch"),
        ("kto", f"{BOX['bullet']} KTO (Preference Learning) - Refine existing"),
    ], "Select training method:")

    if not method:
        return

    # Load config from YAML
    trainer_dir = REPO_ROOT / "Trainers" / f"rtx3090_{method}"
    config_path = trainer_dir / "configs" / "config.yaml"

    if not config_path.exists():
        print_error(f"Config not found: {config_path}")
        return

    # Parse config for display
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_name = config.get('model', {}).get('model_name', 'Unknown')
        dataset_file = config.get('dataset', {}).get('local_file', 'Unknown')
        epochs = config.get('training', {}).get('num_train_epochs', 1)
        batch_size = config.get('training', {}).get('per_device_train_batch_size', 4)
        lr = config.get('training', {}).get('learning_rate', 0)

        # Extract model size from name
        model_display = model_name.split('/')[-1] if '/' in model_name else model_name

        print_config({
            "Method": method.upper(),
            "Model": model_display,
            "Dataset": Path(dataset_file).name if dataset_file else "Unknown",
            "Epochs": str(epochs),
            "Batch Size": str(batch_size),
            "Learning Rate": str(lr),
            "Config": str(config_path.relative_to(REPO_ROOT)),
        }, "Training Configuration (from config.yaml)")

    except Exception as e:
        print_error(f"Failed to parse config: {e}")
        return

    if not confirm("Start training with this configuration?"):
        print_info("Training cancelled.")
        return

    # Execute training - no CLI overrides, uses config.yaml directly
    python = get_conda_python()
    cmd = [python, f"train_{method}.py"]

    print_info(f"Running: {' '.join(cmd)}")
    print()
    subprocess.run(cmd, cwd=str(trainer_dir))


def _train_mac():
    """Mac/Apple Silicon training flow - pulls config directly from config.yaml."""
    print_header("MAC TRAINING", "Apple Silicon MLX LoRA")

    trainer_dir = REPO_ROOT / "Trainers" / "mistral_lora_mac"
    config_path = trainer_dir / "config" / "config.yaml"

    if not config_path.exists():
        print_error(f"Config not found: {config_path}")
        return

    # Parse config for display
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_name = config.get('model', {}).get('model_name', 'Unknown')
        dataset_path = config.get('data', {}).get('dataset_path', 'Unknown')
        epochs = config.get('training', {}).get('num_epochs', 1)
        batch_size = config.get('training', {}).get('per_device_batch_size', 2)
        lr = config.get('training', {}).get('learning_rate', 0)

        model_display = model_name.split('/')[-1] if '/' in model_name else model_name

        print_config({
            "Platform": "Apple Silicon (MLX)",
            "Method": "LoRA",
            "Model": model_display,
            "Dataset": Path(dataset_path).name if dataset_path else "Unknown",
            "Epochs": str(epochs),
            "Batch Size": str(batch_size),
            "Learning Rate": str(lr),
            "Config": str(config_path.relative_to(REPO_ROOT)),
        }, "Training Configuration (from config.yaml)")

    except Exception as e:
        print_error(f"Failed to parse config: {e}")
        return

    # Resume option (only prompt needed)
    resume = prompt("Resume from checkpoint (Enter to skip)", "")

    if not confirm("Start training with this configuration?"):
        print_info("Training cancelled.")
        return

    # Execute training
    cmd = [sys.executable, "main.py", "--config", str(config_path)]
    if resume:
        cmd.extend(["--resume", resume])

    print_info(f"Running: {' '.join(cmd)}")
    print()
    subprocess.run(cmd, cwd=str(trainer_dir))


# =============================================================================
# UPLOAD MENU
# =============================================================================

def _select_model_checkpoint(run_dir: Path) -> Path:
    """
    Select between final_model and specific checkpoints.

    Args:
        run_dir: Path to training run directory

    Returns:
        Path to selected model/checkpoint, or None if cancelled
    """
    # Build list of available models
    options = []

    # Final model (if exists)
    final_model = run_dir / "final_model"
    if final_model.exists():
        options.append(("final", f"{BOX['star']} final_model (recommended)"))

    # Checkpoints
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
            reverse=True  # Most recent first
        )
        for cp in checkpoints:
            step = cp.name.split("-")[1]
            options.append((cp.name, f"{BOX['bullet']} {cp.name} (step {step})"))

    if not options:
        print_error("No models found in training run")
        return None

    # If only final_model exists, use it directly
    if len(options) == 1 and options[0][0] == "final":
        print_info("Using final_model")
        return final_model

    # Let user choose
    choice = print_menu(options, "Select model to upload:")

    if not choice:
        return None

    if choice == "final":
        return final_model
    else:
        return checkpoints_dir / choice


def upload_menu():
    """Upload submenu - push model to HuggingFace."""
    print_header("UPLOAD", "Push your model to HuggingFace")

    # Check token
    load_env()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not hf_token:
        print_error("HF_TOKEN not found in .env file")
        print_info("Create .env with: HF_TOKEN=hf_your_token_here")
        return

    print_success("HuggingFace token found")

    # Select model type
    choice = print_menu([
        ("sft", f"{BOX['bullet']} SFT model"),
        ("kto", f"{BOX['bullet']} KTO model"),
    ], "Select model type:")

    if not choice:
        return

    # List available runs
    runs = list_training_runs(choice)
    if not runs:
        print_error(f"No training runs found for {choice.upper()}")
        return

    # Display runs
    if RICH_AVAILABLE:
        from rich.table import Table
        from rich import box as rich_box
        from ui import STYLES

        table = Table(show_header=True, header_style=STYLES["header"], box=rich_box.ROUNDED, border_style=COLORS["cello"])
        table.add_column("#", style=COLORS["orange"], width=4)
        table.add_column("Training Run", style="white")
        for i, run in enumerate(runs, 1):
            table.add_row(str(i), run.name)
        console.print()
        console.print(table)
        console.print()
    else:
        print("\nAvailable training runs:")
        for i, run in enumerate(runs, 1):
            print(f"  [{i}] {run.name}")
        print()

    # Select run
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

    # Select checkpoint or final model
    model_path = _select_model_checkpoint(selected_run)
    if not model_path:
        return

    # Get repo ID
    repo_id = prompt("HuggingFace repo ID (username/model-name)")
    if not repo_id or "/" not in repo_id:
        print_error("Invalid repo ID format")
        return

    # Save method
    save_method = print_menu([
        ("merged_16bit", f"{BOX['star']} Merged 16-bit (~14GB) - Full quality"),
        ("merged_4bit", f"{BOX['bullet']} Merged 4-bit (~3.5GB) - Smaller"),
        ("lora", f"{BOX['bullet']} LoRA adapters only (~320MB) - Fastest"),
    ], "Select save method:")

    if not save_method:
        return

    # GGUF option
    create_gguf = confirm("Create GGUF versions?")

    # Confirmation
    print_config({
        "Model": str(model_path),
        "Repository": repo_id,
        "Save Method": save_method,
        "GGUF": "Yes" if create_gguf else "No",
    }, "Upload Configuration")

    if not confirm("Start upload?"):
        print_info("Upload cancelled.")
        return

    # Execute upload
    from upload.cli.upload_cli import main as upload_main
    args = [str(model_path), repo_id, "--save-method", save_method]
    if create_gguf:
        args.append("--create-gguf")
    upload_main(args)


# =============================================================================
# EVALUATION MENU
# =============================================================================

def eval_menu():
    """Evaluation submenu - test model performance."""
    print_header("EVALUATION", "Test your model's performance")

    backend = print_menu([
        ("ollama", f"{BOX['bullet']} Ollama (local)"),
        ("lmstudio", f"{BOX['bullet']} LM Studio (local)"),
    ], "Select backend:")

    if not backend:
        return

    # Model name
    model = prompt("Model name (as shown in Ollama/LM Studio)")
    if not model:
        print_error("Model name required")
        return

    # Prompt set
    prompt_set = print_menu([
        ("baseline", f"{BOX['bullet']} Baseline - General scenarios"),
        ("behavior_rubric", f"{BOX['bullet']} Behavior rubric - Behavior evaluation"),
    ], "Select prompt set:")

    if not prompt_set:
        return

    prompt_file = REPO_ROOT / "Evaluator" / "prompts" / f"{prompt_set}.json"

    # Confirmation
    print_config({
        "Backend": backend,
        "Model": model,
        "Prompts": prompt_file.name,
    }, "Evaluation Configuration")

    if not confirm("Start evaluation?"):
        print_info("Evaluation cancelled.")
        return

    # Execute evaluation
    python = get_conda_python()
    cmd = [python, "-m", "Evaluator.cli", "--backend", backend, "--model", model, "--prompt-set", str(prompt_file)]

    print_info(f"Running: {' '.join(cmd)}")
    print()
    subprocess.run(cmd, cwd=str(REPO_ROOT))


# =============================================================================
# PIPELINE MENU
# =============================================================================

def pipeline_menu():
    """Full pipeline: Train -> Upload -> Evaluate."""
    print_header("FULL PIPELINE", "Train -> Upload -> Evaluate")

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

    if not confirm("Continue with full pipeline?"):
        return

    # Step 1
    print_header("STEP 1: TRAINING")
    train_menu()

    if not confirm("Continue to upload?"):
        return

    # Step 2
    print_header("STEP 2: UPLOAD")
    upload_menu()

    if not confirm("Continue to evaluation?"):
        return

    # Step 3
    print_header("STEP 3: EVALUATION")
    eval_menu()

    print_header("PIPELINE COMPLETE", "All steps finished successfully!")


# =============================================================================
# MAIN MENU
# =============================================================================

def main_menu():
    """Main interactive menu."""
    env = detect_environment()
    env_loaded = load_env()

    # Build status info
    status_info = {
        "Environment": env.upper(),
        "Repo": str(REPO_ROOT),
    }
    if env_loaded:
        status_info["Config"] = ".env loaded"

    # Menu options
    menu_options = [
        ("train", f"{BOX['star']} Train a model (SFT, KTO, or MLX)"),
        ("upload", f"{BOX['bullet']} Upload model to HuggingFace"),
        ("eval", f"{BOX['bullet']} Evaluate a model"),
        ("pipeline", f"{BOX['bullet']} Full pipeline (Train -> Upload -> Eval)"),
    ]

    # First menu shows animated logo with bubbling test tube
    first_run = True

    while True:
        if first_run:
            # Animated menu with bubbling test tube
            choice = animated_menu(
                menu_options,
                "What would you like to do?",
                status_info
            )
            first_run = False
        else:
            # Static menu after first choice
            print()
            choice = print_menu(menu_options, "What would you like to do?")

        if not choice:
            if RICH_AVAILABLE:
                console.print()
                console.print(f"  [{COLORS['purple']}]Thanks for using Synaptic Tuner![/{COLORS['purple']}]")
                console.print(f"  [{COLORS['cello']}]Goodbye![/{COLORS['cello']}]")
                console.print()
            else:
                print("\n  Thanks for using Synaptic Tuner!")
                print("  Goodbye!\n")
            break

        if choice == "train":
            train_menu()
        elif choice == "upload":
            upload_menu()
        elif choice == "eval":
            eval_menu()
        elif choice == "pipeline":
            pipeline_menu()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Synaptic Tuner - Fine-tuning CLI for the Claudesidian MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (none)    Interactive menu
  train     Training submenu
  upload    Upload submenu
  eval      Evaluation submenu
  pipeline  Full pipeline (train -> upload -> eval)

Examples:
  python tuner.py           # Interactive mode
  python tuner.py train     # Go directly to training
  python tuner.py upload    # Go directly to upload
"""
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "upload", "eval", "pipeline"],
        help="Command to run (optional, defaults to interactive menu)"
    )

    args = parser.parse_args()

    # Show logo for direct commands
    if args.command:
        print_logo(small=True)

    if args.command == "train":
        train_menu()
    elif args.command == "upload":
        upload_menu()
    elif args.command == "eval":
        eval_menu()
    elif args.command == "pipeline":
        pipeline_menu()
    else:
        main_menu()


if __name__ == "__main__":
    main()
