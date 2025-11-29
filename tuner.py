#!/usr/bin/env python3
"""
Toolset-Training Unified CLI

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


def detect_environment():
    """Detect if running in WSL, native Linux, or Windows."""
    if sys.platform == "win32":
        return "windows"
    elif Path("/mnt/c").exists():
        return "wsl"
    else:
        return "linux"


def print_header(title):
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_menu(options, title="Select an option"):
    """Print a menu and get user selection."""
    print(title)
    print()
    for i, (key, desc) in enumerate(options, 1):
        print(f"  [{i}] {desc}")
    print(f"  [0] Back / Exit")
    print()

    while True:
        try:
            choice = input("Enter choice: ").strip()
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except (ValueError, IndexError):
            pass
        print("Invalid choice. Try again.")


def get_conda_python():
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

    # Fallback to system python
    return sys.executable


def load_env():
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


# =============================================================================
# TRAINING
# =============================================================================

def list_training_runs(trainer_type):
    """List available training runs."""
    output_dir = REPO_ROOT / "Trainers" / f"rtx3090_{trainer_type}" / f"{trainer_type}_output_rtx3090"
    if not output_dir.exists():
        return []

    runs = []
    for d in sorted(output_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "final_model").exists():
            runs.append(d)
    return runs[:10]


def train_menu():
    """Training submenu."""
    print_header("TRAINING")

    # Platform selection
    env = detect_environment()
    platform = print_menu([
        ("rtx", "NVIDIA GPU (RTX 3090 / CUDA) - SFT or KTO training"),
        ("mac", "Apple Silicon (M1/M2/M3) - MLX LoRA training"),
    ], "Select platform:")

    if not platform:
        return

    if platform == "mac":
        train_mac_menu()
        return

    # RTX training continues below
    print()
    choice = print_menu([
        ("sft", "SFT (Supervised Fine-Tuning) - Teach tool-calling from scratch"),
        ("kto", "KTO (Preference Learning) - Refine existing tool-calling"),
    ], "Select training method:")

    if not choice:
        return

    # Model size
    print()
    size = print_menu([
        ("3b", "3B - Fast iteration (~8GB VRAM)"),
        ("7b", "7B - Recommended (~10GB VRAM)"),
        ("13b", "13B - High quality (~16GB VRAM)"),
    ], "Select model size:")

    if not size:
        return

    # Dataset
    print()
    print("Dataset options:")
    print()

    default_dataset = f"Datasets/{'tools_sft_v1.3_11.27.25.jsonl' if choice == 'sft' else 'behavior_merged_kto_v1.3.jsonl'}"
    print(f"  Default: {default_dataset}")
    custom = input("  Custom dataset path (or Enter for default): ").strip()
    dataset = custom if custom else default_dataset

    # Confirm
    print()
    print_header("TRAINING CONFIGURATION")
    print(f"  Platform: NVIDIA RTX")
    print(f"  Method: {choice.upper()}")
    print(f"  Model size: {size}")
    print(f"  Dataset: {dataset}")
    print()

    confirm = input("Start training? (y/N): ").strip().lower()
    if confirm != "y":
        print("Training cancelled.")
        return

    # Run training
    trainer_dir = REPO_ROOT / "Trainers" / f"rtx3090_{choice}"
    python = get_conda_python()

    cmd = [
        python,
        f"train_{choice}.py",
        "--model-size", size,
        "--local-file", str(REPO_ROOT / dataset),
    ]

    print()
    print(f"Running: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, cwd=str(trainer_dir))


def train_mac_menu():
    """Mac (Apple Silicon) training submenu."""
    print()
    print_header("MAC TRAINING (MLX)")

    # Check for config
    config_path = REPO_ROOT / "Trainers" / "mistral_lora_mac" / "config" / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return

    # Dataset
    print("Dataset options:")
    print()
    default_dataset = "Datasets/tools_sft_v1.3_11.27.25.jsonl"
    print(f"  Default: {default_dataset}")
    custom = input("  Custom dataset path (or Enter for default): ").strip()
    dataset = custom if custom else default_dataset

    # Resume from checkpoint?
    print()
    resume = input("Resume from checkpoint? (path or Enter to skip): ").strip()

    # Confirm
    print()
    print_header("MAC TRAINING CONFIGURATION")
    print(f"  Platform: Apple Silicon (MLX)")
    print(f"  Method: LoRA")
    print(f"  Config: {config_path}")
    print(f"  Dataset: {dataset}")
    if resume:
        print(f"  Resume from: {resume}")
    print()

    confirm = input("Start training? (y/N): ").strip().lower()
    if confirm != "y":
        print("Training cancelled.")
        return

    # Run training
    trainer_dir = REPO_ROOT / "Trainers" / "mistral_lora_mac"

    cmd = [
        sys.executable,
        "main.py",
        "--config", str(config_path),
    ]
    if resume:
        cmd.extend(["--resume", resume])

    print()
    print(f"Running: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, cwd=str(trainer_dir))


# =============================================================================
# UPLOAD
# =============================================================================

def upload_menu():
    """Upload submenu."""
    print_header("UPLOAD TO HUGGINGFACE")

    # Check for HF token
    load_env()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not hf_token:
        print("Error: HF_TOKEN not found in .env file")
        print("Create .env with: HF_TOKEN=hf_your_token_here")
        return

    # Select trainer type
    choice = print_menu([
        ("sft", "SFT model"),
        ("kto", "KTO model"),
    ], "Select model type:")

    if not choice:
        return

    # List runs
    runs = list_training_runs(choice)
    if not runs:
        print(f"No training runs found for {choice.upper()}")
        return

    print()
    print("Available training runs:")
    print()
    for i, run in enumerate(runs, 1):
        print(f"  [{i}] {run.name}")
    print()

    while True:
        try:
            sel = input(f"Select run (1-{len(runs)}): ").strip()
            idx = int(sel) - 1
            if 0 <= idx < len(runs):
                selected_run = runs[idx]
                break
        except ValueError:
            pass
        print("Invalid selection.")

    model_path = selected_run / "final_model"

    # Get repo ID
    print()
    repo_id = input("HuggingFace repo ID (username/model-name): ").strip()
    if not repo_id or "/" not in repo_id:
        print("Invalid repo ID format")
        return

    # Save method
    print()
    save_method = print_menu([
        ("merged_16bit", "Merged 16-bit (~14GB) - Full quality"),
        ("merged_4bit", "Merged 4-bit (~3.5GB) - Smaller"),
        ("lora", "LoRA adapters only (~320MB) - Fastest"),
    ], "Select save method:")

    if not save_method:
        return

    # GGUF
    print()
    create_gguf = input("Create GGUF versions? (y/N): ").strip().lower() == "y"

    # Confirm
    print()
    print_header("UPLOAD CONFIGURATION")
    print(f"  Model: {model_path}")
    print(f"  Repository: {repo_id}")
    print(f"  Save method: {save_method}")
    print(f"  GGUF: {'Yes' if create_gguf else 'No'}")
    print()

    confirm = input("Start upload? (y/N): ").strip().lower()
    if confirm != "y":
        print("Upload cancelled.")
        return

    # Run upload using shared framework
    from upload.cli.upload_cli import main as upload_main

    args = [
        str(model_path),
        repo_id,
        "--save-method", save_method,
    ]
    if create_gguf:
        args.append("--create-gguf")

    upload_main(args)


# =============================================================================
# EVALUATION
# =============================================================================

def eval_menu():
    """Evaluation submenu."""
    print_header("EVALUATION")

    # Backend selection
    backend = print_menu([
        ("ollama", "Ollama (local)"),
        ("lmstudio", "LM Studio (local)"),
    ], "Select backend:")

    if not backend:
        return

    # Model name
    print()
    model = input("Model name (as shown in Ollama/LM Studio): ").strip()
    if not model:
        print("Model name required")
        return

    # Prompt set
    print()
    prompt_set = print_menu([
        ("baseline", "Baseline - General scenarios"),
        ("behavior_rubric", "Behavior rubric - Behavior evaluation"),
    ], "Select prompt set:")

    if not prompt_set:
        return

    prompt_file = REPO_ROOT / "Evaluator" / "prompts" / f"{prompt_set}.json"

    # Confirm
    print()
    print_header("EVALUATION CONFIGURATION")
    print(f"  Backend: {backend}")
    print(f"  Model: {model}")
    print(f"  Prompts: {prompt_file.name}")
    print()

    confirm = input("Start evaluation? (y/N): ").strip().lower()
    if confirm != "y":
        print("Evaluation cancelled.")
        return

    # Run evaluation
    python = get_conda_python()
    cmd = [
        python, "-m", "Evaluator.cli",
        "--backend", backend,
        "--model", model,
        "--prompt-set", str(prompt_file),
    ]

    print()
    print(f"Running: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, cwd=str(REPO_ROOT))


# =============================================================================
# PIPELINE
# =============================================================================

def pipeline_menu():
    """Full pipeline: Train -> Upload -> Evaluate."""
    print_header("FULL PIPELINE")
    print("This will run: Training -> Upload -> Evaluation")
    print()

    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != "y":
        return

    # Run each step
    print_header("STEP 1: TRAINING")
    train_menu()

    print()
    cont = input("Continue to upload? (y/N): ").strip().lower()
    if cont != "y":
        return

    print_header("STEP 2: UPLOAD")
    upload_menu()

    print()
    cont = input("Continue to evaluation? (y/N): ").strip().lower()
    if cont != "y":
        return

    print_header("STEP 3: EVALUATION")
    eval_menu()

    print_header("PIPELINE COMPLETE")


# =============================================================================
# MAIN
# =============================================================================

def main_menu():
    """Main interactive menu."""
    print_header("TOOLSET-TRAINING CLI")

    env = detect_environment()
    print(f"Environment: {env}")
    print(f"Repo root: {REPO_ROOT}")

    if load_env():
        print("Loaded .env file")

    while True:
        print()
        choice = print_menu([
            ("train", "Train a model (SFT or KTO)"),
            ("upload", "Upload model to HuggingFace"),
            ("eval", "Evaluate a model"),
            ("pipeline", "Full pipeline (Train -> Upload -> Eval)"),
        ], "What would you like to do?")

        if not choice:
            print("Goodbye!")
            break

        if choice == "train":
            train_menu()
        elif choice == "upload":
            upload_menu()
        elif choice == "eval":
            eval_menu()
        elif choice == "pipeline":
            pipeline_menu()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Toolset-Training Unified CLI",
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
