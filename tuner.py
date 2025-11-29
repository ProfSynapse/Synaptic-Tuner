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
        if d.is_dir():
            # Include runs with final_model OR checkpoints
            has_final = (d / "final_model").exists()
            has_checkpoints = (d / "checkpoints").exists() and any((d / "checkpoints").iterdir())
            if has_final or has_checkpoints:
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

def _load_checkpoint_metrics(run_dir: Path) -> dict:
    """Load training metrics from logs for checkpoint analysis."""
    metrics = {}
    logs_dir = run_dir / "logs"

    if not logs_dir.exists():
        return metrics

    # Find the training log file
    log_files = list(logs_dir.glob("training_*.jsonl"))
    if not log_files:
        return metrics

    try:
        import json
        with open(log_files[0]) as f:
            for line in f:
                entry = json.loads(line)
                step = entry.get("step", 0)
                metrics[step] = entry
    except Exception:
        pass

    return metrics


def _detect_training_type(run_dir: Path) -> str:
    """Detect if this is SFT or KTO based on path."""
    path_str = str(run_dir).lower()
    if "kto" in path_str:
        return "kto"
    elif "sft" in path_str:
        return "sft"
    return "unknown"


def _select_model_checkpoint(run_dir: Path) -> Path:
    """
    Select between final_model and specific checkpoints with metrics display.

    Args:
        run_dir: Path to training run directory

    Returns:
        Path to selected model/checkpoint, or None if cancelled
    """
    # Load metrics from training logs
    metrics = _load_checkpoint_metrics(run_dir)
    training_type = _detect_training_type(run_dir)

    # Build list of available models
    options = []
    checkpoint_info = []

    # Final model (if exists)
    final_model = run_dir / "final_model"
    if final_model.exists():
        options.append(("final", f"{BOX['star']} final_model (recommended)"))
        checkpoint_info.append(None)

    # Checkpoints
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = sorted(
            [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
            reverse=False  # Ascending order for table display
        )
        for cp in checkpoints:
            step = int(cp.name.split("-")[1])
            step_metrics = metrics.get(step, {})
            checkpoint_info.append((cp.name, step, step_metrics))

    if not options and not checkpoint_info:
        print_error("No models found in training run")
        return None

    # If only final_model exists, use it directly
    if len(options) == 1 and not checkpoint_info:
        print_info("Using final_model")
        return final_model

    # Display checkpoint metrics table
    if checkpoint_info and any(c for c in checkpoint_info if c):
        print()
        if RICH_AVAILABLE:
            from rich.table import Table
            from rich import box as rich_box

            table = Table(
                title="Available Checkpoints",
                box=rich_box.ROUNDED,
                border_style=COLORS["cello"],
                show_header=True,
                header_style=f"bold {COLORS['aqua']}"
            )

            table.add_column("#", style=COLORS["orange"], width=4, justify="center")
            table.add_column("Checkpoint", style="white")
            table.add_column("Step", style=COLORS["sky"], justify="right")
            table.add_column("Loss", justify="right")

            if training_type == "kto":
                table.add_column("KL", justify="right")
                table.add_column("Margin", justify="right")
                table.add_column("Score", justify="right", style=COLORS["aqua"])
            else:
                table.add_column("LR", justify="right")

            idx = 1
            if final_model.exists():
                table.add_row(str(idx), f"{BOX['star']} final_model", "-", "-", "-" if training_type != "kto" else "-", "-" if training_type == "kto" else None, "-" if training_type == "kto" else None)
                idx += 1

            for info in checkpoint_info:
                if info:
                    cp_name, step, m = info
                    loss = f"{m.get('loss', 0):.4f}" if m else "-"

                    if training_type == "kto":
                        kl = m.get('kl', 0) if m else 0
                        margin = m.get('rewards/margins', 0) if m else 0
                        score = margin / kl if kl > 0 else 0

                        kl_str = f"{kl:.2f}" if m else "-"
                        margin_str = f"{margin:.2f}" if m else "-"
                        score_str = f"{score:.2f}" if m else "-"

                        table.add_row(str(idx), cp_name, str(step), loss, kl_str, margin_str, score_str)
                    else:
                        lr = m.get('learning_rate', 0) if m else 0
                        lr_str = f"{lr:.2e}" if m else "-"
                        table.add_row(str(idx), cp_name, str(step), loss, lr_str)

                    options.append((cp_name, f"{BOX['bullet']} {cp_name}"))
                    idx += 1

            console.print()
            console.print(table)

            if training_type == "kto":
                console.print(f"\n  [dim]Score = Margin/KL (higher is better: high margin, low KL)[/dim]")
            console.print()
        else:
            # Fallback text display
            print("\nAvailable Checkpoints:")
            print("-" * 60)
            if training_type == "kto":
                print(f"{'#':<4} {'Checkpoint':<20} {'Step':<8} {'Loss':<10} {'KL':<8} {'Margin':<8} {'Score':<8}")
            else:
                print(f"{'#':<4} {'Checkpoint':<20} {'Step':<8} {'Loss':<10} {'LR':<12}")
            print("-" * 60)

            idx = 1
            if final_model.exists():
                print(f"{idx:<4} {'final_model':<20} {'-':<8} {'-':<10} {'-':<8}")
                idx += 1

            for info in checkpoint_info:
                if info:
                    cp_name, step, m = info
                    loss = f"{m.get('loss', 0):.4f}" if m else "-"

                    if training_type == "kto":
                        kl = m.get('kl', 0) if m else 0
                        margin = m.get('rewards/margins', 0) if m else 0
                        score = margin / kl if kl > 0 else 0
                        print(f"{idx:<4} {cp_name:<20} {step:<8} {loss:<10} {kl:<8.2f} {margin:<8.2f} {score:<8.2f}")
                    else:
                        lr = m.get('learning_rate', 0) if m else 0
                        print(f"{idx:<4} {cp_name:<20} {step:<8} {loss:<10} {lr:<12.2e}")

                    options.append((cp_name, f"{BOX['bullet']} {cp_name}"))
                    idx += 1
            print()

    # Let user choose by number
    while True:
        try:
            sel = prompt(f"Select checkpoint (1-{len(options)})", "1")
            idx = int(sel) - 1
            if 0 <= idx < len(options):
                choice = options[idx][0]
                break
        except ValueError:
            pass
        print_error("Invalid selection.")

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

    # Get repo ID (use HF_USERNAME from .env if available)
    hf_username = os.environ.get("HF_USERNAME", "")
    if hf_username:
        print_info(f"HuggingFace username: {hf_username}")
        model_name = prompt("Model name", "")
        if not model_name:
            print_error("Model name required")
            return
        repo_id = f"{hf_username}/{model_name}"
    else:
        repo_id = prompt("HuggingFace repo ID (username/model-name)")
        if not repo_id or "/" not in repo_id:
            print_error("Invalid repo ID format. Add HF_USERNAME to .env for easier input.")
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

    # Execute upload using conda Python (needs GPU/unsloth)
    python = get_conda_python()
    upload_script = REPO_ROOT / "Trainers" / "shared" / "upload" / "cli" / "upload_cli.py"

    cmd = [
        python,
        str(upload_script),
        str(model_path),
        repo_id,
        "--save-method", save_method,
    ]
    if create_gguf:
        cmd.append("--create-gguf")

    print_info(f"Running: {' '.join(cmd)}")
    print()
    subprocess.run(cmd, cwd=str(REPO_ROOT))


# =============================================================================
# EVALUATION MENU
# =============================================================================

def _list_ollama_models() -> list:
    """List available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return []

        models = []
        lines = result.stdout.strip().split("\n")
        for line in lines[1:]:  # Skip header
            if line.strip():
                # Format: NAME ID SIZE MODIFIED
                parts = line.split()
                if parts:
                    models.append(parts[0])  # Model name
        return models
    except Exception:
        return []


def _list_lmstudio_models() -> list:
    """List available LM Studio models via API."""
    try:
        import urllib.request
        import json

        # LM Studio API endpoint
        req = urllib.request.Request(
            "http://localhost:1234/v1/models",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [m.get("id", "") for m in data.get("data", [])]
            return [m for m in models if m]
    except Exception:
        return []


def _list_prompt_sets() -> list:
    """List available prompt sets with their prompt counts."""
    import json
    prompts_dir = REPO_ROOT / "Evaluator" / "prompts"
    prompt_sets = []

    # Define prompt sets with descriptions (order = display order)
    sets_info = [
        ("full_coverage", "Tool Coverage - Comprehensive tool testing"),
        ("behavior_rubric", "Behavior Rubric - Behavioral pattern evaluation"),
        ("behavioral_patterns", "Behavioral Patterns - Additional behaviors"),
        ("baseline", "Baseline - Quick smoke test"),
    ]

    for name, description in sets_info:
        filepath = prompts_dir / f"{name}.json"
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                # Handle both list format and dict with "prompts" key
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    count = len(data.get("prompts", data.get("cases", [])))
                else:
                    count = 0
                prompt_sets.append((name, description, count))
            except Exception:
                pass

    return prompt_sets


def eval_menu():
    """Evaluation submenu - test model performance."""
    print_header("EVALUATION", "Test your model's performance")

    backend = print_menu([
        ("ollama", f"{BOX['bullet']} Ollama (local)"),
        ("lmstudio", f"{BOX['bullet']} LM Studio (local)"),
    ], "Select backend:")

    if not backend:
        return

    # List available models from selected backend
    print_info(f"Fetching models from {backend}...")

    if backend == "ollama":
        models = _list_ollama_models()
    else:
        models = _list_lmstudio_models()

    if not models:
        print_error(f"No models found. Is {backend} running?")
        if backend == "lmstudio":
            print_info("Make sure LM Studio server is running on http://localhost:1234")
        return

    # Display model selection
    if RICH_AVAILABLE:
        from rich.table import Table
        from rich import box as rich_box

        table = Table(
            title=f"Available {backend.title()} Models",
            box=rich_box.ROUNDED,
            border_style=COLORS["cello"],
        )
        table.add_column("#", style=COLORS["orange"], width=4, justify="center")
        table.add_column("Model", style="white")

        for i, m in enumerate(models, 1):
            table.add_row(str(i), m)

        console.print()
        console.print(table)
        console.print()
    else:
        print(f"\nAvailable {backend} models:")
        for i, m in enumerate(models, 1):
            print(f"  [{i}] {m}")
        print()

    # Select model
    while True:
        try:
            sel = prompt(f"Select model (1-{len(models)})", "1")
            idx = int(sel) - 1
            if 0 <= idx < len(models):
                model = models[idx]
                break
        except ValueError:
            pass
        print_error("Invalid selection.")

    # Prompt set - show all available with counts
    prompt_sets = _list_prompt_sets()

    if not prompt_sets:
        print_error("No prompt sets found in Evaluator/prompts/")
        return

    # Display prompt sets with counts
    if RICH_AVAILABLE:
        from rich.table import Table
        from rich import box as rich_box

        table = Table(
            title="Available Prompt Sets",
            box=rich_box.ROUNDED,
            border_style=COLORS["cello"],
        )
        table.add_column("#", style=COLORS["orange"], width=4, justify="center")
        table.add_column("Name", style="white")
        table.add_column("Description", style="dim")
        table.add_column("Tests", style=COLORS["aqua"], justify="right")

        for i, (name, desc, count) in enumerate(prompt_sets, 1):
            table.add_row(str(i), name, desc, str(count))

        console.print()
        console.print(table)
        console.print()
    else:
        print("\nAvailable prompt sets:")
        for i, (name, desc, count) in enumerate(prompt_sets, 1):
            print(f"  [{i}] {name} ({count} tests) - {desc}")
        print()

    # Select prompt set
    while True:
        try:
            sel = prompt(f"Select prompt set (1-{len(prompt_sets)})", "1")
            idx = int(sel) - 1
            if 0 <= idx < len(prompt_sets):
                prompt_set = prompt_sets[idx][0]
                prompt_count = prompt_sets[idx][2]
                break
        except ValueError:
            pass
        print_error("Invalid selection.")

    prompt_file = REPO_ROOT / "Evaluator" / "prompts" / f"{prompt_set}.json"

    # Confirmation
    print_config({
        "Backend": backend,
        "Model": model,
        "Prompts": f"{prompt_file.name} ({prompt_count} tests)",
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
