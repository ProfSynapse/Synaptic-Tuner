"""
Location: Trainers/cloud/train_modal.py

Purpose:
    Standalone Modal wrapper script for running SFT or KTO training in the cloud.
    Can be invoked directly via `modal run Trainers/cloud/train_modal.py` or
    programmatically from the ModalBackend in tuner/backends/training/cloud/.

    The script defines a Modal App with GPU configuration, persistent volumes for
    caching model weights, and a container image with all training dependencies.
    Training runs execute the existing train_sft.py or train_kto.py scripts inside
    the Modal container, then push results to HuggingFace Hub.

Usage:
    # Run SFT training on an L40S GPU (default)
    modal run Trainers/cloud/train_modal.py --trainer-type sft

    # Run KTO training on an A100 GPU
    modal run Trainers/cloud/train_modal.py --trainer-type kto --gpu A100

    # Specify a custom model and dataset
    modal run Trainers/cloud/train_modal.py \\
        --trainer-type sft \\
        --model-name "unsloth/mistral-7b-v0.3-bnb-4bit" \\
        --dataset-path "Datasets/my_dataset.jsonl" \\
        --hub-repo "myuser/my-model"

Dependencies:
    - modal (pip install modal)
    - Modal account with token configured (modal setup)
    - HF_TOKEN environment variable (for model downloads and Hub uploads)
"""

import os
import re
import subprocess
import sys

try:
    import modal
except ImportError:
    print(
        "Error: modal package is not installed.\n"
        "Install it with: pip install modal\n"
        "Then authenticate with: modal setup"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Modal App Configuration
# ---------------------------------------------------------------------------

HOURS = 60 * 60  # seconds

# Persistent volume for caching HuggingFace model weights between runs.
# This avoids re-downloading multi-GB models on every training run, saving
# both time and bandwidth costs.
model_cache = modal.Volume.from_name(
    "toolset-model-cache", create_if_missing=True
)

# Persistent volume for storing training checkpoints between runs.
# Allows resuming interrupted training without losing progress.
checkpoint_cache = modal.Volume.from_name(
    "toolset-checkpoints", create_if_missing=True
)

# Container image with all training dependencies pre-installed.
# Using debian_slim as the base keeps the image small while providing
# a stable foundation. Dependencies are installed via pip_install since
# unsloth has complex CUDA dependency resolution that works better with pip.
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git",
        "trl>=0.15",
        "transformers>=4.46",
        "datasets",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub>=0.25",
        "pyyaml",
        "python-dotenv",
        "rich",
    )
    .apt_install("git")
)

app = modal.App("toolset-training", image=training_image)


# ---------------------------------------------------------------------------
# GPU Pricing Reference (display-only cost estimation for local entrypoint)
# ---------------------------------------------------------------------------
# Approximate Modal GPU prices per hour as of early 2026.
# Canonical pricing data is in tuner/backends/training/cloud/base_cloud.py
# GPU_PRICING["modal"]. This standalone script cannot import from tuner.*
# (runs via `modal run`), so a display-only copy is kept here.
# Check https://modal.com/pricing for current rates.
GPU_PRICING = {
    "T4": 0.59,
    "L4": 0.73,
    "A10G": 1.10,
    "L40S": 1.40,
    "A100": 2.78,       # A100-40GB
    "A100-80GB": 3.72,
    "H100": 4.89,
}

# Valid GPU types that can be passed to Modal
VALID_GPU_TYPES = ["T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100"]

DEFAULT_GPU = "L40S"


# ---------------------------------------------------------------------------
# Training Function (runs remotely on Modal)
# ---------------------------------------------------------------------------

@app.function(
    gpu=DEFAULT_GPU,
    timeout=6 * HOURS,
    volumes={
        "/cache/huggingface": model_cache,
        "/cache/checkpoints": checkpoint_cache,
    },
    # Scope secrets to only the env vars needed for training, rather than
    # exposing the entire .env file via Secret.from_dotenv().
    secrets=[modal.Secret.from_dict({
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
    })],
)
def run_training(
    trainer_type: str = "sft",
    repo_url: str = "",
    repo_branch: str = "main",
    model_name: str = "",
    dataset_path: str = "",
    hub_repo: str = "",
    config_overrides: dict = None,
):
    """Run SFT or KTO training inside the Modal container.

    This function executes remotely on Modal's GPU infrastructure. It:
    1. Clones the Toolset-Training repo into the container
    2. Sets up the HuggingFace cache to use the persistent volume
    3. Runs the appropriate training script (train_sft.py or train_kto.py)
    4. Uploads results to HuggingFace Hub if hub_repo is specified

    Args:
        trainer_type: Training method - "sft" or "kto"
        repo_url: Git URL to clone the training repo from
        repo_branch: Git branch to checkout
        model_name: Override model name (uses config.yaml default if empty)
        dataset_path: Override dataset path relative to repo root
        hub_repo: HuggingFace Hub repo ID to push trained model to
        config_overrides: Dict of CLI argument overrides (learning_rate, epochs, etc.)
    """
    if config_overrides is None:
        config_overrides = {}

    # Validate trainer type
    if trainer_type not in ("sft", "kto"):
        raise ValueError(f"Invalid trainer_type: {trainer_type}. Must be 'sft' or 'kto'.")

    # Point HuggingFace cache at the persistent volume to avoid re-downloading models
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

    # Clone the repo
    workspace = "/workspace/toolset-training"
    if repo_url:
        print(f"[Modal] Cloning repo: {repo_url} (branch: {repo_branch})")
        clone_result = subprocess.run(
            ["git", "clone", "--branch", repo_branch, "--depth", "1", repo_url, workspace],
            capture_output=True,
            text=True,
        )
        if clone_result.returncode != 0:
            # Scrub any credentials from stderr before logging/raising
            safe_stderr = re.sub(r'https?://[^@\s]+@', 'https://[REDACTED]@', clone_result.stderr)
            print(f"[Modal] Git clone failed: {safe_stderr}")
            raise RuntimeError(f"Failed to clone repo: {safe_stderr}")
    else:
        print("[Modal] No repo_url provided, skipping clone.")
        print("[Modal] Ensure training code is available in the container.")
        raise ValueError(
            "repo_url is required. Provide the git URL of your Toolset-Training repo."
        )

    # Determine trainer directory and script
    trainer_dir = os.path.join(workspace, "Trainers", f"rtx3090_{trainer_type}")
    train_script = f"train_{trainer_type}.py"

    if not os.path.isfile(os.path.join(trainer_dir, train_script)):
        raise FileNotFoundError(
            f"Training script not found: {os.path.join(trainer_dir, train_script)}"
        )

    # Build training command with CLI overrides
    cmd = ["python", train_script]

    if model_name:
        # Model name override requires modifying config before training.
        # For now, this is noted as a future enhancement -- the config.yaml
        # in the repo should be pre-configured with the desired model.
        print(f"[Modal] Note: model_name override '{model_name}' requires config.yaml update")

    if dataset_path:
        # Convert relative dataset path to absolute within the workspace
        abs_dataset = os.path.join(workspace, dataset_path)
        cmd.extend(["--local-file", abs_dataset])

    # Apply config overrides as CLI arguments
    if config_overrides.get("learning_rate"):
        cmd.extend(["--learning-rate", str(config_overrides["learning_rate"])])
    if config_overrides.get("num_epochs"):
        cmd.extend(["--num-epochs", str(config_overrides["num_epochs"])])
    if config_overrides.get("batch_size"):
        cmd.extend(["--batch-size", str(config_overrides["batch_size"])])
    if config_overrides.get("max_seq_length"):
        cmd.extend(["--max-seq-length", str(config_overrides["max_seq_length"])])
    if config_overrides.get("max_steps"):
        cmd.extend(["--max-steps", str(config_overrides["max_steps"])])

    print(f"[Modal] Running: {' '.join(cmd)}")
    print(f"[Modal] Working directory: {trainer_dir}")

    # Run training
    process = subprocess.run(
        cmd,
        cwd=trainer_dir,
        env={**os.environ},
    )

    if process.returncode != 0:
        print(f"[Modal] Training failed with exit code {process.returncode}")
        # Commit checkpoint volume so partial work is preserved
        checkpoint_cache.commit()
        raise RuntimeError(f"Training script exited with code {process.returncode}")

    print("[Modal] Training completed successfully")

    # Upload to HuggingFace Hub if requested
    if hub_repo:
        _upload_to_hub(trainer_type, trainer_dir, hub_repo)

    # Commit volumes to persist cached models and checkpoints
    model_cache.commit()
    checkpoint_cache.commit()

    return {"status": "completed", "trainer_type": trainer_type}


def _upload_to_hub(trainer_type: str, trainer_dir: str, hub_repo: str):
    """Upload trained model to HuggingFace Hub.

    Looks for the most recent training run output directory and uploads
    the final_model subdirectory to the specified Hub repository.

    Args:
        trainer_type: "sft" or "kto" (determines output directory name)
        trainer_dir: Path to the trainer directory
        hub_repo: HuggingFace Hub repo ID (e.g., "username/model-name")
    """
    from pathlib import Path
    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[Modal] Warning: HF_TOKEN not set, skipping Hub upload")
        return

    output_dir_name = f"{trainer_type}_output_rtx3090"
    output_base = Path(trainer_dir) / output_dir_name

    if not output_base.exists():
        print(f"[Modal] Warning: Output directory not found: {output_base}")
        return

    # Find the most recent run directory (sorted by timestamp name)
    run_dirs = sorted(output_base.iterdir(), reverse=True)
    if not run_dirs:
        print(f"[Modal] Warning: No training runs found in {output_base}")
        return

    latest_run = run_dirs[0]
    final_model = latest_run / "final_model"

    if not final_model.exists():
        print(f"[Modal] Warning: No final_model directory in {latest_run}")
        return

    print(f"[Modal] Uploading {final_model} to {hub_repo}")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hub_repo, exist_ok=True, private=True)
    api.upload_folder(
        folder_path=str(final_model),
        repo_id=hub_repo,
        commit_message=f"Modal cloud training ({trainer_type})",
    )
    print(f"[Modal] Upload complete: https://huggingface.co/{hub_repo}")


# ---------------------------------------------------------------------------
# Local Entrypoint (runs locally, dispatches to Modal)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    trainer_type: str = "sft",
    gpu: str = DEFAULT_GPU,
    repo_url: str = "",
    repo_branch: str = "main",
    model_name: str = "",
    dataset_path: str = "",
    hub_repo: str = "",
    learning_rate: float = 0.0,
    num_epochs: int = 0,
    batch_size: int = 0,
    max_steps: int = 0,
    timeout_hours: int = 6,
):
    """Local entrypoint for Modal cloud training.

    This function runs on your local machine and dispatches the training
    job to Modal's cloud infrastructure. Use `modal run` to invoke it.

    Args:
        trainer_type: Training method - "sft" or "kto"
        gpu: GPU type (T4, L4, A10G, L40S, A100, A100-80GB, H100)
        repo_url: Git URL to clone (auto-detected from local git remote if empty)
        repo_branch: Git branch to checkout (default: main)
        model_name: Override model name in config
        dataset_path: Override dataset path (relative to repo root)
        hub_repo: HuggingFace Hub repo ID for uploading trained model
        learning_rate: Override learning rate (0 = use config default)
        num_epochs: Override number of epochs (0 = use config default)
        batch_size: Override batch size (0 = use config default)
        max_steps: Override max training steps (0 = use config default)
        timeout_hours: Maximum job duration in hours (default: 6)
    """
    # Validate GPU type
    if gpu not in VALID_GPU_TYPES:
        print(f"Error: Invalid GPU type '{gpu}'")
        print(f"Valid options: {', '.join(VALID_GPU_TYPES)}")
        sys.exit(1)

    # Auto-detect repo URL from local git remote if not provided.
    # Mirrors tuner/backends/training/cloud/base_cloud.resolve_repo_url()
    # but kept inline because this standalone script cannot import from tuner.*.
    if not repo_url:
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                repo_url = result.stdout.strip()
                print(f"[Modal] Auto-detected repo URL: {repo_url}")
            else:
                print("Error: No --repo-url provided and could not auto-detect from git remote.")
                print("Set CLOUD_REPO_URL env var or pass --repo-url explicitly.")
                sys.exit(1)
        except FileNotFoundError:
            print("Error: git not found. Please provide --repo-url explicitly.")
            sys.exit(1)

    # Also check CLOUD_REPO_URL env var as fallback
    if not repo_url:
        repo_url = os.environ.get("CLOUD_REPO_URL", "")
        if not repo_url:
            print("Error: Could not determine repo URL.")
            sys.exit(1)

    # Build config overrides dict (only include non-default values)
    config_overrides = {}
    if learning_rate > 0:
        config_overrides["learning_rate"] = learning_rate
    if num_epochs > 0:
        config_overrides["num_epochs"] = num_epochs
    if batch_size > 0:
        config_overrides["batch_size"] = batch_size
    if max_steps > 0:
        config_overrides["max_steps"] = max_steps

    # Display job configuration
    estimated_cost = GPU_PRICING.get(gpu, 0) * timeout_hours
    print("\n" + "=" * 60)
    print("MODAL CLOUD TRAINING")
    print("=" * 60)
    print(f"  Trainer:     {trainer_type.upper()}")
    print(f"  GPU:         {gpu}")
    print(f"  Timeout:     {timeout_hours}h")
    print(f"  Est. Cost:   ~${estimated_cost:.2f} (max)")
    print(f"  Repo:        {repo_url}")
    print(f"  Branch:      {repo_branch}")
    if model_name:
        print(f"  Model:       {model_name}")
    if dataset_path:
        print(f"  Dataset:     {dataset_path}")
    if hub_repo:
        print(f"  Hub Repo:    {hub_repo}")
    if config_overrides:
        print(f"  Overrides:   {config_overrides}")
    print("=" * 60 + "\n")

    # Override the GPU and timeout for this specific invocation.
    # Modal's `with_options` allows runtime overrides of function config.
    training_fn = run_training.with_options(
        gpu=gpu,
        timeout=timeout_hours * HOURS,
    )

    # Dispatch to Modal (this call blocks until training completes)
    print("[Modal] Submitting training job...")
    result = training_fn.remote(
        trainer_type=trainer_type,
        repo_url=repo_url,
        repo_branch=repo_branch,
        model_name=model_name,
        dataset_path=dataset_path,
        hub_repo=hub_repo,
        config_overrides=config_overrides,
    )

    print(f"\n[Modal] Job result: {result}")
    print("[Modal] Done.")
