#!/usr/bin/env python3
"""Interactive launcher for the RTX 3090 SFT trainer."""

import argparse
import io
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

from configs.config_loader import load_config
from huggingface_hub import HfApi

# -- CLI -------------------------------------------------------------------- #


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the RTX 3090 SFT trainer.",
        epilog="Run `python3 train.py` to load the default config and start training.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the training config YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Run setup only (no training).")
    parser.add_argument("-y", "--auto-confirm", action="store_true", help="Skip the confirmation prompt.")
    return parser.parse_args(argv)


# -- Preflight + Formatting -------------------------------------------------- #

ANSI_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "red": "\033[31m",
}


def supports_ansi() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def color(text: str, name: str) -> str:
    if not supports_ansi():
        return text
    start = ANSI_CODES.get(name, "")
    end = ANSI_CODES["reset"] if start else ""
    return f"{start}{text}{end}"


def print_banner() -> None:
    width = 44
    top = "+" + "=" * (width - 2) + "+"
    mid1 = "|" + "RTX 3090 SFT Trainer".center(width - 2) + "|"
    mid2 = "|" + "Interactive Launcher".center(width - 2) + "|"
    lines = [top, mid1, mid2, top]
    if supports_ansi():
        lines = [color(line, "magenta") for line in lines]
    print("\n".join(lines) + "\n")




def run_pip(args: list[str], label: str) -> bool:
    cmd = [sys.executable, "-m", "pip"] + args
    print(color(f"Installing {label}: {' '.join(args)}", "cyan"))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(color(f"pip install for {label} failed (exit {proc.returncode})", "red"))
        return False
    return True


TORCH_DEFAULT_SPEC = "torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121"
TORCH_DEFAULT_INDEX = "https://download.pytorch.org/whl/cu121"

def install_cuda_stack() -> bool:
    spec = os.getenv("TORCH_CUDA_SPEC", TORCH_DEFAULT_SPEC)
    index = os.getenv("TORCH_CUDA_INDEX", TORCH_DEFAULT_INDEX)
    args = spec.split() + ["-f", index]
    ok = run_pip(args, "CUDA PyTorch stack")
    if not ok:
        return False
    return run_pip(["unsloth[cuda]"], "Unsloth (CUDA)")


def ensure_cuda_tooling() -> bool:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return True
        print(color("PyTorch found but CUDA not available; installing CUDA build...", "yellow"))
    except Exception:
        print(color("PyTorch not found; installing CUDA build...", "yellow"))

    if not install_cuda_stack():
        return False

    try:
        import torch  # type: ignore
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(color(f"PyTorch import failed after install: {exc}", "red"))
        return False

    if not torch.cuda.is_available():
        print(color("CUDA still unavailable after install. Check NVIDIA drivers or pick a GPU runtime.", "red"))
        return False

    try:
        import unsloth  # type: ignore  # noqa: F401
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(color(f"Unsloth import failed after install: {exc}", "red"))
        return False

    return True

def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def dataset_summary(config, base_dir: Path) -> Tuple[str, Path | None]:
    dataset_path = None
    if getattr(config.dataset, "local_file", None):
        dataset_path = resolve_path(config.dataset.local_file, base_dir)
        label = f"Local file ({config.dataset.local_file})"
    else:
        label = f"HF: {config.dataset.dataset_name}"
        if getattr(config.dataset, "dataset_file", None):
            label += f" :: {config.dataset.dataset_file}"
    return label, dataset_path


def preflight_checks(config, base_dir: Path) -> Tuple[List[Tuple[str, str, str]], bool]:
    """Return (check list, has_errors)."""
    checks: List[Tuple[str, str, str]] = []
    has_errors = False

    # Dataset
    dataset_label, dataset_path = dataset_summary(config, base_dir)
    if dataset_path:
        if dataset_path.exists():
            checks.append(("Dataset file", str(dataset_path), "ok"))
        else:
            checks.append(("Dataset file", f"Missing: {dataset_path}", "error"))
            has_errors = True
    else:
        checks.append(("Dataset source", dataset_label, "ok"))

    # Output directory
    output_dir = resolve_path(config.training.output_dir, base_dir)
    if output_dir.exists():
        checks.append(("Output directory", str(output_dir), "ok"))
    else:
        parent = output_dir.parent
        if parent.exists() and os.access(parent, os.W_OK):
            checks.append(("Output directory", f"Will create: {output_dir}", "ok"))
        else:
            checks.append(("Output directory", f"Cannot write to: {output_dir}", "error"))
            has_errors = True

    # PyTorch + CUDA
    try:
        import torch  # pylint: disable=import-error

        checks.append(("PyTorch", torch.__version__, "ok"))
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            mem_gb = gpu.total_memory / 1e9
            checks.append(("CUDA", f"{gpu.name} ({mem_gb:.1f} GB)", "ok"))
        else:
            checks.append(("CUDA", "CUDA not available (training expects a GPU)", "warn"))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        checks.append(("PyTorch", f"Import failed: {exc}", "error"))
        has_errors = True

    # Hugging Face token (only warn if relying on HF hub)
    if not getattr(config.dataset, "local_file", None) and getattr(config.dataset, "dataset_name", None):
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
        if not hf_token:
            checks.append(("HF token", "Not set (HF Hub access may require HF_TOKEN)", "warn"))

    # Weights & Biases
    use_wandb = getattr(config, "use_wandb", False)
    if use_wandb:
        if os.getenv("WANDB_API_KEY"):
            checks.append(("Weights & Biases", "Enabled (WANDB_API_KEY found)", "ok"))
        else:
            checks.append(("Weights & Biases", "Enabled but WANDB_API_KEY is missing", "warn"))
    else:
        checks.append(("Weights & Biases", "Disabled", "ok"))

    return checks, has_errors


def print_summary(config, config_path: Path, base_dir: Path, checks: List[Tuple[str, str, str]], dry_run: bool) -> None:
    print_banner()

    dataset_label, dataset_path = dataset_summary(config, base_dir)
    output_dir = resolve_path(config.training.output_dir, base_dir)
    effective_batch = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps

    print(color("Configuration", "cyan"))
    print(f"  Config: {config_path}")
    print(f"  Model: {config.model.model_name}")
    print(f"  Dataset: {dataset_label}")
    if dataset_path:
        print(f"  Dataset path: {dataset_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  W&B: {'on' if getattr(config, 'use_wandb', False) else 'off'}")

    print("\n" + color("Training plan", "cyan"))
    print(f"  Max seq length: {config.model.max_seq_length}")
    print(f"  Batch: {config.training.per_device_train_batch_size} x {config.training.gradient_accumulation_steps} (effective {effective_batch})")
    print(f"  LR / epochs: {config.training.learning_rate} / {config.training.num_train_epochs}")
    print(f"  LoRA r/alpha: {config.lora.r} / {config.lora.lora_alpha}")
    print(f"  Scheduler: {config.training.lr_scheduler_type}")
    print(f"  Dry run: {'yes' if dry_run else 'no'}")

    print("\n" + color("Preflight checks", "cyan"))
    for label, value, status in checks:
        color_name = {"ok": "green", "warn": "yellow", "error": "red"}.get(status, "")
        tag = {"ok": "[OK]", "warn": "[WARN]", "error": "[ERROR]"}.get(status, "[INFO]")
        print(f"  {color(tag, color_name)} {label}: {value}")
    print()


def prompt_start(auto_confirm: bool) -> bool:
    if auto_confirm:
        return True
    choice = input("Start training now? [Y/n]: ").strip().lower()
    return choice in ("", "y", "yes")


# -- Upload helpers --------------------------------------------------------- #


def _slugify(value: str) -> str:
    keep = []
    for ch in value.lower():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in ("/", " ", "."):
            keep.append("-")
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "model"


def default_repo_id(config) -> str:
    username = os.getenv("HF_USERNAME") or ""
    base = Path(str(config.model.model_name)).name
    base_slug = _slugify(base)
    repo = f"{base_slug}-sft"
    if username:
        return f"{username}/{repo}"
    return repo


def prompt_repo_id(default: str) -> str:
    val = input(f"Hugging Face repo id (username/name) [{default}]: ").strip()
    return val or default


def prompt_save_method() -> str:
    options = {
        "1": "merged_16bit",
        "2": "merged_4bit",
        "3": "lora",
    }
    print("Save method:")
    print("  [1] merged_16bit (default)")
    print("  [2] merged_4bit")
    print("  [3] lora (upload adapters only)")
    choice = input("Select (1/2/3): ").strip()
    return options.get(choice, "merged_16bit")


def prompt_private(default: bool = False) -> bool:
    val = input(f"Make repository private? [{'Y' if default else 'y'}/N]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes")


def build_model_card(repo_id: str, config, run_info: dict) -> str:
    dataset_label, dataset_path = dataset_summary(config, Path(__file__).parent)
    train_size = run_info.get("train_size")
    eval_size = run_info.get("eval_size")
    effective_batch = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps

    lines = [
        f"# {repo_id}",
        "",
        "## Model",
        f"- Base: `{config.model.model_name}`",
        f"- Max seq length: `{config.model.max_seq_length}`",
        f"- dtype: `{config.model.dtype or 'auto'}`",
        f"- 4-bit loading: `{config.model.load_in_4bit}`",
        "",
        "## Data",
        f"- Source: {dataset_label}",
        f"- Train examples: {train_size if train_size is not None else 'unknown'}",
        f"- Eval examples: {eval_size if eval_size is not None else 'none'}",
        f"- Split dataset: `{config.dataset.split_dataset}` (test_size={config.dataset.test_size})",
        f"- Filter desirable: `{config.dataset.filter_desirable}`",
        "",
        "## Training",
        f"- Effective batch size: {effective_batch} ({config.training.per_device_train_batch_size} x {config.training.gradient_accumulation_steps})",
        f"- Learning rate: {config.training.learning_rate}",
        f"- Epochs: {config.training.num_train_epochs}",
        "- Max steps: CLI override if provided",
        f"- Scheduler: {config.training.lr_scheduler_type}",
        f"- Warmup ratio: {config.training.warmup_ratio}",
        f"- Optimizer: {config.training.optim}",
        f"- Gradient checkpointing: {config.training.gradient_checkpointing}",
        f"- FP16/BF16: {config.training.fp16}/{config.training.bf16}",
        f"- Logging steps: {config.training.logging_steps}",
        f"- Save steps: {config.training.save_steps} (keep {config.training.save_total_limit})",
        f"- Seed: {config.seed}",
        "",
        "## LoRA",
        f"- r: {config.lora.r}",
        f"- alpha: {config.lora.lora_alpha}",
        f"- dropout: {config.lora.lora_dropout}",
        f"- bias: {config.lora.bias}",
        "",
        "## Artifacts",
        f"- Run directory: `{run_info.get('run_dir')}`",
        f"- Final model: `{run_info.get('final_model_dir')}`",
        f"- Logs: `{run_info.get('logs_dir')}`",
        "",
        "> Generated by the RTX 3090 SFT interactive CLI.",
    ]
    return "\n".join(lines)


def push_model_card(repo_id: str, token: str, content: str, private: bool) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)
    api.upload_file(
        path_or_fileobj=io.BytesIO(content.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )


def maybe_upload(config, run_info: dict) -> None:
    if not run_info or not run_info.get("final_model_dir"):
        print(color("Upload skipped: no run information available.", "yellow"))
        return

    choice = input("Upload merged model to Hugging Face? [y/N]: ").strip().lower()
    if choice not in ("y", "yes"):
        return

    try:
        from src.upload_to_hf import upload_standard_model  # Import lazily to avoid early Unsloth load
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(color(f"Unable to import upload helper: {exc}", "red"))
        return

    repo_id = prompt_repo_id(default_repo_id(config))
    save_method = prompt_save_method()
    private = prompt_private(default=False)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    if not hf_token:
        hf_token = input("HF token (write): ").strip()
    if not hf_token:
        print(color("HF token is required to upload. Skipping upload.", "red"))
        return

    model_path = run_info.get("final_model_dir")
    print(color(f"
Uploading {model_path} -> {repo_id} ({save_method})", "cyan"))
    upload_standard_model(
        model_path=model_path,
        repo_id=repo_id,
        hf_token=hf_token,
        save_method=save_method,
        private=private,
    )

    print(color("
Pushing model card...", "cyan"))
    card = build_model_card(repo_id, config, run_info)
    push_model_card(repo_id, hf_token, card, private)
    print(color("?? Upload complete with model card.", "green"))


# -- Main ------------------------------------------------------------------- #

def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (base_dir / config_path).resolve()

    if not ensure_cuda_tooling():
        return 1

    if not config_path.exists():
        print(color(f"Config file not found: {config_path}", "red"), file=sys.stderr)
        return 1

    try:
        config = load_config(str(config_path))
    except Exception as exc:  # pylint: disable-broad-exception-caught
        print(color(f"Unable to load config: {exc}", "red"), file=sys.stderr)
        return 1

    checks, has_errors = preflight_checks(config, base_dir)
    print_summary(config, config_path, base_dir, checks, args.dry_run)

    if has_errors:
        print(color("Fix the error(s) above before starting training.", "red"), file=sys.stderr)
        return 1

    if not prompt_start(args.auto_confirm):
        print("Aborted.")
        return 0

    try:
        import train_sft
    except Exception as exc:  # pylint: disable-broad-exception-caught
        print(color(f"Unable to import training module: {exc}", "red"), file=sys.stderr)
        return 1

    cli_args = ["--config", str(config_path)]
    if args.dry_run:
        cli_args.append("--dry-run")

    train_args = train_sft.parse_args(cli_args)
    run_info = train_sft.run(train_args)
    if not args.dry_run:
        maybe_upload(config, run_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
