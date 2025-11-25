"""Interactive CLI for model evaluations with LM Studio or vLLM backends."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .cli_utils import (
    build_metadata,
    build_settings_kwargs,
    color,
    count_behavior_patterns,
    count_prompts,
    determine_exit_code,
    model_output_paths,
    passfail_color,
    print_banner,
    print_record_progress,
    prompt_run_count,
    select_model,
)
from .config import (
    EvaluatorConfig,
    LMStudioSettings,
    VLLMSettings,
    PromptFilter,
    expand_path,
)
from .enums import BackendType
from .lmstudio_client import LMStudioClient
from .vllm_client import VLLMClient
from .prompt_sets import load_prompt_cases
from .reporting import build_run_payload, console_summary, render_markdown, write_json
from .runner import evaluate_cases

# Default paths
DEFAULT_PROMPT_SET = Path(__file__).resolve().parent / "prompts" / "full_coverage.json"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive model evaluator with LM Studio or vLLM backends.",
        epilog="Just run `python -m Evaluator` to pick a backend, model, and evaluation.",
    )
    parser.add_argument("--backend", choices=["lmstudio", "vllm"],
                        help="Backend to use (defaults to interactive selection)")
    parser.add_argument("--host", help="Server host (defaults to env var or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Server port (defaults to env var)")
    parser.add_argument("--model", help="Optional model ID to skip the selection prompt.")
    parser.add_argument("--model-path", help="Path to local model (vLLM only)")
    parser.add_argument("--runs", type=int, help="How many times to run the suite (default: ask).")
    parser.add_argument("--prompt-set", default=str(DEFAULT_PROMPT_SET),
                        help="Prompt set to use (default: full_coverage.json).")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="Directory for JSON/MD artifacts.")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="HTTP timeout in seconds (default: 120 for vLLM)")
    parser.add_argument("--retries", type=int, default=2, help="Retry attempts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip backend calls (schema validation only).")
    parser.add_argument("--validate-context", action="store_true",
                        help="Validate that model uses IDs from system prompt")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Main entry point for interactive CLI."""
    args = parse_args(argv or sys.argv[1:])

    print_banner("Model Evaluator", "Interactive CLI")

    # Select backend
    backend = _select_backend(args.backend)
    if backend is None:
        return 1

    # Route to appropriate handler
    if backend == BackendType.VLLM:
        return _run_vllm_evaluation(args)
    else:
        return _run_lmstudio_evaluation(args)


def _select_backend(preset: Optional[str]) -> Optional[BackendType]:
    """Let user select backend or use preset.

    Args:
        preset: Preset backend from CLI args

    Returns:
        Selected BackendType or None on error
    """
    if preset:
        try:
            return BackendType(preset.lower())
        except ValueError:
            print(f"Unknown backend: {preset}", file=sys.stderr)
            return None

    print(color("\nSelect inference backend:", "magenta"))
    print(f"{color('[1]', 'yellow')} LM Studio")
    print(f"     {color('OpenAI-compatible local server (requires LM Studio running)', 'cyan')}")
    print(f"{color('[2]', 'yellow')} vLLM")
    print(f"     {color('High-performance inference (auto-starts server, loads from training outputs)', 'cyan')}")

    while True:
        choice = input("\nEnter a number (default 1): ").strip()
        if not choice or choice == "1":
            return BackendType.LMSTUDIO
        if choice == "2":
            return BackendType.VLLM
        print("Please enter 1 or 2.", file=sys.stderr)


# ---------------------------------------------------------------------------
# LM Studio Evaluation
# ---------------------------------------------------------------------------

def _run_lmstudio_evaluation(args: argparse.Namespace) -> int:
    """Run evaluation using LM Studio backend."""
    print(color("\n--- LM Studio Backend ---", "cyan"))

    # Select prompt set
    prompt_set_selection = _select_prompt_set() if args.prompt_set == str(DEFAULT_PROMPT_SET) else args.prompt_set

    # Handle "Run All" option
    if isinstance(prompt_set_selection, list):
        prompt_set_paths = [expand_path(p) for p in prompt_set_selection]
        run_all_suites = True
    else:
        prompt_set_paths = [expand_path(prompt_set_selection)]
        run_all_suites = False

    results_dir = expand_path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Validate prompt paths
    for prompt_path in prompt_set_paths:
        if not prompt_path.exists():
            print(f"Prompt set not found: {prompt_path}", file=sys.stderr)
            return 1

    settings_kwargs = build_settings_kwargs(args)

    # Create client for model listing
    list_client = LMStudioClient(
        settings=LMStudioSettings(model="__list__", **settings_kwargs),
        timeout=args.timeout,
        retries=args.retries,
    )

    # Check if LM Studio is running
    if not list_client.is_server_running():
        print(color("\nLM Studio server is not running!", "red"))
        print("Please start LM Studio and load a model, then try again.")
        return 1

    # Select model
    model_name = args.model or select_model(list_client)
    if not model_name:
        return 1

    run_count = args.runs if args.runs and args.runs > 0 else prompt_run_count()

    settings = LMStudioSettings(
        model=model_name,
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
        seed=None,
        **settings_kwargs,
    )
    client = LMStudioClient(settings=settings, timeout=args.timeout, retries=args.retries)

    return _run_evaluation_loop(
        client=client,
        settings=settings,
        prompt_set_paths=prompt_set_paths,
        run_all_suites=run_all_suites,
        results_dir=results_dir,
        run_count=run_count,
        args=args,
        backend_name="lmstudio",
    )


# ---------------------------------------------------------------------------
# vLLM Evaluation
# ---------------------------------------------------------------------------

def _run_vllm_evaluation(args: argparse.Namespace) -> int:
    """Run evaluation using vLLM backend."""
    from . import vllm_setup

    print(color("\n--- vLLM Backend ---", "cyan"))

    # Check vLLM status
    status = vllm_setup.get_vllm_status()

    # Display status
    print(f"\nvLLM installed: {color('Yes' if status.is_installed else 'No', 'green' if status.is_installed else 'red')}")
    if status.version:
        print(f"Version: {status.version}")
    print(f"CUDA available: {color('Yes' if status.cuda_available else 'No', 'green' if status.cuda_available else 'red')}")
    if status.cuda_available:
        print(f"GPU: {vllm_setup.format_gpu_info(status)}")

    # Install vLLM if needed
    if not status.is_installed:
        if not _prompt_install_vllm():
            return 1
        # Re-check status
        status = vllm_setup.get_vllm_status()
        if not status.is_installed:
            print(color("vLLM installation failed.", "red"))
            return 1

    # Check CUDA
    if not status.cuda_available:
        print(color("\nWarning: CUDA not available. vLLM requires a CUDA GPU.", "yellow"))
        print("vLLM may not work correctly without CUDA.")
        if not _prompt_continue():
            return 1

    # Select prompt set
    prompt_set_selection = _select_prompt_set() if args.prompt_set == str(DEFAULT_PROMPT_SET) else args.prompt_set

    if isinstance(prompt_set_selection, list):
        prompt_set_paths = [expand_path(p) for p in prompt_set_selection]
        run_all_suites = True
    else:
        prompt_set_paths = [expand_path(prompt_set_selection)]
        run_all_suites = False

    results_dir = expand_path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for prompt_path in prompt_set_paths:
        if not prompt_path.exists():
            print(f"Prompt set not found: {prompt_path}", file=sys.stderr)
            return 1

    # Select model
    model_path, model_name, lora_path = _select_vllm_model(args, status)
    if not model_path:
        return 1

    run_count = args.runs if args.runs and args.runs > 0 else prompt_run_count()

    # Start vLLM server if not already running
    server_started = False
    settings_kwargs = build_settings_kwargs(args)
    host = settings_kwargs.get("host", "127.0.0.1")
    port = settings_kwargs.get("port", 8000)

    if not status.server_running:
        print(color(f"\nStarting vLLM server with model: {model_path}", "cyan"))

        lora_modules = None
        if lora_path:
            lora_modules = {"finetuned": str(lora_path)}

        if not vllm_setup.start_vllm_server(
            model=str(model_path),
            host=host,
            port=port,
            lora_modules=lora_modules,
            timeout=180,  # 3 minutes for large models
        ):
            print(color("Failed to start vLLM server.", "red"))
            return 1
        server_started = True

        # Use LoRA adapter name if we loaded one
        if lora_path:
            model_name = "finetuned"
    else:
        print(color(f"\nvLLM server already running at {status.server_url}", "green"))

    settings = VLLMSettings(
        model=model_name,
        host=host,
        port=port,
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
        model_path=str(model_path) if model_path else None,
        lora_adapter=str(lora_path) if lora_path else None,
    )
    client = VLLMClient(settings=settings, timeout=args.timeout, retries=args.retries)

    try:
        return _run_evaluation_loop(
            client=client,
            settings=settings,
            prompt_set_paths=prompt_set_paths,
            run_all_suites=run_all_suites,
            results_dir=results_dir,
            run_count=run_count,
            args=args,
            backend_name="vllm",
        )
    finally:
        # Stop server if we started it
        if server_started:
            print(color("\nStopping vLLM server...", "cyan"))
            vllm_setup.stop_vllm_server()


def _prompt_install_vllm() -> bool:
    """Prompt user to install vLLM.

    Returns:
        True if user wants to continue (installed or skipped)
    """
    from . import vllm_setup

    print(color("\nvLLM is not installed.", "yellow"))
    print("vLLM is required for high-performance local inference.")
    print("Installation requires ~2GB and takes a few minutes.")

    while True:
        choice = input("\nInstall vLLM now? [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            return vllm_setup.install_vllm()
        if choice in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def _prompt_continue() -> bool:
    """Prompt user to continue despite warnings.

    Returns:
        True if user wants to continue
    """
    while True:
        choice = input("\nContinue anyway? [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def _select_vllm_model(
    args: argparse.Namespace,
    status,
) -> Tuple[Optional[Path], str, Optional[Path]]:
    """Select model for vLLM evaluation.

    Returns:
        Tuple of (model_path, model_name, lora_path)
    """
    from . import vllm_setup

    # If model-path provided via CLI
    if args.model_path:
        path = Path(args.model_path)
        if not path.exists():
            print(f"Model path not found: {path}", file=sys.stderr)
            return None, "", None
        name = path.name
        return path, name, None

    # If server is already running, list models from it
    if status.server_running:
        print(color("\nSelect model from running vLLM server:", "magenta"))
        settings = VLLMSettings(model="__list__")
        client = VLLMClient(settings=settings, timeout=30, retries=2)
        try:
            models = client.list_models()
            if models:
                for idx, model in enumerate(models, 1):
                    print(f"{color(f'[{idx}]', 'yellow')} {model}")
                while True:
                    choice = input("\nEnter a number (default 1): ").strip()
                    if not choice:
                        return Path("."), models[0], None
                    try:
                        idx = int(choice)
                        if 1 <= idx <= len(models):
                            return Path("."), models[idx - 1], None
                    except ValueError:
                        pass
                    print("Please enter a valid number.")
        except Exception as e:
            print(f"Could not list models: {e}")

    # Discover training runs
    training_runs = vllm_setup.discover_training_runs()

    print(color("\nSelect model source:", "magenta"))
    print(f"{color('[1]', 'yellow')} Training output (local fine-tuned model)")
    print(f"     {color(f'{len(training_runs)} training runs found', 'cyan')}")
    print(f"{color('[2]', 'yellow')} HuggingFace model")
    print(f"     {color('Download and run a pretrained model', 'cyan')}")
    print(f"{color('[3]', 'yellow')} Custom path")
    print(f"     {color('Enter a local path to model directory', 'cyan')}")

    while True:
        choice = input("\nEnter a number (default 1): ").strip()
        if not choice or choice == "1":
            return _select_training_run(training_runs)
        if choice == "2":
            return _select_huggingface_model()
        if choice == "3":
            return _select_custom_path()
        print("Please enter 1, 2, or 3.")


def _select_training_run(
    training_runs: List,
) -> Tuple[Optional[Path], str, Optional[Path]]:
    """Select from discovered training runs.

    Returns:
        Tuple of (model_path, model_name, lora_path)
    """
    if not training_runs:
        print(color("\nNo training runs found!", "yellow"))
        print("Train a model first using Trainers/rtx3090_sft or rtx3090_kto.")
        return None, "", None

    print(color("\nAvailable training runs:", "magenta"))
    for idx, run in enumerate(training_runs, 1):
        status_parts = []
        if run.has_merged_16bit:
            status_parts.append(color("merged-16bit", "green"))
        if run.has_final_model:
            status_parts.append(color("LoRA", "cyan"))

        status_str = ", ".join(status_parts) if status_parts else color("incomplete", "red")
        print(f"{color(f'[{idx}]', 'yellow')} {run.display_name}")
        print(f"     Path: {run.path}")
        print(f"     Available: {status_str}")

    while True:
        choice = input("\nEnter a number: ").strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(training_runs):
                run = training_runs[idx - 1]
                model_path = run.best_model_path
                if not model_path:
                    print(color("No usable model found in this run.", "red"))
                    continue

                # Check if we should use LoRA
                lora_path = None
                if run.has_lora and not run.has_merged_16bit:
                    # Need base model + LoRA
                    print(color("\nThis run only has LoRA adapters.", "yellow"))
                    print("You'll need to specify the base model.")
                    base = _select_huggingface_model()
                    if base[0]:
                        return base[0], base[1], run.lora_path
                    return None, "", None

                return model_path, run.display_name, lora_path
        except ValueError:
            pass
        print("Please enter a valid number.")


def _select_huggingface_model() -> Tuple[Optional[Path], str, Optional[Path]]:
    """Select a HuggingFace model.

    Returns:
        Tuple of (model_path, model_name, lora_path)
    """
    from . import vllm_setup

    models = vllm_setup.discover_huggingface_models()

    print(color("\nRecommended HuggingFace models:", "magenta"))
    for idx, model in enumerate(models, 1):
        print(f"{color(f'[{idx}]', 'yellow')} {model}")
    print(f"{color('[0]', 'yellow')} Enter custom model ID")

    while True:
        choice = input("\nEnter a number: ").strip()
        if choice == "0":
            custom = input("Enter HuggingFace model ID: ").strip()
            if custom:
                return Path(custom), custom, None
            continue
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                model = models[idx - 1]
                return Path(model), model, None
        except ValueError:
            pass
        print("Please enter a valid number.")


def _select_custom_path() -> Tuple[Optional[Path], str, Optional[Path]]:
    """Select a custom model path.

    Returns:
        Tuple of (model_path, model_name, lora_path)
    """
    while True:
        path_str = input("\nEnter path to model directory: ").strip()
        if not path_str:
            return None, "", None
        path = Path(path_str).expanduser().resolve()
        if path.exists():
            return path, path.name, None
        print(f"Path not found: {path}")


# ---------------------------------------------------------------------------
# Shared Evaluation Loop
# ---------------------------------------------------------------------------

def _run_evaluation_loop(
    client,
    settings,
    prompt_set_paths: List[Path],
    run_all_suites: bool,
    results_dir: Path,
    run_count: int,
    args: argparse.Namespace,
    backend_name: str,
) -> int:
    """Run the evaluation loop for any backend.

    Returns:
        Exit code (0 = success, 2 = failures, 3 = errors)
    """
    all_records = []
    model_name = settings.model

    for suite_idx, prompt_path in enumerate(prompt_set_paths, 1):
        try:
            base_config = EvaluatorConfig(
                prompts_path=prompt_path,
                output_path=None,
                save_markdown=True,
                filter=PromptFilter(),
                retries=args.retries,
                request_timeout=args.timeout,
                dry_run=args.dry_run,
            )
            base_config.validate()
        except Exception as exc:
            print(f"Invalid configuration: {exc}", file=sys.stderr)
            return 1

        cases = load_prompt_cases(base_config.prompts_path)
        prompt_set_name = prompt_path.stem.replace('_', ' ').title()

        if run_all_suites:
            print(color(f"\n{'='*60}", "cyan"))
            print(color(f"Test Suite {suite_idx}/{len(prompt_set_paths)}: {prompt_set_name}", "cyan"))
            print(color(f"{'='*60}", "cyan"))

        print(color(f"\nRunning evaluation for: {model_name}", "cyan"))
        print(color(f"Backend: {backend_name}", "cyan"))
        print(color(f"Test suite: {prompt_set_name} ({len(cases)} prompts)", "cyan"))
        print(color(f"Prompt file: {prompt_path}", "cyan"))
        print(color(f"Runs: {run_count}\n", "cyan"))

        for idx in range(run_count):
            suffix = f"_{prompt_path.stem}" if run_all_suites else ""
            json_path, md_path = model_output_paths(
                model_name, results_dir, idx, run_count, suffix=suffix
            )

            config = EvaluatorConfig(
                prompts_path=base_config.prompts_path,
                output_path=json_path,
                save_markdown=True,
                filter=base_config.filter,
                retries=base_config.retries,
                request_timeout=base_config.request_timeout,
                dry_run=base_config.dry_run,
            )
            config.ensure_output_parent()
            md_path.parent.mkdir(parents=True, exist_ok=True)

            print(color(f"--- Run {idx + 1}/{run_count} ---", "magenta"))
            records = evaluate_cases(
                cases,
                client=client,
                dry_run=config.dry_run,
                on_record=print_record_progress,
                validate_context=args.validate_context,
            )
            all_records.extend(records)

            metadata = build_metadata(config, settings, len(cases), len(cases), backend=backend_name)
            payload = build_run_payload(records, metadata=metadata)
            write_json(config.output_path, payload)
            md_path.write_text(render_markdown(records), encoding="utf-8")

            print(color(console_summary(records), passfail_color(records)))
            print(color(f"JSON: {json_path}", "yellow"))
            print(color(f"Markdown: {md_path}\n", "yellow"))

    return determine_exit_code(all_records)


# ---------------------------------------------------------------------------
# Prompt Set Selection
# ---------------------------------------------------------------------------

def _select_prompt_set() -> Union[str, List[str]]:
    """Let user choose which test suite to run.

    Returns:
        Path string or list of paths for 'all'.
    """
    paths = {
        "1": "Evaluator/prompts/behavior_rubric.json",
        "2": "Evaluator/prompts/full_coverage.json",
        "3": "Evaluator/prompts/baseline.json",
        "4": "Evaluator/prompts/tool_combos.json",
    }

    counts = {k: count_prompts(v) for k, v in paths.items()}
    behavior_pattern_count = count_behavior_patterns(paths["1"])
    total_count = sum(counts.values())

    prompt_sets = {
        "1": {
            "name": "Behavior Rubric Tests",
            "path": paths["1"],
            "desc": f"{counts['1']} prompts testing {behavior_pattern_count} behavior patterns (Recommended)",
        },
        "2": {
            "name": "Full Tool Coverage",
            "path": paths["2"],
            "desc": f"{counts['2']} prompts - one test per tool",
        },
        "3": {
            "name": "Baseline Tests",
            "path": paths["3"],
            "desc": f"{counts['3']} general prompts with behavior expectations",
        },
        "4": {
            "name": "Multi-Step Workflows",
            "path": paths["4"],
            "desc": f"{counts['4']} prompts testing complex tool sequences",
        },
        "5": {
            "name": "Run All Tests",
            "path": "ALL",
            "desc": f"Run all 4 test suites sequentially ({total_count} total prompts)",
        },
    }

    print(color("\nSelect test suite:", "magenta"))
    for key in sorted(prompt_sets.keys()):
        pset = prompt_sets[key]
        print(f"{color(f'[{key}]', 'yellow')} {pset['name']}")
        print(f"     {color(pset['desc'], 'cyan')}")

    while True:
        choice = input("\nEnter a number (default 1): ").strip()
        if not choice:
            return prompt_sets["1"]["path"]
        if choice in prompt_sets:
            selected = prompt_sets[choice]
            print(color(f"Selected: {selected['name']}", "green"))
            if selected["path"] == "ALL":
                return [
                    prompt_sets["1"]["path"],
                    prompt_sets["2"]["path"],
                    prompt_sets["3"]["path"],
                    prompt_sets["4"]["path"],
                ]
            return selected["path"]
        print("Please enter a valid option (1-5).", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
