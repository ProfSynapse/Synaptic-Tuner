"""LM Studio-focused CLI for evaluator runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .cli_utils import (
    build_metadata,
    build_settings_kwargs,
    determine_exit_code,
    model_output_paths,
    select_model,
)
from .config import EvaluatorConfig, LMStudioSettings, PromptFilter, expand_path
from .lmstudio_client import LMStudioClient, LMStudioError
from .prompt_sets import load_prompt_cases
from .reporting import build_run_payload, console_summary, render_markdown, write_json
from .runner import evaluate_cases

DEFAULT_PROMPT_SET = Path(__file__).resolve().parent / "prompts" / "tool_prompts.json"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full-coverage evaluations against LM Studio models.",
        epilog="Tip: run `python -m Evaluator.lmstudio_cli list-models` to see what's loaded in LM Studio.",
    )
    parser.add_argument("--host", help="LM Studio host (defaults to LMSTUDIO_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="LM Studio port (defaults to LMSTUDIO_PORT or 1234)")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retry attempts for LM Studio calls")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="List models visible to LM Studio.")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the full coverage prompt set and export JSON + Markdown artifacts.",
    )
    run_parser.add_argument("--model", help="Model ID to evaluate (if omitted, you will be prompted).")
    run_parser.add_argument(
        "--prompt-set",
        default=str(DEFAULT_PROMPT_SET),
        help="Path to a prompt set file (default: tool_prompts.json).",
    )
    run_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where artifacts are written (default: Evaluator/results).",
    )
    run_parser.add_argument("--temperature", type=float, default=0.2)
    run_parser.add_argument("--top-p", type=float, default=0.9)
    run_parser.add_argument("--max-tokens", type=int, default=1024)
    run_parser.add_argument("--seed", type=int, help="Optional generation seed")
    run_parser.add_argument("--dry-run", action="store_true", help="Skip backend calls (schema validation only).")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv or sys.argv[1:])

    if args.command == "list-models":
        return list_models_command(args)
    if args.command == "run":
        return run_full_coverage_command(args)
    print("Unknown command", file=sys.stderr)
    return 1


def list_models_command(args: argparse.Namespace) -> int:
    """List models available in LM Studio."""
    settings_kwargs = build_settings_kwargs(args)
    client = LMStudioClient(
        settings=LMStudioSettings(model="__list__", **settings_kwargs),
        timeout=args.timeout,
        retries=args.retries,
    )
    try:
        models = client.list_models()
    except LMStudioError as exc:
        print(f"Failed to list models: {exc}", file=sys.stderr)
        return 1

    print("Available LM Studio models:")
    for idx, model in enumerate(models, start=1):
        print(f"[{idx}] {model}")
    return 0


def run_full_coverage_command(args: argparse.Namespace) -> int:
    """Run full coverage evaluation."""
    prompt_path = expand_path(args.prompt_set)
    results_dir = expand_path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    settings_kwargs = build_settings_kwargs(args)

    # Create client for model selection
    base_client = LMStudioClient(
        settings=LMStudioSettings(model=args.model or "__list__", **settings_kwargs),
        timeout=args.timeout,
        retries=args.retries,
    )
    model_name = args.model or select_model(base_client)
    if not model_name:
        return 1

    settings = LMStudioSettings(
        model=model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        **settings_kwargs,
    )
    client = LMStudioClient(settings=settings, timeout=args.timeout, retries=args.retries)

    json_path, md_path = model_output_paths(model_name, results_dir)
    config = EvaluatorConfig(
        prompts_path=prompt_path,
        output_path=json_path,
        save_markdown=True,
        filter=PromptFilter(),
        retries=args.retries,
        request_timeout=args.timeout,
        dry_run=args.dry_run,
    )

    try:
        config.validate()
    except Exception as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 1
    config.ensure_output_parent()
    md_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_prompt_cases(config.prompts_path)
    records = evaluate_cases(cases, client=client, dry_run=config.dry_run)

    metadata = build_metadata(config, settings, len(cases), len(cases), backend="lmstudio")
    payload = build_run_payload(records, metadata=metadata)
    write_json(config.output_path, payload)
    md_path.write_text(render_markdown(records), encoding="utf-8")

    print(console_summary(records))
    print(f"\nJSON results: {json_path}")
    print(f"Markdown summary: {md_path}")

    return determine_exit_code(records)


if __name__ == "__main__":
    raise SystemExit(main())
